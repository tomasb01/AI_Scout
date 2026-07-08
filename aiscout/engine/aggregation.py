"""Aggregation boundary — solution = application/service (Sprint 0.3).

Directory grouping stays the *mechanism* (one directory with AI code =
one component); this layer decides which components belong to the same
application so the report counts applications, not folders
(Spec v13 §3.4).

Two deterministic rules, applied in order:

1. **Tutorial collapse** — when the repo character detector classified
   the repo as ``tutorial_example`` (confidence ≥ medium) and directory
   grouping produced many components, the whole repo becomes ONE
   solution ("teaching collection, N examples"). A course repo stops
   reporting as 120 pseudo-solutions; every finding keeps its file:line
   evidence inside the single solution.

2. **Manifest roots** — a component is folded into its nearest ancestor
   directory that carries a dependency manifest (requirements.txt,
   pyproject.toml, package.json, setup.py — the paths the dependency
   scanner already emits as findings). A service with api/ + workers/ +
   rag/ under one pyproject is one solution; components with no manifest
   ancestor keep their directory identity unchanged (conservative
   default — no manifest, no merge).

The solution ID is re-derived from the aggregation root
(``sol-<hash>(repo | root)``), which is the accepted one-time ID
re-baseline documented in README_BUNDLE.md and models/assets.py.
Finding IDs are location-hashed and unaffected.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath

from aiscout.engine.repo_character import KIND_TUTORIAL, RepoCharacter
from aiscout.models import AIAsset, AssetType, FindingType, RiskStatus

# Directory-asset count from which a tutorial repo collapses. Below this
# the per-directory view is still readable and collapsing loses more
# than it gains.
TUTORIAL_COLLAPSE_MIN_COMPONENTS = 8

_STATUS_ORDER = {
    RiskStatus.CRITICAL: 0, RiskStatus.REVIEW: 1, RiskStatus.NO_FINDINGS: 2,
}
# Merged solution type: the most specific component type wins.
_TYPE_PRIORITY = (
    AssetType.MCP_SERVER, AssetType.AGENT, AssetType.AUTOMATION,
    AssetType.LOCAL_MODEL, AssetType.CUSTOM_CODE, AssetType.COMMERCIAL_SAAS,
)


def aggregate_assets(
    assets: list[AIAsset],
    repo_name: str,
    character: RepoCharacter,
    stable_hash,
) -> list[AIAsset]:
    """Fold directory-level assets into application-level solutions.

    ``stable_hash`` is the scanner's hash helper so IDs stay in one
    hashing scheme (``sol-<hash12>``).
    """
    if not assets:
        return assets

    # ── Rule 1: tutorial chapter collapse ───────────────────────────────
    # A course repo explodes at the sub-example level (1-Intro/1_FNN/
    # 1_single, 1-Intro/1_FNN/2_multi, ...), but its top-level chapters
    # ARE the distinct solutions the reader needs to tell apart — each
    # lesson covers a different approach/tech stack. So sub-examples fold
    # into their top-level directory, never into one repo-wide blob:
    # AI-developer-3 reports ~12 chapter solutions, not 1 and not 144.
    if (
        character.kind == KIND_TUTORIAL
        and character.confidence in ("high", "medium")
        and len(assets) >= TUTORIAL_COLLAPSE_MIN_COMPONENTS
    ):
        by_chapter: dict[str, list[AIAsset]] = defaultdict(list)
        for asset in assets:
            by_chapter[_top_level_dir(_solution_dir(asset))].append(asset)

        result: list[AIAsset] = []
        for chapter, group in by_chapter.items():
            if len(group) == 1 and _solution_dir(group[0]) == chapter:
                result.append(group[0])
                continue
            merged = _merge(
                group, repo_name, root=chapter, stable_hash=stable_hash
            )
            if len(group) > 1:
                merged.name = f"{merged.name} ({len(group)} examples)"
            merged.tags = sorted(set(merged.tags) | {"tutorial_collection"})
            result.append(merged)

        result.sort(
            key=lambda a: (_STATUS_ORDER[a.risk_status], a.name.lower(), a.id)
        )
        return result

    # ── Rule 2: manifest roots ──────────────────────────────────────────
    manifest_dirs = _manifest_dirs(assets)
    by_root: dict[str, list[AIAsset]] = defaultdict(list)
    for asset in assets:
        by_root[_aggregation_root(_solution_dir(asset), manifest_dirs)].append(asset)

    result: list[AIAsset] = []
    for root, group in by_root.items():
        if len(group) == 1 and _solution_dir(group[0]) == root:
            # Nothing folded — keep the directory asset untouched (and its
            # ID stable relative to pre-0.3 scans).
            result.append(group[0])
            continue
        result.append(_merge(group, repo_name, root=root, stable_hash=stable_hash))

    result.sort(
        key=lambda a: (_STATUS_ORDER[a.risk_status], a.name.lower(), a.id)
    )
    return result


def _solution_dir(asset: AIAsset) -> str:
    """The directory-grouping key the scanner used for this asset."""
    first_file = asset.file_path.split(", ")[0] if asset.file_path else ""
    parts = PurePosixPath(first_file).parts
    return str(PurePosixPath(*parts[:-1])) if len(parts) > 1 else "."


def _top_level_dir(solution_dir: str) -> str:
    """Top-level chapter directory for the tutorial collapse."""
    if solution_dir in (".", ""):
        return "."
    return PurePosixPath(solution_dir).parts[0]


def _manifest_dirs(assets: list[AIAsset]) -> set[str]:
    """Directories that carry a dependency manifest.

    Derived from the dependency findings the scanner already produced —
    no filesystem access, works identically for local and remote scans.
    """
    dirs: set[str] = set()
    for asset in assets:
        for f in asset.raw_findings:
            if f.type == FindingType.DEPENDENCY_DETECTED:
                parts = PurePosixPath(f.file_path).parts
                dirs.add(str(PurePosixPath(*parts[:-1])) if len(parts) > 1 else ".")
    return dirs


def _aggregation_root(solution_dir: str, manifest_dirs: set[str]) -> str:
    """Nearest manifest ancestor (or the directory itself)."""
    if solution_dir in manifest_dirs:
        return solution_dir
    path = PurePosixPath(solution_dir)
    for ancestor in path.parents:
        key = str(ancestor)
        if key in manifest_dirs:
            return key
    return solution_dir


def _merge(
    group: list[AIAsset], repo_name: str, root: str, stable_hash
) -> AIAsset:
    """Merge component assets into one application-level solution."""
    group = sorted(group, key=lambda a: a.file_path)
    component_dirs = sorted({_solution_dir(a) for a in group})

    files = sorted({fp for a in group for fp in a.file_path.split(", ") if fp})
    findings = [f for a in group for f in a.raw_findings]
    deps = sorted({d for a in group for d in a.dependencies})
    users = sorted({u for a in group for u in a.users})
    tags = sorted({t for a in group for t in a.tags})
    owners = sorted({
        o.strip() for a in group for o in a.owner.split(",")
        if o.strip() and o.strip() != "unknown"
    })

    asset_type = next(
        (t for t in _TYPE_PRIORITY if any(a.type == t for a in group)),
        AssetType.CUSTOM_CODE,
    )
    risk_status = min((a.risk_status for a in group), key=_STATUS_ORDER.get)
    provider = next((a.provider for a in group if a.provider), None)

    # Name: root directory name for manifest merges; the tutorial
    # collapse overwrites this with the collection label. Lazy import —
    # the scanner imports this module inside scan(), so a module-level
    # import back into the scanner would be circular.
    from aiscout.scanners.git_scanner import _clean_dir_name

    if root in (".", ""):
        name = repo_name
    else:
        name = _clean_dir_name(PurePosixPath(root).name) or repo_name

    return AIAsset(
        id="sol-" + stable_hash(repo_name, root),
        name=name,
        type=asset_type,
        owner=", ".join(owners) if owners else "unknown",
        users=users,
        provider=provider,
        risk_status=risk_status,
        discovered_via=sorted({d for a in group for d in a.discovered_via}),
        repository=repo_name,
        file_path=", ".join(files),
        dependencies=deps,
        raw_findings=findings,
        tags=tags,
        component_dirs=component_dirs,
        # code_contexts / data_flow are attached by the analysis stages,
        # which run after aggregation and see the merged file list.
    )
