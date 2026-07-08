"""Aggregation boundary — solution = application/service (Sprint 0.3).

Identity order is **purpose → tech stack → structure**: what the code
does is the solution's identity, the framework is an attribute, and the
repo location is context. Ten LangGraph agents with ten different
purposes are ten solutions; the same flow rewritten in another agent
framework is a *separate* solution surfaced by overlap detection, never
silently merged.

Because identity comes from purpose, aggregation runs AFTER code
analysis and data-flow mapping (``aggregate_scan_result`` is called by
the pipeline once ``DataFlowMap``s exist — not inside the scanner).
Directory grouping stays the mechanism underneath.

Merging is allowed only when BOTH hold — "aggregate only where we are
100% sure":

1. **Same structural boundary** — the components sit in one subtree:
   the top-level chapter of a detected teaching repo, or the subtree of
   a subdirectory dependency manifest (one service). The repo root is
   never a boundary: a root manifest must not swallow a monorepo, and
   two identical flows in different branches are an overlap insight
   ("2 implementations of the same thing"), not one card.
2. **Same functional fingerprint** — identical DataFlowMap shape (sink
   providers + sink types + processing steps + data categories). Same
   boundary + same flow = provably variants of one thing. Components
   without a usable flow map never merge (conservative default).

The merged solution ID hashes repo + boundary + fingerprint; unmerged
components keep their directory-hash ID unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath

from aiscout.engine.repo_character import KIND_TUTORIAL, RepoCharacter
from aiscout.models import AIAsset, AssetType, FindingType, RiskStatus, ScanResult

# Directory-asset count from which a repo classified as teaching
# material activates chapter boundaries. Below this the per-directory
# view is readable as-is.
TUTORIAL_COLLAPSE_MIN_COMPONENTS = 8

_STATUS_ORDER = {
    RiskStatus.CRITICAL: 0, RiskStatus.REVIEW: 1, RiskStatus.NO_FINDINGS: 2,
}
# Merged solution type: the most specific component type wins.
_TYPE_PRIORITY = (
    AssetType.MCP_SERVER, AssetType.AGENT, AssetType.AUTOMATION,
    AssetType.LOCAL_MODEL, AssetType.CUSTOM_CODE, AssetType.COMMERCIAL_SAAS,
)


def aggregate_scan_result(result: ScanResult) -> None:
    """Fold provably-duplicate components inside one scan result.

    Mutates ``result.assets``. Call after ``build_data_flows`` — without
    flow maps nothing merges (by design).
    """
    if not result.assets:
        return
    repo_name = result.metadata.get("repository", "unknown")
    raw = result.metadata.get("repo_character") or {}
    character = RepoCharacter(
        kind=raw.get("kind", "unknown"),
        confidence=raw.get("confidence", "low"),
        signals=list(raw.get("signals", [])),
    )
    result.assets = aggregate_assets(result.assets, repo_name, character)


def aggregate_assets(
    assets: list[AIAsset],
    repo_name: str,
    character: RepoCharacter,
) -> list[AIAsset]:
    """Merge same-boundary, same-fingerprint components into variant
    groups; everything else passes through untouched."""
    if not assets:
        return assets

    tutorial = (
        character.kind == KIND_TUTORIAL
        and character.confidence in ("high", "medium")
        and len(assets) >= TUTORIAL_COLLAPSE_MIN_COMPONENTS
    )
    manifest_dirs = _manifest_dirs(assets)

    groups: dict[tuple[str, str], list[AIAsset]] = defaultdict(list)
    passthrough: list[AIAsset] = []
    for asset in assets:
        fingerprint = _flow_fingerprint(asset)
        if not fingerprint:
            passthrough.append(asset)
            continue
        boundary = _boundary(_solution_dir(asset), tutorial, manifest_dirs)
        groups[(boundary, fingerprint)].append(asset)

    result: list[AIAsset] = list(passthrough)
    for (boundary, fingerprint), group in groups.items():
        if len(group) == 1:
            result.append(group[0])
            continue
        result.append(
            _merge(group, repo_name, boundary=boundary, fingerprint=fingerprint)
        )

    result.sort(
        key=lambda a: (_STATUS_ORDER[a.risk_status], a.name.lower(), a.id)
    )
    return result


def _solution_dir(asset: AIAsset) -> str:
    """The directory-grouping key the scanner used for this asset."""
    if asset.root_path:
        return asset.root_path
    first_file = asset.file_path.split(", ")[0] if asset.file_path else ""
    parts = PurePosixPath(first_file).parts
    return str(PurePosixPath(*parts[:-1])) if len(parts) > 1 else "."


def _top_level_dir(solution_dir: str) -> str:
    if solution_dir in (".", ""):
        return "."
    return PurePosixPath(solution_dir).parts[0]


def _manifest_dirs(assets: list[AIAsset]) -> set[str]:
    """Subdirectories that carry a dependency manifest.

    Derived from the dependency findings the scanner already produced —
    no filesystem access. The repo root is deliberately EXCLUDED: a root
    manifest would make the whole monorepo one boundary and hide the
    granularity Scout exists to show (a hook-triggered agent in hooks/,
    a chatbot in services/ — distinct solutions with distinct paths).
    """
    dirs: set[str] = set()
    for asset in assets:
        for f in asset.raw_findings:
            if f.type == FindingType.DEPENDENCY_DETECTED:
                parts = PurePosixPath(f.file_path).parts
                if len(parts) > 1:
                    dirs.add(str(PurePosixPath(*parts[:-1])))
    return dirs


def _boundary(solution_dir: str, tutorial: bool, manifest_dirs: set[str]) -> str:
    """Structural boundary inside which same-flow components may merge."""
    if tutorial:
        return _top_level_dir(solution_dir)
    if solution_dir in manifest_dirs:
        return solution_dir
    for ancestor in PurePosixPath(solution_dir).parents:
        key = str(ancestor)
        if key in manifest_dirs:
            return key
    return solution_dir


def _flow_fingerprint(asset: AIAsset) -> str:
    """Functional fingerprint from the DataFlowMap (Sprint 5 shape).

    Empty string = no usable flow = the component never merges.
    """
    flow = asset.data_flow
    if not flow or (not flow.sinks and not flow.processing_steps):
        return ""
    return "|".join([
        "sp:" + ",".join(sorted({s.provider for s in flow.sinks if s.provider})),
        "st:" + ",".join(sorted({s.type for s in flow.sinks})),
        "steps:" + ",".join(sorted(set(flow.processing_steps))),
        "cat:" + ",".join(sorted(flow.data_categories)),
    ])


def _merge(
    group: list[AIAsset], repo_name: str, boundary: str, fingerprint: str
) -> AIAsset:
    """Merge provably-duplicate components into one variant group.

    The components share a functional fingerprint, so any component's
    DataFlowMap describes the group; purpose-based naming downstream
    (enrichment) therefore stays crisp instead of averaging a soup of
    unrelated code.
    """
    group = sorted(group, key=lambda a: a.file_path)
    component_dirs = sorted({_solution_dir(a) for a in group})

    files = sorted({fp for a in group for fp in a.file_path.split(", ") if fp})
    findings = [f for a in group for f in a.raw_findings]
    deps = sorted({d for a in group for d in a.dependencies})
    users = sorted({u for a in group for u in a.users})
    tags = sorted({t for a in group for t in a.tags} | {"variant_group"})
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
    base = group[0]

    # ID: repo + boundary + fingerprint — two different variant groups
    # inside one chapter stay distinct, deterministically.
    from aiscout.scanners.git_scanner import _stable_hash

    return AIAsset(
        id="sol-" + _stable_hash(repo_name, boundary, fingerprint),
        name=base.name,
        type=asset_type,
        owner=", ".join(owners) if owners else "unknown",
        users=users,
        provider=provider,
        risk_status=risk_status,
        discovered_via=sorted({d for a in group for d in a.discovered_via}),
        repository=repo_name,
        root_path=boundary,
        file_path=", ".join(files),
        dependencies=deps,
        raw_findings=findings,
        tags=tags,
        component_dirs=component_dirs,
        # Shared-fingerprint group: the first component's flow describes
        # all of them; contexts are pooled for tech-stack extraction.
        data_flow=base.data_flow,
        code_contexts=[c for a in group for c in a.code_contexts],
    )
