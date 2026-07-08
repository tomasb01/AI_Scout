"""Tests for Sprint 0.3 — repo character detector + aggregation boundary.

Solution = application/service: directory grouping stays the mechanism,
the aggregation layer folds components into applications (manifest
roots) and collapses teaching repos into one collection so a course
with 100 lesson folders stops reporting as 100 AI solutions.
"""

from pathlib import Path

from aiscout.engine.aggregation import aggregate_assets
from aiscout.engine.enrichment import enrich_assets
from aiscout.engine.repo_character import (
    KIND_EXPERIMENT,
    KIND_PRODUCTION,
    KIND_TUTORIAL,
    KIND_UNKNOWN,
    RepoCharacter,
    detect_repo_character,
)
from aiscout.models import AIAsset, Finding, FindingType, RiskStatus
from aiscout.report.insights import build_insights, collect_stats
from aiscout.scanners.git_scanner import GitScanner, _stable_hash

FIXTURES_TUTORIAL = Path(__file__).parent / "fixtures_tutorial"


# ── Repo character detector ────────────────────────────────────────────────


def test_detects_tutorial_repo():
    files = [f"{i:02d}-lesson-x/example.py" for i in range(1, 11)]
    dirs = [f"{i:02d}-lesson-x" for i in range(1, 11)]
    result = detect_repo_character(
        files, dirs, readme_text="# My Course\nA step-by-step tutorial."
    )
    assert result.kind == KIND_TUTORIAL
    assert result.confidence == "high"
    assert "sequential_dir_naming" in result.signals


def test_production_signals_veto_tutorial():
    """A course-shaped repo that ships CI + tests stays out of the
    tutorial bucket — misclassifying production is the expensive error."""
    files = [f"{i:02d}-lesson-x/example.py" for i in range(1, 11)]
    files += [".github/workflows/ci.yml", "tests/test_app.py", "Dockerfile"]
    dirs = [f"{i:02d}-lesson-x" for i in range(1, 11)]
    result = detect_repo_character(files, dirs, readme_text="# Tutorial")
    assert result.kind != KIND_TUTORIAL


def test_nested_deploy_artifacts_do_not_veto_tutorial():
    """Validated on AI-developer-3 / AI-Agents-2: lessons ship lockfiles,
    Dockerfiles and tests as teaching material — deploy evidence counts
    only at repo level."""
    files = [f"{i:02d}-lesson-x/example.py" for i in range(1, 11)]
    files += [
        "03-lesson-x/uv.lock", "05-lesson-x/Dockerfile",
        "07-lesson-x/test/test_tool.py",
    ]
    dirs = [f"{i:02d}-lesson-x" for i in range(1, 11)]
    result = detect_repo_character(files, dirs, readme_text="# AI Course")
    assert result.kind == KIND_TUTORIAL


def test_shape_alone_never_collapses_a_real_app():
    """Validated on Fleurdin_AI: numbered pipeline folders (0-Scripts,
    4-RAG_Pipeline, 5-Backend) are a real-app convention — without a
    semantic teaching signal the repo must not classify as tutorial."""
    files = [f"{i}-stage/main.py" for i in range(9)]
    dirs = [f"{i}-stage" for i in range(9)]
    result = detect_repo_character(files, dirs, readme_text="")
    assert result.kind != KIND_TUTORIAL


def test_detects_production_repo():
    files = [
        "api/app.py", "api/llm.py", ".github/workflows/deploy.yml",
        "tests/test_api.py", "Dockerfile", "uv.lock",
    ]
    result = detect_repo_character(files, ["api"])
    assert result.kind == KIND_PRODUCTION
    assert result.confidence == "high"


def test_detects_experiment_repo():
    files = ["scratch.ipynb", "try_rag.ipynb", "utils.py"]
    result = detect_repo_character(files, ["."])
    assert result.kind == KIND_EXPERIMENT


def test_weak_evidence_stays_unknown():
    result = detect_repo_character(["bot/main.py"], ["bot"])
    assert result.kind == KIND_UNKNOWN
    assert result.confidence == "low"


# ── Aggregation: manifest roots ────────────────────────────────────────────


def _dir_asset(repo: str, solution_dir: str, files: list[str],
               findings: list[Finding] | None = None) -> AIAsset:
    return AIAsset(
        id="sol-" + _stable_hash(repo, solution_dir),
        name=solution_dir,
        repository=repo,
        file_path=", ".join(files),
        raw_findings=findings or [],
    )


def _dep_finding(manifest_path: str) -> Finding:
    return Finding(
        type=FindingType.DEPENDENCY_DETECTED,
        file_path=manifest_path, content="openai>=1.0", provider="openai",
    )


def test_components_fold_into_manifest_root():
    repo = "svc"
    assets = [
        _dir_asset(repo, "app", ["app/pyproject.toml"],
                   [_dep_finding("app/pyproject.toml")]),
        _dir_asset(repo, "app/workers", ["app/workers/embed.py"]),
        _dir_asset(repo, "app/rag", ["app/rag/retrieve.py"]),
    ]
    merged = aggregate_assets(
        assets, repo, RepoCharacter(KIND_PRODUCTION, "high"), _stable_hash
    )
    assert len(merged) == 1
    solution = merged[0]
    assert solution.id == "sol-" + _stable_hash(repo, "app")
    assert solution.component_dirs == ["app", "app/rag", "app/workers"]
    assert "app/workers/embed.py" in solution.file_path


def test_no_manifest_ancestor_keeps_directory_identity():
    """Conservative default: no manifest, no merge — and the pre-0.3
    solution ID stays byte-identical."""
    repo = "misc"
    assets = [
        _dir_asset(repo, "bot", ["bot/main.py"]),
        _dir_asset(repo, "tools", ["tools/summarize.py"]),
    ]
    merged = aggregate_assets(
        assets, repo, RepoCharacter(KIND_UNKNOWN, "low"), _stable_hash
    )
    assert {a.id for a in merged} == {a.id for a in assets}
    assert all(a.component_dirs == [] for a in merged)


def test_sibling_manifests_stay_separate():
    repo = "monorepo"
    assets = [
        _dir_asset(repo, "svc-a", ["svc-a/requirements.txt"],
                   [_dep_finding("svc-a/requirements.txt")]),
        _dir_asset(repo, "svc-a/handlers", ["svc-a/handlers/llm.py"]),
        _dir_asset(repo, "svc-b", ["svc-b/requirements.txt"],
                   [_dep_finding("svc-b/requirements.txt")]),
    ]
    merged = aggregate_assets(
        assets, repo, RepoCharacter(KIND_PRODUCTION, "high"), _stable_hash
    )
    assert len(merged) == 2
    roots = sorted(a.id for a in merged)
    assert roots == sorted([
        "sol-" + _stable_hash(repo, "svc-a"),
        "sol-" + _stable_hash(repo, "svc-b"),
    ])


def test_merge_takes_worst_risk_status():
    repo = "svc"
    critical = _dir_asset(repo, "app/keys", ["app/keys/leak.py"])
    critical.risk_status = RiskStatus.CRITICAL
    assets = [
        _dir_asset(repo, "app", ["app/requirements.txt"],
                   [_dep_finding("app/requirements.txt")]),
        critical,
    ]
    merged = aggregate_assets(
        assets, repo, RepoCharacter(KIND_PRODUCTION, "high"), _stable_hash
    )
    assert len(merged) == 1
    assert merged[0].risk_status == RiskStatus.CRITICAL


# ── Aggregation: tutorial collapse ─────────────────────────────────────────


def test_tutorial_repo_collapses_below_threshold_does_not():
    repo = "course"
    assets = [
        _dir_asset(repo, f"{i:02d}-lesson", [f"{i:02d}-lesson/ex.py"])
        for i in range(1, 5)  # only 4 components — keep them visible
    ]
    merged = aggregate_assets(
        assets, repo, RepoCharacter(KIND_TUTORIAL, "high"), _stable_hash
    )
    assert len(merged) == 4


def test_tutorial_collapse_end_to_end_on_fixture():
    """QA spec acceptance: the tutorial fixture reports ONE solution,
    every finding keeps its file:line evidence, and I-01 counts 1."""
    result = GitScanner(repo_path=str(FIXTURES_TUTORIAL)).scan()
    assert result.metadata["repo_character"]["kind"] == KIND_TUTORIAL

    assert len(result.assets) == 1
    collection = result.assets[0]
    assert "teaching collection (10 examples)" in collection.name
    assert "tutorial_collection" in collection.tags
    assert len(collection.component_dirs) == 10
    assert len(collection.raw_findings) == 10  # one import per lesson

    insights = enrich_assets(result.assets)
    # enrichment must not overwrite the deliberate collection label
    assert insights[collection.id].solution_name == collection.name

    stats = collect_stats(
        result.assets, insights,
        repos=1, files_scanned=result.metadata["files_scanned"],
    )
    catalog = build_insights(stats)
    i01 = next(i for i in catalog if i.id == "I-01")
    assert "1 AI solution " in i01.text  # not "10 AI solutions"


def test_tutorial_collapse_is_deterministic():
    first = GitScanner(repo_path=str(FIXTURES_TUTORIAL)).scan()
    second = GitScanner(repo_path=str(FIXTURES_TUTORIAL)).scan()
    assert [a.id for a in first.assets] == [a.id for a in second.assets]
    assert first.metadata["repo_character"] == second.metadata["repo_character"]
