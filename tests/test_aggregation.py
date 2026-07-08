"""Tests for Sprint 0.3 — repo character detector + aggregation boundary.

Identity order: purpose → tech stack → structure. Aggregation runs
AFTER code analysis and merges only components that share BOTH a
structural boundary (tutorial chapter / subdirectory-manifest subtree)
AND a functional DataFlowMap fingerprint — i.e. provable variants of
one thing. Ten agents with ten different flows stay ten solutions; the
same flow in two repo branches stays two solutions (overlap insight).
"""

from pathlib import Path

from aiscout.engine.aggregation import aggregate_assets, aggregate_scan_result
from aiscout.engine.code_analyzer import analyze_assets
from aiscout.engine.data_flow import build_data_flows
from aiscout.engine.enrichment import enrich_assets
from aiscout.engine.repo_character import (
    KIND_EXPERIMENT,
    KIND_PRODUCTION,
    KIND_TUTORIAL,
    KIND_UNKNOWN,
    RepoCharacter,
    detect_repo_character,
)
from aiscout.models import (
    AIAsset,
    DataFlowMap,
    Finding,
    FindingType,
    FlowSink,
    RiskStatus,
)
from aiscout.report.insights import build_insights, collect_stats
from aiscout.scanners.git_scanner import GitScanner, _stable_hash

FIXTURES_TUTORIAL = Path(__file__).parent / "fixtures_tutorial"

PRODUCTION = RepoCharacter(KIND_PRODUCTION, "high")
TUTORIAL = RepoCharacter(KIND_TUTORIAL, "high")


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
    """A course-shaped repo that ships repo-level CI + tests stays out of
    the tutorial bucket — misclassifying production is the expensive
    error."""
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


# ── Aggregation unit tests ─────────────────────────────────────────────────


def _flow(provider: str = "openai", steps: tuple = ("Send prompt to LLM API",),
          categories: tuple = ()) -> DataFlowMap:
    return DataFlowMap(
        sinks=[FlowSink(type="ai_api", name=provider, provider=provider)],
        processing_steps=list(steps),
        data_categories=list(categories),
    )


def _dir_asset(repo: str, solution_dir: str, files: list[str],
               findings: list[Finding] | None = None,
               flow: DataFlowMap | None = None) -> AIAsset:
    return AIAsset(
        id="sol-" + _stable_hash(repo, solution_dir),
        name=solution_dir,
        repository=repo,
        root_path=solution_dir,
        file_path=", ".join(files),
        raw_findings=findings or [],
        data_flow=flow,
    )


def _dep_finding(manifest_path: str) -> Finding:
    return Finding(
        type=FindingType.DEPENDENCY_DETECTED,
        file_path=manifest_path, content="openai>=1.0", provider="openai",
    )


def test_same_boundary_same_flow_merges_into_variant_group():
    repo = "svc"
    assets = [
        _dir_asset(repo, "app", ["app/pyproject.toml", "app/main.py"],
                   [_dep_finding("app/pyproject.toml")], flow=_flow()),
        _dir_asset(repo, "app/handlers", ["app/handlers/chat.py"], flow=_flow()),
        _dir_asset(repo, "app/legacy", ["app/legacy/chat_old.py"], flow=_flow()),
    ]
    merged = aggregate_assets(assets, repo, PRODUCTION)
    assert len(merged) == 1
    solution = merged[0]
    assert solution.root_path == "app"
    assert solution.component_dirs == ["app", "app/handlers", "app/legacy"]
    assert "variant_group" in solution.tags
    # deterministic ID from repo + boundary + fingerprint
    again = aggregate_assets(assets, repo, PRODUCTION)
    assert again[0].id == solution.id


def test_same_boundary_different_flows_stay_separate():
    """Ten LangGraph agents, each with a different purpose, are ten
    solutions — a shared directory and framework never merge them."""
    repo = "svc"
    assets = [
        _dir_asset(repo, "app", ["app/pyproject.toml"],
                   [_dep_finding("app/pyproject.toml")], flow=_flow("openai")),
        _dir_asset(repo, "app/support-agent", ["app/support-agent/main.py"],
                   flow=_flow("anthropic")),
        _dir_asset(repo, "app/rag", ["app/rag/pipeline.py"],
                   flow=_flow("openai", steps=("Query vector DB", "Send prompt to LLM API"))),
    ]
    merged = aggregate_assets(assets, repo, PRODUCTION)
    assert len(merged) == 3
    assert {a.id for a in merged} == {a.id for a in assets}


def test_same_flow_across_boundaries_never_merges():
    """The same flow in two repo branches is TWO deployments — that is
    the overlap insight ('2 implementations of the same thing'), never
    one card."""
    repo = "monorepo"
    assets = [
        _dir_asset(repo, "hooks/on-ticket", ["hooks/on-ticket/agent.py"],
                   flow=_flow()),
        _dir_asset(repo, "services/chatbot", ["services/chatbot/bot.py"],
                   flow=_flow()),
    ]
    merged = aggregate_assets(assets, repo, PRODUCTION)
    assert len(merged) == 2
    assert sorted(a.root_path for a in merged) == [
        "hooks/on-ticket", "services/chatbot",
    ]


def test_root_manifest_is_not_a_boundary():
    """A root requirements.txt must not make the whole monorepo one
    mergeable boundary."""
    repo = "monorepo"
    assets = [
        _dir_asset(repo, ".", ["requirements.txt"],
                   [_dep_finding("requirements.txt")], flow=_flow()),
        _dir_asset(repo, "hooks", ["hooks/agent.py"], flow=_flow()),
        _dir_asset(repo, "services", ["services/bot.py"], flow=_flow()),
    ]
    merged = aggregate_assets(assets, repo, PRODUCTION)
    assert len(merged) == 3
    assert {a.id for a in merged} == {a.id for a in assets}


def test_components_without_flow_never_merge():
    repo = "svc"
    assets = [
        _dir_asset(repo, "app", ["app/pyproject.toml"],
                   [_dep_finding("app/pyproject.toml")]),
        _dir_asset(repo, "app/scripts", ["app/scripts/run.py"]),
    ]
    merged = aggregate_assets(assets, repo, PRODUCTION)
    assert len(merged) == 2


def test_merged_variant_group_takes_worst_risk_status():
    repo = "svc"
    leaky = _dir_asset(repo, "app/keys", ["app/keys/leak.py"], flow=_flow())
    leaky.risk_status = RiskStatus.CRITICAL
    assets = [
        _dir_asset(repo, "app", ["app/requirements.txt", "app/main.py"],
                   [_dep_finding("app/requirements.txt")], flow=_flow()),
        leaky,
    ]
    merged = aggregate_assets(assets, repo, PRODUCTION)
    assert len(merged) == 1
    assert merged[0].risk_status == RiskStatus.CRITICAL


def test_tutorial_below_component_threshold_does_not_activate_chapters():
    repo = "course"
    assets = [
        _dir_asset(repo, f"{i:02d}-lesson", [f"{i:02d}-lesson/ex.py"],
                   flow=_flow())
        for i in range(1, 5)  # only 4 components
    ]
    merged = aggregate_assets(assets, repo, TUTORIAL)
    assert len(merged) == 4


def test_tutorial_chapter_merges_same_flow_variants_only():
    repo = "course"
    assets = []
    # chapter 01: three same-flow variants + one genuinely different flow
    assets.append(_dir_asset(repo, "01-intro/a", ["01-intro/a/ex.py"], flow=_flow()))
    assets.append(_dir_asset(repo, "01-intro/b", ["01-intro/b/ex.py"], flow=_flow()))
    assets.append(_dir_asset(repo, "01-intro/c", ["01-intro/c/ex.py"], flow=_flow()))
    assets.append(_dir_asset(
        repo, "01-intro/rag", ["01-intro/rag/ex.py"],
        flow=_flow("openai", steps=("Query vector DB", "Send prompt to LLM API")),
    ))
    # chapter 02: same flow as chapter 01 variants — different chapter,
    # no merge across chapters
    for sub in ("a", "b", "c", "d"):
        assets.append(_dir_asset(
            repo, f"02-agents/{sub}", [f"02-agents/{sub}/ex.py"], flow=_flow()
        ))
    merged = aggregate_assets(assets, repo, TUTORIAL)
    # 01-intro: 3 variants -> 1 group + 1 distinct flow; 02-agents: 4 -> 1
    assert len(merged) == 3
    groups = {a.root_path: a for a in merged if "variant_group" in a.tags}
    assert set(groups) == {"01-intro", "02-agents"}
    assert len(groups["01-intro"].component_dirs) == 3
    assert len(groups["02-agents"].component_dirs) == 4


# ── End-to-end on the tutorial fixture ─────────────────────────────────────


def _run_pipeline(root: Path):
    result = GitScanner(repo_path=str(root)).scan()
    analyze_assets(result.assets, str(root))
    build_data_flows(result.assets)
    aggregate_scan_result(result)
    return result


def test_tutorial_fixture_end_to_end_variant_groups():
    """10 lessons × 2 same-flow sub-examples → 10 solutions (one variant
    group per chapter), purpose-first names, I-01 counts 10."""
    result = _run_pipeline(FIXTURES_TUTORIAL)
    assert result.metadata["repo_character"]["kind"] == KIND_TUTORIAL

    assert len(result.assets) == 10
    for solution in result.assets:
        assert len(solution.component_dirs) == 2
        assert len(solution.raw_findings) == 2
        assert "variant_group" in solution.tags

    insights = enrich_assets(result.assets)
    names = sorted(insights[a.id].solution_name for a in result.assets)
    assert all("(2 variants)" in n for n in names)
    assert len(set(names)) == 10  # display names never collide

    stats = collect_stats(
        result.assets, insights,
        repos=1, files_scanned=result.metadata["files_scanned"],
    )
    catalog = build_insights(stats)
    i01 = next(i for i in catalog if i.id == "I-01")
    assert "10 AI solutions" in i01.text  # not 20, not 1


def test_pipeline_is_deterministic():
    first = _run_pipeline(FIXTURES_TUTORIAL)
    second = _run_pipeline(FIXTURES_TUTORIAL)
    assert [a.id for a in first.assets] == [a.id for a in second.assets]
    assert first.metadata["repo_character"] == second.metadata["repo_character"]


# ── Dependency evidence (a manifest is not a solution) ─────────────────────


def test_manifest_without_code_is_dependency_evidence_not_a_solution():
    repo = "app"
    manifest_only = _dir_asset(
        repo, ".", ["requirements.txt"], [_dep_finding("requirements.txt")]
    )
    real = _dir_asset(repo, "bot", ["bot/main.py"], flow=_flow())
    real.raw_findings = [Finding(
        type=FindingType.IMPORT_DETECTED, file_path="bot/main.py",
        content="import openai", provider="openai",
    )]
    insights = enrich_assets([manifest_only, real])

    assert "dependency_evidence" in manifest_only.tags
    assert insights[manifest_only.id].category == "Dependency Evidence"
    assert insights[manifest_only.id].solution_name == (
        "Dependency manifest — repo root"
    )

    # excluded from solution counts: I-01 says 1, not 2
    stats = collect_stats([manifest_only, real], insights, repos=1,
                          files_scanned=5)
    assert stats.total == 1
    catalog = build_insights(stats)
    assert "1 AI solution " in catalog[0].text
