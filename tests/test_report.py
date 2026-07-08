"""Tests for HTML Report Generator."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from aiscout.models import (
    AIAsset,
    ClassificationResult,
    Confidence,
    DataCategory,
    Finding,
    FindingType,
    ProviderInfo,
    ScanResult,
)
from aiscout.report.html import ReportGenerator


def _make_scan_result(repo: str = "test-repo", assets: list | None = None) -> ScanResult:
    if assets is None:
        assets = [
            AIAsset(
                name="openai usage",
                provider=ProviderInfo(name="openai"),
                risk_score=0.7,
                repository=repo,
                file_path="app.py",
                raw_findings=[
                    Finding(
                        type=FindingType.IMPORT_DETECTED,
                        file_path="app.py",
                        line_number=1,
                        content="import openai",
                        provider="openai",
                    ),
                ],
            ),
            AIAsset(
                name="langchain usage",
                provider=ProviderInfo(name="langchain"),
                risk_score=0.3,
                repository=repo,
                file_path="chain.py",
                raw_findings=[
                    Finding(
                        type=FindingType.DEPENDENCY_DETECTED,
                        file_path="requirements.txt",
                        content="langchain>=0.1",
                        provider="langchain",
                    ),
                ],
            ),
        ]
    return ScanResult(
        scanner="git_scanner",
        started_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        completed_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        assets=assets,
        metadata={"repository": repo, "files_scanned": 42},
    )


def test_generate_creates_file():
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        output = f.name

    result = _make_scan_result()
    gen = ReportGenerator([result], output_path=output)
    path = gen.generate()

    content = Path(path).read_text()
    assert "AI Scout" in content
    assert "openai usage" in content
    # the langchain asset carries only a dependency finding and no code,
    # so it renders as dependency evidence, not as a solution (Sprint 0.3)
    assert "Dependency manifest" in content
    Path(path).unlink()


def test_context_risk_counts():
    result = _make_scan_result()
    gen = ReportGenerator([result])
    ctx = gen._build_context()
    # The dependency-manifest-only asset is evidence, not a solution —
    # excluded from every headline count (still rendered in the table).
    assert ctx["total_assets"] == 1
    assert len(ctx["assets"]) == 2
    # All counted assets fall into one of the three categories
    assert ctx["critical_count"] + ctx["warning_count"] + ctx["ok_count"] == 1


def test_cross_repo_overlap_detection():
    r1 = _make_scan_result(repo="repo-a", assets=[
        AIAsset(name="openai in A", provider=ProviderInfo(name="openai"), repository="repo-a"),
    ])
    r2 = _make_scan_result(repo="repo-b", assets=[
        AIAsset(name="openai in B", provider=ProviderInfo(name="openai"), repository="repo-b"),
    ])
    gen = ReportGenerator([r1, r2])
    ctx = gen._build_context()
    assert "openai" in ctx["cross_repo_overlaps"]
    assert set(ctx["cross_repo_overlaps"]["openai"]) == {"repo-a", "repo-b"}


def test_empty_scan():
    result = _make_scan_result(assets=[])
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        output = f.name

    gen = ReportGenerator([result], output_path=output)
    path = gen.generate()
    content = Path(path).read_text()
    assert "No AI solutions found" in content
    Path(path).unlink()


def test_no_llm_data():
    result = _make_scan_result()
    gen = ReportGenerator([result])
    ctx = gen._build_context()
    assert ctx["has_llm_data"] is False


def test_org_inventory_totals_aggregated():
    inv = [
        {"owner": "acme", "total_seen": 40, "scanned": 30,
         "skipped_archived": 6, "skipped_forks": 4, "skipped_over_limit": 0,
         "skipped_blocked": 0},
        {"owner": "globex", "total_seen": 10, "scanned": 8,
         "skipped_archived": 1, "skipped_forks": 1, "skipped_over_limit": 0,
         "skipped_blocked": 0},
    ]
    gen = ReportGenerator([_make_scan_result()], org_inventory=inv)
    totals = gen._build_context()["org_inventory_totals"]
    assert totals["total_seen"] == 50
    assert totals["scanned"] == 38
    assert totals["skipped_archived"] == 7
    assert totals["owners"] == ["acme", "globex"]


def test_org_inventory_renders_section():
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        output = f.name
    inv = [{"owner": "acme", "total_seen": 47, "scanned": 35,
            "skipped_archived": 8, "skipped_forks": 4, "skipped_over_limit": 0,
            "skipped_blocked": 0}]
    gen = ReportGenerator([_make_scan_result()], output_path=output, org_inventory=inv)
    content = Path(gen.generate()).read_text()
    assert "GitHub Coverage" in content
    assert "Repos found" in content
    assert "47" in content
    Path(output).unlink()


def test_no_org_inventory_omits_section():
    gen = ReportGenerator([_make_scan_result()])
    ctx = gen._build_context()
    assert ctx["org_inventory_totals"] is None


def test_structural_table_groups():
    """Large repos get collapsible top-level-dir groups in the table —
    presentation only: identities, IDs and counts stay granular."""
    from aiscout.models import DataFlowMap, FlowSink

    def asset(i, top, critical=False):
        a = AIAsset(
            name=f"solution {i}",
            provider=ProviderInfo(name="openai"),
            repository="big-repo",
            root_path=f"{top}/ex{i}",
            file_path=f"{top}/ex{i}/main.py",
            raw_findings=[Finding(
                type=FindingType.API_KEY_DETECTED if critical else FindingType.IMPORT_DETECTED,
                file_path=f"{top}/ex{i}/main.py",
                content="sk-XXX" if critical else "import openai",
                redacted_content="sk-...XXX" if critical else None,
                provider="openai",
            )],
        )
        return a

    assets = [asset(i, "01-chapter") for i in range(5)]
    assets += [asset(10 + i, "02-chapter", critical=(i == 0)) for i in range(4)]
    assets += [asset(20, "misc")]  # below member threshold — stays flat

    result = _make_scan_result(repo="big-repo", assets=assets)
    gen = ReportGenerator([result])
    ctx = gen._build_context()

    headers = list(ctx["group_headers"].values())
    assert {h["label"] for h in headers} == {"01-chapter", "02-chapter"}
    ch1 = next(h for h in headers if h["label"] == "01-chapter")
    ch2 = next(h for h in headers if h["label"] == "02-chapter")
    assert ch1["count"] == 5 and ch2["count"] == 4
    assert ch2["critical"] == 1
    # groups without critical start collapsed; critical ones start open
    assert ch1["id"] in ctx["collapsed_groups"]
    assert ch2["id"] not in ctx["collapsed_groups"]
    # the misc solution has no group
    misc = next(a for a in ctx["assets"] if a.root_path.startswith("misc"))
    assert misc.id not in ctx["row_groups"]

    # renders group header rows into the HTML
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "r.html"
        ReportGenerator([_make_scan_result(repo="big-repo", assets=assets)],
                        output_path=str(out)).generate()
        html = out.read_text()
    assert 'class="grp-row' in html
    assert "01-chapter/" in html
