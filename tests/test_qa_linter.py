"""Tests for the report linter L-01–L-10 and the QA degradation pipeline
(Sprint 0.2, QA spec §2) — including fact strips, the QA appendix and the
--strict CI gate.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from click.testing import CliRunner

from aiscout.cli import cli
from aiscout.engine.enrichment import enrich_assets
from aiscout.models import (
    AIAsset,
    ClassificationResult,
    Confidence,
    DataCategory,
    DataFlowMap,
    Finding,
    FindingType,
    FlowSink,
    FlowSource,
    ProviderInfo,
    ScanResult,
)
from aiscout.report.html import ReportGenerator
from aiscout.report.json_export import JSONExporter
from aiscout.report.linter import (
    ERROR,
    WARN,
    QAReport,
    LintIssue,
    lint_duplicate_summaries,
    lint_text,
)
from aiscout.report.qa import build_fact_strip, prepare_qa

CLEAN = "Found 4 AI solutions across 1 repository (11 files scanned)."


def _rules(text: str, **kwargs) -> set[str]:
    return {i.rule for i in lint_text(text, **kwargs)}


# ── Individual rules ───────────────────────────────────────────────────────


def test_clean_sentence_passes():
    assert lint_text(CLEAN) == []


def test_l01_doubled_word():
    assert "L-01" in _rules("Data is is processed externally.")
    assert "L-01" in _rules("Data is Is processed externally.")  # case-insensitive
    assert "L-01" not in _rules("The work he had had done was reviewed.")


def test_l02_unpaired_brackets_and_quotes():
    assert "L-02" in _rules("Solutions (3 found are critical.")
    assert "L-02" in _rules('He said "this is critical.')
    assert "L-02" not in _rules("Solutions (3 found) are critical.")


def test_l03_unresolved_placeholder():
    assert "L-03" in _rules("Found {count} AI solutions in the scan.")
    assert "L-03" in _rules("Provider is None for this integration.")
    assert "L-03" in _rules("Detected undefined solutions in the scan.")
    assert "L-03" in _rules("Rendering gave [object Object] in output.")


def test_l04_empty_parens_and_orphan_punctuation():
    assert "L-04" in _rules("Solutions () require attention today.")
    assert "L-04" in _rules("Requires immediate attention ,")
    assert "L-04" in _rules(", starts with a comma here.")


def test_l05_truncated_sentence():
    assert "L-05" in _rules("Found 4 AI solutions across the scan")  # no punctuation
    assert "L-05" in _rules("The data is sent to the.")  # ends on stop word
    assert "L-05" not in _rules(CLEAN)
    # labels are exempt from sentence-shape rules
    assert "L-05" not in _rules("RAG pipeline", kind="label")


def test_l06_numeric_nonsense():
    assert "L-06" in _rules("One developer created 137% of solutions.")
    assert "L-06" in _rules("One developer created over 100% of solutions.")
    assert "L-06" in _rules("Found -3 solutions in the repository.")
    assert "L-06" not in _rules("One developer created 100% of solutions.")


def test_l07_code_leakage():
    assert "L-07" in _rules("Processes data via fetch_user_records here.")
    assert "L-07" in _rules("Uses the getUserData helper for requests.")
    assert "L-07" in _rules("Fails on TODO markers inside the scanner.")
    assert "L-07" in _rules("Reads the file src/pipeline/loader.py directly.")
    assert "L-07" in _rules("Runs SELECT name FROM customers on the database.")
    # whitelisted abbreviations and safe entity values do not trip it
    assert "L-07" not in _rules("Verify GDPR and DPA coverage for PII data.")
    assert "L-07" not in _rules(
        "One contributor (john_doe) created 3 of 4 solutions (75%).",
        safe_tokens=["john_doe"],
    )
    # LLM-provenance prose skips code-leak heuristics entirely
    assert "L-07" not in _rules(
        "Summarizes fetch_user_records output for support agents.",
        apply_code_leak=False,
    )


def test_l09_length_bounds_warn_only():
    short = lint_text("Too short.")
    assert any(i.rule == "L-09" and i.severity == WARN for i in short)
    long_text = "Data " + "very " * 60 + "long sentence."
    assert any(i.rule == "L-09" for i in lint_text(long_text))
    title = "T" * 91
    assert any(i.rule == "L-09" for i in lint_text(title, kind="action_title"))


def test_l10_plural_after_one():
    assert "L-10" in _rules("Scan found 1 solutions in the repository.")
    assert "L-10" in _rules("Report lists 1 developers as authors.")
    assert "L-10" not in _rules("Scan found 1 solution in the repository.")
    # non-countable words are not the linter's business
    assert "L-10" not in _rules("Sends data to 1 address in logs.")


def test_l08_duplicate_summaries():
    issues, degrade = lint_duplicate_summaries({
        "a": "AI solution using OpenAI.",
        "b": "AI solution using OpenAI.",
        "c": "Unique fine-tuning pipeline.",
    })
    assert len(issues) == 1 and issues[0].severity == WARN
    assert degrade == {"a", "b"}

    issues, degrade = lint_duplicate_summaries({
        "a": "AI solution using OpenAI.",
        "b": "AI solution using OpenAI.",
        "c": "AI solution using OpenAI.",
    })
    assert issues[0].severity == ERROR
    assert degrade == {"a", "b", "c"}


def test_qa_report_counts():
    report = QAReport(issues=[
        LintIssue(rule="L-01", severity=ERROR, message="", text=""),
        LintIssue(rule="L-09", severity=WARN, message="", text=""),
    ])
    assert report.counts() == {"suppressed": 1, "warnings": 1}


# ── Fact strip (QA spec §1.4) ──────────────────────────────────────────────


def _flow_asset() -> AIAsset:
    return AIAsset(
        name="support bot",
        owner="dev",
        provider=ProviderInfo(name="openai"),
        repository="repo",
        file_path="bot.py",
        tags=["chatbot", "rag"],
        data_flow=DataFlowMap(
            sources=[
                FlowSource(type="database", name="customers_db", detail="raw"),
                FlowSource(type="mystery", name="??", detail="raw"),
            ],
            sinks=[
                FlowSink(type="ai_api", name="OpenAI", provider="openai"),
                FlowSink(type="file", name="out.json"),
            ],
            processing_steps=["LLM call"],
        ),
    )


def test_fact_strip_uses_controlled_vocabulary_only():
    strip = build_fact_strip(_flow_asset(), None)
    assert strip.sources == ["database (SQL)", "unclassified"]
    assert strip.sinks[0].startswith("OpenAI (")  # KB label + residency
    assert "file output" in strip.sinks
    assert strip.pattern == "RAG pipeline"  # rag outranks chatbot
    # no raw strings from the repo may appear anywhere in the strip
    flat = " ".join(strip.sources + strip.sinks + [strip.pattern] + strip.tech)
    assert "customers_db" not in flat and "out.json" not in flat


def test_fact_strip_without_flow_falls_back_to_provider():
    asset = AIAsset(name="bare", provider=ProviderInfo(name="anthropic"),
                    repository="repo", file_path="a.py")
    strip = build_fact_strip(asset, None)
    assert strip.sources == ["unclassified"]
    assert strip.sinks[0].startswith("Anthropic")
    assert strip.pattern == "AI integration"


# ── prepare_qa: degradation pipeline ───────────────────────────────────────


def _asset(name, summary_via_llm=None, owner="dev") -> AIAsset:
    classification = None
    if summary_via_llm is not None:
        classification = ClassificationResult(
            categories=[DataCategory.INTERNAL],
            confidence=Confidence.HIGH,
            details=summary_via_llm,
        )
    return AIAsset(
        name=name, owner=owner,
        provider=ProviderInfo(name="openai"),
        repository="repo", file_path=f"{name}.py",
        data_classification=classification,
    )


def test_no_llm_mode_shows_fact_strip_not_pseudo_summary():
    assets = [_asset("a"), _asset("b")]
    insights = enrich_assets(assets)
    qa = prepare_qa(assets, insights, repos=1, files_scanned=2)
    for asset in assets:
        assert qa.summary_display[asset.id] is None  # P-3: facts by default
        assert qa.fact_strips[asset.id].pattern


def test_llm_prose_shows_when_unique_and_clean():
    assets = [
        _asset("a", summary_via_llm="Answers customer questions about invoices."),
        _asset("b", summary_via_llm="Transcribes call-center recordings for QA review."),
    ]
    insights = enrich_assets(assets)
    qa = prepare_qa(assets, insights, repos=1, files_scanned=2)
    shown = [qa.summary_display[a.id] for a in assets]
    assert all(shown), "unique clean LLM prose must display"


def test_duplicated_llm_summaries_degrade_to_fact_strip():
    """QA spec acceptance: tutorial-repo fixture must not show identical
    summaries on different solutions."""
    assets = [
        _asset(f"lesson{i}", summary_via_llm="Demonstrates a basic OpenAI chat call.")
        for i in range(3)
    ]
    insights = enrich_assets(assets)
    qa = prepare_qa(assets, insights, repos=1, files_scanned=3)
    assert all(qa.summary_display[a.id] is None for a in assets)
    assert any(i.rule == "L-08" and i.severity == ERROR
               for i in qa.qa_report.issues)


def test_broken_llm_prose_is_suppressed_and_logged():
    assets = [_asset("a", summary_via_llm="Handles {count} records records daily")]
    insights = enrich_assets(assets)
    qa = prepare_qa(assets, insights, repos=1, files_scanned=1)
    assert qa.summary_display[assets[0].id] is None
    rules = {i.rule for i in qa.qa_report.suppressed}
    assert {"L-01", "L-03", "L-05"} <= rules
    assert qa.counts()["suppressed"] >= 3


def test_exec_insights_exclude_suppressed_sentences():
    assets = [_asset("a"), _asset("b")]
    insights = enrich_assets(assets)
    qa = prepare_qa(assets, insights, repos=1, files_scanned=2)
    assert qa.exec_insights  # I-01 at minimum
    assert all(not i.suppressed for i in qa.exec_insights)


# ── Report integration ─────────────────────────────────────────────────────


def _scan_result(assets) -> ScanResult:
    return ScanResult(
        scanner="git_scanner",
        started_at=datetime(2026, 7, 8, tzinfo=timezone.utc),
        completed_at=datetime(2026, 7, 8, tzinfo=timezone.utc),
        assets=assets,
        metadata={"repository": "test-repo", "files_scanned": 11},
    )


def _leaky_asset() -> AIAsset:
    return AIAsset(
        name="leaky", owner="solodev",
        provider=ProviderInfo(name="openai"),
        repository="test-repo", file_path="app.py",
        raw_findings=[Finding(
            type=FindingType.API_KEY_DETECTED, file_path="app.py",
            line_number=3, content="sk-XXX", redacted_content="sk-...XXX",
            provider="openai",
        )],
    )


def test_html_report_renders_insights_fact_strip_and_qa_footer():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.html"
        gen = ReportGenerator([_scan_result([_leaky_asset()])], output_path=str(out))
        gen.generate()
        html = out.read_text(encoding="utf-8")
    assert "Executive Summary" in html
    assert "hardcoded API keys." in html          # I-02 sentence
    assert "fact-strip" in html                   # P-3 default detail
    assert "QA: 0 suppressed / 0 warnings" in html
    assert gen.qa_result is not None
    assert gen.qa_result.counts()["suppressed"] == 0


def test_html_report_is_deterministic_across_runs():
    scans = [_scan_result([_leaky_asset()])]
    with tempfile.TemporaryDirectory() as tmp:
        out1, out2 = Path(tmp) / "a.html", Path(tmp) / "b.html"
        ReportGenerator(scans, output_path=str(out1)).generate()
        ReportGenerator(scans, output_path=str(out2)).generate()
        assert out1.read_bytes() == out2.read_bytes()


def test_json_export_carries_typed_insights_and_qa_counts():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.json"
        scan = _scan_result([_leaky_asset()])
        # the CLI always passes enrichment insights in; mirror that here
        gen = JSONExporter([scan], output_path=str(out),
                           insights=enrich_assets(scan.assets))
        gen.generate()
        import json
        data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.2.0"
    assert data["qa"] == {"suppressed": 0, "warnings": 0}
    ids = [i["id"] for i in data["insights"]]
    assert "I-01" in ids and "I-02" in ids
    for insight in data["insights"]:
        assert insight["template_id"]
        assert not insight["suppressed"]


# ── --strict CI gate ───────────────────────────────────────────────────────

FIXTURES = Path(__file__).parent / "fixtures"


def test_strict_passes_on_clean_scan():
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        output = f.name
    result = runner.invoke(cli, [
        "scan", "--local", str(FIXTURES), "--no-llm", "--strict",
        "--output", output,
    ])
    assert result.exit_code == 0, result.output


def test_strict_fails_when_linter_suppressed(monkeypatch):
    from aiscout.report import html as html_mod
    from aiscout.report.qa import QAResult

    real_prepare = html_mod.prepare_qa

    def sabotage(*args, **kwargs):
        qa = real_prepare(*args, **kwargs)
        qa.qa_report.issues.append(LintIssue(
            rule="L-03", severity=ERROR,
            message="injected for test", text="Found {count} solutions.",
        ))
        return qa

    monkeypatch.setattr(html_mod, "prepare_qa", sabotage)
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        output = f.name
    result = runner.invoke(cli, [
        "scan", "--local", str(FIXTURES), "--no-llm", "--strict",
        "--output", output,
    ])
    assert result.exit_code == 2
    assert "suppressed" in result.output.lower()

    # without --strict the same scan degrades but exits 0 (P-4)
    result = runner.invoke(cli, [
        "scan", "--local", str(FIXTURES), "--no-llm", "--output", output,
    ])
    assert result.exit_code == 0
