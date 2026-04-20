"""Tests for Sprint 5 Data Flow Mapper."""

from aiscout.engine.code_analyzer import analyze_assets
from aiscout.engine.data_flow import build_data_flow, build_data_flows
from aiscout.models import AIAsset, CodeContext, Confidence, Finding, FindingType, ProviderInfo
from aiscout.scanners.git_scanner import GitScanner

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_fleurdin_chatbot_sources():
    """The Fleurdin chatbot fixture must produce DB + REST endpoint sources."""
    s = GitScanner(repo_path=str(FIXTURES))
    r = s.scan()
    analyze_assets(r.assets, str(FIXTURES))
    asset = next(a for a in r.assets if "sample_chatbot.py" in a.file_path)
    flow = build_data_flow(asset)
    source_types = {s.type for s in flow.sources}
    assert "database" in source_types
    assert "user_input" in source_types


def test_fleurdin_chatbot_sinks():
    """Must detect AI API sink + DB write + HTTP response."""
    s = GitScanner(repo_path=str(FIXTURES))
    r = s.scan()
    analyze_assets(r.assets, str(FIXTURES))
    asset = next(a for a in r.assets if "sample_chatbot.py" in a.file_path)
    flow = build_data_flow(asset)
    sink_types = {s.type for s in flow.sinks}
    assert "ai_api" in sink_types
    assert "database" in sink_types
    assert "http_response" in sink_types


def test_fleurdin_chatbot_steps_ordered():
    """Processing steps must be in the right execution order."""
    s = GitScanner(repo_path=str(FIXTURES))
    r = s.scan()
    analyze_assets(r.assets, str(FIXTURES))
    asset = next(a for a in r.assets if "sample_chatbot.py" in a.file_path)
    flow = build_data_flow(asset)

    assert len(flow.processing_steps) >= 4
    step_text = " ".join(flow.processing_steps).lower()
    # These must appear in order: receive → query → send → store
    receive_pos = step_text.find("receive")
    query_pos = step_text.find("query")
    send_pos = step_text.find("send")
    store_pos = step_text.find("store")
    assert receive_pos < query_pos < send_pos < store_pos


def test_fleurdin_chatbot_purpose_mentions_fleurdin():
    """Purpose must mention the specific role, not generic 'chatbot'."""
    s = GitScanner(repo_path=str(FIXTURES))
    r = s.scan()
    analyze_assets(r.assets, str(FIXTURES))
    asset = next(a for a in r.assets if "sample_chatbot.py" in a.file_path)
    flow = build_data_flow(asset)
    assert "Fleurdin" in flow.solution_purpose
    assert "florist" in flow.solution_purpose.lower()


def test_fleurdin_chatbot_confidence_high():
    """Chatbot fixture has AST + sources + sinks + steps = high confidence."""
    s = GitScanner(repo_path=str(FIXTURES))
    r = s.scan()
    analyze_assets(r.assets, str(FIXTURES))
    asset = next(a for a in r.assets if "sample_chatbot.py" in a.file_path)
    flow = build_data_flow(asset)
    assert flow.confidence == Confidence.HIGH


def test_asset_without_code_context_gets_low_confidence():
    """An asset with no code context should get a fallback low-confidence flow."""
    asset = AIAsset(
        name="bare import",
        provider=ProviderInfo(name="openai"),
        repository="test",
        raw_findings=[Finding(
            type=FindingType.IMPORT_DETECTED, file_path="x.py",
            content="import openai", provider="openai",
        )],
    )
    flow = build_data_flow(asset)
    assert flow.confidence == Confidence.LOW
    assert "OpenAI" in flow.solution_purpose


def test_build_data_flows_mutates_assets():
    """build_data_flows() must attach data_flow to each asset in place."""
    s = GitScanner(repo_path=str(FIXTURES))
    r = s.scan()
    analyze_assets(r.assets, str(FIXTURES))
    assert all(a.data_flow is None for a in r.assets)
    build_data_flows(r.assets)
    assert all(a.data_flow is not None for a in r.assets)


def test_ai_api_sink_includes_model_name():
    """AI API sink should mention the specific model when code_analyzer found it."""
    s = GitScanner(repo_path=str(FIXTURES))
    r = s.scan()
    analyze_assets(r.assets, str(FIXTURES))
    asset = next(a for a in r.assets if "sample_chatbot.py" in a.file_path)
    flow = build_data_flow(asset)
    ai_sinks = [s for s in flow.sinks if s.type == "ai_api"]
    assert ai_sinks
    assert "claude-sonnet" in ai_sinks[0].name.lower() or "claude" in ai_sinks[0].name.lower()


def test_insert_not_in_sources():
    """INSERT INTO operations must not appear as data sources."""
    s = GitScanner(repo_path=str(FIXTURES))
    r = s.scan()
    analyze_assets(r.assets, str(FIXTURES))
    asset = next(a for a in r.assets if "sample_chatbot.py" in a.file_path)
    flow = build_data_flow(asset)
    for src in flow.sources:
        assert "INSERT" not in src.detail.upper(), f"INSERT in source: {src.detail}"
