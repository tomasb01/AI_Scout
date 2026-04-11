"""Tests for asset enrichment — summary, risk reasoning, recommendations."""

from aiscout.engine.enrichment import enrich_asset, enrich_assets
from aiscout.knowledge.providers import get_provider
from aiscout.models import (
    AIAsset,
    Finding,
    FindingType,
    ProviderInfo,
)


def _make_asset(provider: str = "openai", findings: list | None = None) -> AIAsset:
    if findings is None:
        findings = [
            Finding(
                type=FindingType.IMPORT_DETECTED,
                file_path="app.py",
                line_number=1,
                content="import openai",
                provider=provider,
            ),
        ]
    return AIAsset(
        name=f"{provider} usage",
        provider=ProviderInfo(name=provider),
        repository="test-repo",
        file_path="app.py",
        raw_findings=findings,
    )


def test_enrich_generates_summary():
    asset = _make_asset("openai")
    insight = enrich_asset(asset)
    assert insight.summary
    # Summary describes what it does, provider goes to tech_stack
    assert len(insight.summary) > 10


def test_enrich_api_key_critical():
    asset = _make_asset("openai", findings=[
        Finding(
            type=FindingType.API_KEY_DETECTED,
            file_path="config.py",
            content="sk-abc123",
            redacted_content="sk-abc...123",
            provider="openai",
        ),
    ])
    insight = enrich_asset(asset)
    # Should have a critical risk reason
    critical_reasons = [r for r in insight.risk_reasons if r.severity == "critical"]
    assert len(critical_reasons) >= 1
    assert "Hardcoded" in critical_reasons[0].title
    # Should have key rotation recommendation
    assert any("Rotate" in r or "rotate" in r for r in insight.recommendations)


def test_enrich_local_runtime_low_risk():
    asset = _make_asset("ollama")
    insight = enrich_asset(asset)
    # Ollama is local — should have info-level reason about no data egress
    info_reasons = [r for r in insight.risk_reasons if r.severity == "info"]
    assert any("local" in r.title.lower() or "no data egress" in r.title.lower() for r in info_reasons)
    # Risk score should be low
    assert asset.risk_score < 0.4


def test_enrich_framework_notes_downstream():
    asset = _make_asset("langchain")
    insight = enrich_asset(asset)
    # Should mention downstream providers
    assert any("framework" in r.title.lower() or "downstream" in r.detail.lower() for r in insight.risk_reasons)


def test_enrich_external_api_data_residency():
    asset = _make_asset("openai")
    insight = enrich_asset(asset)
    # OpenAI is US-only — should mention data leaving EU
    warning_reasons = [r for r in insight.risk_reasons if r.severity == "warning"]
    assert len(warning_reasons) >= 1
    # Provider profile should be attached
    assert insight.provider_profile is not None
    assert insight.provider_profile.display_name == "OpenAI"


def test_enrich_assets_returns_dict():
    assets = [_make_asset("openai"), _make_asset("anthropic")]
    result = enrich_assets(assets)
    assert len(result) == 2
    for asset in assets:
        assert asset.id in result


def test_provider_knowledge_base():
    openai = get_provider("openai")
    assert openai.display_name == "OpenAI"
    assert "US" in openai.data_residency
    assert openai.category == "llm_api"

    unknown = get_provider("nonexistent_provider")
    assert unknown.name == "unknown"


def test_risk_score_recalculated():
    asset = _make_asset("openai", findings=[
        Finding(type=FindingType.API_KEY_DETECTED, file_path="x.py",
                content="sk-abc", redacted_content="sk-...bc", provider="openai"),
        Finding(type=FindingType.IMPORT_DETECTED, file_path="x.py",
                content="import openai", provider="openai"),
    ])
    asset.risk_score = 0.0  # start at zero
    enrich_asset(asset)
    # Should be recalculated based on reasons (critical + warnings)
    assert asset.risk_score > 0.4
