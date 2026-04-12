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
    """Sprint 3 policy: plain inference asset using a known LLM provider
    gets ONE info-level 'External AI API' reason describing residency
    and training policy. No reflexive warnings — those only appear when
    an amplifying signal (PII, free tier with PII, MCP server) is also
    detected. The provider profile is still attached to the insight."""
    asset = _make_asset("openai")
    insight = enrich_asset(asset)

    external_info = [
        r for r in insight.risk_reasons
        if r.severity == "info" and r.title.startswith("External AI API")
    ]
    assert external_info, "Expected one info line describing the external API"
    assert "OpenAI" in external_info[0].title
    assert "Residency:" in external_info[0].detail

    assert insight.provider_profile is not None
    assert insight.provider_profile.display_name == "OpenAI"

    # Without PII/free-tier signal the score should land in the OK band
    assert asset.risk_score < 0.40


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


# ── Sprint 2: task_type + tag derivation ─────────────────────────────────


def test_task_type_defaults_to_inference():
    from aiscout.models import TaskType
    asset = _make_asset("openai")
    asset.code_contexts = []
    enrich_asset(asset)
    assert TaskType.INFERENCE in asset.task_types


def test_task_type_detects_fine_tuning_from_code():
    from aiscout.models import CodeContext, TaskType
    asset = _make_asset("huggingface")
    asset.code_contexts = [CodeContext(
        file_path="finetune.py",
        language="python",
        functions=[{
            "name": "main",
            "body_preview": "trainer = Trainer(model=m); trainer.train(); peft.save_pretrained()",
        }],
    )]
    asset.file_path = "scripts/finetune.py"
    enrich_asset(asset)
    assert TaskType.FINE_TUNING in asset.task_types
    titles = {r.title for r in enrich_asset(asset).risk_reasons}
    assert any("Fine-tuning" in t or "Training" in t for t in titles)


def test_tags_include_rag_when_vector_db_used():
    from aiscout.models import CodeContext
    asset = _make_asset("pinecone")
    asset.code_contexts = [CodeContext(
        file_path="rag.py",
        language="python",
        functions=[{"name": "search", "body_preview": "pinecone.query(embedding=vec)"}],
    )]
    enrich_asset(asset)
    assert "rag" in asset.tags


def test_tags_include_chatbot_for_conversational_code():
    from aiscout.models import CodeContext
    asset = _make_asset("openai")
    asset.code_contexts = [CodeContext(
        file_path="bot.py",
        language="python",
        functions=[{
            "name": "chat",
            "body_preview": "client.chat.completions.create(messages=[{'role':'user',...}])",
        }],
        prompts=["You are a helpful assistant"],
    )]
    enrich_asset(asset)
    assert "chatbot" in asset.tags


def test_mcp_finding_raises_risk_reason():
    asset = _make_asset("mcp", findings=[
        Finding(
            type=FindingType.CONFIG_DETECTED,
            file_path="mcp.json",
            content="mcp server: filesystem",
            provider="mcp",
        ),
    ])
    insight = enrich_asset(asset)
    assert any("MCP" in r.title for r in insight.risk_reasons)
    assert "mcp" in asset.tags


# ── Sprint 3: risk scoring review ────────────────────────────────────────


def test_pii_with_free_tier_provider_is_critical():
    from aiscout.models import CodeContext
    asset = _make_asset("openai")
    asset.code_contexts = [CodeContext(
        file_path="chat.py",
        language="python",
        functions=[{
            "name": "answer",
            "body_preview": "load customer email and phone from database",
        }],
    )]
    insight = enrich_asset(asset)
    critical = [r for r in insight.risk_reasons if r.severity == "critical"]
    assert any("Personal data" in r.title for r in critical), (
        "PII + provider with 'may be used' training policy should escalate to critical"
    )
    assert asset.risk_score >= 0.70


def test_plain_anthropic_inference_is_not_warning():
    """Anthropic API does not use data for training — plain inference
    asset with no PII / training signal should land in OK band."""
    asset = _make_asset("anthropic")
    insight = enrich_asset(asset)
    warnings = [r for r in insight.risk_reasons if r.severity == "warning"]
    assert not warnings, f"Unexpected warnings: {[r.title for r in warnings]}"
    assert asset.risk_score < 0.40


def test_mcp_server_vs_client_severity():
    from aiscout.models import CodeContext
    # Client: uses MultiServerMCPClient to call configured servers
    client = _make_asset("langchain")
    client.code_contexts = [CodeContext(
        file_path="client.py",
        language="python",
        functions=[{
            "name": "main",
            "body_preview": "from langchain_mcp_adapters.client import MultiServerMCPClient",
        }],
    )]
    client.raw_findings.append(Finding(
        type=FindingType.CONFIG_DETECTED,
        file_path="mcp.json",
        content="mcp server: filesystem",
        provider="mcp",
    ))
    client_insight = enrich_asset(client)
    assert any(
        r.severity == "info" and "MCP client" in r.title
        for r in client_insight.risk_reasons
    )

    # Server: exposes tools via fastmcp / @server.tool
    server = _make_asset("mcp")
    server.code_contexts = [CodeContext(
        file_path="server.py",
        language="python",
        functions=[{
            "name": "take_screenshot",
            "body_preview": "from mcp.server import Server; @server.tool()",
        }],
    )]
    server_insight = enrich_asset(server)
    assert any(
        r.severity == "warning" and "MCP server" in r.title
        for r in server_insight.risk_reasons
    )


def test_risk_score_floors_match_severity():
    from aiscout.engine.enrichment import _calculate_risk_score, RiskReason
    assert _calculate_risk_score([RiskReason("critical", "t", "d")]) >= 0.70
    assert _calculate_risk_score([RiskReason("warning", "t", "d")]) >= 0.40
    assert _calculate_risk_score([RiskReason("warning", "t", "d")]) < 0.70
    assert _calculate_risk_score([RiskReason("info", "t", "d")]) < 0.40
    assert _calculate_risk_score([]) == 0.10
