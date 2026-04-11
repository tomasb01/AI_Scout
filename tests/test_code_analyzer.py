"""Tests for Code Context Extractor."""

from pathlib import Path

from aiscout.engine.code_analyzer import (
    _analyze_python,
    _analyze_generic,
    analyze_assets,
)
from aiscout.models import AIAsset, Finding, FindingType, ProviderInfo


FIXTURES = Path(__file__).parent / "fixtures"


def test_analyze_python_functions():
    content = FIXTURES.joinpath("sample_chatbot.py").read_text()
    ctx = _analyze_python("app.py", content)

    func_names = [f["name"] for f in ctx.functions]
    assert "chat" in func_names
    assert "load_history" in func_names
    assert "save_to_db" in func_names


def test_analyze_python_decorators():
    content = FIXTURES.joinpath("sample_chatbot.py").read_text()
    ctx = _analyze_python("app.py", content)

    chat_func = next(f for f in ctx.functions if f["name"] == "chat")
    assert any("route" in d for d in chat_func.get("decorators", []))


def test_analyze_prompts():
    content = FIXTURES.joinpath("sample_chatbot.py").read_text()
    ctx = _analyze_python("app.py", content)

    assert len(ctx.prompts) > 0
    assert any("Fleurdin" in p for p in ctx.prompts)


def test_analyze_api_calls():
    content = FIXTURES.joinpath("sample_chatbot.py").read_text()
    ctx = _analyze_python("app.py", content)

    assert len(ctx.api_calls) > 0
    targets = [c["target"] for c in ctx.api_calls]
    assert any("messages.create" in t or "client" in t for t in targets)


def test_analyze_db_operations():
    content = FIXTURES.joinpath("sample_chatbot.py").read_text()
    ctx = _analyze_python("app.py", content)

    assert len(ctx.data_sources) > 0
    # Should detect SQL operations
    all_details = " ".join(d.get("detail", "") for d in ctx.data_sources)
    assert "SELECT" in all_details.upper() or "execute" in all_details


def test_analyze_generic_js():
    content = """
const OpenAI = require('openai');
const client = new OpenAI();

const SYSTEM_PROMPT = "You are a helpful coding assistant";

async function generateCode(prompt) {
    const response = await client.chat.completions.create({
        model: "gpt-4",
        messages: [
            { role: "system", content: SYSTEM_PROMPT },
            { role: "user", content: prompt },
        ],
    });
    return response.choices[0].message.content;
}

module.exports = { generateCode };
"""
    ctx = _analyze_generic("app.js", content, "javascript")

    # Should find the function
    func_names = [f["name"] for f in ctx.functions]
    assert "generateCode" in func_names

    # Should find the prompt
    assert any("helpful coding assistant" in p for p in ctx.prompts)


def test_analyze_asset_end_to_end():
    asset = AIAsset(
        name="Test Bot",
        provider=ProviderInfo(name="anthropic"),
        repository="test-repo",
        file_path="sample_chatbot.py",
        raw_findings=[
            Finding(
                type=FindingType.IMPORT_DETECTED,
                file_path="sample_chatbot.py",
                content="import anthropic",
                provider="anthropic",
            ),
        ],
    )

    analyze_assets([asset], str(FIXTURES))

    assert len(asset.code_contexts) > 0
    ctx = asset.code_contexts[0]
    assert ctx.language == "python"
    assert len(ctx.functions) > 0
    assert len(ctx.prompts) > 0


def test_analyze_nonexistent_path():
    asset = AIAsset(
        name="Missing",
        file_path="nonexistent.py",
    )
    analyze_assets([asset], "/tmp/nonexistent_repo")
    assert asset.code_contexts == []


def test_enrichment_uses_code_context():
    """Verify enrichment produces better summaries with code context."""
    from aiscout.engine.enrichment import enrich_asset

    asset = AIAsset(
        name="Test Bot",
        provider=ProviderInfo(name="anthropic"),
        repository="test-repo",
        file_path="sample_chatbot.py",
        raw_findings=[
            Finding(
                type=FindingType.IMPORT_DETECTED,
                file_path="sample_chatbot.py",
                content="import anthropic",
                provider="anthropic",
            ),
        ],
    )

    # Analyze code first
    analyze_assets([asset], str(FIXTURES))

    # Then enrich
    insight = enrich_asset(asset)

    # Should mention Fleurdin from the prompt
    assert "Fleurdin" in insight.summary or "florist" in insight.summary.lower()
