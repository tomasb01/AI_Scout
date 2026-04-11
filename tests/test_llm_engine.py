"""Tests for LLM Analysis Engine."""

import json

import pytest

from aiscout.engine.llm import LLMEngine
from aiscout.models import (
    AIAsset,
    Confidence,
    DataCategory,
    Finding,
    FindingType,
    ProviderInfo,
)


VALID_LLM_RESPONSE = json.dumps({
    "data_categories": ["internal", "source_code"],
    "confidence": "medium",
    "risk_score": 0.5,
    "summary": "OpenAI integration for code analysis.",
    "recommendations": ["Move API key to env var"],
})


def _make_asset() -> AIAsset:
    return AIAsset(
        name="openai usage",
        provider=ProviderInfo(name="openai"),
        repository="test-repo",
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
    )


def test_classify_ollama(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:11434/api/generate",
        json={"response": VALID_LLM_RESPONSE},
    )
    engine = LLMEngine(mode="ollama")
    result = engine.classify(_make_asset())

    assert DataCategory.INTERNAL in result.categories
    assert DataCategory.SOURCE_CODE in result.categories
    assert result.confidence == Confidence.MEDIUM
    assert result.risk_score == 0.5
    assert "OpenAI" in result.details


def test_classify_openai(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:11434/v1/chat/completions",
        json={
            "choices": [{"message": {"content": VALID_LLM_RESPONSE}}],
        },
    )
    engine = LLMEngine(mode="openai", url="http://localhost:11434", api_key="test-key")
    result = engine.classify(_make_asset())

    assert result.confidence == Confidence.MEDIUM
    assert result.risk_score == 0.5


def test_classify_parse_failure(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:11434/api/generate",
        json={"response": "this is not valid json at all"},
    )
    engine = LLMEngine(mode="ollama")
    result = engine.classify(_make_asset())

    assert result.confidence == Confidence.LOW
    assert "could not be parsed" in result.details


def test_check_health_success(httpx_mock):
    httpx_mock.add_response(url="http://localhost:11434/api/tags", status_code=200)
    engine = LLMEngine(mode="ollama")
    assert engine.check_health() is True


def test_check_health_failure(httpx_mock):
    httpx_mock.add_response(url="http://localhost:11434/api/tags", status_code=500)
    engine = LLMEngine(mode="ollama")
    assert engine.check_health() is False


def test_prompt_truncation():
    asset = _make_asset()
    # Add 30 findings
    asset.raw_findings = [
        Finding(
            type=FindingType.IMPORT_DETECTED,
            file_path=f"file_{i}.py",
            content=f"import openai  # {i}",
            provider="openai",
        )
        for i in range(30)
    ]
    engine = LLMEngine(mode="ollama")
    prompt = engine._build_prompt(asset)
    # Should contain "and 10 more findings"
    assert "10 more findings" in prompt


def test_api_key_redaction_in_prompt():
    asset = _make_asset()
    asset.raw_findings = [
        Finding(
            type=FindingType.API_KEY_DETECTED,
            file_path="config.py",
            content="sk-abcdefghijklmnop1234567890",
            redacted_content="sk-abcde...7890",
            provider="openai",
        ),
    ]
    engine = LLMEngine(mode="ollama")
    prompt = engine._build_prompt(asset)
    # Prompt should use redacted version
    assert "sk-abcde...7890" in prompt
    assert "sk-abcdefghijklmnop1234567890" not in prompt
