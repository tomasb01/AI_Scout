"""Tests for AI Scout data models."""

from datetime import datetime, timezone

from aiscout.models import (
    AIAsset,
    AssetType,
    ClassificationResult,
    Confidence,
    DataCategory,
    Finding,
    FindingType,
    ScanResult,
)


def test_ai_asset_gets_uuid():
    asset = AIAsset(name="test")
    assert asset.id
    assert len(asset.id) == 36  # UUID format


def test_ai_asset_unique_ids():
    a = AIAsset(name="a")
    b = AIAsset(name="b")
    assert a.id != b.id


def test_ai_asset_defaults():
    asset = AIAsset(name="test")
    assert asset.type == AssetType.CUSTOM_CODE
    assert asset.owner == "unknown"
    assert asset.risk_score == 0.0
    assert asset.raw_findings == []


def test_scan_result_merge():
    r1 = ScanResult(
        scanner="git",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        assets=[AIAsset(name="a1")],
        errors=["err1"],
    )
    r2 = ScanResult(
        scanner="git",
        started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        assets=[AIAsset(name="a2"), AIAsset(name="a3")],
        errors=[],
    )
    merged = r1.merge(r2)
    assert len(merged.assets) == 3
    assert merged.errors == ["err1"]
    assert merged.started_at == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_enum_serialization():
    asset = AIAsset(name="test", type=AssetType.AGENT)
    data = asset.model_dump()
    assert data["type"] == "agent"

    restored = AIAsset.model_validate(data)
    assert restored.type == AssetType.AGENT


def test_finding_with_redacted_content():
    finding = Finding(
        type=FindingType.API_KEY_DETECTED,
        file_path="config.py",
        line_number=10,
        content="sk-abcdefghijklmnopqrstuvwxyz",
        redacted_content="sk-abcde...wxyz",
        provider="openai",
    )
    assert finding.redacted_content == "sk-abcde...wxyz"
    assert finding.provider == "openai"


def test_classification_result_defaults():
    cr = ClassificationResult()
    assert cr.categories == []
    assert cr.confidence == Confidence.LOW
    assert cr.risk_score == 0.0


def test_data_category_values():
    assert DataCategory.PII == "pii"
    assert DataCategory.CONFIDENTIAL == "confidential"
