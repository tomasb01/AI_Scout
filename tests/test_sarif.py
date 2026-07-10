"""Tests for the SARIF 2.1.0 export (Sprint 1).

Asserts the structural contract GitHub code scanning requires: rules
referenced by every result, levels, relative URIs, stable partial
fingerprints, redacted messages — and byte-identical output across
runs (determinism is a precondition for diff and upload dedup).
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from click.testing import CliRunner

from aiscout.cli import cli
from aiscout.engine.enrichment import enrich_assets
from aiscout.models import (
    AIAsset,
    Finding,
    FindingType,
    ProviderInfo,
    ScanResult,
    Severity,
)
from aiscout.report.sarif_export import SarifExporter
from aiscout.scanners.git_scanner import _assign_stable_ids

FIXTURES = Path(__file__).parent / "fixtures"


def _scan_result(repo: str = "test-repo") -> ScanResult:
    key_finding = Finding(
        type=FindingType.API_KEY_DETECTED,
        file_path="app/main.py", line_number=12,
        content="sk-live-secret", redacted_content="sk-...ret",
        provider="openai",
    )
    import_finding = Finding(
        type=FindingType.IMPORT_DETECTED,
        file_path="app/main.py", line_number=1,
        content="import openai", provider="openai",
    )
    dep_finding = Finding(
        type=FindingType.DEPENDENCY_DETECTED,
        file_path="requirements.txt",
        content="openai>=1.0", provider="openai",
    )
    _assign_stable_ids([key_finding, import_finding, dep_finding], repo)
    asset = AIAsset(
        name="payment bot", provider=ProviderInfo(name="openai"),
        repository=repo, root_path="app", file_path="app/main.py",
        raw_findings=[key_finding, import_finding, dep_finding],
    )
    return ScanResult(
        scanner="git_scanner",
        started_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
        completed_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
        assets=[asset],
        metadata={"repository": repo, "files_scanned": 3},
    )


def _generate(scan_results, **kwargs) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.sarif"
        SarifExporter(scan_results, output_path=str(out), **kwargs).generate()
        return json.loads(out.read_text())


def test_sarif_shape_and_github_requirements():
    doc = _generate([_scan_result()])
    assert doc["version"] == "2.1.0"
    assert "sarif-2.1.0" in doc["$schema"]
    run = doc["runs"][0]
    driver = run["tool"]["driver"]
    assert driver["name"] == "AI Scout"
    assert driver["semanticVersion"]
    assert run["columnKind"] == "utf16CodeUnits"

    # default: security findings only (the key), no discovery noise
    assert len(run["results"]) == 1
    result = run["results"][0]
    assert result["ruleId"] == "SEC-KEY-001"
    assert result["level"] == "error"

    # every result's ruleId + ruleIndex resolves in the rules array
    rules = driver["rules"]
    assert rules[result["ruleIndex"]]["id"] == result["ruleId"]
    key_rule = rules[result["ruleIndex"]]
    assert key_rule["defaultConfiguration"]["level"] == "error"
    assert key_rule["properties"]["security-severity"] == "9.1"
    assert "security" in key_rule["properties"]["tags"]

    # location: repo-relative URI + line region
    loc = result["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uri"] == "app/main.py"
    assert not loc["artifactLocation"]["uri"].startswith("/")
    assert loc["region"]["startLine"] == 12


def test_sarif_messages_are_redacted_and_name_the_solution():
    insights = None
    scan = _scan_result()
    insights = enrich_assets(scan.assets)
    doc = _generate([scan], insights=insights)
    message = doc["runs"][0]["results"][0]["message"]["text"]
    assert "sk-live-secret" not in message  # raw key never leaves
    assert "sk-...ret" in message
    assert "Rotate" in message


def test_sarif_fingerprints_use_stable_finding_ids():
    scan = _scan_result()
    finding_id = scan.assets[0].raw_findings[0].id
    assert finding_id.startswith("f-")
    doc = _generate([scan])
    fp = doc["runs"][0]["results"][0]["partialFingerprints"]
    assert fp["aiScoutFindingId/v1"] == finding_id


def test_sarif_include_discovery_adds_note_results():
    doc = _generate([_scan_result()], include_discovery=True)
    run = doc["runs"][0]
    levels = [r["level"] for r in run["results"]]
    assert levels.count("error") == 1
    assert levels.count("note") == 2  # import + dependency
    rule_ids = {r["id"] for r in run["tool"]["driver"]["rules"]}
    assert {"SEC-KEY-001", "DISC-IMP-001", "DISC-DEP-001"} <= rule_ids
    # results without a line number carry no region (allowed by SARIF)
    dep = next(r for r in run["results"] if r["ruleId"] == "DISC-DEP-001")
    assert "region" not in dep["locations"][0]["physicalLocation"]


def test_sarif_multi_repo_runs_carry_automation_details():
    doc = _generate([_scan_result("repo-a"), _scan_result("repo-b")])
    assert len(doc["runs"]) == 2
    ids = {r["automationDetails"]["id"] for r in doc["runs"]}
    assert ids == {"aiscout/repo-a", "aiscout/repo-b"}


def test_sarif_output_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.sarif", Path(tmp) / "b.sarif"
        SarifExporter([_scan_result()], output_path=str(a)).generate()
        SarifExporter([_scan_result()], output_path=str(b)).generate()
        assert a.read_bytes() == b.read_bytes()


def test_cli_autodetects_sarif_extension():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "scan.sarif"
        result = runner.invoke(cli, [
            "scan", "--local", str(FIXTURES), "--no-llm",
            "--output", str(out),
        ])
        assert result.exit_code == 0, result.output
        doc = json.loads(out.read_text())
    assert doc["version"] == "2.1.0"
    results = doc["runs"][0]["results"]
    # the fixture tree contains one hardcoded key
    assert any(r["ruleId"] == "SEC-KEY-001" for r in results)
    assert all(r["level"] != "note" for r in results)  # default: no inventory


def test_cli_sarif_include_discovery_flag():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "scan.sarif"
        result = runner.invoke(cli, [
            "scan", "--local", str(FIXTURES), "--no-llm",
            "--sarif-include-discovery", "--output", str(out),
        ])
        assert result.exit_code == 0, result.output
        doc = json.loads(out.read_text())
    assert any(
        r["level"] == "note" for r in doc["runs"][0]["results"]
    )
