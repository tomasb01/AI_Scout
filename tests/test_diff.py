"""Tests for Sprint 2 — scan diff, finding workflow states, baseline delta.

Acceptance (dev plan): two scans of the same org produce a correct
delta report; accepted_risk survives across scans.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from click.testing import CliRunner

from aiscout.cli import cli
from aiscout.engine.diff import ScanDelta, diff_exports
from aiscout.engine.enrichment import enrich_assets
from aiscout.engine.findings_state import FindingsState
from aiscout.models import (
    AIAsset,
    Finding,
    FindingType,
    ProviderInfo,
    RiskStatus,
    ScanResult,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _solution(sid, name, *, risk="no_findings", findings=None, models=None,
              provider="openai", role="application"):
    return {
        "id": sid, "name": name, "repository": "repo", "path": name,
        "risk_status": risk, "category": "AI Agents", "role": role,
        "provider": {"name": provider},
        "findings": findings or [],
        "model_refs": [{"model": m} for m in (models or [])],
    }


def _key_finding(fid, path="app.py", line=3):
    return {
        "id": fid, "rule": {"id": "SEC-KEY-001", "version": 1},
        "type": "api_key_detected", "severity": "critical",
        "file_path": path, "line_number": line,
        "content": "sk-...XXX", "provider": "openai",
    }


def _export(solutions):
    return {
        "schema_version": "1.5.0",
        "generated_at": "2026-07-01T12:00:00+00:00",
        "solutions": solutions,
    }


# ── Diff engine ────────────────────────────────────────────────────────────


def test_diff_added_removed_changed():
    old = _export([
        _solution("sol-a", "keeper"),
        _solution("sol-b", "goner"),
        _solution("sol-c", "changer", risk="no_findings"),
    ])
    new = _export([
        _solution("sol-a", "keeper"),
        _solution("sol-c", "changer", risk="critical",
                  findings=[_key_finding("f-new1")]),
        _solution("sol-d", "newcomer", provider="anthropic"),
    ])
    delta = diff_exports(old, new)

    assert [s["id"] for s in delta.added_solutions] == ["sol-d"]
    assert [s["id"] for s in delta.removed_solutions] == ["sol-b"]
    assert [s["id"] for s in delta.changed_solutions] == ["sol-c"]
    changes = delta.changed_solutions[0]["changes"]
    assert changes["risk_status"] == {"from": "no_findings", "to": "critical"}
    assert changes["findings"]["added"] == ["f-new1"]
    assert delta.new_providers == ["anthropic"]
    assert [f["id"] for f in delta.new_key_findings] == ["f-new1"]


def test_diff_resolved_keys_and_model_changes():
    old = _export([
        _solution("sol-a", "app", risk="critical",
                  findings=[_key_finding("f-old1")], models=["gpt-4o"]),
    ])
    new = _export([
        _solution("sol-a", "app", risk="no_findings", models=["gpt-4o-mini"]),
    ])
    delta = diff_exports(old, new)
    assert [f["id"] for f in delta.resolved_key_findings] == ["f-old1"]
    assert delta.resolved_key_findings[0]["status"] == "resolved"
    changes = delta.changed_solutions[0]["changes"]
    assert changes["model_refs"]["added"] == ["gpt-4o-mini"]
    assert changes["model_refs"]["removed"] == ["gpt-4o"]


def test_diff_ignores_dependency_manifests_in_solution_counts():
    old = _export([])
    new = _export([
        _solution("sol-m", "deps", role="dependency_manifest"),
        _solution("sol-a", "real app"),
    ])
    delta = diff_exports(old, new)
    assert [s["id"] for s in delta.added_solutions] == ["sol-a"]
    assert delta.new_total == 1


def test_diff_no_change_is_empty():
    export = _export([_solution("sol-a", "app")])
    delta = diff_exports(export, export)
    assert delta.counts() == {
        "added": 0, "removed": 0, "changed": 0,
        "new_providers": 0, "new_key_findings": 0,
        "resolved_key_findings": 0,
    }


def test_delta_insight_values_feed_i09():
    delta = ScanDelta(old_generated_at="2026-06-01T10:00:00+00:00")
    delta.added_solutions = [{"id": "x"}]
    delta.new_providers = ["groq"]
    values = delta.insight_values()
    assert values == {
        "prev_date": "2026-06-01", "added": 1, "removed": 0,
        "new_providers": 1,
    }


# ── Findings state ─────────────────────────────────────────────────────────


def _asset_with_key(repo="repo"):
    from aiscout.scanners.git_scanner import _assign_stable_ids
    finding = Finding(
        type=FindingType.API_KEY_DETECTED, file_path="app.py", line_number=3,
        content="sk-XXX", redacted_content="sk-...XXX", provider="openai",
    )
    _assign_stable_ids([finding], repo)
    return AIAsset(
        name="leaky", provider=ProviderInfo(name="openai"),
        repository=repo, root_path="app", file_path="app.py",
        raw_findings=[finding],
    )


def test_accepted_risk_survives_across_scans(tmp_path):
    """Dev-plan acceptance: accepted_risk persists between scans."""
    state_file = tmp_path / "findings.json"
    asset = _asset_with_key()
    fid = asset.raw_findings[0].id

    state = FindingsState.load(state_file)
    state.accept(fid, note="rotated quarterly")
    state.save()

    # a later, fresh scan (new process): load state, stamp assets
    asset2 = _asset_with_key()  # same repo → same stable finding id
    state2 = FindingsState.load(state_file)
    counts = state2.apply_to_assets([asset2])
    assert asset2.raw_findings[0].status == "accepted_risk"
    assert counts["accepted_risk"] == 1
    assert state2.entries[fid]["note"] == "rotated quarterly"


def test_first_seen_stamped_once(tmp_path):
    state_file = tmp_path / "findings.json"
    asset = _asset_with_key()
    state = FindingsState.load(state_file)
    state.apply_to_assets([asset])
    state.save()
    first = asset.raw_findings[0].first_seen
    assert first

    asset2 = _asset_with_key()
    state2 = FindingsState.load(state_file)
    state2.apply_to_assets([asset2])
    assert asset2.raw_findings[0].first_seen == first  # not re-stamped


def test_accepted_key_downgrades_critical_and_stays_visible():
    asset = _asset_with_key()
    asset.raw_findings[0].status = "accepted_risk"
    insights = enrich_assets([asset])
    assert asset.risk_status == RiskStatus.REVIEW  # not critical anymore
    titles = [r.title for r in insights[asset.id].risk_reasons]
    assert "Hardcoded API key accepted as risk" in titles
    assert "Hardcoded API key in source code" not in titles


def test_open_key_still_critical():
    asset = _asset_with_key()
    enrich_assets([asset])
    assert asset.risk_status == RiskStatus.CRITICAL


def test_reopen_returns_to_open(tmp_path):
    state_file = tmp_path / "findings.json"
    state = FindingsState.load(state_file)
    state.accept("f-abc")
    state.reopen("f-abc")
    state.save()
    assert FindingsState.load(state_file).status_of("f-abc") == "open"


# ── CLI ────────────────────────────────────────────────────────────────────


def _write_export(path: Path, solutions):
    path.write_text(json.dumps(_export(solutions)))
    return path


def test_cli_diff_and_fail_on_new_critical(tmp_path):
    old = _write_export(tmp_path / "old.json", [_solution("sol-a", "app")])
    new = _write_export(tmp_path / "new.json", [
        _solution("sol-a", "app"),
        _solution("sol-b", "leaky", risk="critical",
                  findings=[_key_finding("f-k1")]),
    ])
    runner = CliRunner()
    result = runner.invoke(cli, [
        "diff", str(old), str(new), "-o", str(tmp_path / "delta.json"),
    ])
    assert result.exit_code == 0, result.output
    delta = json.loads((tmp_path / "delta.json").read_text())
    assert delta["counts"]["added"] == 1
    assert delta["counts"]["new_key_findings"] == 1

    result = runner.invoke(cli, [
        "diff", str(old), str(new), "--fail-on-new-critical",
    ])
    assert result.exit_code == 3

    # reversed direction: keys resolved, no failure
    result = runner.invoke(cli, [
        "diff", str(new), str(old), "--fail-on-new-critical",
    ])
    assert result.exit_code == 0


def test_cli_findings_accept_and_list(tmp_path):
    state_file = tmp_path / "state.json"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "findings", "accept", "f-12345", "--note", "known demo key",
        "--state-file", str(state_file),
    ])
    assert result.exit_code == 0, result.output
    result = runner.invoke(cli, [
        "findings", "list", "--state-file", str(state_file),
    ])
    assert result.exit_code == 0
    assert "f-12345" in result.output
    assert "accepted_risk" in result.output


def test_cli_scan_with_baseline_emits_delta_and_i09(tmp_path):
    runner = CliRunner()
    base = tmp_path / "base.json"
    result = runner.invoke(cli, [
        "scan", "--local", str(FIXTURES), "--no-llm", "--output", str(base),
    ])
    assert result.exit_code == 0, result.output

    after = tmp_path / "after.json"
    result = runner.invoke(cli, [
        "scan", "--local", str(FIXTURES), "--no-llm",
        "--baseline", str(base), "--output", str(after),
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(after.read_text())
    assert data["schema_version"] == "1.5.0"
    assert data["delta"]["counts"] == {
        "added": 0, "removed": 0, "changed": 0,
        "new_providers": 0, "new_key_findings": 0,
        "resolved_key_findings": 0,
    }
    i09 = [i for i in data["insights"] if i["id"] == "I-09"]
    assert i09 and "no new solutions" in i09[0]["text"]
    assert not i09[0]["suppressed"]


def test_cli_scan_findings_state_downgrades_accepted(tmp_path):
    runner = CliRunner()
    base = tmp_path / "base.json"
    result = runner.invoke(cli, [
        "scan", "--local", str(FIXTURES), "--no-llm", "--output", str(base),
    ])
    assert result.exit_code == 0, result.output
    key_ids = [
        f["id"] for s in json.loads(base.read_text())["solutions"]
        for f in s["findings"] if f["rule"]["id"] == "SEC-KEY-001"
    ]
    assert key_ids

    state_file = tmp_path / "state.json"
    for fid in key_ids:
        runner.invoke(cli, [
            "findings", "accept", fid, "--state-file", str(state_file),
        ])

    after = tmp_path / "after.json"
    result = runner.invoke(cli, [
        "scan", "--local", str(FIXTURES), "--no-llm",
        "--findings-state", str(state_file), "--output", str(after),
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(after.read_text())
    statuses = {
        f["id"]: f["status"] for s in data["solutions"]
        for f in s["findings"]
    }
    for fid in key_ids:
        assert statuses[fid] == "accepted_risk"
    # no open criticals left → I-02 gone, critical count zero
    assert data["summary"]["critical"] == 0
    assert not any(i["id"] == "I-02" for i in data["insights"])


def test_html_report_renders_delta_box_and_accepted_badge(tmp_path):
    from aiscout.report.html import ReportGenerator
    from aiscout.engine.diff import ScanDelta

    asset = _asset_with_key()
    asset.raw_findings[0].status = "accepted_risk"
    scan = ScanResult(
        scanner="git_scanner",
        started_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
        assets=[asset],
        metadata={"repository": "repo", "files_scanned": 1},
    )
    delta = ScanDelta(old_generated_at="2026-06-01T00:00:00+00:00")
    delta.added_solutions = [{
        "id": "sol-x", "name": "newcomer", "repository": "repo",
        "path": "svc", "risk_status": "no_findings", "category": "AI Agents",
    }]
    out = tmp_path / "r.html"
    ReportGenerator([scan], output_path=str(out), delta=delta).generate()
    html = out.read_text()
    assert "Changes since 2026-06-01" in html
    assert "newcomer" in html
    assert "ACCEPTED" in html
