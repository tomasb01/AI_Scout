"""Tests for Sprint 3 — MCP & Agent scanner (launch feature).

Covers MCP server detection (config + code), agent-instruction files,
IDE surface markers, agent-framework detection, the autonomy
classification (tool_calling │ supervised_agent │ autonomous_loop), and
the live environment scanner.
"""

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from aiscout.cli import cli
from aiscout.engine.enrichment import enrich_assets
from aiscout.models import AIAsset, CodeContext, Finding, FindingType, ProviderInfo
from aiscout.scanners.agent_detect import (
    AUTONOMY_AUTONOMOUS,
    AUTONOMY_NONE,
    AUTONOMY_SUPERVISED,
    AUTONOMY_TOOL_CALLING,
    classify_autonomy,
    detect_agent_findings,
    detect_agent_frameworks,
)
from aiscout.scanners.mcp_env import scan_mcp_environment

FIXTURES_AGENTS = Path(__file__).parent / "fixtures_agents"


# ── Per-file detectors ─────────────────────────────────────────────────────


def test_mcp_config_servers_with_transport():
    content = json.dumps({"mcpServers": {
        "fs": {"command": "npx"},
        "remote": {"url": "https://mcp.example.com/sse", "type": "http"},
    }})
    findings = detect_agent_findings(".mcp.json", ".mcp.json", content)
    contents = {f.content for f in findings}
    assert "mcp server: fs (stdio)" in contents
    assert "mcp server: remote (remote)" in contents
    assert all(f.provider == "mcp" for f in findings)


def test_mcp_config_embedded_in_settings_json():
    content = json.dumps({"editor": "x", "mcpServers": {"db": {"command": "psql"}}})
    findings = detect_agent_findings(".cursor/settings.json", "settings.json", content)
    assert any("mcp server: db" in f.content for f in findings)


def test_empty_mcp_config_still_recorded():
    findings = detect_agent_findings("mcp.json", "mcp.json", json.dumps({"mcpServers": {}}))
    assert any("no servers" in f.content for f in findings)


def test_agent_instruction_files():
    for name in ("CLAUDE.md", ".cursorrules", "copilot-instructions.md", "AGENTS.md"):
        findings = detect_agent_findings(f"sub/{name}", name, "You are an agent.")
        assert any(f.provider == "agent" and "instructions" in f.content
                   for f in findings)


def test_ide_surface_markers():
    findings = detect_agent_findings(".cursor/mcp.json", "mcp.json", "{}")
    assert any("Cursor" in f.content for f in findings)
    findings = detect_agent_findings(".claude/settings.json", "settings.json", "{}")
    assert any("Claude" in f.content for f in findings)


# ── Framework detection ────────────────────────────────────────────────────


def test_detect_frameworks():
    assert detect_agent_frameworks("import crewai\nfrom autogen import x") == [
        "CrewAI", "AutoGen",
    ]
    assert detect_agent_frameworks("from langgraph.graph import StateGraph") == [
        "LangGraph",
    ]
    # the langchain.agents substring must not be misread as OpenAI Agents SDK
    assert detect_agent_frameworks("from langchain.agents import x") == [
        "LangChain Agents",
    ]


# ── Autonomy classification ────────────────────────────────────────────────


def _asset(name="a", *, deps=None, body="", tags=None, findings=None):
    return AIAsset(
        name=name, repository="repo", file_path=f"{name}.py",
        dependencies=deps or [], tags=tags or [],
        raw_findings=findings or [],
        code_contexts=[CodeContext(
            file_path=f"{name}.py", language="python",
            functions=[{"name": "run", "body_preview": body}],
        )],
    )


def test_autonomy_autonomous_loop():
    asset = _asset(
        deps=["langgraph"],
        body="agent = create_react_agent(llm, tools=[t])\nwhile True: agent.invoke(x)",
    )
    level, conf = classify_autonomy(asset)
    assert level == AUTONOMY_AUTONOMOUS
    assert conf == "high"


def test_autonomy_supervised_when_human_in_loop():
    asset = _asset(
        deps=["langchain"],
        body="result = agent.invoke(task)\napproval = input('Approve? ')",
        tags=["agent"],
    )
    level, conf = classify_autonomy(asset)
    assert level == AUTONOMY_SUPERVISED


def test_autonomy_tool_calling_without_framework():
    asset = _asset(body="tools=[search]\nclient.chat.completions.create(tools=tools)")
    level, conf = classify_autonomy(asset)
    assert level == AUTONOMY_TOOL_CALLING


def test_autonomy_none_for_plain_inference():
    asset = _asset(body="client.chat.completions.create(model='gpt-4o')")
    level, conf = classify_autonomy(asset)
    assert level == AUTONOMY_NONE


def test_hitl_signal_caps_autonomy_even_with_loop():
    asset = _asset(
        deps=["langgraph"],
        body="while True:\n  r = agent.invoke(x)\n  if require_approval(): input('ok?')",
    )
    level, _ = classify_autonomy(asset)
    assert level == AUTONOMY_SUPERVISED  # not autonomous_loop


# ── Enrichment integration ─────────────────────────────────────────────────


def test_enrichment_stamps_autonomy_and_frameworks():
    asset = _asset(
        deps=["langgraph"],
        body="agent = create_react_agent(llm, tools=[t])\nwhile True: agent.invoke(x)",
    )
    enrich_assets([asset])
    assert asset.autonomy == AUTONOMY_AUTONOMOUS
    assert "LangGraph" in asset.agent_frameworks


# ── End-to-end on fixtures ─────────────────────────────────────────────────


def _scan(path):
    from aiscout.engine.aggregation import aggregate_scan_result
    from aiscout.engine.code_analyzer import analyze_assets
    from aiscout.engine.data_flow import build_data_flows
    from aiscout.scanners.git_scanner import GitScanner

    result = GitScanner(repo_path=str(path)).scan()
    analyze_assets(result.assets, str(path))
    build_data_flows(result.assets)
    aggregate_scan_result(result)
    enrich_assets(result.assets)
    return result


def test_fixtures_end_to_end():
    result = _scan(FIXTURES_AGENTS)
    by_name = {a.name: a for a in result.assets}

    mcp = next(a for a in result.assets if "mcp server" in
               " ".join(f.content for f in a.raw_findings))
    servers = [f.content for f in mcp.raw_findings if f.content.startswith("mcp server")]
    assert len(servers) == 3
    assert any("(remote)" in s for s in servers)

    lg = next(a for a in result.assets if "LangGraph" in a.agent_frameworks)
    assert lg.autonomy == AUTONOMY_AUTONOMOUS
    # the CLAUDE.md instruction file was detected
    assert any("agent instructions: CLAUDE.md" in f.content for f in lg.raw_findings)


def test_json_export_carries_autonomy():
    from aiscout.report.json_export import JSONExporter

    result = _scan(FIXTURES_AGENTS)
    insights = enrich_assets(result.assets)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "r.json"
        JSONExporter([result], output_path=str(out), insights=insights).generate()
        data = json.loads(out.read_text())
    assert data["schema_version"] == "1.6.0"
    autonomies = {s["autonomy"]["level"] for s in data["solutions"]}
    assert AUTONOMY_AUTONOMOUS in autonomies
    lg = next(s for s in data["solutions"] if "LangGraph" in s["autonomy"]["frameworks"])
    assert lg["autonomy"]["confidence"] == "high"


# ── Live environment scanner ───────────────────────────────────────────────


def test_mcp_env_scanner_reads_custom_config(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "notion": {"command": "npx"},
        "sentry": {"url": "https://mcp.sentry.dev/sse", "type": "http"},
    }}))
    result = scan_mcp_environment(extra_paths=[str(cfg)])
    names = {s.name for s in result.servers}
    assert {"notion", "sentry"} <= names
    sentry = next(s for s in result.servers if s.name == "sentry")
    assert sentry.transport == "remote"
    assert sentry.command == "mcp.sentry.dev"  # host only, no token/path


def test_mcp_env_scanner_handles_missing_paths(tmp_path):
    # A nonexistent extra path must not crash; known locations are still
    # enumerated. (The dev machine may itself have live MCP configs — we
    # assert robustness, not a specific server count.)
    result = scan_mcp_environment(extra_paths=[str(tmp_path / "nope.json")])
    assert result.configs_checked
    assert str(tmp_path / "nope.json") in result.configs_checked


def test_cli_mcp_command_with_custom_path(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"db": {"command": "psql"}}}))
    out = tmp_path / "env.json"
    runner = CliRunner()
    result = runner.invoke(cli, ["mcp", "--path", str(cfg), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert "db" in result.output
    data = json.loads(out.read_text())
    assert data["server_count"] >= 1
