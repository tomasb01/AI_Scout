"""MCP & Agent surface detection (Sprint 3, launch feature).

The first tool to map an organization's MCP/agent landscape from code:
MCP server configs, IDE agent directories (.claude/.cursor/.aider),
agent-instruction files, tool definitions and agent frameworks — plus
an **autonomy classification** (tool-calling │ supervised agent │
autonomous loop), an inventory attribute the G7 AIBOM guidance flagged
as a candidate for its next revision.

Two entry points:
* ``detect_agent_findings(path, name, content)`` — per-file detectors
  the Git scanner calls during its walk.
* ``classify_autonomy(asset)`` — post-analysis label from the code
  context (imports, api calls, loop signals).

Discipline: evidence, never a verdict. Autonomy is a structural
observation with confidence, and raw strings stay in finding content
(redaction handled by the caller), never in prose.
"""

from __future__ import annotations

import json
import re

from aiscout.models import AIAsset, Finding, FindingType

# Files that name an agent/MCP surface. Matched by exact name during the
# walk (added to the scanner's SPECIAL_FILENAMES).
AGENT_CONFIG_FILENAMES = {
    # MCP server configs (Claude Desktop / Code, Cursor, generic)
    "mcp.json", ".mcp.json", "claude_desktop_config.json",
    # Agent instruction / rules files
    "CLAUDE.md", ".cursorrules", ".clinerules", ".windsurfrules",
    "copilot-instructions.md", ".aider.conf.yml", "AGENTS.md",
    "GEMINI.md",
}

# Directories whose mere presence marks an AI coding-agent surface.
AGENT_DIR_MARKERS = {
    ".claude": "Claude Code / Desktop",
    ".cursor": "Cursor",
    ".aider": "Aider",
    ".windsurf": "Windsurf",
    ".github/copilot": "GitHub Copilot",
}

# Agent frameworks: import substring → (framework label, is_autonomous_capable)
_FRAMEWORK_SIGNALS: list[tuple[str, str]] = [
    ("crewai", "CrewAI"),
    ("autogen", "AutoGen"),
    ("langgraph", "LangGraph"),
    ("langchain.agents", "LangChain Agents"),
    ("llama_index.core.agent", "LlamaIndex Agent"),
    ("semantic_kernel", "Semantic Kernel"),
    ("openai_agents", "OpenAI Agents SDK"),
    ("from agents import", "OpenAI Agents SDK"),
    ("smolagents", "smolagents"),
    ("pydantic_ai", "PydanticAI"),
    ("google.adk", "Google ADK"),
    ("strands", "Strands Agents"),
]

# Tool-definition signals — the code exposes callable tools to an LLM.
_TOOL_DEF_SIGNALS = (
    "@tool", "@function_tool", "tool(", "tools=[", "toolset",
    "function_declarations", "tool_calls", "@mcp.tool", "list_tools",
    "register_tool", "structuredtool", "functiondeclaration",
)

# Autonomy: loop / long-running signals that lift "tool-calling" to a
# self-directed agent.
_AUTONOMY_LOOP_SIGNALS = (
    "while true", "while not done", "agentexecutor", "create_react_agent",
    "graph.stream", "workflow.run", "crew.kickoff", "run_until_complete",
    "max_iterations", "max_turns", "autonomous", "self-heal", "replan",
    ".invoke(", "runnablewithmessagehistory",
)
# Human-in-the-loop signals — a supervised agent, not a free-running loop.
_SUPERVISED_SIGNALS = (
    "human_input", "input(", "interrupt", "approval", "confirm",
    "human_in_the_loop", "ask_user", "require_approval", "breakpoint",
)

AUTONOMY_TOOL_CALLING = "tool_calling"
AUTONOMY_SUPERVISED = "supervised_agent"
AUTONOMY_AUTONOMOUS = "autonomous_loop"
AUTONOMY_NONE = "none"


def detect_agent_findings(
    file_path: str, name: str, content: str
) -> list[Finding]:
    """Per-file MCP/agent detectors (called from the scanner walk).

    ``name`` is the bare filename; ``file_path`` the repo-relative path.
    """
    findings: list[Finding] = []

    # ── MCP server config (JSON) ──
    if name in ("mcp.json", ".mcp.json", "claude_desktop_config.json") or (
        name.endswith(".json") and '"mcpServers"' in content
    ):
        findings.extend(_detect_mcp_servers(file_path, content))

    # ── Agent instruction / rules files ──
    if name in AGENT_CONFIG_FILENAMES and not name.endswith(".json"):
        findings.append(Finding(
            type=FindingType.CONFIG_DETECTED,
            file_path=file_path,
            content=f"agent instructions: {name}",
            provider="agent",
        ))

    # ── IDE agent directory markers (path-based) ──
    for marker, label in AGENT_DIR_MARKERS.items():
        if f"{marker}/" in f"{file_path}/" or file_path.startswith(f"{marker}/"):
            findings.append(Finding(
                type=FindingType.CONFIG_DETECTED,
                file_path=file_path,
                content=f"agent surface: {label}",
                provider="agent",
            ))
            break

    return findings


def _detect_mcp_servers(file_path: str, content: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return findings
    if not isinstance(data, dict):
        return findings
    servers = data.get("mcpServers") or data.get("mcp_servers") or {}
    if isinstance(servers, dict) and servers:
        for name in sorted(servers.keys()):
            entry = servers[name] if isinstance(servers[name], dict) else {}
            transport = (
                "remote" if entry.get("url") or entry.get("type") == "http"
                else "stdio"
            )
            findings.append(Finding(
                type=FindingType.CONFIG_DETECTED,
                file_path=file_path,
                content=f"mcp server: {name} ({transport})",
                provider="mcp",
            ))
    elif "mcpServers" in data or "mcp_servers" in data:
        findings.append(Finding(
            type=FindingType.CONFIG_DETECTED,
            file_path=file_path,
            content="mcp config (no servers)",
            provider="mcp",
        ))
    return findings


def detect_agent_frameworks(text_lower: str) -> list[str]:
    """Framework labels present in a lowercased code blob."""
    found: list[str] = []
    for needle, label in _FRAMEWORK_SIGNALS:
        if needle in text_lower and label not in found:
            found.append(label)
    return found


def frameworks_for_asset(asset: AIAsset) -> list[str]:
    """Framework labels for an asset — reads the same text as the
    autonomy classifier (imports, deps, calls, findings)."""
    return detect_agent_frameworks(_asset_text(asset))


def classify_autonomy(asset: AIAsset) -> tuple[str, str]:
    """Classify an asset's agent autonomy → (level, confidence).

    G7 discussion-section candidate: tool_calling │ supervised_agent │
    autonomous_loop. Returns ``none`` when there is no agent/tool
    evidence at all.
    """
    text = _asset_text(asset)
    tags = set(asset.tags)

    has_tools = (
        any(sig in text for sig in _TOOL_DEF_SIGNALS)
        or "mcp" in tags
    )
    frameworks = detect_agent_frameworks(text)
    is_agentish = bool(frameworks) or "agent" in tags or has_tools

    if not is_agentish:
        return AUTONOMY_NONE, "high"

    loop = any(sig in text for sig in _AUTONOMY_LOOP_SIGNALS)
    supervised = any(sig in text for sig in _SUPERVISED_SIGNALS)

    # A human-in-the-loop signal caps autonomy at "supervised" even when
    # a loop framework is present — the loop yields to a person.
    if loop and not supervised:
        confidence = "high" if frameworks else "medium"
        return AUTONOMY_AUTONOMOUS, confidence
    if (frameworks or "agent" in tags) and supervised:
        return AUTONOMY_SUPERVISED, "medium"
    if frameworks or "agent" in tags:
        # Framework present but no clear loop and no HITL — treat as
        # supervised by default (conservative: don't over-claim autonomy).
        return AUTONOMY_SUPERVISED, "low"
    return AUTONOMY_TOOL_CALLING, "medium" if has_tools else "low"


def _asset_text(asset: AIAsset) -> str:
    parts: list[str] = [asset.name.lower(), asset.file_path.lower()]
    parts.extend(d.lower() for d in asset.dependencies)
    for ctx in asset.code_contexts:
        for func in ctx.functions:
            parts.append(func.get("name", "").lower())
            parts.append(func.get("body_preview", "").lower())
        for call in ctx.api_calls:
            parts.append(call.get("target", "").lower())
        parts.extend(s.lower() for s in ctx.raw_snippets)
        parts.extend(p.lower() for p in ctx.prompts)
    for f in asset.raw_findings:
        parts.append(f.content.lower())
    return " ".join(parts)
