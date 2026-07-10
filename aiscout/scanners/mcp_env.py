"""Live MCP / agent environment scanner (Sprint 3).

Scans the well-known locations where AI coding agents keep their MCP
server configuration on the machine running Scout — Claude Desktop,
Claude Code, Cursor, VS Code, Windsurf — and reports which MCP servers
are wired in, with what transport. This is the "what agents are
configured on this box" companion to the repo scanner's "what agents
live in this code".

Read-only, offline, self-hosted: reads config files, never launches a
server. Paths are the documented defaults per platform; missing files
are simply absent from the result (not an error).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class McpServerEntry:
    name: str
    source: str            # which app/config declared it
    config_path: str
    transport: str         # stdio | remote
    command: str = ""      # redacted-safe: command/url without args/secrets

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "config_path": self.config_path,
            "transport": self.transport,
            "command": self.command,
        }


@dataclass
class McpEnvResult:
    servers: list[McpServerEntry] = field(default_factory=list)
    configs_found: list[str] = field(default_factory=list)
    configs_checked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "servers": [s.to_dict() for s in self.servers],
            "configs_found": self.configs_found,
            "configs_checked": self.configs_checked,
            "server_count": len(self.servers),
        }


def _candidate_configs() -> list[tuple[str, Path]]:
    """(source label, path) for known agent MCP configs on this platform."""
    home = Path.home()
    candidates: list[tuple[str, Path]] = []

    # Claude Desktop
    if os.name == "nt":
        appdata = Path(os.environ.get("APPDATA", home / "AppData/Roaming"))
        candidates.append(("Claude Desktop", appdata / "Claude/claude_desktop_config.json"))
    else:
        candidates.append((
            "Claude Desktop",
            home / "Library/Application Support/Claude/claude_desktop_config.json",
        ))
        candidates.append((
            "Claude Desktop",
            home / ".config/Claude/claude_desktop_config.json",
        ))

    # Claude Code (project + user scope)
    candidates.append(("Claude Code (user)", home / ".claude.json"))
    candidates.append(("Claude Code (user)", home / ".claude/settings.json"))
    candidates.append(("Claude Code (project)", Path(".mcp.json")))

    # Cursor
    candidates.append(("Cursor (global)", home / ".cursor/mcp.json"))
    candidates.append(("Cursor (project)", Path(".cursor/mcp.json")))

    # VS Code (workspace + user)
    candidates.append(("VS Code (workspace)", Path(".vscode/mcp.json")))
    if os.name != "nt":
        candidates.append((
            "VS Code (user)",
            home / "Library/Application Support/Code/User/mcp.json",
        ))
        candidates.append((
            "VS Code (user)",
            home / ".config/Code/User/mcp.json",
        ))

    # Windsurf
    candidates.append(("Windsurf", home / ".codeium/windsurf/mcp_config.json"))

    return candidates


def scan_mcp_environment(extra_paths: list[str] | None = None) -> McpEnvResult:
    """Scan known agent MCP configs; ``extra_paths`` adds custom files."""
    result = McpEnvResult()
    candidates = _candidate_configs()
    for p in (extra_paths or []):
        candidates.append(("Custom", Path(p)))

    seen_paths: set[str] = set()
    for source, path in candidates:
        rp = str(path)
        result.configs_checked.append(rp)
        if not path.is_file() or path.is_symlink():
            continue
        if rp in seen_paths:
            continue
        seen_paths.add(rp)
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        result.configs_found.append(rp)
        result.servers.extend(_extract_servers(source, rp, data))

    result.servers.sort(key=lambda s: (s.source, s.name))
    return result


def _extract_servers(source: str, path: str, data: dict) -> list[McpServerEntry]:
    """Pull MCP server entries from a config dict (schema varies by app)."""
    # Claude Code stores servers per project under projects[cwd].mcpServers;
    # every other app uses a top-level mcpServers/servers/mcp.servers block.
    blocks: list[dict] = []
    for key in ("mcpServers", "mcp_servers", "servers"):
        if isinstance(data.get(key), dict):
            blocks.append(data[key])
    if isinstance(data.get("mcp"), dict) and isinstance(data["mcp"].get("servers"), dict):
        blocks.append(data["mcp"]["servers"])
    if isinstance(data.get("projects"), dict):
        for proj in data["projects"].values():
            if isinstance(proj, dict) and isinstance(proj.get("mcpServers"), dict):
                blocks.append(proj["mcpServers"])

    entries: list[McpServerEntry] = []
    seen: set[str] = set()
    for servers in blocks:
        for name, cfg in servers.items():
            if name in seen:
                continue
            seen.add(name)
            cfg = cfg if isinstance(cfg, dict) else {}
            if cfg.get("url") or cfg.get("type") in ("http", "sse", "remote"):
                transport = "remote"
                command = _safe_host(cfg.get("url", ""))
            else:
                transport = "stdio"
                command = str(cfg.get("command", "")).split("/")[-1]
            entries.append(McpServerEntry(
                name=name, source=source, config_path=path,
                transport=transport, command=command,
            ))
    return entries


def _safe_host(url: str) -> str:
    """Host only — never query strings or embedded tokens."""
    from urllib.parse import urlsplit

    try:
        return urlsplit(url).netloc or ""
    except ValueError:
        return ""
