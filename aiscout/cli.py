"""CLI entry point for AI Scout."""

from __future__ import annotations

import ipaddress
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from aiscout import __version__
from aiscout.engine.code_analyzer import analyze_assets
from aiscout.engine.data_flow import build_data_flows
from aiscout.engine.enrichment import enrich_assets
from aiscout.engine.llm import LLMEngine
from aiscout.knowledge.providers import get_provider
from aiscout.models import FindingType, RiskStatus
from aiscout.report.html import ReportGenerator
from aiscout.scanners.git_scanner import GitScanner
from aiscout.scanners.github_org import OrgEnumerationError, enumerate_org_repos

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="aiscout")
def cli():
    """AI Scout — Enterprise AI Discovery & Security Assessment Tool."""
    pass


@cli.command()
@click.option("--repo", "-r", multiple=True, help="Git repo URL (repeatable)")
@click.option("--local", "-l", multiple=True, help="Local repo path (repeatable)")
@click.option(
    "--org",
    multiple=True,
    help="GitHub organization or user (URL or name) — scans all visible repos (repeatable)",
)
@click.option("--config", "-c", type=click.Path(exists=True), help="YAML config file")
@click.option("--token", "-t", envvar="AISCOUT_GIT_TOKEN", help="Git access token")
@click.option("--branch", "-b", default="main", help="Default branch to scan")
@click.option("--include-archived", is_flag=True, help="Include archived repos in --org scans")
@click.option("--include-forks", is_flag=True, help="Include forked repos in --org scans")
@click.option(
    "--max-repos",
    type=int,
    default=200,
    help="Cap on repos enumerated per --org (safety limit)",
)
@click.option("--output", "-o", default="aiscout_report.html", help="Output path (.html or .json)")
@click.option(
    "--llm-url",
    default="http://localhost:11434",
    help="LLM backend URL (Ollama default; any OpenAI-compatible host "
         "for --llm-mode openai, e.g. http://vllm:8000)",
)
@click.option("--llm-model", default="qwen2.5-coder:7b", help="LLM model name / deployment id")
@click.option(
    "--llm-mode",
    type=click.Choice(["ollama", "openai"]),
    default="ollama",
    help="LLM transport. 'ollama' = native Ollama REST API. "
         "'openai' = any OpenAI-compatible /v1/chat/completions endpoint "
         "(vLLM, LocalAI, LM Studio, llama.cpp, TGI, Together, Groq, …)",
)
@click.option(
    "--llm-key",
    envvar="AISCOUT_LLM_KEY",
    help="Bearer token for OpenAI-compatible backends that require auth",
)
@click.option("--no-llm", is_flag=True, help="Skip LLM classification")
@click.option(
    "--manifests-only",
    is_flag=True,
    help="Low-sensitivity scan: read only dependency manifests "
         "(requirements.txt, package.json, pyproject.toml), never source code",
)
@click.option(
    "--strict",
    is_flag=True,
    help="CI mode: exit non-zero when the report QA linter suppressed "
         "any sentence (production default is degrade, not fail)",
)
def scan(
    repo, local, org, config, token, branch,
    include_archived, include_forks, max_repos, output,
    llm_url, llm_model, llm_mode, llm_key, no_llm, manifests_only, strict,
):
    """Scan Git repositories for AI assets."""
    # Build list of repos to scan
    repos, org_inventory = _build_repo_list(
        repo, local, org, config, token, branch,
        include_archived=include_archived,
        include_forks=include_forks,
        max_repos=max_repos,
    )

    if not repos:
        console.print("[red]Error:[/] No repositories specified.")
        console.print("Use --repo, --local, or --config to specify repositories.")
        sys.exit(1)

    # Override LLM/output settings from config if present
    llm_config, output = _apply_config_overrides(
        config, llm_url, llm_model, llm_mode, llm_key, output
    )
    if llm_config:
        llm_url = llm_config.get("url", llm_url)
        llm_model = llm_config.get("model", llm_model)
        llm_mode = llm_config.get("mode", llm_mode)
        llm_key = llm_config.get("key", llm_key)

    mode_note = "  ·  manifests-only" if manifests_only else ""
    console.print(Panel(
        f"[bold]Scanning {len(repos)} repositor{'y' if len(repos) == 1 else 'ies'}[/]{mode_note}\n"
        f"LLM: {'disabled' if no_llm else f'{llm_mode} ({llm_model})'}",
        title="AI Scout",
        border_style="blue",
    ))

    # Scan each repo
    scan_results = []
    scanners = []  # keep for cleanup after code analysis
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for entry in repos:
            name = entry.get("name", entry.get("url", entry.get("path", "unknown")))
            task = progress.add_task(f"Scanning {name}...", total=None)

            scanner = GitScanner(
                repo_path=entry.get("path"),
                repo_url=entry.get("url"),
                branch=entry.get("branch", branch),
                token=entry.get("token", token),
                manifests_only=manifests_only,
            )

            result = scanner.scan()
            scan_results.append(result)
            scanners.append(scanner)

            if result.errors:
                for err in result.errors:
                    console.print(f"  [red]Error:[/] {err}")
            else:
                console.print(
                    f"  Found [bold]{len(result.assets)}[/] AI asset(s) "
                    f"in {result.metadata.get('files_scanned', 0)} files"
                )

            progress.remove_task(task)

    # Code context analysis (reads files from repo before cleanup).
    # Skipped in manifests-only mode — it reads source, which that mode avoids.
    if not manifests_only:
        for result in scan_results:
            repo_root = result.metadata.get("repo_root")
            if repo_root and result.assets:
                analyze_assets(result.assets, repo_root)

        # Build data flow maps (Step 2 — rule-based, no LLM)
        for result in scan_results:
            if result.assets:
                build_data_flows(result.assets)

    # Aggregation boundary (Sprint 0.3) — AFTER analysis, because solution
    # identity comes from code purpose: only same-boundary components with
    # an identical data-flow fingerprint merge into a variant group.
    # Without flow maps (manifests-only) nothing merges, by design.
    from aiscout.engine.aggregation import aggregate_scan_result
    for result in scan_results:
        aggregate_scan_result(result)

    # Cleanup cloned repos
    for scanner in scanners:
        scanner.cleanup()

    # Aggregate results
    if not scan_results:
        console.print("[red]No scan results.[/]")
        sys.exit(1)

    # LLM classification
    all_assets = [a for r in scan_results for a in r.assets]
    if not no_llm and all_assets:
        engine = LLMEngine(
            mode=llm_mode, url=llm_url, model=llm_model, api_key=llm_key
        )

        if engine.check_health():
            console.print(f"\n[blue]Classifying {len(all_assets)} asset(s) via LLM...[/]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("LLM classification...", total=len(all_assets))
                for asset in all_assets:
                    try:
                        asset.data_classification = engine.classify(asset)
                    except Exception as e:
                        console.print(
                            f"  [yellow]Warning:[/] Classification failed for "
                            f"'{asset.name}': {e}"
                        )
                    progress.advance(task)
        else:
            console.print(
                "[yellow]Warning:[/] LLM not available, skipping classification. "
                f"Tried {llm_mode} at {llm_url}"
            )

    # Enrich assets with insights (summary, risk reasoning, recommendations)
    if not all_assets:
        all_assets = [a for r in scan_results for a in r.assets]
    insights = enrich_assets(all_assets)

    # Generate report — auto-detect format from extension
    if output.endswith(".json"):
        from aiscout.report.json_export import JSONExporter
        gen = JSONExporter(scan_results, output_path=output, insights=insights)
    else:
        gen = ReportGenerator(
            scan_results, output_path=output, insights=insights,
            org_inventory=org_inventory,
        )
    report_path = gen.generate()

    # Print summary
    _print_summary(scan_results, report_path)

    # QA gate (Sprint 0.2): degradation never fails a production scan
    # (P-4), but CI can insist the report needed no suppression.
    qa_counts = gen.qa_result.counts() if gen.qa_result else {"suppressed": 0, "warnings": 0}
    if qa_counts["suppressed"] or qa_counts["warnings"]:
        console.print(
            f"[yellow]QA linter:[/] {qa_counts['suppressed']} sentence(s) suppressed, "
            f"{qa_counts['warnings']} warning(s) — see the QA appendix in the report."
        )
    if strict and qa_counts["suppressed"]:
        console.print("[red]--strict:[/] failing because the QA linter suppressed output.")
        sys.exit(2)


_ALLOWED_URL_SCHEMES = {"https", "http", "ssh", "git"}
_FORBIDDEN_SYSTEM_PATHS = {
    "/", "/etc", "/var", "/usr", "/bin", "/sbin", "/boot", "/dev", "/proc",
    "/sys", "/root", "/System", "/Library", "/private", "/opt",
    # macOS resolves /etc and /var into /private/* via firmlinks
    "/private/etc", "/private/var",
}


class InputValidationError(click.UsageError):
    """Raised when a CLI-provided repo URL or local path fails validation."""


def _validate_repo_url(url: str) -> str:
    """Reject URLs that could be used for SSRF or local file disclosure.

    Only ``https``/``http``/``ssh``/``git`` schemes are allowed. ``file://``,
    ``gopher://``, and other git-accepted protocols that can read local files
    or hit arbitrary network services are rejected. Hosts resolving to the
    loopback, link-local, or cloud metadata address (``169.254.169.254``)
    are rejected as well.
    """
    if not url or not isinstance(url, str):
        raise InputValidationError("Repository URL must be a non-empty string.")

    url = url.strip()

    # scp-like syntax: git@github.com:org/repo.git — not a real URL, special-case
    if "@" in url and "://" not in url:
        host = url.split("@", 1)[1].split(":", 1)[0]
        if not host or _is_blocked_host(host):
            raise InputValidationError(
                f"Refusing to clone from restricted host: {host or '<empty>'}"
            )
        return url

    parts = urlsplit(url)
    if parts.scheme.lower() not in _ALLOWED_URL_SCHEMES:
        raise InputValidationError(
            f"Unsupported repository URL scheme '{parts.scheme}'. "
            f"Allowed: {sorted(_ALLOWED_URL_SCHEMES)}."
        )
    if not parts.hostname:
        raise InputValidationError(f"Repository URL is missing a hostname: {url}")
    if _is_blocked_host(parts.hostname):
        raise InputValidationError(
            f"Refusing to clone from restricted host: {parts.hostname}"
        )
    return url


def _is_blocked_host(host: str) -> bool:
    host = host.strip().lower().rstrip(".")
    if host in ("localhost", "ip6-localhost", "ip6-loopback"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        return True
    # Cloud metadata endpoints
    if str(ip) in {"169.254.169.254", "fd00:ec2::254"}:
        return True
    return False


def _validate_local_path(raw: str) -> Path:
    """Reject local paths that point at the filesystem root or system dirs."""
    if not raw:
        raise InputValidationError("Local repository path must not be empty.")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise InputValidationError(f"Local path does not exist: {path}")
    if not path.is_dir():
        raise InputValidationError(f"Local path is not a directory: {path}")
    if str(path) in _FORBIDDEN_SYSTEM_PATHS:
        raise InputValidationError(
            f"Refusing to scan system directory: {path}"
        )
    # Require at least one path component below the drive/root
    if len(path.parts) < 2:
        raise InputValidationError(f"Refusing to scan filesystem root: {path}")
    return path


def _build_repo_list(
    repo_urls: tuple,
    local_paths: tuple,
    orgs: tuple,
    config_path: str | None,
    default_token: str | None,
    default_branch: str,
    *,
    include_archived: bool = False,
    include_forks: bool = False,
    max_repos: int = 200,
) -> list[dict]:
    """Build a normalized list of repos from CLI args and/or YAML config.

    Returns ``(repos, org_inventory)`` — the second element holds one entry
    per ``--org`` describing what was enumerated vs. scanned vs. skipped, for
    surfacing in the report.
    """
    repos = []
    org_inventory: list[dict] = []

    # From CLI --repo flags
    for url in repo_urls:
        validated = _validate_repo_url(url)
        repos.append({"url": validated, "token": default_token, "branch": default_branch,
                       "name": validated.rstrip("/").split("/")[-1].removesuffix(".git")})

    # From CLI --org flags (enumerate all visible repos via GitHub API)
    for org in orgs:
        entries, inventory = _enumerate_org(
            org, default_token, default_branch,
            include_archived=include_archived,
            include_forks=include_forks,
            max_repos=max_repos,
        )
        repos.extend(entries)
        if inventory:
            org_inventory.append(inventory)

    # From CLI --local flags
    for path in local_paths:
        abs_path = _validate_local_path(path)
        repos.append({"path": str(abs_path), "branch": default_branch,
                       "name": abs_path.name})

    # From YAML config
    if config_path:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            for entry in cfg.get("repositories", []):
                item: dict = {}
                if "url" in entry:
                    item["url"] = _validate_repo_url(entry["url"])
                    item["name"] = item["url"].rstrip("/").split("/")[-1].removesuffix(".git")
                elif "path" in entry:
                    validated_path = _validate_local_path(entry["path"])
                    item["path"] = str(validated_path)
                    item["name"] = validated_path.name
                else:
                    continue

                item["branch"] = entry.get("branch", default_branch)

                # Token from env var reference
                token_env = entry.get("token_env")
                if token_env:
                    item["token"] = os.environ.get(token_env, default_token)
                else:
                    item["token"] = default_token

                repos.append(item)
        except InputValidationError:
            raise
        except Exception as e:
            console.print(f"[red]Error loading config:[/] {e}")

    return repos, org_inventory


def _enumerate_org(
    org: str,
    token: str | None,
    default_branch: str,
    *,
    include_archived: bool,
    include_forks: bool,
    max_repos: int,
) -> tuple[list[dict], dict | None]:
    """Resolve a GitHub org/user into validated repo entries for the scan loop.

    Returns ``(entries, inventory)`` where ``inventory`` summarizes the
    enumeration (or ``None`` if it failed entirely).
    """
    try:
        enum = enumerate_org_repos(
            org, token,
            include_archived=include_archived,
            include_forks=include_forks,
            max_repos=max_repos,
        )
    except OrgEnumerationError as e:
        console.print(f"[red]Error enumerating '{org}':[/] {e}")
        return [], None

    skipped = []
    if enum.skipped_archived:
        skipped.append(f"{enum.skipped_archived} archived")
    if enum.skipped_forks:
        skipped.append(f"{enum.skipped_forks} forks")
    if enum.skipped_over_limit:
        skipped.append(f"{enum.skipped_over_limit} over --max-repos")
    skip_note = f" (skipped {', '.join(skipped)})" if skipped else ""
    console.print(
        f"[blue]{enum.owner}:[/] enumerated {enum.total_seen} repo(s), "
        f"scanning {len(enum.repos)}{skip_note}"
    )

    entries = []
    blocked = 0
    for item in enum.repos:
        # Run API-provided clone URLs through the same SSRF/scheme guard.
        try:
            item["url"] = _validate_repo_url(item["url"])
        except InputValidationError as e:
            console.print(f"  [yellow]Skipping {item.get('name')}:[/] {e}")
            blocked += 1
            continue
        item.setdefault("branch", default_branch)
        entries.append(item)

    inventory = {
        "owner": enum.owner,
        "total_seen": enum.total_seen,
        "scanned": len(entries),
        "skipped_archived": enum.skipped_archived,
        "skipped_forks": enum.skipped_forks,
        "skipped_over_limit": enum.skipped_over_limit,
        "skipped_blocked": blocked,
    }
    return entries, inventory


def _apply_config_overrides(
    config_path: str | None,
    llm_url: str, llm_model: str, llm_mode: str, llm_key: str | None,
    output: str,
) -> tuple[dict | None, str]:
    """Extract LLM and output overrides from YAML config (CLI args take priority)."""
    if not config_path:
        return None, output

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return None, output

    llm_config = cfg.get("llm")
    output_config = cfg.get("output", {})

    # Only use config output if CLI didn't override default
    if output == "aiscout_report.html" and output_config.get("path"):
        output = output_config["path"]

    return llm_config, output


def _print_summary(scan_results: list, report_path: Path):
    """Print a Rich summary table to the console."""
    console.print()

    table = Table(title="Scan Summary", show_lines=True)
    table.add_column("Repository", style="bold")
    table.add_column("Assets", justify="center")
    table.add_column("Critical", justify="center", style="red")
    table.add_column("Review", justify="center", style="yellow")
    table.add_column("No findings", justify="center", style="green")
    table.add_column("Errors", justify="center")

    total_assets = 0
    total_critical = 0
    total_warning = 0
    total_ok = 0

    for result in scan_results:
        repo = result.metadata.get("repository", "unknown")
        n = len(result.assets)
        crit = sum(1 for a in result.assets if a.risk_status == RiskStatus.CRITICAL)
        warn = sum(1 for a in result.assets if a.risk_status == RiskStatus.REVIEW)
        ok = sum(1 for a in result.assets if a.risk_status == RiskStatus.NO_FINDINGS)

        table.add_row(repo, str(n), str(crit), str(warn), str(ok), str(len(result.errors)))
        total_assets += n
        total_critical += crit
        total_warning += warn
        total_ok += ok

    if len(scan_results) > 1:
        table.add_row(
            "[bold]TOTAL[/]",
            str(total_assets),
            str(total_critical),
            str(total_warning),
            str(total_ok),
            "",
            style="bold",
        )

    console.print(table)
    console.print(f"\nReport saved to [bold blue]{report_path}[/]")


# Data Flow Mapper category vocabulary (see engine/data_flow.py) that should
# never leave the perimeter for an external LLM.
_SENSITIVE_CATEGORIES = {
    "personal_data", "credentials", "financial_data", "medical_data",
}


@cli.command()
@click.option("--path", "-p", default=".", help="Local path to check (default: current dir)")
@click.option("--warn-only", is_flag=True, help="Report issues but always exit 0")
def check(path, warn_only):
    """Pre-commit / CI guardrail for AI code.

    Statically scans a local working tree and fails (exit 1) when it finds
    a hardcoded API key or code that sends sensitive data (PII, financial,
    confidential) to an external LLM. Fully rule-based — no LLM, no network.
    """
    abs_path = _validate_local_path(path)
    scanner = GitScanner(repo_path=str(abs_path))
    result = scanner.scan()

    repo_root = result.metadata.get("repo_root")
    if repo_root and result.assets:
        analyze_assets(result.assets, repo_root)
        build_data_flows(result.assets)
    scanner.cleanup()

    if result.errors:
        for err in result.errors:
            console.print(f"[red]Error:[/] {err}")
        sys.exit(2)

    key_issues, egress_issues = _evaluate_guardrail(result.assets)
    _print_guardrail(key_issues, egress_issues)

    if (key_issues or egress_issues) and not warn_only:
        sys.exit(1)


def _evaluate_guardrail(assets: list) -> tuple[list[dict], list[dict]]:
    """Split assets into hardcoded-key issues and sensitive-egress issues."""
    key_issues: list[dict] = []
    egress_issues: list[dict] = []

    for asset in assets:
        for f in asset.raw_findings:
            if f.type == FindingType.API_KEY_DETECTED:
                key_issues.append({
                    "file": f.file_path,
                    "line": f.line_number,
                    "provider": f.provider or "unknown",
                    "redacted": f.redacted_content or "",
                })

        flow = asset.data_flow
        if not flow:
            continue
        sensitive = sorted(_SENSITIVE_CATEGORIES.intersection(
            c.lower() for c in flow.data_categories
        ))
        if not sensitive:
            continue
        external = _external_llm_sinks(flow.sinks)
        if external:
            egress_issues.append({
                "name": asset.name,
                "file": asset.file_path,
                "categories": sensitive,
                "sinks": external,
            })

    return key_issues, egress_issues


def _external_llm_sinks(sinks: list) -> list[str]:
    """Names of AI-API sinks that leave the perimeter (excludes local runtimes)."""
    external = []
    for sink in sinks:
        if sink.type != "ai_api":
            continue
        # A known local runtime (e.g. Ollama) keeps data inside the perimeter.
        if sink.provider and get_provider(sink.provider).category == "local_runtime":
            continue
        external.append(sink.name or sink.provider or "external LLM")
    return external


def _print_guardrail(key_issues: list[dict], egress_issues: list[dict]) -> None:
    if not key_issues and not egress_issues:
        console.print(Panel(
            "[bold green]No issues found.[/]\n"
            "No hardcoded API keys; no sensitive data sent to external LLMs.",
            title="AI Scout — guardrail PASSED",
            border_style="green",
        ))
        return

    if key_issues:
        console.print("\n[bold red]Hardcoded API keys[/]")
        for i in key_issues:
            loc = f"{i['file']}:{i['line']}" if i["line"] else i["file"]
            console.print(f"  [red]✗[/] {loc} — {i['provider']} key ({i['redacted']})")

    if egress_issues:
        console.print("\n[bold red]Sensitive data sent to external LLM[/]")
        for i in egress_issues:
            cats = ", ".join(i["categories"])
            sinks = ", ".join(i["sinks"])
            console.print(f"  [red]✗[/] {i['name']} ([dim]{i['file']}[/]) — {cats} → {sinks}")

    total = len(key_issues) + len(egress_issues)
    console.print(Panel(
        f"[bold red]{total} issue(s) found.[/]",
        title="AI Scout — guardrail FAILED",
        border_style="red",
    ))


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8080, help="Port to run on")
def web(host, port):
    """Start the AI Scout Web UI."""
    from aiscout.web.app import run_server

    console.print(Panel(
        f"[bold]Starting Web UI[/]\n"
        f"Open [blue]http://localhost:{port}[/] in your browser",
        title="AI Scout Web",
        border_style="blue",
    ))
    run_server(host=host, port=port)
