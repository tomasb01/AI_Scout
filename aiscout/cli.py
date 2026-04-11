"""CLI entry point for AI Scout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from aiscout import __version__
from aiscout.engine.code_analyzer import analyze_assets
from aiscout.engine.enrichment import enrich_assets
from aiscout.engine.llm import LLMEngine
from aiscout.report.html import ReportGenerator
from aiscout.scanners.git_scanner import GitScanner

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="aiscout")
def cli():
    """AI Scout — Enterprise AI Discovery & Security Assessment Tool."""
    pass


@cli.command()
@click.option("--repo", "-r", multiple=True, help="Git repo URL (repeatable)")
@click.option("--local", "-l", multiple=True, help="Local repo path (repeatable)")
@click.option("--config", "-c", type=click.Path(exists=True), help="YAML config file")
@click.option("--token", "-t", envvar="AISCOUT_GIT_TOKEN", help="Git access token")
@click.option("--branch", "-b", default="main", help="Default branch to scan")
@click.option("--output", "-o", default="aiscout_report.html", help="Output report path")
@click.option("--llm-url", default="http://localhost:11434", help="LLM API URL")
@click.option("--llm-model", default="qwen2.5-coder:7b", help="LLM model name")
@click.option(
    "--llm-mode",
    type=click.Choice(["ollama", "openai"]),
    default="ollama",
    help="LLM backend mode",
)
@click.option("--llm-key", envvar="AISCOUT_LLM_KEY", help="LLM API key (OpenAI mode)")
@click.option("--no-llm", is_flag=True, help="Skip LLM classification")
def scan(
    repo, local, config, token, branch, output,
    llm_url, llm_model, llm_mode, llm_key, no_llm,
):
    """Scan Git repositories for AI assets."""
    # Build list of repos to scan
    repos = _build_repo_list(repo, local, config, token, branch)

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

    console.print(Panel(
        f"[bold]Scanning {len(repos)} repositor{'y' if len(repos) == 1 else 'ies'}[/]\n"
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

    # Code context analysis (reads files from repo before cleanup)
    for result in scan_results:
        repo_root = result.metadata.get("repo_root")
        if repo_root and result.assets:
            analyze_assets(result.assets, repo_root)

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
                        result = engine.classify(asset)
                        asset.data_classification = result
                        if result.risk_score > 0:
                            asset.risk_score = max(asset.risk_score, result.risk_score)
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

    # Generate report
    gen = ReportGenerator(scan_results, output_path=output, insights=insights)
    report_path = gen.generate()

    # Print summary
    _print_summary(scan_results, report_path)


def _build_repo_list(
    repo_urls: tuple,
    local_paths: tuple,
    config_path: str | None,
    default_token: str | None,
    default_branch: str,
) -> list[dict]:
    """Build a normalized list of repos from CLI args and/or YAML config."""
    repos = []

    # From CLI --repo flags
    for url in repo_urls:
        repos.append({"url": url, "token": default_token, "branch": default_branch,
                       "name": url.rstrip("/").split("/")[-1].removesuffix(".git")})

    # From CLI --local flags
    for path in local_paths:
        abs_path = str(Path(path).resolve())
        repos.append({"path": abs_path, "branch": default_branch,
                       "name": Path(path).name})

    # From YAML config
    if config_path:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            for entry in cfg.get("repositories", []):
                item: dict = {}
                if "url" in entry:
                    item["url"] = entry["url"]
                    item["name"] = entry["url"].rstrip("/").split("/")[-1].removesuffix(".git")
                elif "path" in entry:
                    item["path"] = str(Path(entry["path"]).resolve())
                    item["name"] = Path(entry["path"]).name
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
        except Exception as e:
            console.print(f"[red]Error loading config:[/] {e}")

    return repos


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
    table.add_column("Warning", justify="center", style="yellow")
    table.add_column("OK", justify="center", style="green")
    table.add_column("Errors", justify="center")

    total_assets = 0
    total_critical = 0
    total_warning = 0
    total_ok = 0

    for result in scan_results:
        repo = result.metadata.get("repository", "unknown")
        n = len(result.assets)
        crit = sum(1 for a in result.assets if a.risk_score >= 0.7)
        warn = sum(1 for a in result.assets if 0.4 <= a.risk_score < 0.7)
        ok = sum(1 for a in result.assets if a.risk_score < 0.4)

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
