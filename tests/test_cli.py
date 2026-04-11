"""Tests for AI Scout CLI."""

import tempfile
from pathlib import Path

from click.testing import CliRunner

from aiscout.cli import cli


FIXTURES = Path(__file__).parent / "fixtures"


def test_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "aiscout" in result.output
    assert "0.1.0" in result.output


def test_scan_no_repos_error():
    runner = CliRunner()
    result = runner.invoke(cli, ["scan"])
    assert result.exit_code != 0
    assert "No repositories specified" in result.output


def test_scan_local_no_llm():
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        output = f.name

    result = runner.invoke(cli, [
        "scan", "--local", str(FIXTURES), "--no-llm", "--output", output,
    ])

    assert result.exit_code == 0
    assert Path(output).exists()
    content = Path(output).read_text()
    assert "AI Scout" in content
    Path(output).unlink()


def test_scan_with_yaml_config():
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        output = f.name

    config_content = f"""
repositories:
  - path: {FIXTURES}
    branch: main

llm:
  mode: ollama
  model: qwen2.5-coder:14b

output:
  path: {output}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as cf:
        cf.write(config_content)
        config_path = cf.name

    result = runner.invoke(cli, [
        "scan", "--config", config_path, "--no-llm",
    ])

    assert result.exit_code == 0
    assert Path(output).exists()
    Path(output).unlink()
    Path(config_path).unlink()
