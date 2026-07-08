"""Tests for AI Scout CLI."""

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from aiscout.cli import (
    InputValidationError,
    _evaluate_guardrail,
    _external_llm_sinks,
    _validate_local_path,
    _validate_repo_url,
    cli,
)
from aiscout.models import (
    AIAsset,
    DataFlowMap,
    Finding,
    FindingType,
    FlowSink,
    ProviderInfo,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "aiscout" in result.output
    assert "0.11.0" in result.output


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


# ── Guardrail (aiscout check) ────────────────────────────────────────────

def test_external_llm_sinks_excludes_local_runtime():
    sinks = [
        FlowSink(type="ai_api", provider="openai", name="OpenAI API"),
        FlowSink(type="ai_api", provider="ollama", name="Ollama (local)"),
        FlowSink(type="database", provider="", name="Postgres"),
    ]
    external = _external_llm_sinks(sinks)
    assert external == ["OpenAI API"]


def test_evaluate_guardrail_flags_key_and_egress():
    asset = AIAsset(
        name="chatbot",
        repository="r",
        file_path="app.py",
        provider=ProviderInfo(name="openai"),
        raw_findings=[
            Finding(
                type=FindingType.API_KEY_DETECTED,
                file_path="app.py",
                line_number=2,
                content="sk-x",
                redacted_content="sk-x...x",
                provider="openai",
            ),
        ],
        data_flow=DataFlowMap(
            data_categories=["personal_data", "user_messages"],
            sinks=[FlowSink(type="ai_api", provider="openai", name="OpenAI API")],
        ),
    )
    keys, egress = _evaluate_guardrail([asset])
    assert len(keys) == 1 and keys[0]["provider"] == "openai"
    assert len(egress) == 1 and egress[0]["categories"] == ["personal_data"]


def test_evaluate_guardrail_local_egress_is_clean():
    asset = AIAsset(
        name="local-bot", repository="r", file_path="app.py",
        data_flow=DataFlowMap(
            data_categories=["personal_data"],
            sinks=[FlowSink(type="ai_api", provider="ollama", name="Ollama")],
        ),
    )
    keys, egress = _evaluate_guardrail([asset])
    assert keys == []
    assert egress == []


def test_check_command_clean_passes():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "util.py").write_text("def add(a, b):\n    return a + b\n")
        result = runner.invoke(cli, ["check", "--path", d])
    assert result.exit_code == 0
    assert "PASSED" in result.output


def test_check_command_fails_on_hardcoded_key():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "app.py").write_text(
            'import openai\nKEY = "sk-abcdefghijklmnop1234567890abcdef"\n'
        )
        result = runner.invoke(cli, ["check", "--path", d])
    assert result.exit_code == 1
    assert "FAILED" in result.output


def test_check_command_warn_only_exits_zero():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "app.py").write_text(
            'import openai\nKEY = "sk-abcdefghijklmnop1234567890abcdef"\n'
        )
        result = runner.invoke(cli, ["check", "--path", d, "--warn-only"])
    assert result.exit_code == 0


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


# ── Sprint 1: H4 input validation ────────────────────────────────────────


@pytest.mark.parametrize("bad_url", [
    "file:///etc/passwd",
    "http://127.0.0.1/repo.git",
    "http://localhost:8080/repo.git",
    "http://169.254.169.254/latest/meta-data/",
    "gopher://example.com/",
    "ssh://localhost/repo.git",
])
def test_validate_repo_url_rejects_unsafe(bad_url):
    with pytest.raises(InputValidationError):
        _validate_repo_url(bad_url)


@pytest.mark.parametrize("good_url", [
    "https://github.com/org/repo.git",
    "https://gitlab.com/group/project",
    "git@github.com:org/repo.git",
    "ssh://git@github.com/org/repo.git",
])
def test_validate_repo_url_accepts_safe(good_url):
    assert _validate_repo_url(good_url) == good_url


@pytest.mark.parametrize("bad_path", ["/", "/etc", "/System", "/Library"])
def test_validate_local_path_rejects_system_dirs(bad_path):
    if not Path(bad_path).exists():
        pytest.skip(f"{bad_path} not present on this host")
    with pytest.raises(InputValidationError):
        _validate_local_path(bad_path)


def test_validate_local_path_accepts_fixture_dir():
    result = _validate_local_path(str(FIXTURES))
    assert result.is_dir()
