"""Tests for Git Repository Scanner."""

import json
import tempfile
from pathlib import Path

from aiscout.models import FindingType
from aiscout.scanners.git_scanner import GitScanner, _redact_key


FIXTURES = Path(__file__).parent / "fixtures"


def test_detect_imports_python():
    scanner = GitScanner(repo_path=str(FIXTURES))
    content = "import openai\nfrom anthropic import Anthropic\n"
    findings = scanner._detect_imports("test.py", content)
    providers = {f.provider for f in findings}
    assert "openai" in providers
    assert "anthropic" in providers


def test_detect_imports_js():
    scanner = GitScanner(repo_path=str(FIXTURES))
    content = "const openai = require('openai');\n"
    findings = scanner._detect_imports("test.js", content)
    assert len(findings) == 1
    assert findings[0].provider == "openai"


def test_detect_api_keys():
    scanner = GitScanner(repo_path=str(FIXTURES))
    content = 'API_KEY = "sk-abcdefghijklmnop1234567890abcdef"\n'
    findings = scanner._detect_api_keys("config.py", content)
    assert len(findings) == 1
    assert findings[0].type == FindingType.API_KEY_DETECTED
    assert findings[0].provider == "openai"
    assert findings[0].redacted_content == "sk-abcde...cdef"


def test_redact_key():
    assert _redact_key("sk-abcdefghijklmnop1234567890") == "sk-abcde...7890"
    assert _redact_key("short") == "shor...rt"


def test_scan_dependencies_requirements():
    scanner = GitScanner(repo_path=str(FIXTURES))
    content = FIXTURES.joinpath("sample_requirements.txt").read_text()
    findings = scanner._scan_requirements_txt("requirements.txt", content)
    providers = {f.provider for f in findings}
    assert "openai" in providers
    assert "langchain" in providers
    assert "chromadb" in providers
    # flask and requests should NOT be detected
    assert not any("flask" in f.content.lower() for f in findings)


def test_scan_dependencies_package_json():
    scanner = GitScanner(repo_path=str(FIXTURES))
    content = FIXTURES.joinpath("sample_package.json").read_text()
    findings = scanner._scan_package_json("package.json", content)
    providers = {f.provider for f in findings}
    assert "openai" in providers
    assert "anthropic" in providers
    # express and jest should NOT be detected
    assert not any("express" in f.content for f in findings)


def test_walk_files_skips_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create scannable file
        (root / "app.py").write_text("print('hello')")
        # Create file in skipped dir
        (root / "node_modules").mkdir()
        (root / "node_modules" / "lib.js").write_text("module.exports = {}")
        # Create file with wrong extension
        (root / "image.png").write_bytes(b"\x89PNG")

        scanner = GitScanner(repo_path=str(root))
        files = list(scanner._walk_files(root))
        names = [f.name for f in files]
        assert "app.py" in names
        assert "lib.js" not in names
        assert "image.png" not in names


def test_group_findings():
    from aiscout.models import Finding, FindingType
    scanner = GitScanner(repo_path="/tmp")
    # Files in SAME directory → one solution asset
    findings = [
        Finding(type=FindingType.IMPORT_DETECTED, file_path="a.py", content="import openai", provider="openai"),
        Finding(type=FindingType.API_KEY_DETECTED, file_path="a.py", content="sk-xxx", redacted_content="sk-x...x", provider="openai"),
        Finding(type=FindingType.IMPORT_DETECTED, file_path="b.py", content="import anthropic", provider="anthropic"),
    ]
    assets = scanner._group_findings_into_assets(findings, "test-repo")
    # All in root dir → grouped into one solution
    assert len(assets) == 1
    assert assets[0].risk_score == 0.7  # has API key

    # Files in DIFFERENT directories → separate solution assets
    findings2 = [
        Finding(type=FindingType.IMPORT_DETECTED, file_path="backend/app.py", content="import openai", provider="openai"),
        Finding(type=FindingType.IMPORT_DETECTED, file_path="ml/train.py", content="import anthropic", provider="anthropic"),
    ]
    assets2 = scanner._group_findings_into_assets(findings2, "test-repo")
    assert len(assets2) == 2


def test_extract_notebook_source():
    scanner = GitScanner(repo_path="/tmp")
    notebook = json.dumps({
        "cells": [
            {"cell_type": "code", "source": ["import openai\n", "client = openai.Client()\n"]},
            {"cell_type": "markdown", "source": ["# Heading"]},
        ]
    })
    result = scanner._extract_notebook_source(notebook)
    assert "import openai" in result
    assert "# Heading" in result


def test_full_scan_local():
    scanner = GitScanner(repo_path=str(FIXTURES))
    result = scanner.scan()
    assert result.scanner == "git_scanner"
    assert len(result.assets) > 0
    assert result.errors == []
    assert result.metadata["files_scanned"] > 0

    # All fixture files are in the same directory → grouped by solution dir
    all_providers = set()
    for asset in result.assets:
        for f in asset.raw_findings:
            all_providers.add(f.provider)
    assert "openai" in all_providers


# ── Sprint 1: security regression tests ──────────────────────────────────


def test_api_key_not_stored_raw():
    """Sprint 1 / C1: raw API key must never appear in Finding.content."""
    scanner = GitScanner(repo_path="/tmp")
    raw = "sk-abcdefghijklmnop1234567890abcdef"
    findings = scanner._detect_api_keys("config.py", f'KEY = "{raw}"')
    assert len(findings) == 1
    assert findings[0].content != raw
    assert raw not in findings[0].content
    assert findings[0].content == findings[0].redacted_content


def test_walk_files_skips_symlinks(tmp_path):
    """Sprint 1 / H3: symlinks must not be followed or yielded."""
    (tmp_path / "real.py").write_text("import openai")
    outside = tmp_path.parent / "outside_target.py"
    outside.write_text("secret = 'x'")
    try:
        (tmp_path / "evil.py").symlink_to(outside)
    except OSError:
        return  # platform doesn't support symlinks — skip silently
    scanner = GitScanner(repo_path=str(tmp_path))
    names = [p.name for p in scanner._walk_files(tmp_path)]
    assert "real.py" in names
    assert "evil.py" not in names
    outside.unlink()


# ── Sprint 2: new detectors ──────────────────────────────────────────────


def test_detect_azure_openai_import():
    scanner = GitScanner(repo_path="/tmp")
    findings = scanner._detect_imports(
        "x.py",
        "from openai import AzureOpenAI\nclient = AzureOpenAI(azure_endpoint='x')\n",
    )
    providers = [f.provider for f in findings]
    assert "azure_openai" in providers
    assert "openai" not in providers  # more-specific supersedes generic


def test_detect_azure_openai_from_env():
    scanner = GitScanner(repo_path="/tmp")
    findings = scanner._detect_azure_openai_config(
        "x.py",
        'endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]\n',
    )
    assert len(findings) == 1
    assert findings[0].provider == "azure_openai"


def test_detect_mcp_imports_and_config():
    scanner = GitScanner(repo_path="/tmp")
    imp = scanner._detect_imports("x.py", "from mcp.server import Server")
    assert [f.provider for f in imp] == ["mcp"]

    cfg = scanner._detect_mcp_config(
        "mcp.json",
        '{"mcpServers": {"github": {"command": "npx"}, "fs": {"command": "npx"}}}',
    )
    assert len(cfg) == 2
    assert {f.content for f in cfg} == {"mcp server: github", "mcp server: fs"}
    assert all(f.provider == "mcp" for f in cfg)


def test_detect_container_images():
    scanner = GitScanner(repo_path="/tmp")
    compose = (
        "services:\n"
        "  llm: { image: ollama/ollama:latest }\n"
        "  inf: { image: vllm/vllm-openai:latest }\n"
        "  db:  { image: qdrant/qdrant:latest }\n"
    )
    findings = scanner._detect_containers("docker-compose.yml", compose)
    providers = sorted(f.provider for f in findings)
    assert providers == ["ollama", "qdrant", "vllm"]


def test_local_model_finding_not_read(tmp_path):
    """Local model file gets a finding with size, never loads content."""
    model = tmp_path / "weights.gguf"
    model.write_bytes(b"\x00" * 2048)
    scanner = GitScanner(repo_path=str(tmp_path))
    files = list(scanner._walk_files(tmp_path))
    assert model in files
    finding = scanner._local_model_finding(model, "weights.gguf")
    assert finding.type == FindingType.LOCAL_MODEL_DETECTED
    assert "weights.gguf" in finding.content
    assert "KB" in finding.content or "B" in finding.content
    assert finding.provider == "llamacpp"


def test_model_file_over_1mb_still_detected(tmp_path):
    """Unlike source code, model files should never be size-capped."""
    big = tmp_path / "big.safetensors"
    big.write_bytes(b"\x00" * (2 * 1024 * 1024))  # 2 MB > MAX_FILE_SIZE
    scanner = GitScanner(repo_path=str(tmp_path))
    names = [p.name for p in scanner._walk_files(tmp_path)]
    assert "big.safetensors" in names


def test_walk_files_skips_escape_via_symlinked_dir(tmp_path):
    """Sprint 1 / H3: a symlinked *directory* must not cause escape."""
    outside_dir = tmp_path.parent / "outside_escape_dir"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "secret.py").write_text("TOKEN='x'")
    try:
        (tmp_path / "escape").symlink_to(outside_dir, target_is_directory=True)
    except OSError:
        return
    (tmp_path / "inside.py").write_text("import openai")
    scanner = GitScanner(repo_path=str(tmp_path))
    names = [p.name for p in scanner._walk_files(tmp_path)]
    assert "inside.py" in names
    assert "secret.py" not in names
    (outside_dir / "secret.py").unlink()
    outside_dir.rmdir()
