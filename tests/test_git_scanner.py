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
