"""Regression harness: stable snapshot of scan + enrichment output.

Runs the full no-LLM pipeline (GitScanner → code analyzer → enrichment) on
the repo's test fixtures, normalises the result into a deterministic JSON
document and diffs it against ``tests/regression/golden.json``.

Why this exists
---------------
Sprint 1 added security hardening (prompt sanitisation, symlink skip, CLI
validation). None of those should change what Scout *finds* or *classifies*
for a clean repo — but "should" is not "does". This harness makes any drift
immediately visible: if a future refactor silently stops detecting a
provider, misgroups an asset, or shifts a risk score, the diff fails the
build and the change has to be explicit.

Updating the golden
-------------------
When a detection/classification change is intentional, regenerate with:

    AISCOUT_UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/test_regression.py

and commit the updated file.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from aiscout.engine.code_analyzer import analyze_assets
from aiscout.engine.enrichment import enrich_assets
from aiscout.scanners.git_scanner import GitScanner

FIXTURES = Path(__file__).parent / "fixtures"
SPRINT2_FIXTURES = FIXTURES / "sprint2"
SPRINT3_FIXTURES = FIXTURES / "sprint3"
GOLDEN_DIR = Path(__file__).parent / "regression"
GOLDEN = GOLDEN_DIR / "golden.json"
GOLDEN_SPRINT2 = GOLDEN_DIR / "golden_sprint2.json"
GOLDEN_SPRINT3 = GOLDEN_DIR / "golden_sprint3.json"


def _run_pipeline(root: Path = FIXTURES) -> dict:
    """Run the full no-LLM scan pipeline and return a normalised snapshot."""
    scanner = GitScanner(repo_path=str(root))
    result = scanner.scan()
    analyze_assets(result.assets, str(root))
    insights_by_id = enrich_assets(result.assets)
    # Re-key insights from volatile asset UUID → stable asset name so the
    # snapshot diff isn't dominated by UUID churn.
    insights_by_name = {}
    for asset in result.assets:
        ins = insights_by_id.get(asset.id)
        if ins is not None:
            insights_by_name[asset.name] = ins
    return _normalise(result, insights_by_name)


def _normalise(result, insights) -> dict:
    """Strip volatile fields (UUIDs, timestamps, absolute paths) and
    canonicalise ordering so the snapshot is byte-stable across runs."""
    assets_out = []
    for asset in sorted(result.assets, key=lambda a: (a.name, a.file_path)):
        provider_name = asset.provider.name if asset.provider else None
        raw_findings = sorted(
            (
                {
                    "type": f.type.value,
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                    "content": f.content,
                    "redacted_content": f.redacted_content,
                    "provider": f.provider,
                }
                for f in asset.raw_findings
            ),
            key=lambda f: (f["type"], f["file_path"], f["line_number"] or 0, f["content"]),
        )
        code_contexts = sorted(
            (
                {
                    "file_path": c.file_path,
                    "language": c.language,
                    "function_names": sorted(
                        [fn.get("name", "") for fn in c.functions]
                    ),
                    "class_names": sorted(
                        [cl.get("name", "") for cl in c.classes]
                    ),
                    "api_call_targets": sorted(
                        [ac.get("target", "") for ac in c.api_calls]
                    ),
                    "data_source_types": sorted(
                        [ds.get("type", "") for ds in c.data_sources]
                    ),
                    "data_sink_types": sorted(
                        [ds.get("type", "") for ds in c.data_sinks]
                    ),
                    "prompt_count": len(c.prompts),
                    "prompt_lengths": sorted([len(p) for p in c.prompts]),
                    "env_vars": sorted(c.env_vars),
                    "model_names": sorted(set(c.model_names)),
                }
                for c in asset.code_contexts
            ),
            key=lambda c: c["file_path"],
        )

        assets_out.append({
            "name": asset.name,
            "type": asset.type.value,
            "provider": provider_name,
            "risk_score": round(asset.risk_score, 4),
            "discovered_via": sorted(asset.discovered_via),
            "repository": asset.repository,
            "file_paths": sorted(asset.file_path.split(", ")),
            "dependencies": sorted(asset.dependencies),
            "tags": sorted(asset.tags),
            "task_types": sorted(t.value for t in asset.task_types),
            "raw_findings": raw_findings,
            "code_contexts": code_contexts,
        })

    insights_out = []
    for asset_name in sorted(insights.keys()):
        ins = insights[asset_name]
        if hasattr(ins, "model_dump"):
            data = ins.model_dump()
        elif dataclasses.is_dataclass(ins):
            data = dataclasses.asdict(ins)
        else:
            data = dict(ins.__dict__)
        for volatile in ("asset_id", "id", "timestamp", "generated_at"):
            data.pop(volatile, None)
        insights_out.append({"asset_name": asset_name, **data})

    return {
        "scanner": result.scanner,
        "files_scanned": result.metadata.get("files_scanned"),
        "errors": result.errors,
        "asset_count": len(assets_out),
        "assets": assets_out,
        "insights": insights_out,
    }


def _diff_against_golden(snapshot: dict, golden_path: Path, label: str):
    serialised = json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False)

    if os.environ.get("AISCOUT_UPDATE_GOLDEN"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(serialised + "\n", encoding="utf-8")
        pytest.skip(f"Golden updated at {golden_path}")

    if not golden_path.exists():
        pytest.fail(
            f"Golden snapshot missing at {golden_path}. "
            f"Run with AISCOUT_UPDATE_GOLDEN=1 to create it."
        )

    expected = golden_path.read_text(encoding="utf-8").rstrip("\n")
    if expected != serialised:
        import difflib
        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(),
                serialised.splitlines(),
                fromfile=f"{label} (golden)",
                tofile=f"{label} (current)",
                lineterm="",
            )
        )
        pytest.fail(
            f"{label} regression snapshot drifted. Review the diff below. "
            f"If the change is intentional, rerun with "
            f"AISCOUT_UPDATE_GOLDEN=1 to update the golden.\n\n" + diff
        )


def test_regression_snapshot():
    """Sprint 1 fixture tree — chatbot + hardcoded key."""
    _diff_against_golden(_run_pipeline(FIXTURES), GOLDEN, "fixtures")


def test_regression_snapshot_sprint2():
    """Sprint 2 fixture tree — MCP config, Docker/compose, Azure OpenAI,
    fine-tuning script, local model weights. Exercises every Sprint 2
    detector in a single pipeline run."""
    _diff_against_golden(
        _run_pipeline(SPRINT2_FIXTURES), GOLDEN_SPRINT2, "fixtures/sprint2"
    )


def test_regression_snapshot_sprint3():
    """Sprint 3 fixture tree — CI workflow, YAML config with Azure
    deployment, requirements.txt pinned to legacy openai/langchain. One
    snapshot covers CI detection, YAML model parsing and dependency
    advisories."""
    _diff_against_golden(
        _run_pipeline(SPRINT3_FIXTURES), GOLDEN_SPRINT3, "fixtures/sprint3"
    )


def test_snapshot_is_deterministic():
    """Two back-to-back runs must produce byte-identical output."""
    first = json.dumps(_run_pipeline(), indent=2, sort_keys=True, ensure_ascii=False)
    second = json.dumps(_run_pipeline(), indent=2, sort_keys=True, ensure_ascii=False)
    assert first == second, "Pipeline output is not deterministic — cannot snapshot"

    first2 = json.dumps(_run_pipeline(SPRINT2_FIXTURES), indent=2, sort_keys=True, ensure_ascii=False)
    second2 = json.dumps(_run_pipeline(SPRINT2_FIXTURES), indent=2, sort_keys=True, ensure_ascii=False)
    assert first2 == second2, "Sprint 2 pipeline output is not deterministic"
