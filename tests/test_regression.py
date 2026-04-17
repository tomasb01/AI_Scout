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
    """Run the full no-LLM scan pipeline and return the stable snapshot.

    Back-compat shim for tests that only want the golden-diffable shape.
    New code should prefer ``_run_pipeline_both`` which also returns the
    volatile shape for floor assertions.
    """
    stable, _volatile = _run_pipeline_both(root)
    return stable


def _run_pipeline_both(root: Path = FIXTURES) -> tuple[dict, dict]:
    """Run the pipeline once and return (stable, volatile) snapshots."""
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
    stable = _normalise_stable(result, insights_by_name)
    volatile = _normalise_volatile(result, insights_by_name)
    return stable, volatile


def _normalise_stable(result, insights) -> dict:
    """Produce the *stable* snapshot — fields whose drift is a real regression.

    Stable = structural classification the user relies on:
      * asset identity (name, type, provider, tags, task_types, risk_score)
      * raw findings (type + file_path + provider — NOT content text)
      * code context shape (function names, api call targets, env vars,
        prompt *count/length* not prompt text)
      * risk reasons by (severity, title) — NOT detail text
      * insight tech_stack and data_involved lists

    Anything the user reads but where word-level drift is acceptable
    lives in the volatile shape and is checked separately.
    """
    assets_out = []
    for asset in sorted(result.assets, key=lambda a: (a.name, a.file_path)):
        provider_name = asset.provider.name if asset.provider else None
        raw_findings = sorted(
            (
                {
                    "type": f.type.value,
                    "file_path": f.file_path,
                    "provider": f.provider,
                    "has_redaction": f.redacted_content is not None,
                }
                for f in asset.raw_findings
            ),
            key=lambda f: (f["type"], f["file_path"], f["provider"]),
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
            "risk_score": round(asset.risk_score, 2),
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
        if dataclasses.is_dataclass(ins):
            data = dataclasses.asdict(ins)
        else:
            data = dict(ins.__dict__)
        # Risk reasons: keep (severity, title) tuples. Detail text is volatile.
        reasons_stable = sorted(
            [(r["severity"], r["title"]) for r in (data.get("risk_reasons") or [])]
        )
        tech_stack = sorted(data.get("tech_stack") or [])
        data_involved = sorted(data.get("data_involved") or [])
        insights_out.append({
            "asset_name": asset_name,
            "category": data.get("category", ""),
            "solution_name_set": bool((data.get("solution_name") or "").strip()),
            "tech_stack": tech_stack,
            "data_involved": data_involved,
            "risk_reasons": reasons_stable,
            "recommendation_count": len(data.get("recommendations") or []),
            "provider_profile_name": (
                data.get("provider_profile") or {}
            ).get("name") if data.get("provider_profile") else None,
        })

    return {
        "scanner": result.scanner,
        "files_scanned": result.metadata.get("files_scanned"),
        "errors": result.errors,
        "asset_count": len(assets_out),
        "assets": assets_out,
        "insights": insights_out,
    }


def _normalise_volatile(result, insights) -> dict:
    """Produce the *volatile* shape — fields we smoke-check but don't diff.

    Purpose: guard against the regression harness drowning in noise when
    a summary generator (human-authored or LLM-authored) produces
    semantically equivalent but byte-different text. We still want a
    floor check that every asset ended up with a non-empty summary and
    at least one risk reason, just not byte-level equality.
    """
    volatile = {"assets": {}, "insights": {}}
    for asset in result.assets:
        volatile["assets"][asset.name] = {
            "finding_count": len(asset.raw_findings),
            "has_code_context": bool(asset.code_contexts),
        }
    for asset_name, ins in insights.items():
        data = dataclasses.asdict(ins) if dataclasses.is_dataclass(ins) else ins.__dict__
        volatile["insights"][asset_name] = {
            "summary_length": len(data.get("summary") or ""),
            "summary_nonempty": bool((data.get("summary") or "").strip()),
            "reason_count": len(data.get("risk_reasons") or []),
        }
    return volatile


# Keep the old name as an alias so ad-hoc callers (notebooks, debuggers)
# that imported the pre-split helper still work.
_normalise = _normalise_stable


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


def _assert_volatile_floor(volatile: dict, label: str):
    """Floor checks — every asset must produce a non-empty summary and
    at least one risk reason. This catches the 'enrichment returned
    blank' failure mode without pinning the exact text."""
    for asset_name, facts in volatile["insights"].items():
        assert facts["summary_nonempty"], (
            f"{label}: asset '{asset_name}' has an empty summary"
        )
        assert facts["summary_length"] >= 15, (
            f"{label}: asset '{asset_name}' summary is implausibly short "
            f"({facts['summary_length']} chars)"
        )
        assert facts["reason_count"] >= 1, (
            f"{label}: asset '{asset_name}' has no risk reasons"
        )


def test_regression_snapshot():
    """Sprint 1 fixture tree — chatbot + hardcoded key."""
    stable, volatile = _run_pipeline_both(FIXTURES)
    _diff_against_golden(stable, GOLDEN, "fixtures")
    _assert_volatile_floor(volatile, "fixtures")


def test_regression_snapshot_sprint2():
    """Sprint 2 fixture tree — MCP config, Docker/compose, Azure OpenAI,
    fine-tuning script, local model weights. Exercises every Sprint 2
    detector in a single pipeline run."""
    stable, volatile = _run_pipeline_both(SPRINT2_FIXTURES)
    _diff_against_golden(stable, GOLDEN_SPRINT2, "fixtures/sprint2")
    _assert_volatile_floor(volatile, "fixtures/sprint2")


def test_regression_snapshot_sprint3():
    """Sprint 3 fixture tree — CI workflow, YAML config with Azure
    deployment, requirements.txt pinned to legacy openai/langchain. One
    snapshot covers CI detection, YAML model parsing and dependency
    advisories."""
    stable, volatile = _run_pipeline_both(SPRINT3_FIXTURES)
    _diff_against_golden(stable, GOLDEN_SPRINT3, "fixtures/sprint3")
    _assert_volatile_floor(volatile, "fixtures/sprint3")


def test_snapshot_is_deterministic():
    """The stable snapshot must be byte-identical across two runs in the
    same process. We explicitly test the *stable* shape only — the
    volatile shape is intentionally allowed to drift across LLM runs."""
    first = json.dumps(_run_pipeline(), indent=2, sort_keys=True, ensure_ascii=False)
    second = json.dumps(_run_pipeline(), indent=2, sort_keys=True, ensure_ascii=False)
    assert first == second, "Pipeline output is not deterministic — cannot snapshot"

    first2 = json.dumps(_run_pipeline(SPRINT2_FIXTURES), indent=2, sort_keys=True, ensure_ascii=False)
    second2 = json.dumps(_run_pipeline(SPRINT2_FIXTURES), indent=2, sort_keys=True, ensure_ascii=False)
    assert first2 == second2, "Sprint 2 pipeline output is not deterministic"

    first3 = json.dumps(_run_pipeline(SPRINT3_FIXTURES), indent=2, sort_keys=True, ensure_ascii=False)
    second3 = json.dumps(_run_pipeline(SPRINT3_FIXTURES), indent=2, sort_keys=True, ensure_ascii=False)
    assert first3 == second3, "Sprint 3 pipeline output is not deterministic"
