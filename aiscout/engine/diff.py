"""Scan diff — what changed between two scans (Sprint 2).

Compares two JSON exports by the stable IDs introduced in Sprint 0.1
(``sol-…`` solution IDs, ``f-…`` finding IDs) and produces a
deterministic delta: solutions added/removed/changed, providers that
appeared, keys that appeared/disappeared, model references and data
flows that moved. This is the change-management artifact: "what is new
since the last audit".

Works on exported JSON (schema ≥ 1.1), not on live scans — both sides
of a diff are reviewable, signable files. ``resolved`` is an automatic
verdictless observation: the finding is no longer detected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_KEY_RULE = "SEC-KEY-001"


@dataclass
class ScanDelta:
    """Delta between two scan exports (old → new)."""

    old_generated_at: str = ""
    new_generated_at: str = ""
    old_total: int = 0
    new_total: int = 0

    added_solutions: list[dict] = field(default_factory=list)
    removed_solutions: list[dict] = field(default_factory=list)
    changed_solutions: list[dict] = field(default_factory=list)

    new_providers: list[str] = field(default_factory=list)
    removed_providers: list[str] = field(default_factory=list)

    new_key_findings: list[dict] = field(default_factory=list)
    resolved_key_findings: list[dict] = field(default_factory=list)

    def counts(self) -> dict:
        return {
            "added": len(self.added_solutions),
            "removed": len(self.removed_solutions),
            "changed": len(self.changed_solutions),
            "new_providers": len(self.new_providers),
            "new_key_findings": len(self.new_key_findings),
            "resolved_key_findings": len(self.resolved_key_findings),
        }

    def insight_values(self) -> dict:
        """Values for the I-09 SCAN_DELTA insight (QA catalog)."""
        return {
            "prev_date": self.old_generated_at[:10],
            "added": len(self.added_solutions),
            "removed": len(self.removed_solutions),
            "new_providers": len(self.new_providers),
        }

    def to_dict(self) -> dict:
        return {
            "baseline_generated_at": self.old_generated_at,
            "current_generated_at": self.new_generated_at,
            "totals": {"baseline": self.old_total, "current": self.new_total},
            "counts": self.counts(),
            "added_solutions": self.added_solutions,
            "removed_solutions": self.removed_solutions,
            "changed_solutions": self.changed_solutions,
            "new_providers": self.new_providers,
            "removed_providers": self.removed_providers,
            "new_key_findings": self.new_key_findings,
            "resolved_key_findings": self.resolved_key_findings,
        }


def load_export(path: str | Path) -> dict:
    """Load and minimally validate a Scout JSON export."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "solutions" not in data or "schema_version" not in data:
        raise ValueError(
            f"{path}: not an AI Scout JSON export "
            "(missing schema_version/solutions)"
        )
    return data


def diff_exports(old: dict, new: dict) -> ScanDelta:
    """Compute the delta between two loaded exports."""
    delta = ScanDelta(
        old_generated_at=old.get("generated_at", ""),
        new_generated_at=new.get("generated_at", ""),
    )

    old_solutions = {s["id"]: s for s in old.get("solutions", [])}
    new_solutions = {s["id"]: s for s in new.get("solutions", [])}
    # Dependency-evidence rows are facts about the repo, not solutions —
    # keep them out of the solution delta (consistent with all counts).
    old_real = {
        sid: s for sid, s in old_solutions.items()
        if s.get("role", "application") != "dependency_manifest"
    }
    new_real = {
        sid: s for sid, s in new_solutions.items()
        if s.get("role", "application") != "dependency_manifest"
    }
    delta.old_total = len(old_real)
    delta.new_total = len(new_real)

    for sid in sorted(new_real.keys() - old_real.keys()):
        delta.added_solutions.append(_solution_ref(new_real[sid]))
    for sid in sorted(old_real.keys() - new_real.keys()):
        delta.removed_solutions.append(_solution_ref(old_real[sid]))
    for sid in sorted(old_real.keys() & new_real.keys()):
        changes = _solution_changes(old_real[sid], new_real[sid])
        if changes:
            ref = _solution_ref(new_real[sid])
            ref["changes"] = changes
            delta.changed_solutions.append(ref)

    old_providers = _providers(old_solutions.values())
    new_providers = _providers(new_solutions.values())
    delta.new_providers = sorted(new_providers - old_providers)
    delta.removed_providers = sorted(old_providers - new_providers)

    old_keys = _key_findings(old_solutions.values())
    new_keys = _key_findings(new_solutions.values())
    for fid in sorted(new_keys.keys() - old_keys.keys()):
        delta.new_key_findings.append(new_keys[fid])
    for fid in sorted(old_keys.keys() - new_keys.keys()):
        resolved = dict(old_keys[fid])
        resolved["status"] = "resolved"
        delta.resolved_key_findings.append(resolved)

    return delta


def diff_files(old_path: str | Path, new_path: str | Path) -> ScanDelta:
    return diff_exports(load_export(old_path), load_export(new_path))


# ── Helpers ────────────────────────────────────────────────────────────────


def _solution_ref(solution: dict) -> dict:
    return {
        "id": solution["id"],
        "name": solution.get("name", ""),
        "repository": solution.get("repository", ""),
        "path": solution.get("path", ""),
        "risk_status": solution.get("risk_status", ""),
        "category": solution.get("category", ""),
    }


def _solution_changes(old: dict, new: dict) -> dict:
    """Field-level changes worth surfacing for a persisting solution."""
    changes: dict = {}

    if old.get("risk_status") != new.get("risk_status"):
        changes["risk_status"] = {
            "from": old.get("risk_status"), "to": new.get("risk_status"),
        }

    old_findings = {f["id"] for f in old.get("findings", []) if f.get("id")}
    new_findings = {f["id"] for f in new.get("findings", []) if f.get("id")}
    if old_findings != new_findings:
        changes["findings"] = {
            "added": sorted(new_findings - old_findings),
            "resolved": sorted(old_findings - new_findings),
        }

    old_models = {m["model"] for m in old.get("model_refs", [])}
    new_models = {m["model"] for m in new.get("model_refs", [])}
    if old_models != new_models:
        changes["model_refs"] = {
            "added": sorted(new_models - old_models),
            "removed": sorted(old_models - new_models),
        }

    old_provider = (old.get("provider") or {}).get("name", "")
    new_provider = (new.get("provider") or {}).get("name", "")
    if old_provider != new_provider:
        changes["provider"] = {"from": old_provider, "to": new_provider}

    return changes


def _providers(solutions) -> set[str]:
    providers: set[str] = set()
    for s in solutions:
        name = (s.get("provider") or {}).get("name", "")
        if name:
            providers.add(name)
        for f in s.get("findings", []):
            if f.get("provider"):
                providers.add(f["provider"])
    return providers


def _key_findings(solutions) -> dict[str, dict]:
    """Hardcoded-key findings by stable ID (redacted content only)."""
    keys: dict[str, dict] = {}
    for s in solutions:
        for f in s.get("findings", []):
            if f.get("rule", {}).get("id") == _KEY_RULE and f.get("id"):
                keys[f["id"]] = {
                    "id": f["id"],
                    "solution": s.get("name", ""),
                    "repository": s.get("repository", ""),
                    "file_path": f.get("file_path", ""),
                    "line_number": f.get("line_number"),
                    "content": f.get("content", ""),  # already redacted
                    "provider": f.get("provider", ""),
                }
    return keys
