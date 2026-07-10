"""SARIF 2.1.0 export — findings where developers live (Sprint 1).

Emits static-analysis results consumable by GitHub code scanning and
GitLab (via SARIF ingestion/conversion). Design decisions:

* **Security tab gets security findings.** By default only findings of
  severity ≥ medium become SARIF results (today: hardcoded API keys,
  SEC-KEY-001). Discovery findings (imports, dependencies, configs) are
  inventory, not alerts — flooding a security dashboard with hundreds
  of "import detected" notes would bury the real signal. They can be
  included explicitly (``include_discovery=True`` / CLI
  ``--sarif-include-discovery``) and map to level ``note``.
* **Stable fingerprints from stable IDs.** ``partialFingerprints``
  carries the Sprint 0.1 finding ID (hash of repo | rule | location |
  provider), so alerts don't duplicate across scans and survive
  unrelated file edits.
* **Deterministic output** — explicit sorting, no timestamps — two runs
  over the same scan produce byte-identical SARIF (same discipline as
  the JSON export; a precondition for diffing and signing).
* One SARIF ``run`` per scanned repository; with multiple repos each
  run carries ``automationDetails.id`` so upload targets stay separate.
"""

from __future__ import annotations

import json
from pathlib import Path

from aiscout import __version__
from aiscout.knowledge.providers import get_provider
from aiscout.models import Finding, ScanResult, Severity

_SCOUT_URI = "https://github.com/tomasb01/AI_Scout"

# severity → SARIF level + GitHub security-severity score (CVSS-like
# scale GitHub uses to bucket alerts: 9.0+ critical, 7.0+ high,
# 4.0+ medium, <4.0 low).
_LEVELS: dict[Severity, tuple[str, str]] = {
    Severity.CRITICAL: ("error", "9.1"),
    Severity.HIGH: ("error", "7.5"),
    Severity.MEDIUM: ("warning", "5.0"),
    Severity.LOW: ("note", "3.0"),
    Severity.INFO: ("note", "0.0"),
}

_ALERT_SEVERITIES = {Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM}

# Rule catalog metadata — human-written, versioned with the scanner's
# _FINDING_RULES table (rule ids must match).
_RULE_META: dict[str, dict] = {
    "SEC-KEY-001": {
        "name": "HardcodedApiKey",
        "shortDescription": "Hardcoded AI provider API key in source",
        "fullDescription": (
            "An AI provider API key is committed to the repository. "
            "Anyone with repository access can extract and misuse it. "
            "Rotate the key and move it to an environment variable or "
            "a secret manager."
        ),
        "tags": ["security", "secret", "ai"],
    },
    "DISC-IMP-001": {
        "name": "AiImportDetected",
        "shortDescription": "AI SDK or framework import",
        "fullDescription": (
            "The file imports an AI provider SDK or framework — "
            "inventory evidence for the AI solution catalog."
        ),
        "tags": ["discovery", "ai"],
    },
    "DISC-DEP-001": {
        "name": "AiDependencyDetected",
        "shortDescription": "AI package in dependency manifest",
        "fullDescription": (
            "A dependency manifest declares an AI-related package — "
            "inventory evidence for the AI solution catalog."
        ),
        "tags": ["discovery", "ai"],
    },
    "DISC-CFG-001": {
        "name": "AiConfigReference",
        "shortDescription": "AI model reference in configuration",
        "fullDescription": (
            "A configuration file references an AI model or deployment — "
            "inventory evidence for the AI solution catalog."
        ),
        "tags": ["discovery", "ai"],
    },
    "DISC-MDL-001": {
        "name": "LocalModelArtifact",
        "shortDescription": "Local model weights artifact",
        "fullDescription": (
            "A local model artifact (weights file) is present in the "
            "repository — inventory evidence for the AI solution catalog."
        ),
        "tags": ["discovery", "ai"],
    },
    "DISC-CTR-001": {
        "name": "AiContainerService",
        "shortDescription": "AI service in container manifest",
        "fullDescription": (
            "A Docker/compose manifest runs an AI-related service — "
            "inventory evidence for the AI solution catalog."
        ),
        "tags": ["discovery", "ai"],
    },
}


class SarifExporter:
    """Export scan findings as SARIF 2.1.0."""

    def __init__(
        self,
        scan_results: list[ScanResult],
        output_path: str = "aiscout.sarif",
        insights: dict | None = None,
        include_discovery: bool = False,
    ):
        self.scan_results = scan_results
        self.output_path = output_path
        self.insights = insights or {}
        self.include_discovery = include_discovery
        # Interface parity with the other generators (CLI --strict gate);
        # the QA sentence linter has no prose to check here.
        self.qa_result = None

    def generate(self) -> Path:
        doc = self._build_document()
        out = Path(self.output_path)
        out.write_text(
            json.dumps(doc, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return out

    # ── Document assembly ───────────────────────────────────────────────

    def _build_document(self) -> dict:
        multi_repo = len(self.scan_results) > 1
        runs = [
            self._build_run(result, multi_repo)
            for result in self.scan_results
        ]
        return {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": runs,
        }

    def _build_run(self, result: ScanResult, multi_repo: bool) -> dict:
        findings = self._selected_findings(result)
        rule_ids = sorted({f.rule_id for f, _ in findings})
        rule_index = {rid: i for i, rid in enumerate(rule_ids)}

        results = [
            self._build_result(finding, solution_name, rule_index)
            for finding, solution_name in findings
        ]
        results.sort(key=lambda r: (
            r["ruleId"],
            r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"],
            r["locations"][0]["physicalLocation"].get("region", {}).get("startLine", 0),
            r["partialFingerprints"]["aiScoutFindingId/v1"],
        ))

        run: dict = {
            "tool": {
                "driver": {
                    "name": "AI Scout",
                    "informationUri": _SCOUT_URI,
                    "semanticVersion": __version__,
                    "rules": [self._build_rule(rid) for rid in rule_ids],
                }
            },
            "columnKind": "utf16CodeUnits",
            "results": results,
        }
        if multi_repo:
            repo = result.metadata.get("repository", "unknown")
            run["automationDetails"] = {"id": f"aiscout/{repo}"}
        return run

    def _selected_findings(
        self, result: ScanResult
    ) -> list[tuple[Finding, str]]:
        """(finding, solution display name) pairs that become results."""
        selected = []
        for asset in result.assets:
            insight = self.insights.get(asset.id)
            solution_name = (
                insight.solution_name
                if insight and insight.solution_name else asset.name
            )
            for f in asset.raw_findings:
                if f.severity in _ALERT_SEVERITIES or self.include_discovery:
                    selected.append((f, solution_name))
        return selected

    def _build_rule(self, rule_id: str) -> dict:
        meta = _RULE_META.get(rule_id, {
            "name": rule_id,
            "shortDescription": rule_id,
            "fullDescription": rule_id,
            "tags": ["ai"],
        })
        severity = _rule_severity(rule_id)
        level, security_severity = _LEVELS[severity]
        rule = {
            "id": rule_id,
            "name": meta["name"],
            "shortDescription": {"text": meta["shortDescription"]},
            "fullDescription": {"text": meta["fullDescription"]},
            "helpUri": _SCOUT_URI,
            "defaultConfiguration": {"level": level},
            "properties": {"tags": meta["tags"]},
        }
        if "security" in meta["tags"]:
            rule["properties"]["security-severity"] = security_severity
        return rule

    def _build_result(
        self, finding: Finding, solution_name: str, rule_index: dict
    ) -> dict:
        level, _ = _LEVELS[finding.severity]
        location: dict = {
            "physicalLocation": {
                "artifactLocation": {
                    "uri": finding.file_path,
                    "uriBaseId": "%SRCROOT%",
                },
            }
        }
        if finding.line_number is not None:
            location["physicalLocation"]["region"] = {
                "startLine": finding.line_number,
            }
        return {
            "ruleId": finding.rule_id,
            "ruleIndex": rule_index[finding.rule_id],
            "level": level,
            "message": {"text": _result_message(finding, solution_name)},
            "locations": [location],
            # Stable across scans (Sprint 0.1) — prevents duplicate
            # alerts in GitHub code scanning.
            "partialFingerprints": {"aiScoutFindingId/v1": finding.id},
            "properties": {
                "provider": finding.provider,
                "solution": solution_name,
                "confidence": finding.confidence,
                "ruleVersion": finding.rule_version,
            },
        }


# ── Helpers ────────────────────────────────────────────────────────────────


def _rule_severity(rule_id: str) -> Severity:
    """Severity for the rule catalog — mirrors the scanner's table."""
    from aiscout.scanners.git_scanner import _FINDING_RULES

    for _, (rid, _version, severity) in _FINDING_RULES.items():
        if rid == rule_id:
            return severity
    return Severity.INFO


def _result_message(finding: Finding, solution_name: str) -> str:
    """Human message for one result — never contains raw secrets
    (redacted content only, same discipline as the reports)."""
    provider_display = (
        get_provider(finding.provider).display_name
        if finding.provider else "AI provider"
    )
    if finding.rule_id == "SEC-KEY-001":
        redacted = finding.redacted_content or "<redacted>"
        return (
            f"Hardcoded {provider_display} API key in source "
            f"({redacted}), part of solution '{solution_name}'. "
            "Rotate the key and move it to an environment variable or "
            "a secret manager."
        )
    kind = _RULE_META.get(finding.rule_id, {}).get(
        "shortDescription", "AI usage detected"
    )
    return f"{kind}: {provider_display} (solution '{solution_name}')."
