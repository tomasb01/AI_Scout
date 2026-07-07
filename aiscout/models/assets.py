"""Core data models for AI Scout."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


def now_utc() -> datetime:
    """Current UTC time, overridable via ``AISCOUT_TIMESTAMP`` (ISO 8601).

    The override exists so two runs over identical inputs can produce
    bit-identical output — a precondition for signing and diffing
    (datamodel spec §1.5, AIBOM determinism requirement).
    """
    override = os.environ.get("AISCOUT_TIMESTAMP")
    if override:
        ts = datetime.fromisoformat(override)
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


# ── Enums ──────────────────────────────────────────────────────────────────


class AssetType(StrEnum):
    COMMERCIAL_SAAS = "commercial_saas"
    CUSTOM_CODE = "custom_code"
    LOCAL_MODEL = "local_model"
    AUTOMATION = "automation"
    AGENT = "agent"
    MCP_SERVER = "mcp_server"


class TaskType(StrEnum):
    """What the code *does* with the AI model.

    Used to distinguish inference from training/fine-tuning, which have
    very different data-privacy and compute implications.
    """

    INFERENCE = "inference"
    TRAINING = "training"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"
    UNKNOWN = "unknown"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Documentation(StrEnum):
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"


class DataCategory(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PII = "pii"
    FINANCIAL = "financial"
    SOURCE_CODE = "source_code"
    UNKNOWN = "unknown"


class FindingType(StrEnum):
    IMPORT_DETECTED = "import_detected"
    API_KEY_DETECTED = "api_key_detected"
    DEPENDENCY_DETECTED = "dependency_detected"
    CONFIG_DETECTED = "config_detected"
    LOCAL_MODEL_DETECTED = "local_model_detected"
    CONTAINER_DETECTED = "container_detected"


class Severity(StrEnum):
    """Severity of a single finding — one axis of the two-axis model.

    Combined with per-finding ``confidence`` (the second axis) instead of
    a single weighted score: evidence, not verdict (Product Spec v13 §9.3).
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RiskStatus(StrEnum):
    """Categorical solution-level status derived from findings.

    Deliberately three-valued; never "ok" — absence of findings is not a
    safety verdict, just an absence of evidence (spec v13: "No findings").
    """

    CRITICAL = "critical"
    REVIEW = "review"
    NO_FINDINGS = "no_findings"


# ── Helper models ──────────────────────────────────────────────────────────


class DataFlow(BaseModel):
    source: str
    destination: str
    data_types: list[str] = []
    description: str = ""


class ProviderInfo(BaseModel):
    name: str
    region: str = "unknown"
    training_policy: str = ""
    certifications: list[str] = []


class ClassificationResult(BaseModel):
    """LLM data classification: categories + confidence, never a verdict."""

    categories: list[DataCategory] = []
    confidence: Confidence = Confidence.LOW
    details: str = ""
    recommendations: list[str] = []


class Finding(BaseModel):
    # Stable ID: f-<hash12>(repo | rule_id | file:line | provider), assigned
    # by the scanner once the repo is known. Survives across scans — the
    # precondition for diff (Sprint 2) and SARIF fingerprints (Sprint 1).
    id: str = ""
    type: FindingType
    rule_id: str = ""
    rule_version: int = 1
    severity: Severity = Severity.INFO
    confidence: float = 1.0  # deterministic detections are 1.0 (spec §1.4)
    file_path: str
    line_number: int | None = None
    content: str
    redacted_content: str | None = None
    provider: str = ""


# ── Data Flow models (Sprint 5) ───────────────────────────────────────────


class FlowSource(BaseModel):
    """Where data enters the AI solution."""

    type: str = ""  # database, file, api, user_input, env_var, message_queue
    name: str = ""  # human-readable: "Chat message", "Customer DB"
    detail: str = ""  # technical: "POST /chat — request.json['message']"


class FlowSink(BaseModel):
    """Where data leaves the AI solution."""

    type: str = ""  # ai_api, database, file, http_response, webhook, log
    name: str = ""  # human-readable: "Anthropic Claude API", "results.json"
    detail: str = ""  # technical: "claude-sonnet-4-20250514 via messages.create()"
    provider: str = ""  # "anthropic", "openai", "" for non-AI


class DataFlowMap(BaseModel):
    """Structured data flow for one AI solution — the core value entity.

    Built by ``engine/data_flow.py`` from CodeContext data (no LLM
    required). LLM enrichment can later refine ``solution_purpose``
    and ``data_categories`` but the structural flow (sources → steps →
    sinks) is constructed purely from code analysis.
    """

    solution_purpose: str = ""
    sources: list[FlowSource] = []
    sinks: list[FlowSink] = []
    processing_steps: list[str] = []
    data_categories: list[str] = []
    confidence: Confidence = Confidence.MEDIUM


# ── Code analysis models ───────────────────────────────────────────────────


class CodeContext(BaseModel):
    """Structured context extracted from source code analysis."""

    file_path: str
    language: str = ""  # python, javascript, typescript, etc.
    functions: list[dict] = []  # {name, args, docstring, body_preview}
    classes: list[dict] = []  # {name, methods, docstring}
    api_calls: list[dict] = []  # {target, method, args_preview}
    data_sources: list[dict] = []  # {type, name, detail}
    data_sinks: list[dict] = []  # {type, name, detail}
    prompts: list[str] = []  # system/user prompt texts
    env_vars: list[str] = []
    model_names: list[str] = []  # LLM model identifiers found in code (e.g. "gpt-4o", "claude-3-sonnet")
    raw_snippets: list[str] = []  # key code excerpts (truncated)


# ── Primary entities ───────────────────────────────────────────────────────


class AIAsset(BaseModel):
    # Stable ID: sol-<hash12>(repo | normalized solution root path), assigned
    # by the scanner. The uuid4 default only covers ad-hoc construction in
    # tests/tools; the scan pipeline always overwrites it.
    #
    # NOTE (Sprint 0.3 re-baseline, accepted decision — see README_BUNDLE.md):
    # when Sprint 0.3 introduces the aggregation boundary ("solution =
    # application/service" above directory grouping), the root path that
    # feeds this hash is promoted from directory to aggregation root, which
    # CHANGES solution IDs. A one-time golden-snapshot re-baseline after 0.3
    # is expected and cheap because diff (Sprint 2) lands after it. Finding
    # IDs are unaffected — they hash rule + location, not the grouping.
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: AssetType = AssetType.CUSTOM_CODE
    owner: str = "unknown"
    users: list[str] = []
    data_inputs: list[DataFlow] = []
    data_outputs: list[DataFlow] = []
    provider: ProviderInfo | None = None
    risk_status: RiskStatus = RiskStatus.NO_FINDINGS
    data_classification: ClassificationResult | None = None
    discovered_via: list[str] = []
    last_activity: datetime | None = None
    documentation: Documentation = Documentation.NONE
    file_path: str = ""
    repository: str = ""
    dependencies: list[str] = []
    raw_findings: list[Finding] = []
    code_contexts: list[CodeContext] = []
    data_flow: DataFlowMap | None = None  # Sprint 5
    task_types: list[TaskType] = []
    tags: list[str] = []


class ScannerConfig(BaseModel):
    name: str
    required_credentials: list[str] = []
    description: str = ""


class ScanResult(BaseModel):
    # scan_id is internal-only (not rendered into HTML/JSON outputs), so a
    # random UUID does not break the bit-identical-output guarantee.
    scan_id: str = Field(default_factory=lambda: str(uuid4()))
    scanner: str
    started_at: datetime = Field(default_factory=now_utc)
    completed_at: datetime | None = None
    assets: list[AIAsset] = []
    errors: list[str] = []
    metadata: dict = {}

    def merge(self, other: ScanResult) -> ScanResult:
        """Merge two ScanResults into a new combined result."""
        return ScanResult(
            scanner=f"{self.scanner}+{other.scanner}",
            started_at=min(self.started_at, other.started_at),
            completed_at=max(
                self.completed_at or self.started_at,
                other.completed_at or other.started_at,
            ),
            assets=self.assets + other.assets,
            errors=self.errors + other.errors,
            metadata={**self.metadata, **other.metadata},
        )
