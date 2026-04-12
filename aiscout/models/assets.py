"""Core data models for AI Scout."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


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
    categories: list[DataCategory] = []
    confidence: Confidence = Confidence.LOW
    details: str = ""
    recommendations: list[str] = []
    risk_score: float = 0.0


class Finding(BaseModel):
    type: FindingType
    file_path: str
    line_number: int | None = None
    content: str
    redacted_content: str | None = None
    provider: str = ""


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
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: AssetType = AssetType.CUSTOM_CODE
    owner: str = "unknown"
    users: list[str] = []
    data_inputs: list[DataFlow] = []
    data_outputs: list[DataFlow] = []
    provider: ProviderInfo | None = None
    risk_score: float = 0.0
    data_classification: ClassificationResult | None = None
    discovered_via: list[str] = []
    last_activity: datetime | None = None
    documentation: Documentation = Documentation.NONE
    file_path: str = ""
    repository: str = ""
    dependencies: list[str] = []
    raw_findings: list[Finding] = []
    code_contexts: list[CodeContext] = []
    task_types: list[TaskType] = []  # Sprint 2 — training vs inference etc.
    tags: list[str] = []  # Sprint 2 — chatbot/rag/agent/training/…


class ScannerConfig(BaseModel):
    name: str
    required_credentials: list[str] = []
    description: str = ""


class ScanResult(BaseModel):
    scan_id: str = Field(default_factory=lambda: str(uuid4()))
    scanner: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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
