"""Asset enrichment — generates summaries, risk reasoning, and recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field

from aiscout.knowledge.providers import ProviderProfile, get_provider
from aiscout.models import AIAsset, Finding, FindingType


@dataclass
class RiskReason:
    """A single reason contributing to the risk score."""

    severity: str  # critical, warning, info
    title: str
    detail: str


@dataclass
class AssetInsight:
    """Enriched insight for an AI asset — summary, risk reasoning, recommendations."""

    summary: str
    risk_reasons: list[RiskReason] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    provider_profile: ProviderProfile | None = None


def enrich_asset(asset: AIAsset) -> AssetInsight:
    """Generate summary, risk reasoning, and recommendations for an asset."""
    provider = get_provider(asset.provider.name) if asset.provider else None

    summary = _build_summary(asset, provider)
    risk_reasons = _build_risk_reasons(asset, provider)
    recommendations = _build_recommendations(asset, provider, risk_reasons)

    # Recalculate risk score based on reasons
    asset.risk_score = _calculate_risk_score(risk_reasons)

    return AssetInsight(
        summary=summary,
        risk_reasons=risk_reasons,
        recommendations=recommendations,
        provider_profile=provider,
    )


def enrich_assets(assets: list[AIAsset]) -> dict[str, AssetInsight]:
    """Enrich all assets. Returns a dict mapping asset.id -> AssetInsight."""
    return {asset.id: enrich_asset(asset) for asset in assets}


# ── Summary builder ───────────────────────────────────────────────────────


def _build_summary(asset: AIAsset, provider: ProviderProfile | None) -> str:
    """Generate a summary focused on WHAT the solution does, not what the provider is."""
    # Infer purpose from file paths, imports, and directory structure
    purpose = _infer_purpose(asset)
    provider_name = provider.display_name if provider else "unknown provider"

    # Build the summary: what it does + what it's built on
    if purpose:
        summary = f"{purpose} Built on {provider_name}."
    elif provider:
        summary = f"AI solution using {provider_name}."
    else:
        summary = "AI integration detected."

    # Add key stats
    files = asset.file_path.split(", ") if asset.file_path else []
    import_count = sum(1 for f in asset.raw_findings if f.type == FindingType.IMPORT_DETECTED)
    key_count = sum(1 for f in asset.raw_findings if f.type == FindingType.API_KEY_DETECTED)

    stats = []
    if len(files) > 0:
        stats.append(f"{len(files)} file{'s' if len(files) != 1 else ''}")
    if import_count:
        stats.append(f"{import_count} import{'s' if import_count != 1 else ''}")
    if key_count:
        stats.append(f"{key_count} API key{'s' if key_count != 1 else ''} in code")

    if stats:
        summary += f" ({', '.join(stats)})"

    return summary


def _infer_purpose(asset: AIAsset) -> str:
    """Infer what the AI solution does from file paths, imports, and code context."""
    file_paths = asset.file_path.split(", ") if asset.file_path else []
    all_paths_lower = " ".join(file_paths).lower()
    all_imports = " ".join(
        f.content.lower() for f in asset.raw_findings
        if f.type == FindingType.IMPORT_DETECTED
    )

    # Collect signals from file/dir names and imports
    signals: list[str] = []

    # ── Fine-tuning / Training signals ──
    if any(kw in all_paths_lower for kw in ("finetun", "fine_tun", "fine-tun", "train")):
        # Try to find model names from file/dir names
        models = _extract_model_names(all_paths_lower + " " + all_imports)
        if models:
            signals.append(f"Fine-tuning pipeline for {', '.join(models)} model{'s' if len(models) > 1 else ''}.")
        else:
            signals.append("Model fine-tuning and training pipeline.")

    # ── RAG signals ──
    elif any(kw in all_paths_lower for kw in ("rag", "retriev", "embed", "search", "vector")):
        if "embed" in all_paths_lower and "upload" in all_paths_lower:
            signals.append("Embedding generation and vector storage for RAG retrieval.")
        elif "search" in all_paths_lower:
            signals.append("Semantic search and retrieval system.")
        elif "agent" in all_paths_lower:
            signals.append("RAG-powered AI agent pipeline.")
        else:
            signals.append("Retrieval-Augmented Generation (RAG) pipeline.")

    # ── Script / notebook signals (before backend, since scripts may contain "api" in name) ──
    elif any(kw in all_paths_lower for kw in ("script", "notebook", ".ipynb", "homework", "example", "demo")):
        signals.append("AI development scripts and experiments.")

    # ── Backend / API signals ──
    elif any(kw in all_paths_lower for kw in ("backend", "server", "app.py", "main.py")):
        if "chat" in all_paths_lower or "conversation" in all_paths_lower:
            signals.append("Backend API with AI-powered chat/conversation functionality.")
        else:
            signals.append("Backend application with AI integration.")

    # ── Chatbot signals ──
    elif any(kw in all_paths_lower for kw in ("chat", "bot", "assistant", "conversation")):
        signals.append("Conversational AI / chatbot application.")

    # ── Data processing signals ──
    elif any(kw in all_paths_lower for kw in ("pipeline", "process", "etl", "data")):
        signals.append("Data processing pipeline with AI components.")

    # ── Agent signals ──
    elif any(kw in all_paths_lower for kw in ("agent", "crew", "autogen")):
        signals.append("AI agent system.")

    # If we have specific import context, enrich
    if not signals:
        # Generic: describe based on what's imported
        if "transformers" in all_imports or "automodel" in all_imports:
            signals.append("ML model inference using Transformers library.")
        elif "chat" in all_imports or "completion" in all_imports:
            signals.append("AI-powered text generation / chat completion.")
        elif "embedding" in all_imports:
            signals.append("Text embedding and vector operations.")

    return " ".join(signals)


def _extract_model_names(text: str) -> list[str]:
    """Extract known model family names from text."""
    known_models = {
        "mistral": "Mistral",
        "gemma": "Gemma",
        "llama": "Llama",
        "phi": "Phi",
        "qwen": "Qwen",
        "falcon": "Falcon",
        "gpt": "GPT",
        "bert": "BERT",
        "t5": "T5",
        "stable diffusion": "Stable Diffusion",
        "whisper": "Whisper",
    }
    found = []
    for key, display in known_models.items():
        if key in text and display not in found:
            found.append(display)
    return found


# ── Risk reasoning ────────────────────────────────────────────────────────


def _build_risk_reasons(
    asset: AIAsset, provider: ProviderProfile | None
) -> list[RiskReason]:
    """Generate specific reasons for the risk classification."""
    reasons = []

    # ── Critical reasons ──

    # Hardcoded API keys
    key_findings = [f for f in asset.raw_findings if f.type == FindingType.API_KEY_DETECTED]
    if key_findings:
        files = sorted({f.file_path for f in key_findings})
        reasons.append(RiskReason(
            severity="critical",
            title="Hardcoded API key in source code",
            detail=(
                f"Found {len(key_findings)} API key{'s' if len(key_findings) > 1 else ''} "
                f"directly in code ({', '.join(files)}). "
                "Anyone with repository access can extract and misuse these keys. "
                "Keys should be stored in environment variables or a secret manager."
            ),
        ))

    # ── Warning reasons ──

    # Data leaves EU
    if provider and provider.data_residency:
        non_eu = [
            r for r in provider.data_residency
            if "EU" not in r and "local" not in r.lower() and "depends" not in r.lower()
        ]
        if non_eu and not any("EU" in r for r in provider.data_residency):
            reasons.append(RiskReason(
                severity="warning",
                title="Data may leave the EU",
                detail=(
                    f"{provider.display_name} processes data in: "
                    f"{', '.join(provider.data_residency)}. "
                    "If your organization is subject to GDPR or data residency requirements, "
                    "verify that a Data Processing Agreement (DPA) is in place and data "
                    "residency is configured appropriately."
                ),
            ))

    # Training policy risk
    if provider and "may be used" in provider.training_policy.lower():
        reasons.append(RiskReason(
            severity="warning",
            title="Data may be used for model training",
            detail=(
                f"{provider.display_name} training policy: {provider.training_policy} "
                "Verify which tier/plan is being used and whether data opt-out is configured."
            ),
        ))

    # External API usage (data egress)
    if provider and provider.category == "llm_api":
        reasons.append(RiskReason(
            severity="warning" if not key_findings else "info",
            title="Data sent to external AI API",
            detail=(
                f"This integration sends data to {provider.display_name} ({provider.vendor}) "
                f"for processing. Data is transmitted outside your infrastructure. "
                f"Vendor: {provider.vendor}."
            ),
        ))

    # Embedding DB with cloud component
    if provider and provider.category == "embedding_db" and "self-hosted" not in str(provider.data_residency).lower():
        reasons.append(RiskReason(
            severity="warning",
            title="Embeddings stored in external service",
            detail=(
                f"Embeddings sent to {provider.display_name} may encode sensitive information "
                "from your data. While not directly readable, embeddings can potentially be "
                "reversed or used for inference attacks."
            ),
        ))

    # ── Info reasons ──

    # Framework (data flows through to configured provider)
    if provider and provider.category == "framework":
        reasons.append(RiskReason(
            severity="info",
            title="Orchestration framework — risk depends on configured providers",
            detail=(
                f"{provider.display_name} is an orchestration framework that routes data "
                "to configured LLM providers and data stores. The actual data risk depends "
                "on which providers are configured downstream."
            ),
        ))

    # Local runtime (low risk)
    if provider and provider.category == "local_runtime":
        reasons.append(RiskReason(
            severity="info",
            title="Local runtime — no data egress",
            detail=(
                f"{provider.display_name} runs entirely on local hardware. "
                "No data is sent to external servers. This is the lowest-risk deployment model."
            ),
        ))

    # No reasons found — it's just a dependency
    if not reasons:
        reasons.append(RiskReason(
            severity="info",
            title="AI dependency detected",
            detail=(
                f"AI-related code or dependency found for {asset.name}. "
                "No specific risk indicators detected. Review the integration to ensure "
                "it aligns with your organization's AI governance policy."
            ),
        ))

    return reasons


# ── Recommendations ───────────────────────────────────────────────────────


def _build_recommendations(
    asset: AIAsset,
    provider: ProviderProfile | None,
    reasons: list[RiskReason],
) -> list[str]:
    """Generate actionable recommendations based on risk reasons."""
    recs = []
    severities = {r.severity for r in reasons}
    titles = {r.title for r in reasons}

    # API key recommendations
    if "Hardcoded API key in source code" in titles:
        recs.append(
            "Move API keys to environment variables or a secret manager "
            "(e.g., HashiCorp Vault, AWS Secrets Manager, Azure Key Vault)."
        )
        recs.append(
            "Rotate the exposed key(s) immediately — they may already be compromised "
            "if the repository has been shared or is public."
        )
        recs.append("Add API key patterns to your .gitignore and pre-commit hooks.")

    # Data residency
    if "Data may leave the EU" in titles:
        recs.append(
            "Verify GDPR compliance: ensure a DPA is signed with the provider "
            "and data residency is configured for EU region if required."
        )
        if provider and provider.enterprise_note:
            recs.append(f"Consider enterprise tier: {provider.enterprise_note}")

    # Training policy
    if "Data may be used for model training" in titles:
        recs.append(
            "Confirm which tier/plan is being used. Switch to a paid API or enterprise "
            "plan with contractual opt-out from data training."
        )

    # External API
    if "Data sent to external AI API" in titles and "critical" not in severities:
        recs.append(
            "Review what data is being sent to the API. Ensure no PII, confidential, "
            "or regulated data is transmitted without appropriate controls."
        )

    # Framework
    if any("framework" in r.title.lower() for r in reasons):
        recs.append(
            "Audit the downstream LLM providers and data stores configured in this "
            "framework to assess the full data flow risk."
        )

    # Local runtime — positive
    if any("local runtime" in r.title.lower() for r in reasons):
        recs.append(
            "Local runtime is the safest deployment model. Ensure the host machine "
            "is secured and the model weights are from a trusted source."
        )

    # Generic if nothing specific
    if not recs:
        recs.append(
            "Document this AI integration in your organization's AI asset inventory."
        )
        recs.append(
            "Review the integration against your AI governance policy."
        )

    return recs


# ── Risk score calculation ────────────────────────────────────────────────


def _calculate_risk_score(reasons: list[RiskReason]) -> float:
    """Calculate a weighted risk score from risk reasons."""
    if not reasons:
        return 0.1

    weights = {"critical": 0.4, "warning": 0.2, "info": 0.05}
    score = sum(weights.get(r.severity, 0.0) for r in reasons)

    return min(1.0, max(0.0, round(score, 2)))
