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
    solution_name: str = ""  # human-readable name derived from code
    category: str = ""  # chatbot, rag, fine-tuning, agent, script, api, etc.
    tech_stack: list[str] = field(default_factory=list)  # all technologies used
    data_involved: list[str] = field(default_factory=list)  # what data is processed
    risk_reasons: list[RiskReason] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    provider_profile: ProviderProfile | None = None


def enrich_asset(asset: AIAsset) -> AssetInsight:
    """Generate summary, risk reasoning, and recommendations for an asset."""
    provider = get_provider(asset.provider.name) if asset.provider else None

    summary = _build_summary(asset, provider)
    risk_reasons = _build_risk_reasons(asset, provider)
    recommendations = _build_recommendations(asset, provider, risk_reasons)
    tech_stack = _extract_tech_stack(asset)
    data_involved = _extract_data_involved(asset)
    category = _classify_category(asset)
    solution_name = _derive_solution_display_name(asset, category)

    # Recalculate risk score based on reasons
    asset.risk_score = _calculate_risk_score(risk_reasons)

    return AssetInsight(
        summary=summary,
        solution_name=solution_name,
        category=category,
        tech_stack=tech_stack,
        data_involved=data_involved,
        risk_reasons=risk_reasons,
        recommendations=recommendations,
        provider_profile=provider,
    )


def enrich_assets(assets: list[AIAsset]) -> dict[str, AssetInsight]:
    """Enrich all assets. Returns a dict mapping asset.id -> AssetInsight."""
    return {asset.id: enrich_asset(asset) for asset in assets}


# ── Summary builder ───────────────────────────────────────────────────────


def _build_summary(asset: AIAsset, provider: ProviderProfile | None) -> str:
    """Generate a summary focused on WHAT the solution does, not what the provider is.

    Priority: purpose from code > directory context > provider name.
    Provider goes into tech_stack, not into summary.
    """
    purpose = _infer_purpose(asset)

    if purpose:
        return purpose

    # Fallback: directory context + provider
    dir_context = _get_dir_context(asset)
    provider_name = provider.display_name if provider else "unknown"
    if dir_context:
        return f"{dir_context} AI solution using {provider_name}."
    return f"AI solution using {provider_name}."


def _get_dir_context(asset: AIAsset) -> str:
    """Extract meaningful context from directory path."""
    if not asset.file_path:
        return ""

    from aiscout.scanners.git_scanner import _clean_dir_name
    first_file = asset.file_path.split(", ")[0]
    parts = first_file.split("/")

    # Build context from directory hierarchy (skip leaf filename)
    dir_parts = parts[:-1] if len(parts) > 1 else []
    if not dir_parts:
        return ""

    # Clean and filter directory names
    meaningful = []
    for p in dir_parts:
        cleaned = _clean_dir_name(p)
        if cleaned and cleaned.lower() not in ("old", "archive", "src", "lib"):
            meaningful.append(cleaned)

    if meaningful:
        return " > ".join(meaningful) + "."

    return ""


def _infer_purpose(asset: AIAsset) -> str:
    """Infer what the AI solution does from code context, file paths, and imports."""
    # Priority 1: Use code contexts if available (deep analysis)
    if asset.code_contexts:
        purpose = _infer_from_code_contexts(asset)
        if purpose:
            return purpose

    # Priority 2: Fallback to file path / import heuristics
    return _infer_from_paths(asset)


def _infer_from_code_contexts(asset: AIAsset) -> str:
    """Infer purpose from extracted code contexts — functions, prompts, API calls, README."""
    parts = []

    # Collect all context data across files
    all_functions = []
    all_prompts = []
    all_api_calls = []
    all_sources = []
    all_sinks = []
    all_env_vars = []
    all_readmes = []
    all_docstrings = []

    for ctx in asset.code_contexts:
        all_functions.extend(ctx.functions)
        all_prompts.extend(ctx.prompts)
        all_api_calls.extend(ctx.api_calls)
        all_sources.extend(ctx.data_sources)
        all_sinks.extend(ctx.data_sinks)
        all_env_vars.extend(ctx.env_vars)
        # README content
        if ctx.language == "markdown" and ctx.raw_snippets:
            all_readmes.extend(ctx.raw_snippets)
        # Collect docstrings from functions
        for func in ctx.functions:
            if func.get("docstring"):
                all_docstrings.append(func["docstring"])
        for cls in ctx.classes:
            if cls.get("docstring"):
                all_docstrings.append(cls["docstring"])

    # ── Priority 1: README content ──
    if all_readmes:
        readme_text = all_readmes[0]
        # Extract first meaningful paragraph from README
        lines = [l.strip() for l in readme_text.split("\n") if l.strip()]
        # Skip title lines (# headings)
        content_lines = [l for l in lines if not l.startswith("#") and len(l) > 20]
        if content_lines:
            parts.append(content_lines[0][:200] + ("." if not content_lines[0].endswith(".") else ""))

    # ── Prompts tell us about purpose ──
    if all_prompts:
        best_prompt = max(all_prompts, key=len)
        purpose_from_prompt = _extract_purpose_from_prompt(best_prompt)
        if purpose_from_prompt:
            parts.append(purpose_from_prompt)

    # ── Docstrings describe purpose directly ──
    if all_docstrings and not parts:
        best_doc = max(all_docstrings, key=len)
        if len(best_doc) > 15:
            parts.append(best_doc[:200] + ("." if not best_doc.endswith(".") else ""))

    # ── Functions with decorators (Flask routes, FastAPI endpoints) ──
    endpoints = []
    for func in all_functions:
        decorators = func.get("decorators", [])
        for dec in decorators:
            if "route" in dec or "app." in dec or "router." in dec:
                args_str = ", ".join(func.get("args", []))
                endpoints.append(f"{func['name']}({args_str})")

    if endpoints:
        parts.append(f"API endpoints: {', '.join(endpoints[:5])}.")

    # ── Key function names hint at purpose ──
    meaningful_funcs = [
        f for f in all_functions
        if f["name"] not in ("__init__", "main", "setup", "config", "run")
        and not f["name"].startswith("_")
    ]
    if meaningful_funcs and len(parts) < 2:
        func_names = [f["name"] for f in meaningful_funcs[:6]]
        parts.append(f"Key functions: {', '.join(func_names)}.")

    # ── Data flow description ──
    source_types = set()
    sink_types = set()

    for src in all_sources:
        if src["type"] == "database":
            detail = src.get("detail", "").upper()
            if "SELECT" in detail:
                # Extract table name
                import re
                table_match = re.search(r'FROM\s+(\w+)', detail, re.IGNORECASE)
                if table_match:
                    source_types.add(f"database ({table_match.group(1)})")
                else:
                    source_types.add("database")
            else:
                source_types.add("database")
        elif src["type"] == "file":
            source_types.add("file input")
        else:
            source_types.add(src["type"])

    for sink in all_sinks:
        if sink["type"] == "http":
            sink_types.add(f"external API ({sink.get('detail', '')[:40]})")
        elif sink["type"] == "file":
            sink_types.add("file output")
        elif sink["type"] == "database":
            sink_types.add("database")
        else:
            sink_types.add(sink["type"])

    if source_types or sink_types:
        flow_parts = []
        if source_types:
            flow_parts.append(f"Reads from: {', '.join(list(source_types)[:3])}")
        if sink_types:
            flow_parts.append(f"Outputs to: {', '.join(list(sink_types)[:3])}")
        if flow_parts:
            parts.append(". ".join(flow_parts) + ".")

    # ── Key functions describe what the code does ──
    if not parts:
        meaningful_funcs = [
            f for f in all_functions
            if f["name"] not in ("__init__", "main", "setup", "config")
            and not f["name"].startswith("_")
        ]
        if meaningful_funcs:
            func_names = [f["name"] for f in meaningful_funcs[:5]]
            docstrings = [f["docstring"] for f in meaningful_funcs if f.get("docstring")]
            if docstrings:
                parts.append(docstrings[0][:150] + "." if not docstrings[0].endswith(".") else docstrings[0][:150])
            else:
                parts.append(f"Functions: {', '.join(func_names)}.")

    # ── AI API calls ──
    if all_api_calls and not any("API" in p for p in parts):
        targets = set(call.get("target", "")[:50] for call in all_api_calls[:3])
        if targets:
            parts.append(f"AI calls: {', '.join(targets)}.")

    # ── Environment variables hint at integrations ──
    interesting_vars = [v for v in all_env_vars if any(
        kw in v.upper() for kw in ("KEY", "TOKEN", "URL", "HOST", "DB", "SECRET", "API")
    )]
    if interesting_vars and not parts:
        parts.append(f"Uses config: {', '.join(interesting_vars[:4])}.")

    # ── Fallback to file path heuristics if code context didn't produce much ──
    if not parts:
        return _infer_from_paths(asset)

    return " ".join(parts)


def _extract_purpose_from_prompt(prompt_text: str) -> str:
    """Extract what the AI solution DOES from a system prompt.

    Skips generic role descriptions like "You are a helpful assistant"
    and looks for the actual task/purpose instructions.
    """
    import re
    text = prompt_text.strip()
    sentences = [s.strip() for s in re.split(r'[.!\n]', text) if s.strip()]

    # Skip the "You are X" preamble if generic, use the NEXT sentence(s)
    useful_sentences = []
    for s in sentences:
        # Skip generic role statements
        if re.match(r"(?:You are|You're|I am|Act as)\s+(?:a |an )", s, re.IGNORECASE):
            role = re.sub(r"(?:You are|You're|I am|Act as)\s+", "", s, flags=re.IGNORECASE).strip()
            if not _is_generic_role(role) and len(role) > 15:
                useful_sentences.append(f'Role: "{role[:120]}"')
            continue
        # Skip very short sentences
        if len(s) < 10:
            continue
        # Skip meta-instructions to the LLM
        if any(kw in s.lower() for kw in ("respond in json", "return json", "format your", "do not")):
            continue
        useful_sentences.append(s[:150])

    if useful_sentences:
        return ". ".join(useful_sentences[:2]) + "."

    # Fallback: first meaningful chunk of the prompt
    if len(text) > 20:
        return f'Prompt: "{text[:120]}..."'

    return ""


def _infer_from_paths(asset: AIAsset) -> str:
    """Fallback: infer purpose from file paths and import names."""
    file_paths = asset.file_path.split(", ") if asset.file_path else []
    all_paths_lower = " ".join(file_paths).lower()
    all_imports = " ".join(
        f.content.lower() for f in asset.raw_findings
        if f.type == FindingType.IMPORT_DETECTED
    )

    signals: list[str] = []

    if any(kw in all_paths_lower for kw in ("finetun", "fine_tun", "fine-tun", "train")):
        models = _extract_model_names(all_paths_lower + " " + all_imports)
        if models:
            signals.append(f"Fine-tuning pipeline for {', '.join(models)} model{'s' if len(models) > 1 else ''}.")
        else:
            signals.append("Model fine-tuning and training pipeline.")
    elif any(kw in all_paths_lower for kw in ("rag", "retriev", "embed", "search", "vector")):
        if "agent" in all_paths_lower:
            signals.append("RAG-powered AI agent pipeline.")
        elif "search" in all_paths_lower:
            signals.append("Semantic search and retrieval system.")
        else:
            signals.append("Retrieval-Augmented Generation (RAG) pipeline.")
    elif any(kw in all_paths_lower for kw in ("script", "notebook", ".ipynb", "homework", "example", "demo")):
        signals.append("AI development scripts and experiments.")
    elif any(kw in all_paths_lower for kw in ("backend", "server", "app.py", "main.py")):
        signals.append("Backend application with AI integration.")
    elif any(kw in all_paths_lower for kw in ("chat", "bot", "assistant")):
        signals.append("Conversational AI / chatbot application.")
    elif any(kw in all_paths_lower for kw in ("agent", "crew", "autogen")):
        signals.append("AI agent system.")
    elif not signals:
        if "transformers" in all_imports:
            signals.append("ML model inference using Transformers library.")
        elif "chat" in all_imports or "completion" in all_imports:
            signals.append("AI-powered text generation / chat completion.")

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


# ── Tech stack extraction ─────────────────────────────────────────────────


def _extract_tech_stack(asset: AIAsset) -> list[str]:
    """Extract all technologies used in this solution."""
    stack = set()

    # From providers found in findings
    providers_seen = set()
    for f in asset.raw_findings:
        if f.provider:
            providers_seen.add(f.provider)
    for p in providers_seen:
        profile = get_provider(p)
        if profile.name != "unknown":
            stack.add(profile.display_name)

    # From dependencies
    for dep in asset.dependencies:
        pkg = dep.split(">=")[0].split("==")[0].split("@")[0].strip().lower()
        if pkg in ("flask", "fastapi", "django", "express"):
            stack.add(pkg.capitalize())
        elif pkg in ("streamlit",):
            stack.add("Streamlit")
        elif pkg in ("gradio",):
            stack.add("Gradio")
        elif "psycopg" in pkg or "sqlalchemy" in pkg:
            stack.add("PostgreSQL")
        elif "pymongo" in pkg:
            stack.add("MongoDB")
        elif "redis" in pkg:
            stack.add("Redis")

    # From code context — detect frameworks from imports and calls
    all_text = ""
    for ctx in asset.code_contexts:
        for func in ctx.functions:
            all_text += " " + func.get("body_preview", "")
        all_text += " " + " ".join(ctx.env_vars)

    if "flask" in all_text.lower():
        stack.add("Flask")
    if "fastapi" in all_text.lower():
        stack.add("FastAPI")
    if "streamlit" in all_text.lower():
        stack.add("Streamlit")
    if "sqlite" in all_text.lower():
        stack.add("SQLite")
    if "postgres" in all_text.lower() or "psycopg" in all_text.lower():
        stack.add("PostgreSQL")
    if "tavily" in all_text.lower():
        stack.add("Tavily Search")
    if "playwright" in all_text.lower():
        stack.add("Playwright")
    if "puppeteer" in all_text.lower():
        stack.add("Puppeteer")
    if "docker" in all_text.lower():
        stack.add("Docker")
    if "mcp" in all_text.lower():
        stack.add("MCP")

    return sorted(stack)


# ── Data involved extraction ──────────────────────────────────────────────


def _extract_data_involved(asset: AIAsset) -> list[str]:
    """Extract what types of data this solution processes."""
    data_types = set()

    for ctx in asset.code_contexts:
        all_text = " ".join(
            [func.get("body_preview", "") for func in ctx.functions]
            + ctx.prompts
            + [src.get("detail", "") for src in ctx.data_sources]
            + [sink.get("detail", "") for sink in ctx.data_sinks]
            + ctx.raw_snippets
        ).lower()

        # Detect data types from code content
        if any(kw in all_text for kw in ("audio", "wav", "mp3", "transcri", "whisper", "speech")):
            data_types.add("Audio / Speech")
        if any(kw in all_text for kw in ("image", "photo", "png", "jpg", "vision", "multimodal")):
            data_types.add("Images")
        if any(kw in all_text for kw in ("video", "mp4", "frame")):
            data_types.add("Video")
        if any(kw in all_text for kw in ("pdf", "document", "docx", "txt file")):
            data_types.add("Documents")
        if any(kw in all_text for kw in ("csv", "excel", "spreadsheet", "dataframe")):
            data_types.add("Spreadsheet / CSV data")
        if any(kw in all_text for kw in ("stock", "price", "ticker", "dividend", "financial", "investment")):
            data_types.add("Financial data")
        if any(kw in all_text for kw in ("email", "smtp", "inbox")):
            data_types.add("Email")
        if any(kw in all_text for kw in ("customer", "user_name", "personal", "pii", "phone", "address")):
            data_types.add("Personal data / PII")
        if any(kw in all_text for kw in ("password", "credential", "secret", "api_key")):
            data_types.add("Credentials / Secrets")
        if any(kw in all_text for kw in ("select ", "insert ", "database", "table", "cursor", "sql")):
            data_types.add("Database records")
        if any(kw in all_text for kw in ("embedding", "vector", "similarity")):
            data_types.add("Vector embeddings")
        if any(kw in all_text for kw in ("chat", "message", "conversation", "user_input")):
            data_types.add("Chat / Conversation")
        if any(kw in all_text for kw in ("search", "query", "web", "scrape", "crawl", "url")):
            data_types.add("Web content")
        if any(kw in all_text for kw in ("code", "source", "script", "function", "class")):
            data_types.add("Source code")

    return sorted(data_types)


# ── Category classification ───────────────────────────────────────────────


def _classify_category(asset: AIAsset) -> str:
    """Classify the solution into a category for grouping."""
    all_text = asset.name.lower() + " " + asset.file_path.lower()

    # Add code context signals
    for ctx in asset.code_contexts:
        for func in ctx.functions:
            all_text += " " + func.get("name", "").lower()
            all_text += " " + func.get("body_preview", "").lower()
        all_text += " " + " ".join(p.lower() for p in ctx.prompts)

    if any(kw in all_text for kw in ("finetun", "fine_tun", "train", "dataset", "tokenizer")):
        return "Fine-tuning & Training"
    if any(kw in all_text for kw in ("agent", "react", "tool_call", "create_agent", "swarm", "crew")):
        return "AI Agents"
    if any(kw in all_text for kw in ("rag", "retriev", "embed", "vector", "chunk", "similarity_search")):
        return "RAG & Search"
    if any(kw in all_text for kw in ("chat", "conversation", "message", "assistant")):
        return "Chatbot & Conversation"
    if any(kw in all_text for kw in ("web", "playwright", "puppeteer", "browser", "scrape")):
        return "Web Automation"
    if any(kw in all_text for kw in ("workflow", "pipeline", "chain", "graph", "node")):
        return "Workflows & Pipelines"
    if any(kw in all_text for kw in ("model", "inference", "predict", "classif")):
        return "Model & Inference"
    if any(kw in all_text for kw in ("mcp", "server", "client")):
        return "MCP & Integration"

    return "Other AI Solutions"


# ── Solution display name ─────────────────────────────────────────────────


def _derive_solution_display_name(asset: AIAsset, category: str) -> str:
    """Derive a name that describes WHAT the solution does, not what framework it uses.

    Priority:
    1. README title (if meaningful)
    2. Specific prompt role ("Joe Rogan voice clone", "Fleurdin florist assistant")
    3. Purpose from key functions ("Stock Price & Dividend Checker")
    4. Category + directory context ("Image Analysis — Multimodal")
    5. Directory-based name (last resort)
    """
    import re

    # ── 1. README title ──
    for ctx in asset.code_contexts:
        if ctx.language == "markdown" and ctx.raw_snippets:
            for line in ctx.raw_snippets[0].split("\n"):
                line = line.strip()
                if line.startswith("# "):
                    title = line[2:].strip()
                    if _is_meaningful_name(title):
                        return title[:80]

    # ── 2. Specific prompt role ──
    for ctx in asset.code_contexts:
        for prompt in ctx.prompts:
            match = re.match(
                r"(?:You are|You're|I am|Act as)\s+(.+?)(?:\.|,|!|\n)",
                prompt, re.IGNORECASE,
            )
            if match:
                role = match.group(1).strip()
                if _is_meaningful_name(role) and not _is_generic_role(role):
                    # Capitalize first letter
                    return role[0].upper() + role[1:] if role else role

    # ── 3. Purpose from key functions ──
    key_funcs = []
    for ctx in asset.code_contexts:
        for func in ctx.functions:
            name = func.get("name", "")
            if name and name not in (
                "main", "__init__", "setup", "run", "config", "chat",
                "send_message", "append", "invoke",
            ) and not name.startswith("_"):
                key_funcs.append(name)

    if key_funcs:
        # Turn function names into a readable title
        readable = _funcs_to_title(key_funcs[:4])
        if readable:
            return readable

    # ── 4. Category + most specific directory ──
    dir_leaf = _get_leaf_dir_name(asset)
    if dir_leaf and dir_leaf != asset.name:
        return f"{category} — {dir_leaf}"

    # ── 5. Fallback to directory-based name ──
    return asset.name


def _funcs_to_title(func_names: list[str]) -> str:
    """Convert function names to a readable solution title.

    ['get_stock_price', 'get_dividend_date'] → 'Stock Price & Dividend Tool'
    ['encode_image'] → 'Image Encoder'
    ['chat', 'load_history'] → ''  (too generic)
    """
    if not func_names:
        return ""

    # Convert function names to readable phrases
    phrases = []
    noise = {"get", "set", "create", "make", "build", "run", "do", "is", "has",
             "from", "to", "with", "and", "the", "a", "an", "in", "on", "for",
             "completion", "messages", "append", "chat", "send", "message",
             "load", "save", "init", "start", "stop", "update", "delete",
             "process", "handle", "parse", "format", "convert", "check"}

    for name in func_names:
        parts = [p for p in name.replace("_", " ").split() if p.lower() not in noise]
        if parts:
            phrase = " ".join(p.capitalize() for p in parts)
            if len(phrase) > 3 and phrase not in phrases:
                phrases.append(phrase)

    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) <= 3:
        return " & ".join(phrases)
    return ""  # too many, not a clear title


def _get_leaf_dir_name(asset: AIAsset) -> str:
    """Get the most specific (leaf) directory name, cleaned up."""
    if not asset.file_path:
        return ""
    from aiscout.scanners.git_scanner import _clean_dir_name
    first_file = asset.file_path.split(", ")[0]
    parts = first_file.split("/")
    if len(parts) <= 1:
        return ""
    # Take the last directory (most specific)
    leaf = parts[-2]
    return _clean_dir_name(leaf)


# Generic names that should NOT be used as solution names
_GENERIC_NAMES = {
    "readme", "tbd", "finish", "todo", "wip", "test", "main", "app",
    "index", "script", "example", "demo", "description", "run",
    "run (locally)", "untitled",
}

# Generic AI roles from prompts that don't describe a specific solution
_GENERIC_ROLES = {
    "a helpful assistant", "a helpful ai assistant", "an ai assistant",
    "an assistant", "a coder", "a programmer", "a developer",
    "a software developer", "a software engineer", "a coding assistant",
    "a helpful coding assistant", "an expert", "a helpful expert",
}


def _is_meaningful_name(name: str) -> bool:
    """Check if a name is specific enough to use as solution name."""
    cleaned = name.strip().lower()
    return (
        len(cleaned) > 3
        and cleaned not in _GENERIC_NAMES
        and not cleaned.startswith("untitled")
    )


def _is_generic_role(role: str) -> bool:
    """Check if a prompt role is too generic to be useful as a name."""
    cleaned = role.strip().lower()
    # Exact match
    if cleaned in _GENERIC_ROLES:
        return True
    # Starts with generic pattern
    if any(cleaned.startswith(g) for g in (
        "a helpful", "an ai", "a coding", "a software", "a professional",
    )):
        return True
    # Too short to be specific
    if len(cleaned) < 10:
        return True
    return False


# ── Risk score calculation ────────────────────────────────────────────────


def _calculate_risk_score(reasons: list[RiskReason]) -> float:
    """Calculate a weighted risk score from risk reasons."""
    if not reasons:
        return 0.1

    weights = {"critical": 0.4, "warning": 0.2, "info": 0.05}
    score = sum(weights.get(r.severity, 0.0) for r in reasons)

    return min(1.0, max(0.0, round(score, 2)))
