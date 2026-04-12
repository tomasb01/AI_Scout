"""Asset enrichment — generates summaries, risk reasoning, and recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field

import re

from aiscout.knowledge.dependency_advisories import (
    Advisory,
    find_advisories,
)
from aiscout.knowledge.providers import ProviderProfile, get_provider
from aiscout.models import AIAsset, Finding, FindingType, TaskType


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

    # Sprint 2 — attach task_types + tags BEFORE the rest so downstream
    # builders (summary, risk reasons, recommendations) can reference them.
    asset.task_types = _detect_task_types(asset)
    asset.tags = _derive_tags(asset)

    summary = _build_summary(asset, provider)
    risk_reasons = _build_risk_reasons(asset, provider)
    recommendations = _build_recommendations(asset, provider, risk_reasons)
    tech_stack = _extract_tech_stack(asset)
    data_involved = _extract_data_involved(asset)

    # Enrich data_involved from LLM classification if available
    if asset.data_classification and asset.data_classification.categories:
        for cat in asset.data_classification.categories:
            label = cat.value.replace("_", " ").title()
            if label not in data_involved:
                data_involved.append(label)
        data_involved.sort()

    # Use LLM recommendations if available
    if asset.data_classification and asset.data_classification.recommendations:
        recommendations = asset.data_classification.recommendations

    category = _classify_category(asset)
    solution_name = _derive_solution_display_name(asset, category)

    # Risk score: use LLM score if available, otherwise rule-based
    if asset.data_classification and asset.data_classification.risk_score > 0:
        asset.risk_score = max(asset.risk_score, asset.data_classification.risk_score)
    else:
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


# ── Sprint 2: task_type + tag derivation ─────────────────────────────────

_TRAINING_KEYWORDS = (
    "trainer(", "trainingarguments", "sfttrainer", "dpotrainer", "pportrainer",
    "model.fit(", "model.train()", ".backward()", "optimizer.step(",
    "torch.optim", "adamw", "lr_scheduler", "loraconfig", "get_peft_model",
    "accelerate.accelerator", "deepspeed", "fsdp", "huggingface_hub.upload",
)
_FINE_TUNE_KEYWORDS = (
    "fine_tune", "fine-tune", "finetune", "lora", "qlora", "peft",
    "client.fine_tuning", "fine_tuning.jobs", "sfttrainer",
)
_EVAL_KEYWORDS = (
    "model.eval()", "evaluate.load", "sklearn.metrics", "compute_metrics",
    "classification_report", "confusion_matrix", "rouge_score", "bleu_score",
    "lm_eval", "trainer.evaluate",
)

_TAG_RULES = (
    # (tag, list-of-lowercase-substrings that trigger it)
    ("chatbot", (
        "system_prompt", "system prompt", "assistant",
        "messages.create", "chat.completions", "chatmessage",
        "conversation", "chat history", "user_message",
    )),
    ("rag", (
        "chromadb", "pinecone", "qdrant", "weaviate", "faiss", "milvus",
        "vector_store", "vectorstore", "embedding", "retrieval",
        "similarity_search", "as_retriever", "langchain.retrievers",
    )),
    ("agent", (
        "crewai", "autogen", "langgraph", "agentexecutor", "tool_call",
        "tools=[", "function_call", "run_agent", "langchain.agents",
    )),
    ("training", _TRAINING_KEYWORDS),
    ("fine_tuning", _FINE_TUNE_KEYWORDS),
    ("evaluation", _EVAL_KEYWORDS),
    ("transcription", (
        "whisper", "transcribe", "audio_file", "speech_to_text",
    )),
    ("image_generation", (
        "dall-e", "dalle", "stablediffusion", "stable_diffusion",
        "images.generate", "text2image",
    )),
    ("local_model", (
        "ollama", "llama.cpp", "llama_cpp", "vllm", "localai",
    )),
    ("mcp", (
        "mcp.server", "modelcontextprotocol", "mcpserver",
    )),
)


def _collect_code_text(asset: AIAsset) -> str:
    """Concatenate all searchable code-derived text for keyword matching."""
    parts: list[str] = [asset.name, asset.file_path]
    for ctx in asset.code_contexts:
        parts.append(ctx.file_path)
        for func in ctx.functions:
            parts.append(func.get("name", ""))
            parts.append(func.get("body_preview", ""))
            parts.append(func.get("docstring", ""))
            parts.extend(func.get("decorators", []))
        for cls in ctx.classes:
            parts.append(cls.get("name", ""))
        for call in ctx.api_calls:
            parts.append(call.get("target", ""))
            parts.append(call.get("args_preview", ""))
        parts.extend(ctx.prompts)
        parts.extend(ctx.model_names)
        parts.extend(ctx.env_vars)
        parts.extend(ctx.raw_snippets)
    for f in asset.raw_findings:
        parts.append(f.content)
        parts.append(f.file_path)
    return " ".join(p for p in parts if p).lower()


_FINE_TUNE_DEP_MARKERS = (
    "peft", "trl", "bitsandbytes", "auto-gptq", "unsloth",
)
_TRAINING_DEP_MARKERS = (
    "accelerate", "deepspeed", "torch", "tensorflow", "keras",
)
_FINE_TUNE_API_MARKERS = (
    "fine_tuning.jobs", "client.fine_tuning", "finetune", "create_fine_tune",
)


def _detect_task_types(asset: AIAsset) -> list[TaskType]:
    """Infer what the asset *does* with the model.

    Training and fine-tuning are the highest-risk task types because they
    typically ingest large volumes of real customer data. Evaluation is
    lower risk but still worth surfacing because it indicates an active
    ML lifecycle rather than a shipped feature.

    Signal sources (strongest → weakest):
      1. Explicit training classes/calls in code body (Trainer, .train(),
         optimizer.step)
      2. Training-specific dependencies (peft, trl, bitsandbytes)
      3. Fine-tuning API calls (OpenAI fine_tuning.jobs.create, etc.)
      4. File path contains "train" / "finetune"
    """
    text = _collect_code_text(asset)
    deps_lower = " ".join(d.lower() for d in asset.dependencies)
    types: list[TaskType] = []

    # API calls — stable signal even when code body preview is thin
    api_texts: list[str] = []
    for ctx in asset.code_contexts:
        for call in ctx.api_calls:
            api_texts.append(call.get("target", "").lower())
            api_texts.append(call.get("args_preview", "").lower())
    api_text = " ".join(api_texts)

    fine_tune = (
        any(k in text for k in _FINE_TUNE_KEYWORDS)
        or any(m in deps_lower for m in _FINE_TUNE_DEP_MARKERS)
        or any(m in api_text for m in _FINE_TUNE_API_MARKERS)
    )
    if fine_tune:
        types.append(TaskType.FINE_TUNING)

    training = (
        any(k in text for k in _TRAINING_KEYWORDS)
        or ("transformers" in deps_lower and any(
            m in deps_lower for m in _TRAINING_DEP_MARKERS
        ))
    )
    if training and TaskType.FINE_TUNING not in types:
        types.append(TaskType.TRAINING)

    if any(k in text for k in _EVAL_KEYWORDS):
        types.append(TaskType.EVALUATION)

    # Heuristic: leaf directory literally says "train" or "finetune".
    # We *only* look at the last two path components of the FIRST file so
    # grand-parent noise (e.g. a top-level "4-openai-and-finetuning/"
    # dir that also holds plain inference examples) doesn't falsely
    # promote every descendant to training.
    first_file = (asset.file_path or "").split(", ")[0]
    leaf_parts = first_file.lower().split("/")[-3:-1]  # parent + grandparent
    leaf_context = "/".join(leaf_parts)
    train_markers = ("train", "finetun", "fine_tun", "fine-tun", "sft", "rlhf")
    if any(m in leaf_context for m in train_markers):
        if TaskType.TRAINING not in types and TaskType.FINE_TUNING not in types:
            if any(m in leaf_context for m in ("lora", "peft", "qlora", "sft")):
                types.append(TaskType.FINE_TUNING)
            else:
                types.append(TaskType.TRAINING)

    if not types:
        types.append(TaskType.INFERENCE)
    return sorted(set(types), key=lambda t: t.value)


_LOCAL_MODEL_PROVIDERS = {
    "ollama", "huggingface", "llamacpp", "vllm", "localai",
    "pytorch", "onnx", "coreml", "tflite",
}
_VECTOR_PROVIDERS = {
    "chromadb", "pinecone", "qdrant", "weaviate", "faiss", "milvus",
}


def _derive_tags(asset: AIAsset) -> list[str]:
    """Derive human-facing tags for the asset card.

    Tags describe the *kind* of AI workload (chatbot, RAG, agent,
    training, …) — orthogonal to the provider. They feed the report
    filter chips and Sprint-2 sub-component view.

    Signal cascade:
      1. Keyword rules over all collected code text (chatbot, rag, …)
      2. Finding-type promotion (CONTAINER_DETECTED, LOCAL_MODEL_DETECTED)
      3. Fallback from providers / deps / path when the rules return
         nothing — ensures every asset gets at least one meaningful tag
         even when code context is thin (e.g. notebook with no docstrings).
    """
    text = _collect_code_text(asset)
    tags: set[str] = set()
    for tag, keywords in _TAG_RULES:
        if any(k in text for k in keywords):
            tags.add(tag)

    # Promote tags from findings themselves (containers + providers)
    finding_providers: set[str] = set()
    for f in asset.raw_findings:
        if f.type == FindingType.CONTAINER_DETECTED:
            tags.add("local_model")
        if f.type == FindingType.LOCAL_MODEL_DETECTED:
            tags.add("local_model")
        if f.provider == "mcp":
            tags.add("mcp")
        if f.provider:
            finding_providers.add(f.provider)

    deps_lower = " ".join(d.lower() for d in asset.dependencies)
    path_lower = asset.file_path.lower()

    # Fallbacks — fire only when keyword detection didn't cover the
    # asset, so we don't over-tag things that already have strong signals.
    if not any(t in tags for t in ("rag", "agent", "chatbot", "training", "fine_tuning")):
        if finding_providers & _LOCAL_MODEL_PROVIDERS:
            tags.add("local_model")
        if finding_providers & _VECTOR_PROVIDERS:
            tags.add("rag")

    if "agent" not in tags and any(kw in path_lower for kw in (
        "/agent", "agent/", "agents/", "crew", "autogen", "langgraph",
    )):
        tags.add("agent")

    if "chatbot" not in tags and any(kw in path_lower for kw in (
        "chatbot", "chat_bot", "/chat", "conversation",
    )):
        tags.add("chatbot")

    if "rag" not in tags and any(kw in path_lower for kw in (
        "/rag", "rag_", "_rag", "retriev", "embedding",
    )):
        tags.add("rag")

    if not tags:
        # Absolute last resort: tell the reader this is inference code
        if "transformers" in deps_lower or finding_providers & {"huggingface"}:
            tags.add("local_model")

    return sorted(tags)


# ── Summary builder ───────────────────────────────────────────────────────


def _build_summary(asset: AIAsset, provider: ProviderProfile | None) -> str:
    """Generate a summary focused on WHAT the solution does.

    Priority:
      0. LLM classification (highest — most accurate when available)
      1. Synthesised purpose from strong signals (task_type + tags +
         model_names + provider) — deterministic, beats noisy README
         extraction for fine-tuning scripts, local inference notebooks,
         MCP servers, …
      2. Code-context inference (prompts, docstrings, README, functions)
      3. Directory context + provider (last resort)
    """
    if asset.data_classification and asset.data_classification.details:
        return asset.data_classification.details

    synth = _synthesize_purpose(asset, provider)
    if synth:
        # If code context *also* gives us a rich purpose (docstrings,
        # prompts), append it after the synth one-liner — the synth
        # sentence answers "what is this", the inferred part adds "how".
        extra = _infer_purpose(asset)
        if extra and _is_descriptive(extra):
            return f"{synth} {extra}"
        return synth

    purpose = _infer_purpose(asset)
    if purpose:
        return purpose

    dir_context = _get_dir_context(asset)
    provider_name = provider.display_name if provider else "unknown"
    if dir_context:
        return f"{dir_context} AI solution using {provider_name}."
    return f"AI solution using {provider_name}."


_SYNTH_MODELS = (
    ("mistral", "Mistral"), ("llama-3", "Llama 3"), ("llama3", "Llama 3"),
    ("llama-2", "Llama 2"), ("llama2", "Llama 2"), ("llama", "Llama"),
    ("qwen", "Qwen"), ("gemma", "Gemma"), ("phi-3", "Phi-3"), ("phi3", "Phi-3"),
    ("falcon", "Falcon"), ("mixtral", "Mixtral"), ("deepseek", "DeepSeek"),
    ("gpt-4o", "GPT-4o"), ("gpt-4", "GPT-4"), ("gpt-3.5", "GPT-3.5"),
    ("claude-3.5", "Claude 3.5"), ("claude-3", "Claude 3"), ("claude", "Claude"),
    ("gemini", "Gemini"), ("whisper", "Whisper"), ("bert", "BERT"),
    ("stable-diffusion", "Stable Diffusion"), ("dall-e", "DALL-E"),
)


def _guess_model_family(asset: AIAsset) -> str:
    """Find the most-likely model family name referenced by the asset."""
    haystacks = [asset.file_path.lower(), asset.name.lower()]
    for ctx in asset.code_contexts:
        haystacks.extend(m.lower() for m in ctx.model_names)
        for func in ctx.functions:
            haystacks.append(func.get("body_preview", "").lower())
    combined = " ".join(haystacks)
    for needle, display in _SYNTH_MODELS:
        if needle in combined:
            return display
    return ""


def _synthesize_purpose(
    asset: AIAsset, provider: ProviderProfile | None
) -> str:
    """Build a deterministic one-sentence purpose from strong signals.

    Only fires when the asset has clear task-type + tag evidence. The
    output is opinionated: it answers "what does this AI solution do?"
    in a form an auditor can scan quickly, without depending on
    whatever noise a README may contain.

    Returns an empty string if signals are too weak — the caller then
    falls through to code-context / README inference.
    """
    task_types = set(asset.task_types)
    tags = set(asset.tags)
    provider_display = provider.display_name if provider else ""
    model = _guess_model_family(asset)

    def with_model(prefix: str) -> str:
        return f"{prefix} {model}." if model else f"{prefix} a large language model."

    # ── Training / fine-tuning ──
    if TaskType.FINE_TUNING in task_types:
        base = with_model("Fine-tuning")
        if "peft" in " ".join(asset.dependencies).lower() or "lora" in asset.file_path.lower():
            base = base.rstrip(".") + " via LoRA/PEFT."
        return base
    if TaskType.TRAINING in task_types:
        return with_model("Model training pipeline for")

    # ── Agent / tool-using LLM ──
    if "agent" in tags:
        if "mcp" in tags:
            return "AI agent that calls tools via Model Context Protocol (MCP) servers."
        if "local_model" in tags:
            return f"AI agent running on a local model runtime ({provider_display or 'local'})."
        if provider_display:
            return f"AI agent backed by {provider_display}."
        return "AI agent with tool-using capabilities."

    # ── RAG ──
    if "rag" in tags:
        backend = provider_display or model or "an LLM"
        return f"Retrieval-augmented generation pipeline backed by {backend}."

    # ── MCP server (no agent client) ──
    if "mcp" in tags:
        if _asset_is_mcp_server(asset):
            return "Model Context Protocol server exposing tools to LLM clients."
        return "LLM application that consumes tools from MCP servers."

    # ── Chatbot ──
    if "chatbot" in tags:
        who = provider_display or model or "an LLM"
        return f"Conversational chatbot powered by {who}."

    # ── Transcription / image generation ──
    if "transcription" in tags:
        return "Speech-to-text / audio transcription pipeline."
    if "image_generation" in tags:
        return "Image generation pipeline."

    # ── Local-model inference (Model & Inference category) ──
    if "local_model" in tags and TaskType.INFERENCE in task_types:
        return with_model("Local inference on")

    return ""


def _is_descriptive(text: str) -> bool:
    """Reject noisy one-liners that don't describe what the code does."""
    if not text or len(text.strip()) < 12:
        return False
    noisy = (
        "run in terminal", "run the", "install ", "pip install",
        "getting started", "usage:", "todo", "fixme",
        "this folder", "this directory", "step 1", "step 2",
    )
    low = text.lower().strip()
    if any(low.startswith(n) for n in noisy):
        return False
    return True


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
        lines = [l.strip() for l in readme_text.split("\n") if l.strip()]
        # Skip markdown headings, code fences, shell commands, imperatives
        # and other non-descriptive boilerplate so the summary describes
        # the project instead of "Run in terminal commands."
        def _is_descriptive_line(line: str) -> bool:
            if len(line) < 25:
                return False
            if line.startswith(("#", "```", "> ", "- ", "* ", "|", "![")):
                return False
            low = line.lower()
            imperative_prefixes = (
                "run ", "install ", "pip install", "npm install",
                "cd ", "git clone", "python ", "uv ", "poetry ",
                "note:", "todo", "fixme", "click ", "open ",
                "go to ", "first, ", "then, ", "next, ",
            )
            if any(low.startswith(p) for p in imperative_prefixes):
                return False
            # Skip pure shell commands with flags
            if re.match(r"^[\w./-]+\s+(?:-{1,2}[\w-]+\s*)+", line):
                return False
            return True

        descriptive = [l for l in lines if _is_descriptive_line(l)]
        if descriptive:
            chosen = descriptive[0][:200]
            if not chosen.endswith("."):
                chosen += "."
            parts.append(chosen)

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
            flow_parts.append(f"Reads from: {', '.join(sorted(source_types)[:3])}")
        if sink_types:
            flow_parts.append(f"Outputs to: {', '.join(sorted(sink_types)[:3])}")
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
        targets = {call.get("target", "")[:50] for call in all_api_calls[:3]}
        if targets:
            parts.append(f"AI calls: {', '.join(sorted(targets))}.")

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


_PII_KEYWORDS = (
    "personal data", "pii", "email", "phone", "address",
    "customer name", "customer_name", "user_name", "full name",
    "date of birth", "ssn", "passport", "national id",
    "medical", "patient", "health record", "diagnosis",
    "financial", "credit card", "bank account", "iban", "salary",
)


def _asset_processes_pii(asset: AIAsset) -> bool:
    """Return True when the asset likely handles personal / sensitive data.

    Signals are combined: explicit ``data_classification`` from an LLM
    run, data_involved tags produced by ``_extract_data_involved``, and
    keyword matches against code-context text. We prefer *false negative*
    over *false positive* — a PII tag escalates risk severity, so we
    only assert it when the evidence is reasonably clear.
    """
    if asset.data_classification:
        for cat in asset.data_classification.categories:
            if cat.value in ("pii", "financial"):
                return True
    text = _collect_code_text(asset)
    return any(kw in text for kw in _PII_KEYWORDS)


def _asset_is_mcp_server(asset: AIAsset) -> bool:
    """Distinguish MCP *server* (exposes tools) from MCP *client* (uses servers).

    Servers import ``mcp.server`` or declare their entrypoint as an MCP
    server via ``fastmcp`` / ``@server.tool`` decorators. Clients import
    ``langchain_mcp_adapters`` / use ``MultiServerMCPClient`` etc.
    """
    text = _collect_code_text(asset)
    server_markers = (
        "mcp.server", "fastmcp", "server.tool", "@server.", "mcp_server",
        "stdio_server", "sse_server",
    )
    client_markers = (
        "multiservermcpclient", "langchain_mcp_adapters",
        "streamablehttp_client", "stdio_client", "sse_client",
        "mcp.client",
    )
    has_server = any(m in text for m in server_markers)
    has_client = any(m in text for m in client_markers)
    if has_server and not has_client:
        return True
    # If only a config file was found (mcp.json with server list), the
    # asset is a client that *configures* servers.
    if any(f.type == FindingType.CONFIG_DETECTED and f.provider == "mcp"
           for f in asset.raw_findings) and not has_server:
        return False
    return has_server


def _build_risk_reasons(
    asset: AIAsset, provider: ProviderProfile | None
) -> list[RiskReason]:
    """Generate specific reasons for the risk classification.

    Sprint 3 design — risk is *action + context*, not *mere existence*:

    * **critical** = concrete incident: secret committed to source, PII
      flowing to free-tier / untested provider, known-vulnerable dep with
      public exploit. Requires same-day remediation.
    * **warning** = requires review: training/fine-tuning pipelines,
      MCP server exposing tools, deprecated deps, PII + external API with
      unclear tier.
    * **info** = context so the reader understands where data flows, but
      not in itself something to fix (provider residency, framework,
      local runtime, external API to known-safe provider).

    The previous iteration emitted a warning for *every* LLM API call
    ("Data sent to external AI API"), which flooded the report — every
    OpenAI asset got 2-3 reflexive warnings. Now a plain inference asset
    with a known-safe provider surfaces ONE info line with the facts; a
    warning only appears when a risk-amplifying signal (PII flow, free
    tier, training, …) is also present.
    """
    reasons: list[RiskReason] = []

    key_findings = [
        f for f in asset.raw_findings if f.type == FindingType.API_KEY_DETECTED
    ]
    processes_pii = _asset_processes_pii(asset)
    is_free_tier_risk = (
        provider is not None and "may be used" in provider.training_policy.lower()
    )

    # ── CRITICAL — immediate action required ────────────────────────────

    if key_findings:
        files = sorted({f.file_path for f in key_findings})
        reasons.append(RiskReason(
            severity="critical",
            title="Hardcoded API key in source code",
            detail=(
                f"Found {len(key_findings)} API key{'s' if len(key_findings) > 1 else ''} "
                f"directly in code ({', '.join(files)}). Anyone with repository "
                "access can extract and misuse these keys. Rotate immediately and "
                "move to an environment variable or secret manager."
            ),
        ))

    if (
        processes_pii
        and provider is not None
        and provider.category == "llm_api"
        and is_free_tier_risk
    ):
        reasons.append(RiskReason(
            severity="critical",
            title="Personal data flows to provider with training-on-data risk",
            detail=(
                f"This asset processes personal / PII data and sends it to "
                f"{provider.display_name}, whose free tier may use submitted "
                "data for model training. Confirm the paid API / enterprise plan "
                "is in use and that a DPA is signed, or remove PII from prompts."
            ),
        ))

    # ── WARNING — requires review ───────────────────────────────────────

    if (
        processes_pii
        and provider is not None
        and provider.category in ("llm_api", "embedding_db")
        and not any(r.severity == "critical" for r in reasons)
    ):
        reasons.append(RiskReason(
            severity="warning",
            title="Personal data sent to external provider",
            detail=(
                f"Personal / PII data is processed by this asset and flows to "
                f"{provider.display_name}. Verify the relationship has a DPA, "
                "that data residency matches your compliance requirements, and "
                "that operators of the service have appropriate access controls."
            ),
        ))

    if (
        provider is not None
        and provider.category == "embedding_db"
        and "self-hosted" not in str(provider.data_residency).lower()
        and "local" not in str(provider.data_residency).lower()
    ):
        reasons.append(RiskReason(
            severity="warning",
            title="Embeddings stored in external service",
            detail=(
                f"Embeddings sent to {provider.display_name} may encode sensitive "
                "information from the source text. Embedding inversion attacks "
                "can recover partial content — treat embedding stores like any "
                "other data store holding the underlying source data."
            ),
        ))

    # ── INFO — provider context ─────────────────────────────────────────

    if provider is not None and provider.category == "llm_api":
        residency = ", ".join(provider.data_residency) or "unknown"
        reasons.append(RiskReason(
            severity="info",
            title=f"External AI API: {provider.display_name}",
            detail=(
                f"Data is processed by {provider.display_name} ({provider.vendor}). "
                f"Residency: {residency}. Training policy: {provider.training_policy}"
            ),
        ))

    # ── Sprint 2: workload-kind risks ──

    if TaskType.FINE_TUNING in asset.task_types or TaskType.TRAINING in asset.task_types:
        kind = (
            "Fine-tuning" if TaskType.FINE_TUNING in asset.task_types else "Training"
        )
        reasons.append(RiskReason(
            severity="warning",
            title=f"{kind} pipeline ingests training data",
            detail=(
                f"This asset appears to {kind.lower()} a model. Training pipelines "
                "typically consume real production data (logs, customer messages, "
                "documents), so they carry higher privacy, IP-leakage and "
                "compute-cost risk than plain inference. Verify data provenance, "
                "access controls on the training set, and that resulting model "
                "weights are stored in a governed artifact repository."
            ),
        ))

    mcp_findings = [
        f for f in asset.raw_findings if f.provider == "mcp"
    ]
    if mcp_findings:
        # Distinguish MCP *server* (this asset exposes tools) from MCP
        # *client* (this asset just calls configured servers). Servers
        # are higher risk because they bolt new capabilities onto the
        # model; clients inherit the risk of whatever they connect to.
        is_server = _asset_is_mcp_server(asset)
        server_names = sorted({
            f.content.replace("mcp server: ", "")
            for f in mcp_findings
            if f.content.startswith("mcp server:")
        })
        if is_server:
            reasons.append(RiskReason(
                severity="warning",
                title="MCP server exposes tools to LLM clients",
                detail=(
                    "This asset IS an MCP server: it exposes local functions, "
                    "filesystem access or remote API calls as LLM-callable "
                    "tools. Every tool invocation is a potential data-exfil "
                    "vector — audit the tool surface, limit filesystem and "
                    "network scope, and log every call."
                ),
            ))
        else:
            detail = (
                "This asset uses Model Context Protocol to call one or more "
                "external MCP servers. The data sent to each server depends "
                "on the server's implementation — review each server before "
                "granting it access to production prompts."
            )
            if server_names:
                detail += f" Configured servers: {', '.join(server_names)}."
            reasons.append(RiskReason(
                severity="info",
                title="MCP client — consumes external tool servers",
                detail=detail,
            ))

    container_findings = [
        f for f in asset.raw_findings
        if f.type == FindingType.CONTAINER_DETECTED
    ]
    if container_findings:
        images = sorted({f.content for f in container_findings})
        reasons.append(RiskReason(
            severity="info",
            title="Self-hosted AI runtime detected in container manifest",
            detail=(
                "Docker/Compose files declare an AI runtime or vector store: "
                f"{', '.join(images)}. Self-hosting keeps data on-prem but "
                "requires you to patch the runtime, monitor resource use, and "
                "secure the management endpoints."
            ),
        ))

    # Sprint 3 — dependency advisories (deprecated / vulnerable AI libs).
    for advisory in find_advisories(list(asset.dependencies)):
        reasons.append(RiskReason(
            severity=advisory.severity,
            title=advisory.title,
            detail=advisory.detail,
        ))

    local_model_findings = [
        f for f in asset.raw_findings
        if f.type == FindingType.LOCAL_MODEL_DETECTED
    ]
    if local_model_findings:
        reasons.append(RiskReason(
            severity="info",
            title="Local model weights committed to repository",
            detail=(
                f"Found {len(local_model_findings)} model weight file(s) "
                "(.gguf/.safetensors/.onnx/…) tracked in source control. Large "
                "binaries bloat the repo and are often better served via a model "
                "registry or Git LFS; verify licensing of the underlying model."
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
    """Extract all technologies used in this solution.

    Sources (in priority order):
    1. Provider profiles from raw_findings
    2. LLM model names from code_contexts (e.g. "gpt-4o", "claude-3-sonnet")
    3. Dependencies (AI packages + infrastructure)
    4. API call targets from code_contexts
    5. Framework/tool detection from code text
    """
    stack = set()

    # ── 1. Providers from findings ──
    providers_seen = set()
    for f in asset.raw_findings:
        if f.provider:
            providers_seen.add(f.provider)
    for p in providers_seen:
        profile = get_provider(p)
        if profile.name != "unknown":
            stack.add(profile.display_name)

    # ── 2. LLM model names from code analysis ──
    for ctx in asset.code_contexts:
        for model_name in ctx.model_names:
            display = _model_name_to_display(model_name)
            if display:
                stack.add(display)

    # ── 3. Dependencies ──
    for dep in asset.dependencies:
        pkg = dep.split(">=")[0].split("==")[0].split("@")[0].strip().lower()
        # Infrastructure / web frameworks
        dep_display = _DEP_TO_TECH.get(pkg)
        if dep_display:
            stack.add(dep_display)
        # AI packages — resolve via provider profile if not already in stack
        elif pkg in _AI_DEP_TO_PROVIDER:
            provider_name = _AI_DEP_TO_PROVIDER[pkg]
            profile = get_provider(provider_name)
            if profile.name != "unknown":
                stack.add(profile.display_name)

    # ── 4. API call targets from code contexts ──
    for ctx in asset.code_contexts:
        for call in ctx.api_calls:
            target = call.get("target", "").lower()
            args = call.get("args_preview", "").lower()
            # Detect specific services from API call patterns
            if "embedding" in target:
                stack.add("Embeddings")
            if "transcri" in target or "whisper" in target:
                stack.add("Speech-to-Text")
            if "image" in target or "dall" in target:
                stack.add("Image Generation")
            # Model names in API call args (e.g. model='gpt-4o' as string in args_preview)
            if "model=" in args or "model_name=" in args:
                import re
                for m in re.finditer(r"model(?:_name)?=['\"]([^'\"]{3,60})['\"]", args):
                    display = _model_name_to_display(m.group(1))
                    if display:
                        stack.add(display)

    # ── 5. Framework/tool detection from code text ──
    all_text = ""
    for ctx in asset.code_contexts:
        for func in ctx.functions:
            all_text += " " + func.get("body_preview", "")
        all_text += " " + " ".join(ctx.env_vars)
        # Include prompts and data sink targets for broader coverage
        all_text += " " + " ".join(ctx.prompts[:3])
        for sink in ctx.data_sinks:
            all_text += " " + sink.get("detail", "")

    text_lower = all_text.lower()
    for keyword, tech_name in _CODE_TEXT_TECH_MAP.items():
        if keyword in text_lower:
            stack.add(tech_name)

    return sorted(_deduplicate_tech_stack(stack))


# ── Model name → display name mapping ────────────────────────────────────

_MODEL_FAMILY_MAP = {
    "gpt-4": "GPT-4",
    "gpt-4o": "GPT-4o",
    "gpt-4-turbo": "GPT-4 Turbo",
    "gpt-3.5": "GPT-3.5",
    "gpt-3": "GPT-3",
    "o1": "OpenAI o1",
    "o3": "OpenAI o3",
    "o4": "OpenAI o4",
    "claude-3": "Claude 3",
    "claude-3.5": "Claude 3.5",
    "claude-3-5": "Claude 3.5",
    "claude-4": "Claude 4",
    "claude-2": "Claude 2",
    "gemini-1.5": "Gemini 1.5",
    "gemini-2": "Gemini 2",
    "gemini-pro": "Gemini Pro",
    "gemini-flash": "Gemini Flash",
    "mistral-large": "Mistral Large",
    "mistral-small": "Mistral Small",
    "mistral-medium": "Mistral Medium",
    "codestral": "Codestral",
    "mixtral": "Mixtral",
    "llama-3": "Llama 3",
    "llama-2": "Llama 2",
    "llama3": "Llama 3",
    "llama2": "Llama 2",
    "meta-llama": "Llama",
    "qwen2": "Qwen 2",
    "qwen-": "Qwen",
    "phi-3": "Phi-3",
    "phi-4": "Phi-4",
    "phi3": "Phi-3",
    "deepseek": "DeepSeek",
    "command-r": "Cohere Command R",
    "command-light": "Cohere Command",
    "gemma": "Gemma",
    "falcon": "Falcon",
    "whisper": "Whisper",
    "dall-e": "DALL-E",
    "dalle": "DALL-E",
    "tts-1": "OpenAI TTS",
    "text-embedding": "OpenAI Embeddings",
    "stable-diffusion": "Stable Diffusion",
    "sdxl": "Stable Diffusion XL",
}


def _model_name_to_display(model_name: str) -> str:
    """Convert a raw model name to a human-readable display name.

    'gpt-4o-2024-08-06' → 'GPT-4o'
    'claude-3-sonnet-20240229' → 'Claude 3'
    'meta-llama/Meta-Llama-3-8B' → 'Llama 3'
    """
    name = model_name.strip().lower()
    # Try longest prefix match first
    best_match = ""
    best_display = ""
    for prefix, display in _MODEL_FAMILY_MAP.items():
        if name.startswith(prefix) and len(prefix) > len(best_match):
            best_match = prefix
            best_display = display
    if best_display:
        return best_display

    # Check if any key appears anywhere in the name (for HuggingFace-style paths)
    for key, display in _MODEL_FAMILY_MAP.items():
        if key in name:
            return display

    return ""


# ── Tech stack deduplication ──────────────────────────────────────────────

# When a specific model is present, suppress the generic provider name.
# "GPT-4o" makes "OpenAI" redundant; "Claude 3" makes "Anthropic (Claude)" redundant.
_MODEL_SUPPRESSES_PROVIDER = {
    "OpenAI": {"GPT-3", "GPT-3.5", "GPT-4", "GPT-4o", "GPT-4 Turbo", "OpenAI o1", "OpenAI o3", "OpenAI o4", "DALL-E", "Whisper", "OpenAI TTS", "OpenAI Embeddings"},
    "Anthropic (Claude)": {"Claude 2", "Claude 3", "Claude 3.5", "Claude 4"},
    "Google AI (Gemini)": {"Gemini 1.5", "Gemini 2", "Gemini Pro", "Gemini Flash"},
    "Mistral AI": {"Mistral Large", "Mistral Small", "Mistral Medium", "Codestral", "Mixtral"},
    "Cohere": {"Cohere Command R", "Cohere Command"},
}

# More specific version suppresses generic: "Llama 3" suppresses "Llama"
# Newer model in same family suppresses older: "GPT-4o" suppresses "GPT-4" and "GPT-3.5"
_SPECIFIC_SUPPRESSES_GENERIC = {
    "Llama": {"Llama 2", "Llama 3"},
    "Llama 2": {"Llama 3"},
    "Qwen": {"Qwen 2"},
    "GPT-3": {"GPT-3.5", "GPT-4", "GPT-4o", "GPT-4 Turbo"},
    "GPT-3.5": {"GPT-4", "GPT-4o", "GPT-4 Turbo"},
    "GPT-4": {"GPT-4o", "GPT-4 Turbo"},
    "Claude 2": {"Claude 3", "Claude 3.5", "Claude 4"},
    "Claude 3": {"Claude 3.5", "Claude 4"},
    "Gemini 1.5": {"Gemini 2"},
}


def _deduplicate_tech_stack(stack: set[str]) -> set[str]:
    """Remove redundant entries from tech stack.

    Rules:
    - If a specific model is present, remove the generic provider
      (e.g. "GPT-4o" present → remove "OpenAI")
    - If a versioned model is present, remove the unversioned
      (e.g. "Llama 3" present → remove "Llama")
    """
    to_remove = set()

    # Provider suppression by specific models
    for provider, models in _MODEL_SUPPRESSES_PROVIDER.items():
        if provider in stack and stack & models:
            to_remove.add(provider)

    # Generic suppression by specific versions
    for generic, specifics in _SPECIFIC_SUPPRESSES_GENERIC.items():
        if generic in stack and stack & specifics:
            to_remove.add(generic)

    return stack - to_remove


# ── Dependency → tech display name ───────────────────────────────────────

_DEP_TO_TECH = {
    "flask": "Flask",
    "fastapi": "FastAPI",
    "django": "Django",
    "express": "Express",
    "streamlit": "Streamlit",
    "gradio": "Gradio",
    "chainlit": "Chainlit",
    "panel": "Panel",
    "psycopg2": "PostgreSQL",
    "psycopg": "PostgreSQL",
    "asyncpg": "PostgreSQL",
    "sqlalchemy": "SQLAlchemy",
    "pymongo": "MongoDB",
    "motor": "MongoDB",
    "redis": "Redis",
    "celery": "Celery",
    "aiohttp": "aiohttp",
    "uvicorn": "Uvicorn",
    "gunicorn": "Gunicorn",
    "pydantic": "Pydantic",
    "scipy": "SciPy",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "unstructured": "Unstructured",
    "pypdf": "PyPDF",
    "pymupdf": "PyMuPDF",
    "beautifulsoup4": "BeautifulSoup",
    "selenium": "Selenium",
    "playwright": "Playwright",
}

_AI_DEP_TO_PROVIDER = {
    "openai": "openai",
    "anthropic": "anthropic",
    "langchain": "langchain",
    "langchain-core": "langchain",
    "langchain-community": "langchain",
    "langchain-openai": "langchain",
    "langchain-anthropic": "langchain",
    "llama-index": "llamaindex",
    "llamaindex": "llamaindex",
    "transformers": "huggingface",
    "huggingface-hub": "huggingface",
    "diffusers": "huggingface",
    "sentence-transformers": "huggingface",
    "mistralai": "mistral",
    "cohere": "cohere",
    "ollama": "ollama",
    "replicate": "replicate",
    "together": "together",
    "groq": "groq",
    "fireworks-ai": "fireworks",
    "chromadb": "chromadb",
    "pinecone-client": "pinecone",
    "pinecone": "pinecone",
    "qdrant-client": "qdrant",
    "weaviate-client": "weaviate",
    "faiss-cpu": "faiss",
    "faiss-gpu": "faiss",
    "guidance": "guidance",
    "dspy-ai": "dspy",
    "instructor": "instructor",
    "outlines": "outlines",
    "crewai": "crewai",
    "autogen": "autogen",
    "semantic-kernel": "semantic-kernel",
    "google-generativeai": "google_ai",
    "google-cloud-aiplatform": "google_ai",
}

# ── Code text → tech detection ───────────────────────────────────────────

_CODE_TEXT_TECH_MAP = {
    "flask": "Flask",
    "fastapi": "FastAPI",
    "streamlit": "Streamlit",
    "gradio": "Gradio",
    "chainlit": "Chainlit",
    "sqlite": "SQLite",
    "postgres": "PostgreSQL",
    "psycopg": "PostgreSQL",
    "mongodb": "MongoDB",
    "pymongo": "MongoDB",
    "redis": "Redis",
    "tavily": "Tavily Search",
    "playwright": "Playwright",
    "puppeteer": "Puppeteer",
    "selenium": "Selenium",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "mcp": "MCP",
    "supabase": "Supabase",
    "firebase": "Firebase",
    "pinecone": "Pinecone",
    "weaviate": "Weaviate",
    "qdrant": "Qdrant",
    "chroma": "ChromaDB",
    "neo4j": "Neo4j",
    "elasticsearch": "Elasticsearch",
}


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
    _readme_noise_titles = {
        "run", "usage", "install", "setup", "getting started", "example",
        "examples", "todo", "readme", "notes", "overview", "quickstart",
        "quick start", "demo", "run (runpod.io)", "run in colab",
    }
    for ctx in asset.code_contexts:
        if ctx.language == "markdown" and ctx.raw_snippets:
            for line in ctx.raw_snippets[0].split("\n"):
                line = line.strip()
                if line.startswith("# "):
                    title = line[2:].strip()
                    if title.lower().strip(":!") in _readme_noise_titles:
                        break  # skip whole README title section
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
    """Calculate a weighted risk score with explicit severity floors.

    Sprint 3 change — instead of summing fractional weights (which let
    5 "info" lines add up to a warning-level score), we use explicit
    floors keyed on severity plus a small additive contribution so
    multiple issues of the same class still stack:

      * any ``critical``  → score ≥ 0.70, +0.10 per extra critical
      * any ``warning``   → score ≥ 0.40, +0.05 per extra warning
      * only ``info``     → score ≤ 0.25, +0.02 per info
      * nothing           → 0.10 (AI code with no audit signals)

    This produces a cleaner bimodal-to-trimodal distribution: plain
    inference assets sit in the 0.10-0.25 range ("OK"), assets that
    need review land in 0.40-0.60 ("Warning"), and concrete incidents
    start at 0.70 ("Critical").
    """
    if not reasons:
        return 0.10

    crit = sum(1 for r in reasons if r.severity == "critical")
    warn = sum(1 for r in reasons if r.severity == "warning")
    info = sum(1 for r in reasons if r.severity == "info")

    if crit:
        score = 0.70 + 0.10 * (crit - 1) + 0.03 * warn
    elif warn:
        score = 0.40 + 0.05 * (warn - 1) + 0.02 * info
    else:
        score = 0.10 + 0.02 * info

    return min(1.0, max(0.0, round(score, 2)))
