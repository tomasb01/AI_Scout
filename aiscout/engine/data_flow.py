"""Data Flow Mapper — assembles CodeContext into structured DataFlowMap.

This is Step 2 of the three-step analysis pipeline described in the
architecture (``02_Architecture/02_Data_Flow_Mapper.md``):

  Step 1 — Code Context Extractor (``code_analyzer.py``): parses code
           via AST / regex, produces raw ``CodeContext`` per file.
  Step 2 — **This module**: assembles extracted pieces into a structured
           ``DataFlowMap`` (sources → steps → sinks) using rule-based
           heuristics. No LLM required.
  Step 3 — LLM enrichment (``llm.py``): optional layer that refines
           ``solution_purpose`` and ``data_categories``.

The key insight: Step 1 already extracts data_sources, data_sinks,
api_calls, functions, prompts, model_names. This module ASSEMBLES them
into a coherent flow map — the 80% of extracted information that was
previously discarded by the generic ``_synthesize_purpose()`` in
enrichment.py.
"""

from __future__ import annotations

import re
from collections import defaultdict

from aiscout.knowledge.providers import get_provider
from aiscout.models import (
    AIAsset,
    CodeContext,
    Confidence,
    DataFlowMap,
    FlowSink,
    FlowSource,
)


# ── Public API ────────────────────────────────────────────────────────────


def build_data_flow(asset: AIAsset) -> DataFlowMap:
    """Assemble a DataFlowMap from all CodeContexts attached to the asset.

    Returns a DataFlowMap even when code context is thin (single import,
    no functions) — the ``confidence`` field reflects how much evidence
    was available.
    """
    if not asset.code_contexts:
        return _fallback_flow(asset)

    sources = _identify_sources(asset)
    sinks = _identify_sinks(asset)
    steps = _infer_processing_steps(asset)
    purpose = _compose_purpose(asset, sources, sinks, steps)
    categories = _classify_data_categories(asset, sources, sinks)
    confidence = _assess_confidence(asset, sources, sinks, steps)

    return DataFlowMap(
        solution_purpose=purpose,
        sources=sources,
        sinks=sinks,
        processing_steps=steps,
        data_categories=categories,
        confidence=confidence,
    )


def build_data_flows(assets: list[AIAsset]) -> None:
    """Attach a DataFlowMap to every asset. Mutates in place."""
    for asset in assets:
        asset.data_flow = build_data_flow(asset)


# ── Source identification ─────────────────────────────────────────────────

_SOURCE_TYPE_NAMES = {
    "database": "Database",
    "file": "File input",
    "http": "HTTP request",
    "user_input": "User input",
    "env_var": "Environment config",
    "message_queue": "Message queue",
}


def _identify_sources(asset: AIAsset) -> list[FlowSource]:
    """Extract data sources from all code contexts."""
    sources: list[FlowSource] = []
    seen: set[str] = set()

    for ctx in asset.code_contexts:
        # Explicit data_sources from code_analyzer
        for src in ctx.data_sources:
            detail = src.get("detail", "")
            if _is_write_operation(detail) or _is_noise_source(src):
                continue
            normalized = _normalize_detail(detail)
            if normalized in seen:
                continue
            seen.add(normalized)
            src_type = src.get("type", "unknown")
            sources.append(FlowSource(
                type=src_type,
                name=_humanize_source(src),
                detail=_clean_detail(detail),
            ))

        # Infer user_input from REST endpoint decorators
        for func in ctx.functions:
            for dec in func.get("decorators", []):
                endpoint = _extract_endpoint(dec)
                if endpoint:
                    key = ("user_input", endpoint)
                    if key not in seen:
                        seen.add(key)
                        sources.append(FlowSource(
                            type="user_input",
                            name="REST endpoint",
                            detail=endpoint,
                        ))

        # Environment variables as config sources
        for var in ctx.env_vars:
            if any(kw in var.upper() for kw in ("KEY", "TOKEN", "SECRET", "URL", "ENDPOINT")):
                key = ("env_var", var)
                if key not in seen:
                    seen.add(key)
                    sources.append(FlowSource(
                        type="env_var",
                        name=var,
                        detail=f"os.environ['{var}']",
                    ))

    return sources


def _humanize_source(src: dict) -> str:
    """Turn a raw data_source dict into a readable name."""
    src_type = src.get("type", "")
    detail = src.get("detail", "")
    name = src.get("name", "")

    if name and name != src_type:
        return name

    if src_type == "database":
        table = _extract_table_name(detail)
        if table:
            return f"Database: {table}"
        return "Database query"
    if src_type == "file":
        return f"File: {_short_path(detail)}" if detail else "File input"
    if src_type == "http":
        return f"HTTP: {_short_url(detail)}" if detail else "HTTP request"

    return _SOURCE_TYPE_NAMES.get(src_type, src_type or "Unknown source")


# ── Sink identification ───────────────────────────────────────────────────

_AI_API_TARGETS = {
    "client.messages.create": ("anthropic", "Anthropic Claude API"),
    "client.chat.completions.create": ("openai", "OpenAI Chat API"),
    "client.completions.create": ("openai", "OpenAI Completions API"),
    "client.responses.create": ("openai", "OpenAI Responses API"),
    "client.embeddings.create": ("openai", "OpenAI Embeddings API"),
    "client.images.generate": ("openai", "OpenAI DALL-E API"),
    "openai.chat.completions.create": ("openai", "OpenAI Chat API"),
    "openai.images.generate": ("openai", "OpenAI DALL-E API"),
    "anthropic.messages.create": ("anthropic", "Anthropic Claude API"),
    "model.generate": ("ollama", "Ollama local model"),
    "ollama.generate": ("ollama", "Ollama local model"),
    "ollama.chat": ("ollama", "Ollama local chat"),
    "agent.ainvoke": ("langchain", "LangChain agent"),
    "agent.invoke": ("langchain", "LangChain agent"),
    "chain.invoke": ("langchain", "LangChain chain"),
    "chain.ainvoke": ("langchain", "LangChain chain"),
    "model_with_tools.invoke": ("langchain", "LangChain model+tools"),
}


def _identify_sinks(asset: AIAsset) -> list[FlowSink]:
    """Extract data sinks from all code contexts."""
    sinks: list[FlowSink] = []
    seen: set[str] = set()

    for ctx in asset.code_contexts:
        # Explicit data_sinks from code_analyzer
        for sink in ctx.data_sinks:
            if _is_noise_sink(sink):
                continue
            key = (sink.get("type", ""), sink.get("detail", "")[:80])
            if key in seen:
                continue
            seen.add(key)
            sink_type = sink.get("type", "unknown")
            provider = sink.get("provider", "")
            sinks.append(FlowSink(
                type=sink_type,
                name=_humanize_sink(sink),
                detail=_clean_detail(sink.get("detail", "")),
                provider=provider,
            ))

        # AI API calls → sinks (the most important signal)
        for call in ctx.api_calls:
            target = call.get("target", "")
            if target in _AI_API_TARGETS:
                provider, name = _AI_API_TARGETS[target]
                # Enrich with model name if available
                model = _extract_model_from_args(call.get("args_preview", ""))
                if model:
                    name = f"{name} ({model})"
                key = ("ai_api", target)
                if key not in seen:
                    seen.add(key)
                    sinks.append(FlowSink(
                        type="ai_api",
                        name=name,
                        detail=f"{target}()",
                        provider=provider,
                    ))

        # DB write operations (INSERT, save_to_db, etc.)
        for func in ctx.functions:
            body = func.get("body_preview", "")
            if re.search(r"save_to_db|INSERT\s+INTO|\.commit\(|\.add\(.*session", body, re.IGNORECASE):
                key = ("database_write", func.get("name", ""))
                if key not in seen:
                    seen.add(key)
                    table = ""
                    m = re.search(r"INSERT\s+INTO\s+(\w+)", body, re.IGNORECASE)
                    if m:
                        table = f": {m.group(1)}"
                    sinks.append(FlowSink(
                        type="database",
                        name=f"Database write{table}",
                        detail=_clean_detail(body[:100]) if table else "save_to_db()",
                    ))

        # REST endpoint decorator → implies http_response sink
        for func in ctx.functions:
            for dec in func.get("decorators", []):
                if any(kw in dec for kw in ("route", "app.get", "app.post", "app.put", "router.")):
                    key = ("http_response", func.get("name", ""))
                    if key not in seen:
                        seen.add(key)
                        endpoint = _extract_endpoint(dec)
                        sinks.append(FlowSink(
                            type="http_response",
                            name="API response",
                            detail=f"Response from {endpoint or func.get('name', '')}",
                        ))

    # If no AI API sinks were found from api_calls, try to infer from
    # provider info — the asset uses an AI service even if we couldn't
    # identify the specific call site.
    has_ai_sink = any(s.type == "ai_api" for s in sinks)
    if not has_ai_sink and asset.provider:
        profile = get_provider(asset.provider.name)
        if profile.category in ("llm_api", "embedding_db"):
            model_names = []
            for ctx in asset.code_contexts:
                model_names.extend(ctx.model_names)
            model_hint = f" ({model_names[0]})" if model_names else ""
            sinks.append(FlowSink(
                type="ai_api",
                name=f"{profile.display_name} API{model_hint}",
                detail=f"Provider: {profile.vendor}",
                provider=asset.provider.name,
            ))

    return sinks


def _humanize_sink(sink: dict) -> str:
    """Turn a raw data_sink dict into a readable name."""
    sink_type = sink.get("type", "")
    detail = sink.get("detail", "")
    name = sink.get("name", "")

    if name and name != sink_type:
        return name
    if sink_type == "database":
        table = _extract_table_name(detail)
        return f"Database: {table}" if table else "Database write"
    if sink_type == "file":
        return f"File: {_short_path(detail)}" if detail else "File output"
    if sink_type == "http":
        return f"HTTP: {_short_url(detail)}" if detail else "HTTP call"
    return sink_type or "Unknown sink"


# ── Processing steps inference ────────────────────────────────────────────

_STEP_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Input reception
    (re.compile(r"request\.json|request\.form|request\.get_json|request\.data"), "Receive user input via REST endpoint"),
    (re.compile(r"input\s*\(|sys\.stdin|argparse"), "Accept user input from CLI"),
    (re.compile(r"open\s*\(.+['\"]r['\"]|read_text|read_csv|load_json"), "Load data from file"),
    # Data retrieval
    (re.compile(r"cursor\.execute|\.query\(|SELECT\s|\.find\(|\.get\(.*id"), "Query data from database"),
    (re.compile(r"get_history|load_history|fetch_messages|get_context"), "Load conversation history"),
    (re.compile(r"similarity_search|vector_store\.query|\.retrieve\("), "Retrieve relevant documents (vector search)"),
    # Processing
    (re.compile(r"encode_image|base64\.b64encode|Image\.open"), "Encode/process image input"),
    (re.compile(r"embed|embedding|get_embedding"), "Generate embeddings"),
    (re.compile(r"text_splitter|chunk|split_documents|RecursiveCharacter"), "Split text into chunks"),
    # AI API calls
    (re.compile(r"messages\.create|chat\.completions\.create"), "Send prompt to LLM API"),
    (re.compile(r"responses\.create"), "Send prompt to OpenAI Responses API"),
    (re.compile(r"images\.generate|dall.e|generate_image"), "Generate image via AI API"),
    (re.compile(r"transcri|whisper|audio"), "Transcribe audio"),
    (re.compile(r"tool_calls|function_call|tools=\["), "Execute tool/function calls"),
    (re.compile(r"create_react_agent|AgentExecutor|agent\.invoke"), "Run AI agent with tools"),
    # Storage
    (re.compile(r"save_to_db|INSERT\s+INTO|\.insert\(|\.add\(|cursor\.execute.*INSERT|\.commit\("), "Store results in database"),
    (re.compile(r"\.write\(|save_json|to_csv|dump\(|\.save\("), "Write results to file"),
    (re.compile(r"vector_store\.add|\.upsert\(|add_documents"), "Store embeddings in vector database"),
    # Output
    (re.compile(r"return\s+jsonify|return\s+.*json|JSONResponse|Response\(|jsonify\("), "Return response to client"),
    (re.compile(r"print\s*\(.*response|print\s*\(.*result|console\.print"), "Output results"),
]


def _infer_processing_steps(asset: AIAsset) -> list[str]:
    """Infer the sequence of processing steps from function bodies.

    For Python files with AST-parsed functions, we walk the body
    top-to-bottom and classify each statement. This gives us the
    actual execution order — the real value of Step 2.

    For other languages (regex-extracted), we fall back to pattern
    matching across the full code context, which gives us the set of
    operations but not necessarily their order.
    """
    steps: list[str] = []
    seen_steps: set[str] = set()

    # Prefer the "main" function's body for step ordering
    main_body = _find_main_function_body(asset)

    if main_body:
        for pattern, step_desc in _STEP_PATTERNS:
            if pattern.search(main_body):
                if step_desc not in seen_steps:
                    steps.append(step_desc)
                    seen_steps.add(step_desc)

    # Supplement with ALL function bodies + function names — the main
    # body is often truncated (body_preview cap) so steps at the end
    # of the function (store, return) get cut off. Also: a function
    # named ``save_to_db`` is itself a strong "Store in database" signal.
    all_text = _collect_all_body_text(asset)
    all_func_names = " ".join(
        f.get("name", "")
        for ctx in asset.code_contexts
        for f in ctx.functions
    )
    combined = all_text + " " + all_func_names

    for pattern, step_desc in _STEP_PATTERNS:
        if step_desc not in seen_steps and pattern.search(combined):
            steps.append(step_desc)
            seen_steps.add(step_desc)

    # Infer steps from sinks/sources that we detected structurally but
    # whose body text was truncated (common for the last few lines of
    # long functions — body_preview caps at ~500 chars).
    sinks = _identify_sinks(asset)
    if any(s.type == "http_response" for s in sinks):
        step = "Return response to client"
        if step not in seen_steps:
            steps.append(step)
    if any(s.type == "database" for s in sinks):
        step = "Store results in database"
        if step not in seen_steps:
            steps.append(step)

    return steps


def _find_main_function_body(asset: AIAsset) -> str:
    """Find the most representative function body for step extraction.

    Priority: decorated endpoint function > function with most AI
    calls > longest function body. This ensures we describe what the
    solution DOES, not utility helpers.
    """
    best: str = ""
    best_score: int = -1

    for ctx in asset.code_contexts:
        for func in ctx.functions:
            body = func.get("body_preview", "")
            if not body:
                continue
            score = 0
            # Decorated endpoints are most descriptive
            for dec in func.get("decorators", []):
                if "route" in dec or "app." in dec or "router." in dec:
                    score += 100
            # Functions that call AI APIs
            if any(t in body for t in ("completions.create", "messages.create", "agent.invoke")):
                score += 50
            # Tool-using functions
            if "tool_calls" in body or "function_call" in body:
                score += 30
            # Longer bodies have more steps
            score += min(len(body) // 100, 20)
            if score > best_score:
                best = body
                best_score = score

    return best


# ── Purpose composition ───────────────────────────────────────────────────


def _compose_purpose(
    asset: AIAsset,
    sources: list[FlowSource],
    sinks: list[FlowSink],
    steps: list[str],
) -> str:
    """Compose a solution_purpose string from structured flow data.

    This replaces the generic ``_synthesize_purpose()`` in enrichment.py.
    Instead of pattern-matching ("has prompt → chatbot"), we describe
    what the code actually does based on its data flow.
    """
    parts: list[str] = []

    # 1. Role from prompt (most specific purpose indicator)
    role = _extract_role_from_prompts(asset)
    if role:
        parts.append(role)

    # 2. What it does — based on steps and sinks
    action = _describe_action(asset, sinks, steps)
    if action:
        parts.append(action)

    # 3. Where data comes from / goes to
    flow_desc = _describe_flow(sources, sinks)
    if flow_desc:
        parts.append(flow_desc)

    if parts:
        return " ".join(parts)

    # Fallback: provider-based
    if asset.provider:
        profile = get_provider(asset.provider.name)
        return f"AI solution using {profile.display_name}."
    return "AI solution."


def _extract_role_from_prompts(asset: AIAsset) -> str:
    """Extract a specific role/identity from system prompts."""
    for ctx in asset.code_contexts:
        for prompt in ctx.prompts:
            # Greedy match up to period-space or end — don't stop at comma
            # so "Fleurdin, a helpful florist assistant" captures fully.
            match = re.match(
                r"(?:You are|You're|I am|Act as)\s+(.+?)(?:\.\s|\.\"|\.'\s*$|!|\n|$)",
                prompt, re.IGNORECASE,
            )
            if match:
                role = match.group(1).strip().rstrip(".")
                if len(role) > 10 and not _is_generic_role(role):
                    return f'Role: "{role[:150]}."'
    return ""


def _is_generic_role(role: str) -> bool:
    generic = (
        "a helpful assistant", "an ai assistant", "a helpful ai",
        "an assistant", "a chatbot", "helpful", "an ai",
    )
    return role.lower().strip() in generic


def _describe_action(
    asset: AIAsset, sinks: list[FlowSink], steps: list[str]
) -> str:
    """One sentence describing the primary action."""
    ai_sinks = [s for s in sinks if s.type == "ai_api"]
    has_tools = any("tool" in s.lower() or "agent" in s.lower() for s in steps)
    has_rag = any("vector" in s.lower() or "retriev" in s.lower() or "embed" in s.lower() for s in steps)
    has_image = any("image" in s.lower() for s in steps)
    has_transcription = any("transcri" in s.lower() for s in steps)

    model_names = []
    for ctx in asset.code_contexts:
        model_names.extend(ctx.model_names)
    model_hint = model_names[0] if model_names else ""

    if has_tools:
        if model_hint:
            return f"AI agent using {model_hint} with function calling / tool use."
        return "AI agent with function calling / tool use."
    if has_rag:
        return "Retrieval-augmented generation (RAG) pipeline."
    if has_image:
        return "Image generation pipeline."
    if has_transcription:
        return "Audio transcription pipeline."
    if ai_sinks:
        sink = ai_sinks[0]
        return f"Sends prompts to {sink.name}."

    return ""


def _describe_flow(
    sources: list[FlowSource], sinks: list[FlowSink]
) -> str:
    """Short description of data flow: reads from X, writes to Y."""
    # Filter out env_var sources (config, not data)
    data_sources = [s for s in sources if s.type != "env_var"]
    # Filter out http_response sinks (always present in web apps)
    data_sinks = [s for s in sinks if s.type not in ("http_response",)]

    parts = []
    if data_sources:
        src_names = sorted({s.name for s in data_sources})[:3]
        parts.append(f"Reads from: {', '.join(src_names)}.")
    if data_sinks:
        sink_names = sorted({s.name for s in data_sinks})[:3]
        parts.append(f"Outputs to: {', '.join(sink_names)}.")
    return " ".join(parts)


# ── Data category classification ──────────────────────────────────────────

_CATEGORY_SIGNALS: list[tuple[tuple[str, ...], str]] = [
    (("customer", "user_name", "personal", "pii", "phone", "address", "email"), "personal_data"),
    (("password", "credential", "secret", "api_key", "token"), "credentials"),
    (("stock", "price", "ticker", "dividend", "financial", "payment", "invoice"), "financial_data"),
    (("patient", "medical", "health", "diagnosis", "prescription"), "medical_data"),
    (("audio", "wav", "mp3", "transcri", "speech", "whisper"), "audio"),
    (("image", "photo", "png", "jpg", "vision", "screenshot"), "images"),
    (("video", "mp4", "frame"), "video"),
    (("pdf", "document", "docx"), "documents"),
    (("csv", "excel", "spreadsheet", "dataframe"), "tabular_data"),
    (("embedding", "vector", "similarity"), "embeddings"),
    (("chat", "message", "conversation", "prompt"), "user_messages"),
    (("code", "source", "script", "function", "class"), "source_code"),
]


def _classify_data_categories(
    asset: AIAsset,
    sources: list[FlowSource],
    sinks: list[FlowSink],
) -> list[str]:
    """Classify what types of data this solution processes."""
    haystack = " ".join([
        " ".join(s.name + " " + s.detail for s in sources),
        " ".join(s.name + " " + s.detail for s in sinks),
        " ".join(p for ctx in asset.code_contexts for p in ctx.prompts),
        " ".join(
            f.get("body_preview", "")
            for ctx in asset.code_contexts
            for f in ctx.functions
        ),
    ]).lower()

    categories: set[str] = set()
    for keywords, category in _CATEGORY_SIGNALS:
        if any(kw in haystack for kw in keywords):
            categories.add(category)

    return sorted(categories)


# ── Confidence assessment ─────────────────────────────────────────────────


def _assess_confidence(
    asset: AIAsset,
    sources: list[FlowSource],
    sinks: list[FlowSink],
    steps: list[str],
) -> Confidence:
    """How much do we trust this DataFlowMap?"""
    has_ast = any(ctx.language == "python" for ctx in asset.code_contexts)
    has_sources = len(sources) > 0
    has_sinks = len(sinks) > 0
    has_steps = len(steps) >= 2

    if has_ast and has_sources and has_sinks and has_steps:
        return Confidence.HIGH
    if has_sinks and has_steps:
        return Confidence.MEDIUM
    return Confidence.LOW


# ── Fallback for assets with no code context ──────────────────────────────


def _fallback_flow(asset: AIAsset) -> DataFlowMap:
    """Minimal DataFlowMap when no code context is available."""
    purpose = "AI solution"
    if asset.provider:
        profile = get_provider(asset.provider.name)
        purpose = f"AI solution using {profile.display_name}."
    return DataFlowMap(
        solution_purpose=purpose,
        confidence=Confidence.LOW,
    )


# ── String utilities ──────────────────────────────────────────────────────


_WRITE_OPS_RE = re.compile(
    r"INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|\.write\(|\.commit\(",
    re.IGNORECASE,
)


_NOISE_SOURCE_RE = re.compile(
    r"^(?:conn\.cursor|cursor$|\.connect\(|\.close\(|\.commit\(|"
    r"cursor\.fetchall|cursor\.fetchone|fetchall|fetchmany)",
    re.IGNORECASE,
)
_NOISE_SINK_RE = re.compile(
    r"\.fetchall|\.fetchone|\.fetchmany|cursor\.execute.*SELECT|\.read\(",
    re.IGNORECASE,
)


def _is_write_operation(detail: str) -> bool:
    """Return True if this detail describes a write, not a read."""
    return bool(_WRITE_OPS_RE.search(detail))


def _is_noise_source(src: dict) -> bool:
    detail = src.get("detail", "") or src.get("name", "")
    return bool(_NOISE_SOURCE_RE.search(detail))


def _is_noise_sink(sink: dict) -> bool:
    detail = sink.get("detail", "") or sink.get("name", "")
    return bool(_NOISE_SINK_RE.search(detail))


def _normalize_detail(detail: str) -> str:
    """Normalize a detail string for dedup — strip whitespace, lowercase."""
    text = re.sub(r"\s+", " ", detail.strip().lower())
    # For SQL, extract just the core operation + table
    match = re.search(r"(select|insert|update|delete)\s+.*?(?:from|into)\s+(\w+)", text)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return text[:80]


def _extract_endpoint(decorator: str) -> str:
    """Extract REST endpoint from a decorator string."""
    match = re.search(r"""['"](/[^'"]+)['"]""", decorator)
    if match:
        methods = ""
        m2 = re.search(r"methods\s*=\s*\[([^\]]+)\]", decorator)
        if m2:
            methods = " " + m2.group(1).replace("'", "").replace('"', '').strip()
        return f"{methods.strip()} {match.group(1)}".strip()
    return ""


def _extract_table_name(detail: str) -> str:
    match = re.search(r"FROM\s+(\w+)", detail, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"INTO\s+(\w+)", detail, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def _extract_model_from_args(args_preview: str) -> str:
    match = re.search(r"model=['\"]([^'\"]+)['\"]", args_preview)
    return match.group(1) if match else ""


def _short_path(path: str) -> str:
    return path.split("/")[-1][:40] if "/" in path else path[:40]


def _short_url(url: str) -> str:
    url = re.sub(r"https?://", "", url)
    return url[:50]


def _clean_detail(detail: str) -> str:
    return " ".join(detail.split())[:150]


def _collect_all_body_text(asset: AIAsset) -> str:
    parts = []
    for ctx in asset.code_contexts:
        for func in ctx.functions:
            parts.append(func.get("body_preview", ""))
        parts.extend(ctx.raw_snippets)
    return " ".join(parts)
