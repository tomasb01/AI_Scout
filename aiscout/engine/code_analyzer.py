"""Code Context Extractor — analyzes source code to understand what AI solutions do."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from aiscout.models import AIAsset, CodeContext

# ── Language detection ─────────────────────────────────────────────────────

LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".java": "java",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".ipynb": "python",
}

# ── Regex patterns for all languages ───────────────────────────────────────

PROMPT_PATTERNS = [
    re.compile(r'(?:system[_\s]?(?:prompt|message|instruction|content))\s*[:=]\s*(?:f?["\'\`]{1,3})(.*?)(?:["\'\`]{1,3})', re.DOTALL | re.IGNORECASE),
    re.compile(r'["\']role["\']\s*:\s*["\']system["\'].*?["\']content["\']\s*:\s*["\'\`]{1,3}(.*?)["\'\`]{1,3}', re.DOTALL),
    re.compile(r'(?:SYSTEM_PROMPT|SYSTEM_MESSAGE|DEFAULT_PROMPT)\s*=\s*(?:f?["\'\`]{1,3})(.*?)(?:["\'\`]{1,3})', re.DOTALL),
]

DB_PATTERNS = [
    re.compile(r'(?:execute|query|run_query|raw)\s*\(\s*(?:f?["\'])(.*?)["\']', re.DOTALL | re.IGNORECASE),
    re.compile(r'\b(SELECT\s+.{5,80}?\s+FROM\s+\w+)', re.IGNORECASE),
    re.compile(r'\b(INSERT\s+INTO\s+\w+)', re.IGNORECASE),
    re.compile(r'\b(UPDATE\s+\w+\s+SET)', re.IGNORECASE),
    re.compile(r'\b(DELETE\s+FROM\s+\w+)', re.IGNORECASE),
    re.compile(r'\b(CREATE\s+TABLE\s+\w+)', re.IGNORECASE),
]

HTTP_PATTERNS = [
    re.compile(r'(?:requests|httpx|aiohttp)\s*\.\s*(get|post|put|delete|patch)\s*\(\s*(?:f?["\'])(.*?)["\']', re.IGNORECASE),
    re.compile(r'fetch\s*\(\s*(?:f?["\'\`])(.*?)["\'\`]'),
    re.compile(r'axios\s*\.\s*(get|post|put|delete)\s*\(\s*(?:f?["\'])(.*?)["\']'),
]

ENV_VAR_PATTERNS = [
    re.compile(r'os\.environ\s*(?:\[|\.get\s*\(\s*)["\'](\w+)["\']'),
    re.compile(r'os\.getenv\s*\(\s*["\'](\w+)["\']'),
    re.compile(r'process\.env\.(\w+)'),
    re.compile(r'env\s*\(\s*["\'](\w+)["\']'),
]

FILE_IO_PATTERNS = [
    re.compile(r'open\s*\(\s*(?:f?["\'])(.*?)["\']'),
    re.compile(r'(?:pd\.read_csv|pd\.read_excel|pd\.read_json|pd\.read_parquet)\s*\(\s*(?:f?["\'])(.*?)["\']'),
    re.compile(r'(?:json\.load|yaml\.safe_load|toml\.load)\s*\('),
    re.compile(r'(?:json\.dump|yaml\.dump)\s*\('),
    re.compile(r'(?:\.to_csv|\.to_excel|\.to_json|\.to_parquet)\s*\('),
]

# Known AI SDK call patterns
AI_CALL_PATTERNS = [
    re.compile(r'(\w+)\.(chat\.completions|completions|messages|embeddings)\.(create)\s*\(', re.DOTALL),
    re.compile(r'(\w+)\.(generate|invoke|predict|run|call|complete|transcribe)\s*\('),
    re.compile(r'(openai|anthropic|client|model|chain|agent)\.\w+\.\w+\s*\('),
]


# ── Public API ─────────────────────────────────────────────────────────────


def analyze_assets(assets: list[AIAsset], repo_root: str | Path):
    """Analyze code for all assets and populate code_contexts."""
    root = Path(repo_root)
    if not root.exists():
        return

    for asset in assets:
        contexts = _analyze_asset_files(asset, root)
        asset.code_contexts = contexts


def _analyze_asset_files(asset: AIAsset, repo_root: Path) -> list[CodeContext]:
    """Read and analyze all files in an asset."""
    if not asset.file_path:
        return []

    contexts = []
    file_paths = asset.file_path.split(", ")

    for rel_path in file_paths:
        full_path = repo_root / rel_path
        if not full_path.exists() or not full_path.is_file():
            continue

        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        if not content.strip():
            continue

        suffix = full_path.suffix.lower()
        language = LANG_MAP.get(suffix, "")

        # For notebooks, extract source first
        if suffix == ".ipynb":
            content = _extract_notebook_source(content)
            if not content:
                continue

        if language == "python":
            ctx = _analyze_python(rel_path, content)
        else:
            ctx = _analyze_generic(rel_path, content, language)

        if ctx and _has_useful_context(ctx):
            contexts.append(ctx)

    return contexts


def _has_useful_context(ctx: CodeContext) -> bool:
    """Check if context has any meaningful extracted data."""
    return bool(
        ctx.functions or ctx.classes or ctx.api_calls
        or ctx.data_sources or ctx.data_sinks or ctx.prompts
    )


# ── Python AST analysis ───────────────────────────────────────────────────


def _analyze_python(file_path: str, content: str) -> CodeContext:
    """Full AST analysis for Python files."""
    ctx = CodeContext(file_path=file_path, language="python")

    # Try AST parsing
    try:
        tree = ast.parse(content)
    except SyntaxError:
        # Fall back to regex
        return _analyze_generic(file_path, content, "python")

    lines = content.splitlines()

    # Extract functions
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_info = _extract_function(node, lines)
            ctx.functions.append(func_info)

        elif isinstance(node, ast.ClassDef):
            class_info = _extract_class(node, lines)
            ctx.classes.append(class_info)

    # Extract API calls, data sources/sinks from AST
    _extract_calls_from_ast(tree, lines, ctx)

    # Extract prompts from string literals in AST
    _extract_prompts_from_ast(tree, ctx)

    # Regex-based extraction for things AST misses
    _extract_with_regex(content, ctx)

    return ctx


def _extract_function(node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]) -> dict:
    """Extract function info from AST node."""
    args = [a.arg for a in node.args.args if a.arg != "self"]
    docstring = ast.get_docstring(node) or ""

    # Body preview: first 10 lines of function body
    body_start = node.body[0].lineno - 1 if node.body else node.lineno
    # Skip docstring in body preview
    if docstring and node.body and isinstance(node.body[0], ast.Expr):
        body_start = node.body[1].lineno - 1 if len(node.body) > 1 else body_start
    body_end = min(body_start + 10, len(lines))
    body_preview = "\n".join(lines[body_start:body_end]).strip()

    # Detect decorators (route, endpoint, etc.)
    decorators = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
            dec_text = f"@{ast.unparse(dec.func)}({', '.join(ast.unparse(a) for a in dec.args[:2])})"
            decorators.append(dec_text)
        elif isinstance(dec, ast.Attribute):
            decorators.append(f"@{ast.unparse(dec)}")
        elif isinstance(dec, ast.Name):
            decorators.append(f"@{dec.id}")

    return {
        "name": node.name,
        "args": args,
        "docstring": docstring[:200] if docstring else "",
        "body_preview": body_preview[:500],
        "decorators": decorators,
        "is_async": isinstance(node, ast.AsyncFunctionDef),
    }


def _extract_class(node: ast.ClassDef, lines: list[str]) -> dict:
    """Extract class info from AST node."""
    docstring = ast.get_docstring(node) or ""
    methods = [
        n.name for n in node.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    bases = [ast.unparse(b) for b in node.bases]

    return {
        "name": node.name,
        "methods": methods,
        "docstring": docstring[:200] if docstring else "",
        "bases": bases,
    }


def _extract_calls_from_ast(tree: ast.Module, lines: list[str], ctx: CodeContext):
    """Extract API calls, data sources/sinks from AST Call nodes."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        call_str = ""
        try:
            call_str = ast.unparse(node.func)
        except Exception:
            continue

        call_lower = call_str.lower()

        # AI SDK calls
        if any(kw in call_lower for kw in (
            "create", "generate", "invoke", "predict", "complete",
            "transcribe", "embed", "chat", "messages",
        )):
            if any(ai in call_lower for ai in (
                "openai", "anthropic", "client", "model", "chain",
                "agent", "completion", "message", "embedding",
            )):
                args_preview = _get_call_args_preview(node)
                ctx.api_calls.append({
                    "target": call_str,
                    "method": "ai_sdk",
                    "args_preview": args_preview[:300],
                })

        # Database calls
        if any(kw in call_lower for kw in ("execute", "query", "cursor", "session")):
            args_preview = _get_call_args_preview(node)
            ctx.data_sources.append({
                "type": "database",
                "name": call_str,
                "detail": args_preview[:200],
            })

        # HTTP calls
        if any(kw in call_lower for kw in ("requests.", "httpx.", "fetch", "axios.")):
            args_preview = _get_call_args_preview(node)
            ctx.data_sinks.append({
                "type": "http",
                "name": call_str,
                "detail": args_preview[:200],
            })

        # File operations
        if call_str in ("open",) or any(kw in call_lower for kw in (
            "read_csv", "read_json", "read_excel", "to_csv", "to_json",
            "json.load", "json.dump", "yaml.safe_load",
        )):
            args_preview = _get_call_args_preview(node)
            is_read = any(kw in call_lower for kw in ("read", "load", "open"))
            target = ctx.data_sources if is_read else ctx.data_sinks
            target.append({
                "type": "file",
                "name": call_str,
                "detail": args_preview[:200],
            })


def _get_call_args_preview(node: ast.Call) -> str:
    """Get a string preview of call arguments."""
    parts = []
    for arg in node.args[:3]:
        try:
            parts.append(ast.unparse(arg)[:100])
        except Exception:
            pass
    for kw in node.keywords[:3]:
        try:
            parts.append(f"{kw.arg}={ast.unparse(kw.value)[:80]}")
        except Exception:
            pass
    return ", ".join(parts)


def _extract_prompts_from_ast(tree: ast.Module, ctx: CodeContext):
    """Find prompt-like strings in AST."""
    for node in ast.walk(tree):
        # Look for string assignments to prompt-like variables
        if isinstance(node, ast.Assign):
            for target in node.targets:
                target_name = ""
                if isinstance(target, ast.Name):
                    target_name = target.id.lower()
                elif isinstance(target, ast.Attribute):
                    target_name = target.attr.lower()

                if any(kw in target_name for kw in ("prompt", "system", "instruction", "template")):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        text = node.value.value.strip()
                        if len(text) > 15:
                            ctx.prompts.append(text[:500])
                    elif isinstance(node.value, ast.JoinedStr):
                        # f-string — unparse it
                        try:
                            text = ast.unparse(node.value)
                            if len(text) > 15:
                                ctx.prompts.append(text[:500])
                        except Exception:
                            pass

        # Look for long strings that look like prompts (passed as arguments)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.strip()
            if len(text) > 50 and any(kw in text.lower() for kw in (
                "you are", "your role", "your task", "as a", "as an",
                "please", "analyze", "generate", "translate", "summarize",
                "classify", "extract", "help",
            )):
                ctx.prompts.append(text[:500])


# ── Generic regex analysis ─────────────────────────────────────────────────


def _analyze_generic(file_path: str, content: str, language: str) -> CodeContext:
    """Regex-based analysis for non-Python languages."""
    ctx = CodeContext(file_path=file_path, language=language or "unknown")

    # Extract functions (JS/TS style)
    for match in re.finditer(
        r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)',
        content,
    ):
        ctx.functions.append({
            "name": match.group(1),
            "args": [a.strip().split(":")[0].strip() for a in match.group(2).split(",") if a.strip()],
            "docstring": "",
            "body_preview": content[match.end():match.end() + 300].strip()[:300],
        })

    # Arrow functions assigned to const/let
    for match in re.finditer(
        r'(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>',
        content,
    ):
        ctx.functions.append({
            "name": match.group(1),
            "args": [],
            "docstring": "",
            "body_preview": content[match.end():match.end() + 300].strip()[:300],
        })

    # Common regex extraction for all languages
    _extract_with_regex(content, ctx)

    return ctx


def _extract_with_regex(content: str, ctx: CodeContext):
    """Extract prompts, DB ops, HTTP calls, env vars, file I/O using regex."""
    # Prompts
    for pattern in PROMPT_PATTERNS:
        for match in pattern.finditer(content):
            text = match.group(1).strip()
            if len(text) > 15 and text not in ctx.prompts:
                ctx.prompts.append(text[:500])

    # Database operations
    for pattern in DB_PATTERNS:
        for match in pattern.finditer(content):
            detail = match.group(1) if match.lastindex else match.group(0)
            detail = detail.strip()[:200]
            if not any(d["detail"] == detail for d in ctx.data_sources):
                ctx.data_sources.append({
                    "type": "database",
                    "name": "SQL query",
                    "detail": detail,
                })

    # HTTP calls
    for pattern in HTTP_PATTERNS:
        for match in pattern.finditer(content):
            groups = match.groups()
            url = groups[-1] if groups else match.group(0)
            if not any(d["detail"] == url for d in ctx.data_sinks):
                ctx.data_sinks.append({
                    "type": "http",
                    "name": "HTTP request",
                    "detail": url[:200],
                })

    # Environment variables
    for pattern in ENV_VAR_PATTERNS:
        for match in pattern.finditer(content):
            var = match.group(1)
            if var not in ctx.env_vars:
                ctx.env_vars.append(var)

    # File I/O
    for pattern in FILE_IO_PATTERNS:
        for match in pattern.finditer(content):
            if match.lastindex:
                detail = match.group(1)[:200]
                is_write = any(kw in match.group(0).lower() for kw in ("dump", "to_csv", "to_json", "to_excel", "write"))
                target = ctx.data_sinks if is_write else ctx.data_sources
                if not any(d["detail"] == detail for d in target):
                    target.append({
                        "type": "file",
                        "name": "file I/O",
                        "detail": detail,
                    })


# ── Notebook helper ────────────────────────────────────────────────────────


def _extract_notebook_source(content: str) -> str:
    """Extract source code from Jupyter notebook JSON."""
    try:
        nb = json.loads(content)
        sources = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") in ("code", "markdown"):
                source = cell.get("source", [])
                if isinstance(source, list):
                    sources.append("".join(source))
                elif isinstance(source, str):
                    sources.append(source)
        return "\n".join(sources)
    except (json.JSONDecodeError, KeyError):
        return ""
