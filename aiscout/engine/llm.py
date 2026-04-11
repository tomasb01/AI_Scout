"""LLM Analysis Engine — classifies AI assets via Ollama or OpenAI-compatible API."""

from __future__ import annotations

import json
import time

import httpx
from rich.console import Console

from aiscout.models import (
    AIAsset,
    ClassificationResult,
    Confidence,
    DataCategory,
    FindingType,
)

console = Console()

# Map string values to DataCategory enum
_CATEGORY_MAP = {c.value: c for c in DataCategory}


class LLMEngine:
    """Classifies AI assets using an LLM (Ollama or OpenAI-compatible API)."""

    def __init__(
        self,
        mode: str = "ollama",
        url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:7b",
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.mode = mode
        self.url = url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def check_health(self) -> bool:
        """Check if the LLM backend is reachable."""
        try:
            if self.mode == "ollama":
                r = self._client.get(f"{self.url}/api/tags")
            else:
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                r = self._client.get(f"{self.url}/v1/models", headers=headers)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def classify(self, asset: AIAsset) -> ClassificationResult:
        """Classify a single AI asset via LLM."""
        prompt = self._build_prompt(asset)

        if self.mode == "ollama":
            raw = self._call_ollama(prompt)
        else:
            raw = self._call_openai(prompt)

        return self._parse_response(raw)

    def classify_batch(self, assets: list[AIAsset]) -> list[AIAsset]:
        """Classify all assets, attaching results. Returns updated assets."""
        for i, asset in enumerate(assets):
            try:
                result = self.classify(asset)
                asset.data_classification = result
                if result.risk_score > 0:
                    asset.risk_score = max(asset.risk_score, result.risk_score)
            except Exception as e:
                console.print(
                    f"  [yellow]Warning:[/] LLM classification failed for "
                    f"'{asset.name}': {e}"
                )

            # Rate limiting between calls
            if i < len(assets) - 1:
                time.sleep(0.5)

        return assets

    # ── Private helpers ────────────────────────────────────────────────

    def _build_prompt(self, asset: AIAsset) -> str:
        """Build a prompt with full code context for deep analysis."""
        sections = []

        # Basic info
        files = asset.file_path.split(", ")
        sections.append(f"""## Solution
- Name: {asset.name}
- Repository: {asset.repository}
- Files: {len(files)} ({', '.join(files[:5])})""")

        # Code context — the key differentiator
        if asset.code_contexts:
            ctx_parts = []

            # Functions
            all_funcs = []
            for ctx in asset.code_contexts:
                for f in ctx.functions:
                    name = f.get("name", "")
                    args = ", ".join(f.get("args", []))
                    doc = f.get("docstring", "")
                    body = f.get("body_preview", "")
                    decorators = f.get("decorators", [])
                    dec_str = " ".join(decorators) + " " if decorators else ""
                    entry = f"{dec_str}{name}({args})"
                    if doc:
                        entry += f" — {doc[:100]}"
                    if body:
                        entry += f"\n    {body[:200]}"
                    all_funcs.append(entry)

            if all_funcs:
                ctx_parts.append("Functions:\n" + "\n".join(f"- {fn}" for fn in all_funcs[:10]))

            # Prompts (most valuable signal)
            all_prompts = []
            for ctx in asset.code_contexts:
                all_prompts.extend(ctx.prompts)
            if all_prompts:
                # Show the best prompt (longest, most specific)
                best = max(all_prompts, key=len)
                ctx_parts.append(f"System prompt:\n\"{best[:400]}\"")

            # Model names
            all_models = []
            for ctx in asset.code_contexts:
                all_models.extend(ctx.model_names)
            if all_models:
                ctx_parts.append(f"LLM models used: {', '.join(set(all_models))}")

            # API calls
            all_calls = []
            for ctx in asset.code_contexts:
                for call in ctx.api_calls:
                    all_calls.append(f"{call.get('target', '')}({call.get('args_preview', '')[:80]})")
            if all_calls:
                ctx_parts.append("API calls:\n" + "\n".join(f"- {c}" for c in all_calls[:8]))

            # Data sources
            all_sources = []
            for ctx in asset.code_contexts:
                for src in ctx.data_sources:
                    all_sources.append(f"[{src.get('type', '')}] {src.get('name', '')} — {src.get('detail', '')[:80]}")
            if all_sources:
                ctx_parts.append("Data sources:\n" + "\n".join(f"- {s}" for s in all_sources[:6]))

            # Data sinks
            all_sinks = []
            for ctx in asset.code_contexts:
                for sink in ctx.data_sinks:
                    all_sinks.append(f"[{sink.get('type', '')}] {sink.get('name', '')} — {sink.get('detail', '')[:80]}")
            if all_sinks:
                ctx_parts.append("Data outputs:\n" + "\n".join(f"- {s}" for s in all_sinks[:6]))

            # Env vars
            all_env = set()
            for ctx in asset.code_contexts:
                all_env.update(ctx.env_vars)
            if all_env:
                ctx_parts.append(f"Environment variables: {', '.join(sorted(all_env)[:8])}")

            if ctx_parts:
                sections.append("## Code Analysis\n" + "\n\n".join(ctx_parts))

        # Findings (brief)
        findings = []
        for f in asset.raw_findings[:10]:
            content = f.redacted_content if f.redacted_content else f.content
            findings.append(f"- [{f.type.value}] {f.file_path}: {content}")
        if findings:
            sections.append("## Scan Findings\n" + "\n".join(findings))

        # Task
        sections.append("""## Task
Based on the code analysis above, describe this AI solution. Return a JSON object with exactly these keys:

- "summary": 2-3 sentences describing what this solution DOES, what data it processes, and its purpose. Be specific — mention the actual functionality, not just the framework.
- "data_categories": list from: "public", "internal", "confidential", "pii", "financial", "source_code", "unknown"
- "confidence": "high", "medium", or "low"
- "risk_score": float 0.0-1.0 (0=safe, 1=critical risk)
- "recommendations": list of 2-3 actionable recommendations

Example:
{"summary": "Voice transcription pipeline that records customer calls, transcribes audio via Whisper API, extracts key topics using GPT-4, and stores results in PostgreSQL. Processes audio recordings containing personal customer information.", "data_categories": ["pii", "internal"], "confidence": "high", "risk_score": 0.6, "recommendations": ["Ensure customer consent for call recording", "Add PII redaction before storing transcripts"]}

Return ONLY the JSON object.""")

        return "\n\n".join(sections)

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama REST API."""
        response = self._client.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "num_ctx": 8192,
                    "temperature": 0.1,
                },
            },
        )
        response.raise_for_status()
        return response.json()["response"]

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI-compatible API."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._client.post(
            f"{self.url}/v1/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _parse_response(self, raw: str) -> ClassificationResult:
        """Parse LLM JSON response into ClassificationResult."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return ClassificationResult(
                confidence=Confidence.LOW,
                details="LLM response could not be parsed",
            )

        # Parse categories
        categories = []
        for cat in data.get("data_categories", []):
            if cat in _CATEGORY_MAP:
                categories.append(_CATEGORY_MAP[cat])

        # Parse confidence
        conf_str = data.get("confidence", "low")
        try:
            confidence = Confidence(conf_str)
        except ValueError:
            confidence = Confidence.LOW

        # Parse risk score
        try:
            risk_score = float(data.get("risk_score", 0.0))
            risk_score = max(0.0, min(1.0, risk_score))
        except (TypeError, ValueError):
            risk_score = 0.0

        # Build details
        summary = data.get("summary", "")
        recommendations = data.get("recommendations", [])

        return ClassificationResult(
            categories=categories,
            confidence=confidence,
            risk_score=risk_score,
            details=summary,
            recommendations=recommendations,
        )
