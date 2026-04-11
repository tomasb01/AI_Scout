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
        model: str = "qwen2.5-coder:14b",
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
        """Build a structured classification prompt for the LLM."""
        findings_text = []
        count = 0
        for f in asset.raw_findings:
            if count >= 20:
                findings_text.append(f"... and {len(asset.raw_findings) - 20} more findings")
                break
            # Use redacted content for API keys
            content = f.redacted_content if f.redacted_content else f.content
            findings_text.append(
                f"- [{f.type.value}] {f.file_path}"
                f"{f':L{f.line_number}' if f.line_number else ''}: {content}"
            )
            count += 1

        deps_text = ", ".join(asset.dependencies) if asset.dependencies else "none"

        return f"""You are an AI security analyst. Analyze the following AI asset discovered in a code repository and classify it.

## Asset Information
- Name: {asset.name}
- Repository: {asset.repository}
- Provider: {asset.provider.name if asset.provider else 'unknown'}
- Files: {asset.file_path}
- Dependencies: {deps_text}

## Findings
{chr(10).join(findings_text)}

## Task
Based on the findings above, classify this AI asset. Return a JSON object with exactly these keys:

- "data_categories": list of applicable categories from: "public", "internal", "confidential", "pii", "financial", "source_code", "unknown"
- "confidence": one of "high", "medium", "low"
- "risk_score": float between 0.0 and 1.0 (0=no risk, 1=critical)
- "summary": brief description of what this AI integration does and its risk profile
- "recommendations": list of actionable security recommendations

## Example Response
{{"data_categories": ["internal", "source_code"], "confidence": "medium", "risk_score": 0.4, "summary": "OpenAI integration for code analysis with API key stored in source code.", "recommendations": ["Move API key to environment variable or secret manager", "Review data being sent to OpenAI API"]}}

Return ONLY the JSON object, no other text."""

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama REST API."""
        response = self._client.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
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
