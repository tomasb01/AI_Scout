"""Controlled vocabularies for the report QA layer (Sprint 0.2).

QA spec P-1: report prose is composed exclusively from (a) human-written
templates, (b) Provider KB labels, (c) the finite vocabularies below.
Raw strings extracted from a scanned repository never enter a sentence —
they belong in evidence fields rendered as monospace.

These dictionaries are versioned together with the templates (QA spec
§2.2); changing an entry is a conscious, review-visible act.
"""

from __future__ import annotations

# ── L-07: abbreviations allowed in prose (ALL-CAPS tokens ≥ 3 chars) ───────
ABBREVIATIONS = frozenset({
    "PII", "GDPR", "DPA", "API", "APIS", "MCP", "RAG", "SQL", "LLM", "LLMS",
    "SPOF", "REST", "URL", "EOL", "SBOM", "AIBOM", "SARIF", "SSO", "OAUTH",
    "DPIA", "AWS", "GCP", "KEY", "IDE", "PEFT", "GPT", "HTTP", "HTTPS",
    "JSON", "HTML", "CSV", "PDF", "AND",  # "AND" guards Oxford-joiner edge
})

# ── L-10: countable nouns whose plural after "1" is a composition bug ──────
COUNTABLE_NOUNS = frozenset({
    "solutions", "developers", "repositories", "providers", "keys",
    "findings", "areas", "integrations", "contributors", "files",
    "categories", "dependencies", "secrets", "tools", "agents", "scans",
})

# ── L-05: a sentence must not end on these words (truncation signal) ───────
SENTENCE_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at",
    "by", "for", "with", "from", "into", "than", "that", "which", "is",
    "are", "was", "were", "be", "been", "as", "if", "because", "while",
})

# ── L-01: legitimate doubled words (practically empty for EN reports) ──────
LEGIT_DOUBLED_WORDS = frozenset({"had"})

# ── I-02: finite vocabulary of critical-finding reasons ────────────────────
# Maps a stable reason kind → the phrase used in the exec-summary sentence.
CRITICAL_REASON_LABELS = {
    "hardcoded_api_key": "hardcoded API keys",
    "pii_training_risk": "personal data sent to a training-risk provider",
    "secrets_in_config": "secrets in configuration",
    "other": "other critical findings",
}

# ── I-07: sensitive data-category labels (finite; the word "data" is part
# of the label — templates never append it, killing "Financial data data") ─
SENSITIVE_DATA_LABELS = (
    "Personal data / PII",
    "Financial data",
    "Credentials / Secrets",
    "Health data",
)

# ── Fact strip: source-type labels (QA spec §1.4) ──────────────────────────
SOURCE_TYPE_LABELS = {
    "database": "database (SQL)",
    "file": "file input",
    "api": "REST endpoint",
    "user_input": "user input",
    "env_var": "environment variable",
    "message_queue": "message queue",
    "web_search": "web search",
}

# ── Fact strip: sink-type labels for non-AI sinks ──────────────────────────
SINK_TYPE_LABELS = {
    "ai_api": "AI API",
    "database": "database write",
    "file": "file output",
    "http_response": "HTTP response",
    "webhook": "webhook",
    "log": "log output",
}

# ── Fact strip: pattern labels, keyed by asset tags (priority order) ───────
PATTERN_LABELS = (
    # (tag, label) — first matching tag wins
    ("fine_tuning", "fine-tuning pipeline"),
    ("training", "training pipeline"),
    ("rag", "RAG pipeline"),
    ("agent", "agent loop"),
    ("mcp", "MCP integration"),
    ("chatbot", "chat completion"),
    ("transcription", "speech-to-text"),
    ("image_generation", "image generation"),
    ("evaluation", "model evaluation"),
    ("local_model", "local inference"),
)
PATTERN_MCP_SERVER = "MCP server"
PATTERN_FALLBACK = "AI integration"
UNCLASSIFIED = "unclassified"
