"""Report linter L-01–L-10 (Sprint 0.2, QA spec §2).

Runs over the FINAL rendered text of every prose field (insight
sentences, action sentences, displayed summaries). Purely deterministic —
regex + the vocabularies in ``qa_vocab`` — no LLM, no network.

It is a safety net against *composition* errors, not a grammar checker:
general English correctness is guaranteed by construction (human-written
templates + KB labels, QA spec P-1).

Degradation, never failure (P-4): an ERROR suppresses the sentence (the
caller renders a fact strip instead); a WARN lets it through. Both are
logged and surface in the report's QA appendix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from aiscout.report.qa_vocab import (
    ABBREVIATIONS,
    COUNTABLE_NOUNS,
    LEGIT_DOUBLED_WORDS,
    SENTENCE_STOP_WORDS,
)

ERROR = "ERROR"
WARN = "WARN"


@dataclass
class LintIssue:
    rule: str
    severity: str  # ERROR | WARN
    message: str
    text: str
    template_id: str = ""
    entity_id: str = ""

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "text": self.text,
            "template_id": self.template_id,
            "entity_id": self.entity_id,
        }


@dataclass
class QAReport:
    """Aggregated linter outcome for one generated report."""

    issues: list[LintIssue] = field(default_factory=list)

    @property
    def suppressed(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == WARN]

    def counts(self) -> dict:
        return {
            "suppressed": len(self.suppressed),
            "warnings": len(self.warnings),
        }


# ── Rule implementations ───────────────────────────────────────────────────

_DOUBLED_WORD_RE = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
_PLACEHOLDER_TOKENS = ("undefined", "null", "None", "NaN", "[object Object]")
_EMPTY_PARENS_RE = re.compile(r"\(\s*\)")
_ORPHAN_PUNCT_RE = re.compile(r"(\s[,;:]\s*$)|(^\s*[,;:.])")
_PCT_RE = re.compile(r"(\d+)\s*%")
_NEGATIVE_COUNT_RE = re.compile(r"(?<![\w.])-\d+")
_SNAKE_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b")
_CAMEL_RE = re.compile(r"\b[a-z]+[A-Z][a-zA-Z0-9]*\b")
_ALLCAPS_RE = re.compile(r"\b[A-Z]{3,}\b")
_PATH_RE = re.compile(r"(?:^|[\s(])/?(?:[\w.-]+/)+[\w.-]+\.[a-zA-Z]{1,4}\b")
_SQL_RE = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE)\s+.*\b(FROM|INTO|SET|WHERE)\b")
_ONE_PLURAL_RE = re.compile(r"\b1\s+([a-zA-Z]+s)\b")

# Bracket/quote pairs for L-02. Straight quotes are checked by parity,
# curly ones as open/close pairs.
_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"), ("“", "”"))


def lint_text(
    text: str,
    *,
    kind: str = "insight",  # insight | action | label
    template_id: str = "",
    entity_id: str = "",
    safe_tokens: list[str] | None = None,
    apply_code_leak: bool = True,
) -> list[LintIssue]:
    """Apply L-01–L-07, L-09, L-10 to one rendered text field.

    ``safe_tokens`` are entity values from safe domains (git author
    identities, KB labels) — masked before the code-leak heuristics so
    "john_doe" as an author or "LangChain" as a KB label never trips L-07.
    """
    issues: list[LintIssue] = []

    def add(rule: str, severity: str, message: str) -> None:
        issues.append(LintIssue(
            rule=rule, severity=severity, message=message, text=text,
            template_id=template_id, entity_id=entity_id,
        ))

    # L-01 — doubled word
    for m in _DOUBLED_WORD_RE.finditer(text):
        if m.group(1).lower() not in LEGIT_DOUBLED_WORDS:
            add("L-01", ERROR, f"Doubled word: {m.group(0)!r}")

    # L-02 — unpaired brackets / quotes
    for opener, closer in _PAIRS:
        if text.count(opener) != text.count(closer):
            add("L-02", ERROR, f"Unpaired {opener}{closer}")
    if text.count('"') % 2 != 0:
        add("L-02", ERROR, "Unpaired double quote")

    # L-03 — unresolved placeholder
    if "{" in text or "}" in text:
        add("L-03", ERROR, "Unresolved placeholder braces")
    for token in _PLACEHOLDER_TOKENS:
        if re.search(rf"(?<![\w']){re.escape(token)}(?![\w'])", text):
            add("L-03", ERROR, f"Placeholder token {token!r} in output")

    # L-04 — empty parentheses / orphaned punctuation
    if _EMPTY_PARENS_RE.search(text):
        add("L-04", ERROR, "Empty parentheses")
    if _ORPHAN_PUNCT_RE.search(text):
        add("L-04", ERROR, "Orphaned punctuation")

    # L-05 — truncated sentence (insight/action prose only, not labels)
    if kind != "label" and text.strip():
        stripped = text.strip()
        if stripped[-1] not in ".?!":
            add("L-05", ERROR, "Sentence does not end with punctuation")
        else:
            last_word = re.sub(r"[^\w]", "", stripped.split()[-1]).lower()
            if last_word in SENTENCE_STOP_WORDS:
                add("L-05", ERROR, f"Sentence ends on stop word {last_word!r}")

    # L-06 — numeric nonsense
    for m in _PCT_RE.finditer(text):
        if int(m.group(1)) > 100:
            add("L-06", ERROR, f"Percentage over 100: {m.group(0)!r}")
    if re.search(r"\bover 100\s*%", text, re.IGNORECASE):
        add("L-06", ERROR, "'over 100%' phrasing")
    if _NEGATIVE_COUNT_RE.search(text):
        add("L-06", ERROR, "Negative count in text")

    # L-07 — source-code leakage into prose. Skipped for LLM-provenance
    # prose (``apply_code_leak=False``), which legitimately references
    # code identifiers and is labelled as LLM output in the report.
    masked = text if apply_code_leak else ""
    for token in sorted(set(safe_tokens or []), key=len, reverse=True):
        if token:
            masked = masked.replace(token, "SAFEENTITY")
    if _SNAKE_RE.search(masked):
        add("L-07", ERROR, "snake_case identifier in prose")
    if _CAMEL_RE.search(masked):
        add("L-07", ERROR, "camelCase identifier in prose")
    for m in _ALLCAPS_RE.finditer(masked):
        if m.group(0) not in ABBREVIATIONS and m.group(0) != "SAFEENTITY":
            add("L-07", ERROR, f"Unknown ALL-CAPS token {m.group(0)!r} in prose")
    if _PATH_RE.search(masked):
        add("L-07", ERROR, "File path in prose")
    if _SQL_RE.search(masked):
        add("L-07", ERROR, "SQL keywords in prose")

    # L-09 — length bounds. Insight and action sentences share the 20–280
    # window; the tighter 90-char bound applies to short action *titles*
    # (kind="action_title"), which the report will grow with the redesign.
    if kind in ("insight", "action") and text.strip() and not (20 <= len(text) <= 280):
        add("L-09", WARN, f"Sentence length {len(text)} outside 20–280")
    if kind == "action_title" and len(text) > 90:
        add("L-09", WARN, f"Action title length {len(text)} over 90")

    # L-10 — plural mismatch after "1"
    for m in _ONE_PLURAL_RE.finditer(text):
        if m.group(1).lower() in COUNTABLE_NOUNS:
            add("L-10", ERROR, f"Plural after 1: {m.group(0)!r}")

    return issues


def lint_duplicate_summaries(
    summaries: dict[str, str],
) -> tuple[list[LintIssue], set[str]]:
    """L-08 — identical summary across ≥ 2 solutions.

    Returns (issues, entity ids to degrade to fact strip). A duplicated
    "summary" carries no information; at ≥ 3 occurrences it is an ERROR,
    at 2 a WARN — but every occurrence degrades either way (QA spec §2.2).
    """
    by_text: dict[str, list[str]] = {}
    for entity_id, text in summaries.items():
        if text and text.strip():
            by_text.setdefault(text.strip(), []).append(entity_id)

    issues: list[LintIssue] = []
    degrade: set[str] = set()
    for text, ids in sorted(by_text.items()):
        if len(ids) < 2:
            continue
        severity = ERROR if len(ids) >= 3 else WARN
        degrade.update(ids)
        issues.append(LintIssue(
            rule="L-08", severity=severity,
            message=f"Identical summary across {len(ids)} solutions",
            text=text, entity_id=",".join(sorted(ids)),
        ))
    return issues, degrade
