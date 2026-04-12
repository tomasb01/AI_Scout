"""Offline knowledge base of deprecated / vulnerable AI dependency versions.

This is intentionally a small, hand-curated list. Scout's goal is to
surface *known* risks without making network calls — customers run this
tool air-gapped. When a CVE or EOL window is well-known and stable we
encode it here; everything else is left to the customer's SCA tooling
(Dependabot, Renovate, Snyk, Trivy, …) which is better suited to the
daily churn of vulnerability data.

Each advisory describes one package and one constraint. Constraints use
packaging-compatible specifiers (``<1.0``, ``>=0.5,<0.7``) so we can
later swap to pypi-served data without changing the call sites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Advisory:
    package: str            # canonical lowercase name
    constraint: str         # e.g. "<1.0" or ">=0.0,<0.10"
    severity: str           # "critical" | "warning" | "info"
    title: str              # short headline for the report
    detail: str             # actionable explanation


# Keep this list short and high-signal. Every entry should describe
# *why* it matters — auditors should be able to explain the flag from
# the detail text alone, without a secondary lookup.
ADVISORIES: tuple[Advisory, ...] = (
    Advisory(
        package="openai",
        constraint="<1.0",
        severity="warning",
        title="OpenAI SDK < 1.0 (legacy API)",
        detail=(
            "The 0.x OpenAI Python SDK uses the legacy API surface and is "
            "no longer receiving security or compatibility fixes. Upgrade "
            "to >=1.0 and migrate any ``openai.Completion`` / "
            "``openai.ChatCompletion`` calls to the new client API."
        ),
    ),
    Advisory(
        package="langchain",
        constraint="<0.1",
        severity="warning",
        title="LangChain < 0.1 (pre-split, unsupported)",
        detail=(
            "LangChain versions below 0.1 predate the core/community "
            "split and have multiple documented prompt-injection and "
            "code-execution issues (CVE-2023-36095, CVE-2023-44467 et al.)."
            " Upgrade to the latest 0.2+ series and move integration "
            "code to the split packages (``langchain-openai``, …)."
        ),
    ),
    Advisory(
        package="langchain",
        constraint=">=0.1,<0.2",
        severity="info",
        title="LangChain 0.1.x — superseded",
        detail=(
            "Works but is no longer the upstream default. Consider "
            "upgrading to 0.2+ for active bug fixes and the new "
            "middleware API."
        ),
    ),
    Advisory(
        package="transformers",
        constraint="<4.36",
        severity="warning",
        title="HuggingFace transformers < 4.36",
        detail=(
            "Older transformers releases have known model-loading RCE "
            "risks (trust_remote_code defaults) and missing safetensors "
            "support for several architectures. Upgrade to >=4.38."
        ),
    ),
    Advisory(
        package="llama-index",
        constraint="<0.10",
        severity="warning",
        title="LlamaIndex < 0.10 (pre-rewrite)",
        detail=(
            "LlamaIndex 0.10 reorganised the package tree and fixed a "
            "number of SSRF/path-traversal issues in loaders. Upgrade "
            "and migrate loader imports to the new llama_index.readers."
        ),
    ),
    Advisory(
        package="llamaindex",
        constraint="<0.10",
        severity="warning",
        title="LlamaIndex < 0.10 (pre-rewrite)",
        detail=(
            "Alias of llama-index; same guidance as above."
        ),
    ),
    Advisory(
        package="chromadb",
        constraint="<0.4",
        severity="warning",
        title="ChromaDB < 0.4 (unsupported schema)",
        detail=(
            "ChromaDB < 0.4 uses the legacy DuckDB/parquet backend that "
            "is no longer tested upstream and will refuse to load in "
            "newer client versions. Migrate the collection to >=0.4."
        ),
    ),
    Advisory(
        package="pydantic",
        constraint="<2.0",
        severity="info",
        title="Pydantic 1.x with modern LangChain/OpenAI SDKs",
        detail=(
            "Recent LangChain and OpenAI SDK releases require Pydantic "
            "2.x. If this project pins <2.0 you may have fallen behind "
            "on dependent AI libs; review for incompatibilities."
        ),
    ),
    Advisory(
        package="gradio",
        constraint="<4.0",
        severity="warning",
        title="Gradio < 4.0",
        detail=(
            "Pre-4.0 Gradio has multiple published XSS and file-disclosure "
            "advisories (e.g. GHSA-3qqg-pgqq-3695). If a Gradio demo is "
            "shipped with this project, upgrade before exposing it."
        ),
    ),
)


# Pre-index by package name for fast lookup
_BY_PACKAGE: dict[str, list[Advisory]] = {}
for _a in ADVISORIES:
    _BY_PACKAGE.setdefault(_a.package, []).append(_a)


_RANGE_RE = re.compile(r"\s*(<=|>=|==|<|>|!=)\s*([\w.\-+]+)")


def _parse_version(text: str) -> tuple[int, ...]:
    """Return a tuple of integers for the leading numeric version parts.

    We intentionally ignore pre-release tags (``rc``, ``b1``, …) — the
    advisories we ship are all of the form "below x.y", so missing the
    odd beta-tag nuance is acceptable.
    """
    parts: list[int] = []
    for chunk in re.split(r"[.\-+]", text):
        m = re.match(r"(\d+)", chunk)
        if not m:
            break
        parts.append(int(m.group(1)))
    return tuple(parts) or (0,)


def _version_matches(version: str | None, constraint: str) -> bool:
    """Return True when ``version`` satisfies ``constraint``.

    Unknown or unpinned dependencies (version=None) never match — we
    don't flag a project merely because it lists a package. Only actual
    pinned versions count as evidence.
    """
    if not version:
        return False
    ver = _parse_version(version)
    if ver == (0,):
        return False
    for op, bound in _RANGE_RE.findall(constraint):
        target = _parse_version(bound)
        if op == "<" and not ver < target:
            return False
        if op == "<=" and not ver <= target:
            return False
        if op == ">" and not ver > target:
            return False
        if op == ">=" and not ver >= target:
            return False
        if op == "==" and not ver == target:
            return False
        if op == "!=" and not ver != target:
            return False
    return True


_DEP_LINE_RE = re.compile(
    r"^\s*([A-Za-z0-9_.\-/@]+)\s*(?:\[[^\]]*\])?\s*"
    r"(?:(?:==|>=|<=|~=|!=|===)\s*([\w.\-+]+))?"
)


def parse_dep_line(line: str) -> tuple[str, str | None]:
    """Parse a single requirements line into (package, version-or-None)."""
    line = line.split(";", 1)[0].split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        return "", None
    m = _DEP_LINE_RE.match(line)
    if not m:
        return "", None
    name = m.group(1).lower().replace("_", "-")
    # Handle `pkg@1.2.3` npm-style
    if "@" in name and not name.startswith("@"):
        name, _, tail = name.partition("@")
        return name, tail or None
    version = m.group(2)
    return name, version


def find_advisories(dep_lines: list[str]) -> list[Advisory]:
    """Resolve every dependency line against the offline advisory KB.

    Duplicate hits (same advisory fired by two requirements files for
    the same project) are collapsed so the report lists each issue once.
    """
    hits: list[Advisory] = []
    seen: set[tuple[str, str]] = set()
    for line in dep_lines:
        pkg, version = parse_dep_line(line)
        if not pkg:
            continue
        for advisory in _BY_PACKAGE.get(pkg, ()):
            if _version_matches(version, advisory.constraint):
                key = (advisory.package, advisory.title)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(advisory)
    return hits
