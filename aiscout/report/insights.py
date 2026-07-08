"""Typed insight catalog I-01–I-10 for the report QA layer (Sprint 0.2).

QA spec §1: executive-summary sentences are never concatenated by hand.
Each insight is a typed record (metrics + entities from safe domains)
rendered through a versioned ICU MessageFormat template. All arithmetic
happens in the data layer (`build_insights`), which is guarded by the
invariants in `validate_invariants` — templates only render values (P-2).

ICU note: the spec suggests PyICU/babel. PyICU is a C extension against a
system ICU library — a heavy, platform-fragile dependency for a tool that
must install offline in 10 minutes. We instead ship a small deterministic
renderer for the ICU subset the catalog uses (`plural` with =N/one/other,
`select`, `#`, `{var, date, medium}`). The full template corpus is
exercised by edge-case tests, so any divergence from ICU semantics would
surface as a test failure, not a broken report sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from aiscout.knowledge.providers import get_provider
from aiscout.models import AIAsset, RiskStatus
from aiscout.report.qa_vocab import (
    CRITICAL_REASON_LABELS,
    SENSITIVE_DATA_LABELS,
)


class InvariantViolation(ValueError):
    """A data-layer invariant failed — an analysis bug, not a text bug.

    QA spec §3: raised before any template renders; the scan fails hard
    because the numbers themselves cannot be trusted.
    """


# ── ICU MessageFormat subset renderer ──────────────────────────────────────


def format_icu(template: str, values: dict) -> str:
    """Render an ICU MessageFormat template.

    Supported subset: ``{var}``, ``{var, plural, =N {...} one {...}
    other {...}}`` (with ``#``), ``{var, select, key {...} other {...}}``,
    ``{var, date, medium}``. Raises ``KeyError`` on a missing variable and
    ``ValueError`` on a malformed template — a template bug must fail in
    tests, never render half a sentence.
    """
    out: list[str] = []
    i = 0
    while i < len(template):
        ch = template[i]
        if ch != "{":
            out.append(ch)
            i += 1
            continue
        body, i = _read_braced(template, i)
        out.append(_render_argument(body, values))
    return "".join(out)


def _read_braced(text: str, start: int) -> tuple[str, int]:
    """Read a balanced ``{...}`` block starting at ``start``.

    Returns (inner content, index after the closing brace).
    """
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:j], j + 1
    raise ValueError(f"Unbalanced braces in template: {text!r}")


def _render_argument(body: str, values: dict) -> str:
    parts = body.split(",", 2)
    name = parts[0].strip()
    if name not in values:
        raise KeyError(f"Missing template variable: {name!r}")
    value = values[name]

    if len(parts) == 1:
        return _format_value(value)

    kind = parts[1].strip()
    if kind == "date":
        return _format_date(value)
    if kind in ("plural", "select"):
        if len(parts) < 3:
            raise ValueError(f"{kind} argument without options: {body!r}")
        options = _parse_options(parts[2])
        return _select_option(kind, value, options, values)
    raise ValueError(f"Unsupported ICU argument type: {kind!r}")


def _parse_options(text: str) -> dict[str, str]:
    """Parse ``key {value} key {value} ...`` option lists."""
    options: dict[str, str] = {}
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        brace = text.find("{", i)
        if brace == -1:
            raise ValueError(f"Malformed ICU options: {text!r}")
        key = text[i:brace].strip()
        if not key:
            raise ValueError(f"Empty option key in: {text!r}")
        value, i = _read_braced(text, brace)
        options[key] = value
    return options


def _select_option(
    kind: str, value, options: dict[str, str], values: dict
) -> str:
    if kind == "plural":
        exact = f"={value}"
        if exact in options:
            chosen = options[exact]
        elif value == 1 and "one" in options:
            chosen = options["one"]
        else:
            chosen = options["other"]
        # `#` substitutes the formatted number (top level only, not in
        # nested arguments — the catalog never nests inside plural).
        chosen = chosen.replace("#", _format_value(value))
    else:  # select
        chosen = options.get(str(value), options["other"])
    return format_icu(chosen, values)


def _format_value(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


_MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def _format_date(value) -> str:
    """ICU ``date, medium`` — rendered manually so the output does not
    depend on platform strftime quirks (determinism requirement)."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, (date, datetime)):
        raise ValueError(f"Not a date: {value!r}")
    return f"{_MONTHS[value.month - 1]} {value.day}, {value.year}"


# ── Insight model ──────────────────────────────────────────────────────────


@dataclass
class Insight:
    """One typed insight (QA spec §1.1)."""

    id: str
    type: str
    severity: str  # info | warning | critical
    metrics: dict = field(default_factory=dict)
    entities: dict = field(default_factory=dict)
    provenance: str = "deterministic"
    template_id: str = ""
    text: str = ""  # rendered sentence (filled by build_insights)
    suppressed: bool = False  # set by the linter on ERROR

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "severity": self.severity,
            "metrics": self.metrics,
            "entities": self.entities,
            "provenance": self.provenance,
            "template_id": self.template_id,
            "text": self.text,
            "suppressed": self.suppressed,
        }

    def safe_tokens(self) -> list[str]:
        """Entity values from safe domains (git identities, KB labels) —
        the linter masks these before applying code-leak heuristics."""
        tokens: list[str] = []
        for v in self.entities.values():
            if isinstance(v, str):
                tokens.append(v)
            elif isinstance(v, (list, tuple)):
                tokens.extend(str(x) for x in v)
        return tokens


# ── Template catalog (versioned: text change ⇒ new version suffix) ─────────

TEMPLATES: dict[str, str] = {
    "T-INVENTORY-v2": (
        "Found {total, plural, =0 {no AI solutions} one {# AI solution} "
        "other {# AI solutions}} across {repos, plural, one {# repository} "
        "other {# repositories}} ({files, plural, one {# file} "
        "other {# files}} scanned)."
    ),
    "T-CRITICAL-v1": (
        "{critical_count, plural, one {# solution requires} "
        "other {# solutions require}} immediate attention: {reasons}."
    ),
    "T-EGRESS-v1": (
        "{egress_count, plural, one {# solution sends} "
        "other {# solutions send}} data to {region}-based providers "
        "({provider_list}) — verify DPA and data-residency requirements."
    ),
    "T-AUTHOR-CONC-v1": (
        "{top_author_pct, select, 100 {A single contributor ({author}) "
        "created all {total} solutions} other {One contributor ({author}) "
        "created {top_author_count} of {total} solutions "
        "({top_author_pct}%)}} — single-point-of-failure risk."
    ),
    "T-DEP-CONC-v1": (
        "Highest dependency: {tech}, used by {tech_count} of {total} "
        "solutions ({tech_pct}%)."
    ),
    "T-OVERLAP-v1": (
        "{overlap_solutions} solutions functionally overlap in "
        "{group_count, plural, one {# capability area} "
        "other {# capability areas}} — consolidation opportunity."
    ),
    "T-DATA-CAT-v1": (
        "{cat_count, plural, one {# solution processes} "
        "other {# solutions process}} {category} — elevated compliance "
        "attention recommended."
    ),
    "T-UNKNOWN-PROV-v1": (
        "{candidate_count, plural, one {# possible AI integration} "
        "other {# possible AI integrations}} could not be matched to a "
        "known provider — listed under “Manual review”."
    ),
    "T-SCAN-DELTA-v1": (
        "Since {prev_date, date, medium}: {added, plural, "
        "=0 {no new solutions} one {# new solution} "
        "other {# new solutions}}, {removed, plural, =0 {none removed} "
        "one {# removed} other {# removed}}{new_providers, plural, =0 {} "
        "one {, # new provider} other {, # new providers}}."
    ),
    "T-LOCAL-ONLY-v1": (
        "{local_count, plural, one {# solution runs} "
        "other {# solutions run}} fully local (no data egress)."
    ),
}


# ── Data layer ─────────────────────────────────────────────────────────────


def pct(count: int, total: int) -> int:
    """THE percentage function — every pct in the report uses this one
    denominator discipline (kills the '1 developer created over 100%'
    class of bugs, QA spec §3.1)."""
    if total <= 0:
        return 0
    return round(100 * count / total)


def join_oxford(items: list[str]) -> str:
    """Oxford-comma joiner — list composition happens in the data layer,
    never in a template (QA spec I-02)."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def provider_list_label(names: list[str], limit: int = 3) -> str:
    """Max ``limit`` provider names + 'and N more' (data layer, I-03)."""
    if len(names) <= limit:
        return ", ".join(names)
    return f"{', '.join(names[:limit])} and {len(names) - limit} more"


@dataclass
class InsightStats:
    """Validated numbers the insight catalog renders from.

    Built by ``collect_stats`` from the asset list; every field is
    checked by ``validate_invariants`` before any template runs.
    """

    total: int = 0
    repos: int = 1
    files: int = 0
    status_counts: dict = field(default_factory=dict)
    category_counts: dict = field(default_factory=dict)
    critical_count: int = 0
    critical_reasons: list[str] = field(default_factory=list)
    egress_count: int = 0
    egress_region: str = "US"
    egress_providers: list[str] = field(default_factory=list)
    top_author: str = ""
    top_author_count: int = 0
    top_tech: str = ""
    top_tech_count: int = 0
    overlap_solutions: int = 0
    overlap_group_sizes: list[int] = field(default_factory=list)
    sensitive_counts: dict = field(default_factory=dict)
    candidate_count: int = 0
    local_count: int = 0
    local_ids: frozenset = frozenset()
    egress_ids: frozenset = frozenset()
    known_labels: frozenset = frozenset()


def collect_stats(
    assets: list[AIAsset],
    asset_insights: dict,
    *,
    repos: int = 1,
    files_scanned: int = 0,
    overlap_group_sizes: list[int] | None = None,
) -> InsightStats:
    """Compute all insight metrics from one place, one denominator."""
    stats = InsightStats(
        total=len(assets),
        repos=max(1, repos),
        files=files_scanned,
        overlap_group_sizes=sorted(overlap_group_sizes or [], reverse=True),
    )
    stats.overlap_solutions = sum(stats.overlap_group_sizes)

    status_counts = {s.value: 0 for s in RiskStatus}
    category_counts: dict[str, int] = {}
    author_counts: dict[str, int] = {}
    tech_counts: dict[str, int] = {}
    sensitive_counts: dict[str, int] = {}
    critical_kinds: list[str] = []
    egress_providers: set[str] = set()
    egress_ids: set[str] = set()
    local_ids: set[str] = set()
    known_labels: set[str] = set()
    candidate_count = 0

    for asset in assets:
        status_counts[asset.risk_status.value] += 1
        insight = asset_insights.get(asset.id)
        category = (
            insight.category if insight and insight.category
            else "Other AI Solutions"
        )
        category_counts[category] = category_counts.get(category, 0) + 1
        known_labels.add(category)

        if asset.owner and asset.owner != "unknown":
            for author in asset.owner.split(","):
                author = author.strip()
                if author:
                    author_counts[author] = author_counts.get(author, 0) + 1

        if insight:
            for tech in insight.tech_stack:
                tech_counts[tech] = tech_counts.get(tech, 0) + 1
                known_labels.add(tech)
            for label in insight.data_involved:
                if label in SENSITIVE_DATA_LABELS:
                    sensitive_counts[label] = sensitive_counts.get(label, 0) + 1
            if asset.risk_status == RiskStatus.CRITICAL:
                critical_kinds.extend(
                    _reason_kind(r.title)
                    for r in insight.risk_reasons
                    if r.severity == "critical"
                )

        profile = None
        if asset.provider:
            profile = get_provider(asset.provider.name)
        if profile and profile.category == "llm_api" and any(
            "US" in r for r in profile.data_residency
        ):
            egress_ids.add(asset.id)
            egress_providers.add(profile.display_name)
            known_labels.add(profile.display_name)
        elif profile and profile.category == "local_runtime":
            local_ids.add(asset.id)

        # Heuristic safety-net findings (confidence < 1.0) — candidates
        # for manual review, never counted into deterministic totals.
        if asset.raw_findings and all(
            f.confidence < 1.0 for f in asset.raw_findings
        ):
            candidate_count += 1

    stats.status_counts = status_counts
    stats.category_counts = category_counts
    stats.critical_count = status_counts[RiskStatus.CRITICAL.value]
    stats.critical_reasons = _reason_labels(critical_kinds)
    stats.egress_ids = frozenset(egress_ids)
    stats.egress_count = len(egress_ids)
    stats.egress_providers = sorted(egress_providers)
    stats.local_ids = frozenset(local_ids)
    stats.local_count = len(local_ids)
    stats.candidate_count = candidate_count
    stats.sensitive_counts = dict(sorted(sensitive_counts.items()))
    stats.known_labels = frozenset(known_labels)

    if author_counts:
        top = sorted(author_counts.items(), key=lambda x: (-x[1], x[0]))[0]
        stats.top_author, stats.top_author_count = top
    if tech_counts:
        top = sorted(tech_counts.items(), key=lambda x: (-x[1], x[0]))[0]
        stats.top_tech, stats.top_tech_count = top

    return stats


def _reason_kind(title: str) -> str:
    lowered = title.lower()
    if "api key" in lowered:
        return "hardcoded_api_key"
    if "training" in lowered and ("personal" in lowered or "pii" in lowered):
        return "pii_training_risk"
    if "secret" in lowered:
        return "secrets_in_config"
    return "other"


def _reason_labels(kinds: list[str]) -> list[str]:
    """Map reason kinds → finite vocabulary, deduped, catalog order."""
    present = set(kinds)
    return [
        label
        for kind, label in CRITICAL_REASON_LABELS.items()
        if kind in present
    ]


def validate_invariants(stats: InsightStats) -> None:
    """QA spec §3 — hard checks before any template renders."""
    s = stats
    if s.total < 0 or s.files < 0 or s.repos < 1:
        raise InvariantViolation(
            f"Counts out of range: total={s.total} files={s.files} repos={s.repos}"
        )
    if sum(s.status_counts.values()) != s.total:
        raise InvariantViolation(
            f"Status counts {s.status_counts} do not sum to total {s.total}"
        )
    if s.category_counts and sum(s.category_counts.values()) != s.total:
        raise InvariantViolation(
            f"Category counts {s.category_counts} do not sum to total {s.total}"
        )
    subsets = {
        "critical_count": s.critical_count,
        "egress_count": s.egress_count,
        "top_author_count": s.top_author_count,
        "top_tech_count": s.top_tech_count,
        "overlap_solutions": s.overlap_solutions,
        "local_count": s.local_count,
        **{f"sensitive[{k}]": v for k, v in s.sensitive_counts.items()},
    }
    for name, value in subsets.items():
        if value < 0 or value > s.total:
            raise InvariantViolation(
                f"Subset count {name}={value} outside [0, {s.total}]"
            )
    for name, value in (
        ("top_author_pct", pct(s.top_author_count, s.total)),
        ("top_tech_pct", pct(s.top_tech_count, s.total)),
    ):
        if not 0 <= value <= 100:
            raise InvariantViolation(f"{name}={value} outside [0, 100]")
    if s.critical_count > 0 and s.status_counts and not s.critical_reasons:
        raise InvariantViolation(
            "critical_count > 0 but no critical reasons collected"
        )
    if any(size < 2 for size in s.overlap_group_sizes):
        raise InvariantViolation(
            f"Overlap group with < 2 solutions: {s.overlap_group_sizes}"
        )
    if s.local_ids & s.egress_ids:
        raise InvariantViolation(
            f"local ∩ egress not empty: {sorted(s.local_ids & s.egress_ids)}"
        )


def build_insights(
    stats: InsightStats,
    delta: dict | None = None,
) -> list[Insight]:
    """Render the insight catalog from validated stats.

    ``delta`` (I-09) arrives once diff mode lands (Sprint 2): a dict with
    ``prev_date``, ``added``, ``removed``, ``new_providers``.
    """
    validate_invariants(stats)
    insights: list[Insight] = []

    def add(insight: Insight, values: dict) -> None:
        insight.text = format_icu(TEMPLATES[insight.template_id], values)
        insights.append(insight)

    # I-01 · INVENTORY_TOTAL — always, even at 0 (coverage statement)
    add(
        Insight(
            id="I-01", type="INVENTORY_TOTAL", severity="info",
            metrics={
                "total": stats.total, "repos": stats.repos,
                "files": stats.files,
            },
            template_id="T-INVENTORY-v2",
        ),
        {"total": stats.total, "repos": stats.repos, "files": stats.files},
    )

    # I-02 · CRITICAL_FINDINGS
    if stats.critical_count > 0:
        add(
            Insight(
                id="I-02", type="CRITICAL_FINDINGS", severity="critical",
                metrics={
                    "critical_count": stats.critical_count,
                    "total": stats.total,
                },
                entities={"reasons": stats.critical_reasons},
                template_id="T-CRITICAL-v1",
            ),
            {
                "critical_count": stats.critical_count,
                "reasons": join_oxford(stats.critical_reasons),
            },
        )

    # I-03 · DATA_EGRESS_REGION
    if stats.egress_count > 0:
        add(
            Insight(
                id="I-03", type="DATA_EGRESS_REGION", severity="warning",
                metrics={
                    "egress_count": stats.egress_count,
                    "total": stats.total,
                },
                entities={
                    "region": stats.egress_region,
                    "providers": stats.egress_providers,
                },
                template_id="T-EGRESS-v1",
            ),
            {
                "egress_count": stats.egress_count,
                "region": stats.egress_region,
                "provider_list": provider_list_label(stats.egress_providers),
            },
        )

    # I-04 · AUTHOR_CONCENTRATION (SPOF) — concentration is meaningless
    # for a single solution, so total ≥ 2 (also keeps "all 1 solutions"
    # out of the 100% template branch).
    top_author_pct = pct(stats.top_author_count, stats.total)
    if stats.total >= 2 and stats.top_author and top_author_pct >= 50:
        add(
            Insight(
                id="I-04", type="AUTHOR_CONCENTRATION", severity="warning",
                metrics={
                    "top_author_pct": top_author_pct,
                    "top_author_count": stats.top_author_count,
                    "total": stats.total,
                },
                entities={"author": stats.top_author},
                template_id="T-AUTHOR-CONC-v1",
            ),
            {
                "top_author_pct": top_author_pct,
                "top_author_count": stats.top_author_count,
                "total": stats.total,
                "author": stats.top_author,
            },
        )

    # I-05 · DEPENDENCY_CONCENTRATION — total ≥ 2 for the same reason
    # as I-04 ("used by 1 of 1 solutions" is noise, not an insight).
    top_tech_pct = pct(stats.top_tech_count, stats.total)
    if stats.total >= 2 and stats.top_tech and top_tech_pct >= 40:
        add(
            Insight(
                id="I-05", type="DEPENDENCY_CONCENTRATION", severity="info",
                metrics={
                    "tech_pct": top_tech_pct,
                    "tech_count": stats.top_tech_count,
                    "total": stats.total,
                },
                entities={"tech": stats.top_tech},
                template_id="T-DEP-CONC-v1",
            ),
            {
                "tech": stats.top_tech,
                "tech_count": stats.top_tech_count,
                "total": stats.total,
                "tech_pct": top_tech_pct,
            },
        )

    # I-06 · OVERLAP_GROUPS
    if stats.overlap_solutions >= 2 and stats.overlap_group_sizes:
        add(
            Insight(
                id="I-06", type="OVERLAP_GROUPS", severity="info",
                metrics={
                    "overlap_solutions": stats.overlap_solutions,
                    "group_count": len(stats.overlap_group_sizes),
                    "total": stats.total,
                },
                template_id="T-OVERLAP-v1",
            ),
            {
                "overlap_solutions": stats.overlap_solutions,
                "group_count": len(stats.overlap_group_sizes),
            },
        )

    # I-07 · DATA_CATEGORY_VOLUME — one sentence per sensitive category
    for label, count in stats.sensitive_counts.items():
        if count > 0:
            add(
                Insight(
                    id="I-07", type="DATA_CATEGORY_VOLUME",
                    severity="warning",
                    metrics={"cat_count": count, "total": stats.total},
                    entities={"category": label},
                    template_id="T-DATA-CAT-v1",
                ),
                {"cat_count": count, "category": label},
            )

    # I-08 · UNKNOWN_PROVIDER_CANDIDATES
    if stats.candidate_count > 0:
        add(
            Insight(
                id="I-08", type="UNKNOWN_PROVIDER_CANDIDATES",
                severity="info",
                metrics={"candidate_count": stats.candidate_count},
                template_id="T-UNKNOWN-PROV-v1",
            ),
            {"candidate_count": stats.candidate_count},
        )

    # I-09 · SCAN_DELTA — only with a previous scan of the same scope
    if delta is not None:
        add(
            Insight(
                id="I-09", type="SCAN_DELTA", severity="info",
                metrics={
                    "added": delta["added"],
                    "removed": delta["removed"],
                    "new_providers": delta["new_providers"],
                },
                entities={"prev_date": str(delta["prev_date"])},
                template_id="T-SCAN-DELTA-v1",
            ),
            {
                "prev_date": delta["prev_date"],
                "added": delta["added"],
                "removed": delta["removed"],
                "new_providers": delta["new_providers"],
            },
        )

    # I-10 · LOCAL_ONLY_SHARE — positive insight
    if stats.local_count > 0:
        add(
            Insight(
                id="I-10", type="LOCAL_ONLY_SHARE", severity="info",
                metrics={
                    "local_count": stats.local_count, "total": stats.total,
                },
                template_id="T-LOCAL-ONLY-v1",
            ),
            {"local_count": stats.local_count},
        )

    return insights


# Import-time typo check: unbalanced braces in a template must fail at
# import, not at render time in a customer scan.
for _tid, _tpl in TEMPLATES.items():
    if _tpl.count("{") != _tpl.count("}"):
        raise AssertionError(f"Unbalanced template {_tid}")
