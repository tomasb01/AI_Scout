"""QA pipeline — ties insights, linter and fact strips together (Sprint 0.2).

One entry point, ``prepare_qa``, used by both the HTML and JSON
generators so every output shares the same validated numbers, the same
rendered sentences and the same suppression decisions:

    data model → validate invariants → render templates → lint → degrade

Facts are the default, sentences the add-on (P-3): the per-solution
detail always has a fact strip built from controlled vocabulary; prose
summaries are displayed only when they carry information (LLM mode, not
duplicated) and pass the linter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aiscout.engine.enrichment import AssetInsight, _asset_is_mcp_server
from aiscout.knowledge.providers import get_provider
from aiscout.models import AIAsset
from aiscout.report.insights import Insight, InsightStats, build_insights, collect_stats
from aiscout.report.linter import QAReport, lint_duplicate_summaries, lint_text
from aiscout.report.qa_vocab import (
    PATTERN_FALLBACK,
    PATTERN_LABELS,
    PATTERN_MCP_SERVER,
    SINK_TYPE_LABELS,
    SOURCE_TYPE_LABELS,
    UNCLASSIFIED,
)


@dataclass
class FactStrip:
    """Structured facts for one solution (QA spec §1.4).

    Every value comes from a finite vocabulary or the Provider KB —
    never a raw string from the scanned repository. Where no label
    exists, the strip says ``unclassified`` and the evidence fields
    carry the raw detail.
    """

    sources: list[str] = field(default_factory=list)
    sinks: list[str] = field(default_factory=list)
    pattern: str = PATTERN_FALLBACK
    tech: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sources": self.sources,
            "sinks": self.sinks,
            "pattern": self.pattern,
            "tech": self.tech,
        }


@dataclass
class QAResult:
    """Everything the generators need from the QA layer."""

    insights: list[Insight] = field(default_factory=list)
    qa_report: QAReport = field(default_factory=QAReport)
    fact_strips: dict[str, FactStrip] = field(default_factory=dict)
    # asset id → prose summary to display, or None → render the fact strip
    summary_display: dict[str, str | None] = field(default_factory=dict)
    # asset id → recommendations that passed the linter
    recommendations_display: dict[str, list[str]] = field(default_factory=dict)
    stats: InsightStats | None = None

    @property
    def exec_insights(self) -> list[Insight]:
        return [i for i in self.insights if not i.suppressed]

    def counts(self) -> dict:
        return self.qa_report.counts()


def build_fact_strip(asset: AIAsset, insight: AssetInsight | None) -> FactStrip:
    """Build the controlled-vocabulary fact strip for one solution."""
    strip = FactStrip()

    flow = asset.data_flow
    if flow:
        seen: set[str] = set()
        for src in flow.sources:
            label = SOURCE_TYPE_LABELS.get(src.type, UNCLASSIFIED)
            if label not in seen:
                seen.add(label)
                strip.sources.append(label)
        seen = set()
        for sink in flow.sinks:
            if sink.provider:
                profile = get_provider(sink.provider)
                residency = "/".join(profile.data_residency) or "unknown"
                label = f"{profile.display_name} ({residency})"
            else:
                label = SINK_TYPE_LABELS.get(sink.type, UNCLASSIFIED)
            if label not in seen:
                seen.add(label)
                strip.sinks.append(label)
    if not strip.sources:
        strip.sources = [UNCLASSIFIED]
    if not strip.sinks:
        if asset.provider:
            profile = get_provider(asset.provider.name)
            residency = "/".join(profile.data_residency) or "unknown"
            strip.sinks = [f"{profile.display_name} ({residency})"]
        else:
            strip.sinks = [UNCLASSIFIED]

    tags = set(asset.tags)
    if "mcp" in tags and _asset_is_mcp_server(asset):
        strip.pattern = PATTERN_MCP_SERVER
    else:
        for tag, label in PATTERN_LABELS:
            if tag in tags:
                strip.pattern = label
                break

    if insight:
        strip.tech = list(insight.tech_stack)

    return strip


def prepare_qa(
    assets: list[AIAsset],
    asset_insights: dict[str, AssetInsight],
    *,
    repos: int = 1,
    files_scanned: int = 0,
    overlap_group_sizes: list[int] | None = None,
    delta: dict | None = None,
) -> QAResult:
    """Run the full QA pipeline over enriched assets."""
    result = QAResult()

    # ── Data layer: stats + invariants + rendered insight catalog ──────
    stats = collect_stats(
        assets, asset_insights,
        repos=repos, files_scanned=files_scanned,
        overlap_group_sizes=overlap_group_sizes,
    )
    result.stats = stats
    result.insights = build_insights(stats, delta=delta)

    # ── Lint insight sentences; ERROR ⇒ the sentence never renders ─────
    kb_safe_tokens = sorted(stats.known_labels) + [stats.top_author]
    for insight in result.insights:
        issues = lint_text(
            insight.text,
            kind="insight",
            template_id=insight.template_id,
            entity_id=insight.id,
            safe_tokens=kb_safe_tokens + insight.safe_tokens(),
        )
        if any(i.severity == "ERROR" for i in issues):
            insight.suppressed = True
        result.qa_report.issues.extend(issues)

    # ── Fact strips (always built — the no-LLM default detail view) ────
    for asset in assets:
        result.fact_strips[asset.id] = build_fact_strip(
            asset, asset_insights.get(asset.id)
        )

    # ── Summaries: prose only when it carries information ──────────────
    # P-3: without LLM classification the detail shows facts, not a
    # rule-based pseudo-summary — so only LLM-mode summaries are display
    # candidates. L-08 runs over exactly those: a sentence duplicated
    # across solutions is a pseudo-summary and every occurrence degrades
    # to the fact strip.
    candidate_summaries = {
        a.id: asset_insights[a.id].summary
        for a in assets
        if a.id in asset_insights
        and a.data_classification
        and asset_insights[a.id].summary
    }
    dup_issues, degrade_ids = lint_duplicate_summaries(candidate_summaries)
    result.qa_report.issues.extend(dup_issues)

    for asset in assets:
        summary = candidate_summaries.get(asset.id, "")
        show_prose = bool(summary) and asset.id not in degrade_ids
        if show_prose:
            # LLM prose legitimately references code identifiers, so the
            # L-07 code-leak heuristics do not apply (it is marked with
            # LLM provenance in the report); composition errors still
            # degrade it to the fact strip like any other sentence.
            issues = lint_text(
                summary, kind="insight", entity_id=asset.id,
                safe_tokens=kb_safe_tokens, apply_code_leak=False,
            )
            result.qa_report.issues.extend(issues)
            if any(i.severity == "ERROR" for i in issues):
                show_prose = False
        result.summary_display[asset.id] = summary if show_prose else None

    # ── Recommendations (action sentences) ──────────────────────────────
    for asset in assets:
        insight = asset_insights.get(asset.id)
        if not insight:
            result.recommendations_display[asset.id] = []
            continue
        kept: list[str] = []
        for rec in insight.recommendations:
            issues = lint_text(
                rec, kind="action", entity_id=asset.id,
                safe_tokens=kb_safe_tokens,
            )
            result.qa_report.issues.extend(issues)
            if not any(i.severity == "ERROR" for i in issues):
                kept.append(rec)
        result.recommendations_display[asset.id] = kept

    return result
