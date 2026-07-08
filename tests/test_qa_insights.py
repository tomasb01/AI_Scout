"""Tests for the typed insight catalog I-01–I-10 (Sprint 0.2, QA spec §1).

Every template is exercised over its mandatory edge cases (0/1/many,
pct = 100, empty and long lists) and every rendered sentence must pass
the report linter with zero errors — the cartesian safety net demanded
by QA spec §4.1.
"""

from datetime import date

import pytest

from aiscout.models import (
    AIAsset,
    ClassificationResult,
    Confidence,
    DataCategory,
    Finding,
    FindingType,
    ProviderInfo,
    RiskStatus,
)
from aiscout.engine.enrichment import enrich_assets
from aiscout.report.insights import (
    InsightStats,
    InvariantViolation,
    TEMPLATES,
    build_insights,
    collect_stats,
    format_icu,
    join_oxford,
    pct,
    provider_list_label,
    validate_invariants,
)
from aiscout.report.linter import lint_text


def _lint_clean(text: str, safe_tokens: list[str] | None = None) -> None:
    """Assert a rendered sentence produces no linter ERRORs."""
    issues = lint_text(text, kind="insight", safe_tokens=safe_tokens or [])
    errors = [i for i in issues if i.severity == "ERROR"]
    assert not errors, f"{text!r} -> {[e.message for e in errors]}"


def render(template_id: str, values: dict) -> str:
    return format_icu(TEMPLATES[template_id], values)


# ── ICU renderer ───────────────────────────────────────────────────────────


def test_icu_plain_variable():
    assert format_icu("Hello {name}.", {"name": "world"}) == "Hello world."


def test_icu_plural_exact_one_other():
    tpl = "{n, plural, =0 {none} one {# item} other {# items}}"
    assert format_icu(tpl, {"n": 0}) == "none"
    assert format_icu(tpl, {"n": 1}) == "1 item"
    assert format_icu(tpl, {"n": 5}) == "5 items"


def test_icu_select_with_nested_variables():
    tpl = "{p, select, 100 {all {total}} other {{count} of {total}}}"
    assert format_icu(tpl, {"p": 100, "total": 7, "count": 7}) == "all 7"
    assert format_icu(tpl, {"p": 60, "total": 5, "count": 3}) == "3 of 5"


def test_icu_date_medium_is_platform_independent():
    assert format_icu("{d, date, medium}", {"d": date(2026, 7, 8)}) == "Jul 8, 2026"
    assert format_icu("{d, date, medium}", {"d": "2026-01-30"}) == "Jan 30, 2026"


def test_icu_missing_variable_raises():
    with pytest.raises(KeyError):
        format_icu("Hello {name}.", {})


def test_icu_unbalanced_braces_raise():
    with pytest.raises(ValueError):
        format_icu("{n, plural, one {x", {"n": 1})


# ── Data-layer helpers ─────────────────────────────────────────────────────


def test_pct_single_denominator():
    assert pct(144, 144) == 100
    assert pct(1, 3) == 33
    assert pct(0, 10) == 0
    assert pct(5, 0) == 0  # degenerate, never > 100


def test_join_oxford():
    assert join_oxford([]) == ""
    assert join_oxford(["a"]) == "a"
    assert join_oxford(["a", "b"]) == "a and b"
    assert join_oxford(["a", "b", "c"]) == "a, b, and c"


def test_provider_list_label_caps_at_three():
    assert provider_list_label(["A"]) == "A"
    assert provider_list_label(["A", "B", "C"]) == "A, B, C"
    assert provider_list_label(["A", "B", "C", "D", "E"]) == "A, B, C and 2 more"


# ── I-01 INVENTORY_TOTAL ───────────────────────────────────────────────────


@pytest.mark.parametrize("total", [0, 1, 2, 144])
@pytest.mark.parametrize("repos", [1, 2])
@pytest.mark.parametrize("files", [1, 42])
def test_i01_edge_cases(total, repos, files):
    text = render("T-INVENTORY-v2", {"total": total, "repos": repos, "files": files})
    _lint_clean(text)
    if total == 0:
        # 0 findings must still yield a meaningful coverage statement
        assert "no AI solutions" in text
    elif total == 1:
        assert "1 AI solution " in text
    else:
        assert f"{total} AI solutions" in text
    assert ("1 repository" in text) if repos == 1 else ("2 repositories" in text)
    assert ("1 file scanned" in text) if files == 1 else ("42 files scanned" in text)


# ── I-02 CRITICAL_FINDINGS ─────────────────────────────────────────────────


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("n_reasons", [1, 2, 3])
def test_i02_edge_cases(count, n_reasons):
    reasons = ["hardcoded API keys", "secrets in configuration",
               "other critical findings"][:n_reasons]
    text = render("T-CRITICAL-v1", {
        "critical_count": count, "reasons": join_oxford(reasons),
    })
    _lint_clean(text)
    if count == 1:
        assert "1 solution requires" in text
    else:
        assert "2 solutions require" in text
    if n_reasons == 3:
        assert ", and " in text  # Oxford comma from the data layer


# ── I-03 DATA_EGRESS_REGION ────────────────────────────────────────────────


@pytest.mark.parametrize("count", [1, 56])
@pytest.mark.parametrize("n_providers", [1, 3, 5])
@pytest.mark.parametrize("region", ["US", "outside-EU"])
def test_i03_edge_cases(count, n_providers, region):
    providers = ["OpenAI", "Anthropic", "Google AI", "Cohere", "Groq"][:n_providers]
    text = render("T-EGRESS-v1", {
        "egress_count": count, "region": region,
        "provider_list": provider_list_label(providers),
    })
    _lint_clean(text, safe_tokens=providers)
    if n_providers == 5:
        assert "and 2 more" in text
    assert ("1 solution sends" in text) if count == 1 else (f"{count} solutions send" in text)


# ── I-04 AUTHOR_CONCENTRATION — pct=100 is the mandatory test ──────────────


@pytest.mark.parametrize("pct_value,count,total", [(50, 2, 4), (99, 143, 144), (100, 144, 144)])
def test_i04_edge_cases(pct_value, count, total):
    text = render("T-AUTHOR-CONC-v1", {
        "top_author_pct": pct_value, "top_author_count": count,
        "total": total, "author": "lukaskellerstein",
    })
    _lint_clean(text, safe_tokens=["lukaskellerstein"])
    assert "single-point-of-failure risk" in text
    if pct_value == 100:
        # The "over 100%" bug class is structurally impossible here
        assert f"A single contributor (lukaskellerstein) created all {total} solutions" in text
        assert "%" not in text
    else:
        assert f"({pct_value}%)" in text
        assert "over" not in text


def test_i04_pct_over_100_dies_in_validation_never_in_template():
    stats = InsightStats(
        total=86, top_author="dev", top_author_count=144,
        status_counts={"critical": 0, "review": 0, "no_findings": 86},
    )
    with pytest.raises(InvariantViolation):
        validate_invariants(stats)


# ── I-05 DEPENDENCY_CONCENTRATION ──────────────────────────────────────────


@pytest.mark.parametrize("count,total", [(2, 5), (51, 100), (10, 10)])
def test_i05_edge_cases(count, total):
    text = render("T-DEP-CONC-v1", {
        "tech": "LangChain", "tech_count": count, "total": total,
        "tech_pct": pct(count, total),
    })
    _lint_clean(text, safe_tokens=["LangChain"])
    assert f"used by {count} of {total} solutions" in text


# ── I-06 OVERLAP_GROUPS ────────────────────────────────────────────────────


@pytest.mark.parametrize("solutions,groups", [(2, 1), (58, 25)])
def test_i06_edge_cases(solutions, groups):
    text = render("T-OVERLAP-v1", {
        "overlap_solutions": solutions, "group_count": groups,
    })
    _lint_clean(text)
    if groups == 1:
        assert "1 capability area " in text
    else:
        assert f"{groups} capability areas" in text


def test_i06_group_of_one_never_generated():
    stats = InsightStats(
        total=5, overlap_group_sizes=[1],
        status_counts={"critical": 0, "review": 0, "no_findings": 5},
    )
    with pytest.raises(InvariantViolation):
        validate_invariants(stats)


# ── I-07 DATA_CATEGORY_VOLUME — cartesian over the whole label vocabulary ──


@pytest.mark.parametrize("count", [1, 18])
@pytest.mark.parametrize("category", [
    "Personal data / PII", "Financial data", "Credentials / Secrets", "Health data",
])
def test_i07_no_label_doubling(count, category):
    text = render("T-DATA-CAT-v1", {"cat_count": count, "category": category})
    _lint_clean(text, safe_tokens=[category])
    # "Financial data data" class of bug: the label carries the word
    # "data"; the template must not add another one.
    assert "data data" not in text
    assert category in text


# ── I-08 UNKNOWN_PROVIDER_CANDIDATES ───────────────────────────────────────


@pytest.mark.parametrize("count", [1, 7])
def test_i08_edge_cases(count):
    text = render("T-UNKNOWN-PROV-v1", {"candidate_count": count})
    _lint_clean(text)
    if count == 1:
        assert "1 possible AI integration " in text
    else:
        assert "7 possible AI integrations" in text


# ── I-09 SCAN_DELTA ────────────────────────────────────────────────────────


@pytest.mark.parametrize("added,removed,new_providers", [
    (0, 0, 0), (1, 0, 0), (5, 2, 1),
])
def test_i09_edge_cases(added, removed, new_providers):
    text = render("T-SCAN-DELTA-v1", {
        "prev_date": date(2026, 6, 1), "added": added,
        "removed": removed, "new_providers": new_providers,
    })
    _lint_clean(text)
    assert text.startswith("Since Jun 1, 2026:")
    if (added, removed, new_providers) == (0, 0, 0):
        assert "no new solutions" in text and "none removed" in text
    if new_providers == 1:
        assert "1 new provider" in text


def test_i09_not_generated_without_previous_scan():
    stats = InsightStats(
        total=0, status_counts={"critical": 0, "review": 0, "no_findings": 0},
    )
    insights = build_insights(stats, delta=None)
    assert all(i.id != "I-09" for i in insights)


# ── I-10 LOCAL_ONLY_SHARE ──────────────────────────────────────────────────


@pytest.mark.parametrize("count", [1, 10])
def test_i10_edge_cases(count):
    text = render("T-LOCAL-ONLY-v1", {"local_count": count})
    _lint_clean(text)
    if count == 1:
        assert "1 solution runs" in text
    else:
        assert "10 solutions run" in text


# ── Invariants (QA spec §3) ────────────────────────────────────────────────


def _valid_stats(**overrides) -> InsightStats:
    base = dict(
        total=4,
        status_counts={"critical": 1, "review": 1, "no_findings": 2},
        category_counts={"Chatbot & Conversation": 4},
        critical_count=1,
        critical_reasons=["hardcoded API keys"],
    )
    base.update(overrides)
    return InsightStats(**base)


def test_invariants_accept_consistent_stats():
    validate_invariants(_valid_stats())


def test_invariant_status_sum_must_match_total():
    with pytest.raises(InvariantViolation):
        validate_invariants(_valid_stats(
            status_counts={"critical": 1, "review": 1, "no_findings": 5},
        ))


def test_invariant_category_sum_must_match_total():
    with pytest.raises(InvariantViolation):
        validate_invariants(_valid_stats(
            category_counts={"RAG & Search": 1},
        ))


def test_invariant_subset_cannot_exceed_total():
    with pytest.raises(InvariantViolation):
        validate_invariants(_valid_stats(local_count=99))


def test_invariant_local_and_egress_disjoint():
    with pytest.raises(InvariantViolation):
        validate_invariants(_valid_stats(
            local_ids=frozenset({"a"}), egress_ids=frozenset({"a", "b"}),
            local_count=1, egress_count=2,
        ))


def test_invariant_critical_without_reasons_fails():
    with pytest.raises(InvariantViolation):
        validate_invariants(_valid_stats(critical_reasons=[]))


# ── collect_stats + build_insights over real assets ────────────────────────


def _asset(name: str, *, owner="unknown", provider=None, findings=None,
           classification=None) -> AIAsset:
    return AIAsset(
        name=name,
        owner=owner,
        provider=ProviderInfo(name=provider) if provider else None,
        repository="test-repo",
        file_path=f"{name}.py",
        raw_findings=findings or [],
        data_classification=classification,
    )


def _key_finding() -> Finding:
    return Finding(
        type=FindingType.API_KEY_DETECTED,
        file_path="app.py", line_number=3,
        content="sk-XXX", redacted_content="sk-...XXX",
        provider="openai",
    )


def test_single_contributor_fixture_renders_all_n_not_over_100():
    """QA spec acceptance: fixture '1 contributor' → 'created all N'."""
    assets = [
        _asset(f"sol{i}", owner="solodev", provider="openai")
        for i in range(3)
    ]
    insights = enrich_assets(assets)
    stats = collect_stats(assets, insights, repos=1, files_scanned=9)
    catalog = build_insights(stats)
    i04 = next(i for i in catalog if i.id == "I-04")
    assert "A single contributor (solodev) created all 3 solutions" in i04.text
    assert "over" not in i04.text
    assert i04.metrics["top_author_pct"] == 100


def test_critical_insight_uses_finite_reason_vocabulary():
    assets = [
        _asset("leaky", owner="dev", provider="openai",
               findings=[_key_finding()]),
        _asset("clean", owner="dev", provider="ollama"),
    ]
    insights = enrich_assets(assets)
    assert assets[0].risk_status == RiskStatus.CRITICAL
    stats = collect_stats(assets, insights, repos=1, files_scanned=2)
    catalog = build_insights(stats)
    i02 = next(i for i in catalog if i.id == "I-02")
    assert i02.text == (
        "1 solution requires immediate attention: hardcoded API keys."
    )


def test_local_runtime_asset_feeds_i10_and_stays_out_of_egress():
    assets = [
        _asset("local-llm", owner="dev", provider="ollama"),
        _asset("cloud", owner="dev", provider="openai"),
    ]
    insights = enrich_assets(assets)
    stats = collect_stats(assets, insights, repos=1, files_scanned=2)
    assert stats.local_count == 1
    assert stats.egress_count == 1
    assert not (stats.local_ids & stats.egress_ids)
    catalog = build_insights(stats)
    assert any(i.id == "I-10" for i in catalog)
    assert any(i.id == "I-03" for i in catalog)


def test_i01_present_even_for_empty_scan():
    catalog = build_insights(collect_stats([], {}, repos=1, files_scanned=5))
    assert len(catalog) == 1
    assert catalog[0].id == "I-01"
    assert "no AI solutions" in catalog[0].text


def test_low_confidence_findings_count_as_candidates():
    low_conf = Finding(
        type=FindingType.IMPORT_DETECTED, file_path="x.py",
        content="import mystery_ai", confidence=0.4,
    )
    assets = [_asset("maybe-ai", findings=[low_conf])]
    insights = enrich_assets(assets)
    stats = collect_stats(assets, insights)
    assert stats.candidate_count == 1
    catalog = build_insights(stats)
    i08 = next(i for i in catalog if i.id == "I-08")
    assert "1 possible AI integration" in i08.text


def test_every_rendered_insight_passes_the_linter():
    """Integration sweep: a busy org scan renders only lint-clean text."""
    assets = [
        _asset("leaky", owner="solodev", provider="openai",
               findings=[_key_finding()]),
        _asset("bot", owner="solodev", provider="anthropic"),
        _asset("local", owner="solodev", provider="ollama"),
        _asset("classified", owner="solodev", provider="openai",
               classification=ClassificationResult(
                   categories=[DataCategory.PII],
                   confidence=Confidence.HIGH,
                   details="Processes customer support emails.",
               )),
    ]
    insights = enrich_assets(assets)
    stats = collect_stats(assets, insights, repos=2, files_scanned=40)
    catalog = build_insights(stats, delta={
        "prev_date": date(2026, 6, 1), "added": 2, "removed": 0,
        "new_providers": 1,
    })
    assert {i.id for i in catalog} >= {"I-01", "I-02", "I-03", "I-04", "I-09"}
    safe = list(stats.known_labels) + [stats.top_author]
    for insight in catalog:
        _lint_clean(insight.text, safe_tokens=safe + insight.safe_tokens())
