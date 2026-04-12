"""Tests for Sprint 3 offline dependency advisory KB."""

from aiscout.knowledge.dependency_advisories import (
    Advisory,
    _version_matches,
    find_advisories,
    parse_dep_line,
)


def test_version_matches_lt():
    assert _version_matches("0.0.340", "<1.0")
    assert _version_matches("0.9.9", "<1.0")
    assert not _version_matches("1.0.0", "<1.0")
    assert not _version_matches("2.3.4", "<1.0")


def test_version_matches_range():
    assert _version_matches("0.1.2", ">=0.1,<0.2")
    assert not _version_matches("0.2.0", ">=0.1,<0.2")
    assert not _version_matches("0.0.9", ">=0.1,<0.2")


def test_version_matches_unpinned_never_matches():
    # An unpinned requirement has no version to compare — don't flag.
    assert not _version_matches(None, "<1.0")
    assert not _version_matches("", "<1.0")


def test_parse_dep_line_requirements():
    assert parse_dep_line("openai==0.28.1") == ("openai", "0.28.1")
    assert parse_dep_line("langchain>=0.0.340") == ("langchain", "0.0.340")
    assert parse_dep_line("transformers[torch]==4.35.0") == ("transformers", "4.35.0")
    assert parse_dep_line("# a comment") == ("", None)


def test_find_advisories_detects_legacy_openai():
    hits = find_advisories(["openai==0.28.1"])
    assert any("Legacy" in a.title or "< 1.0" in a.title for a in hits)
    assert all(isinstance(a, Advisory) for a in hits)


def test_find_advisories_no_false_positive_on_current_version():
    hits = find_advisories(["openai==1.12.0", "langchain==0.2.5"])
    assert hits == []


def test_find_advisories_deduped_across_files():
    """Same advisory fired by two requirement files collapses to one hit."""
    hits = find_advisories([
        "openai==0.28.0",
        "openai==0.28.0",
    ])
    titles = [a.title for a in hits]
    assert len(titles) == len(set(titles))
