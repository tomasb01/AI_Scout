"""Tests for GitHub organization / user repository enumeration."""

import pytest

from aiscout.scanners.github_org import (
    OrgAccessError,
    OrgNotFoundError,
    enumerate_org_repos,
    parse_owner,
)

_ORG_URL = "https://api.github.com/orgs/acme/repos?per_page=100&type=all"
_USER_URL = "https://api.github.com/users/acme/repos?per_page=100&type=all"


def _repo(name, *, archived=False, fork=False, branch="main"):
    return {
        "name": name,
        "clone_url": f"https://github.com/acme/{name}.git",
        "default_branch": branch,
        "archived": archived,
        "fork": fork,
        "private": False,
    }


# --- parse_owner ---------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("acme", "acme"),
    ("https://github.com/acme", "acme"),
    ("https://github.com/acme/", "acme"),
    ("github.com/acme", "acme"),
    ("https://github.com/acme/some-repo", "acme"),
])
def test_parse_owner(value, expected):
    assert parse_owner(value) == expected


def test_parse_owner_empty():
    from aiscout.scanners.github_org import OrgEnumerationError
    with pytest.raises(OrgEnumerationError):
        parse_owner("")


# --- enumeration ---------------------------------------------------------

def test_enumerate_basic(httpx_mock):
    httpx_mock.add_response(
        url=_ORG_URL,
        json=[_repo("api"), _repo("web", branch="develop")],
    )
    result = enumerate_org_repos("acme", token="ghp_x")

    assert result.owner == "acme"
    assert len(result.repos) == 2
    assert result.repos[0]["url"] == "https://github.com/acme/api.git"
    assert result.repos[1]["branch"] == "develop"
    assert result.repos[0]["token"] == "ghp_x"


def test_enumerate_pagination(httpx_mock):
    page2 = _ORG_URL + "&page=2"
    httpx_mock.add_response(
        url=_ORG_URL,
        json=[_repo("api")],
        headers={"Link": f'<{page2}>; rel="next"'},
    )
    httpx_mock.add_response(url=page2, json=[_repo("web")])

    result = enumerate_org_repos("acme")
    assert {r["name"] for r in result.repos} == {"api", "web"}
    assert result.total_seen == 2


def test_enumerate_user_fallback(httpx_mock):
    httpx_mock.add_response(url=_ORG_URL, status_code=404)
    httpx_mock.add_response(url=_USER_URL, json=[_repo("personal")])

    result = enumerate_org_repos("acme")
    assert len(result.repos) == 1
    assert result.repos[0]["name"] == "personal"


def test_enumerate_skips_archived_and_forks(httpx_mock):
    httpx_mock.add_response(
        url=_ORG_URL,
        json=[
            _repo("live"),
            _repo("old", archived=True),
            _repo("forked", fork=True),
        ],
    )
    result = enumerate_org_repos("acme")

    assert [r["name"] for r in result.repos] == ["live"]
    assert result.skipped_archived == 1
    assert result.skipped_forks == 1
    assert result.total_seen == 3


def test_enumerate_include_archived_and_forks(httpx_mock):
    httpx_mock.add_response(
        url=_ORG_URL,
        json=[_repo("live"), _repo("old", archived=True), _repo("forked", fork=True)],
    )
    result = enumerate_org_repos(
        "acme", include_archived=True, include_forks=True
    )
    assert len(result.repos) == 3
    assert result.skipped_archived == 0
    assert result.skipped_forks == 0


def test_enumerate_max_repos_cap(httpx_mock):
    httpx_mock.add_response(
        url=_ORG_URL,
        json=[_repo(f"r{i}") for i in range(5)],
    )
    result = enumerate_org_repos("acme", max_repos=2)

    assert len(result.repos) == 2
    assert result.skipped_over_limit == 3


def test_enumerate_not_found(httpx_mock):
    httpx_mock.add_response(url=_ORG_URL, status_code=404)
    httpx_mock.add_response(url=_USER_URL, status_code=404)

    with pytest.raises(OrgNotFoundError):
        enumerate_org_repos("acme")


def test_enumerate_auth_error(httpx_mock):
    httpx_mock.add_response(
        url=_ORG_URL,
        status_code=401,
        headers={"X-RateLimit-Remaining": "42"},
    )
    with pytest.raises(OrgAccessError, match="token"):
        enumerate_org_repos("acme", token="bad")


def test_enumerate_rate_limited(httpx_mock):
    httpx_mock.add_response(
        url=_ORG_URL,
        status_code=403,
        headers={"X-RateLimit-Remaining": "0"},
    )
    with pytest.raises(OrgAccessError, match="rate limit"):
        enumerate_org_repos("acme")
