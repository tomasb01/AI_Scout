"""GitHub organization / user repository enumeration.

Resolves a GitHub organization (or user) name into the list of its
repositories via the REST API, so the scan pipeline can iterate over a
whole org instead of a single hand-pasted URL. The enumerated clone URLs
feed the existing multi-repo scan loop unchanged.

Only enumeration lives here — each resulting repo is still cloned and
scanned by ``GitScanner``. ``httpx`` is imported lazily to keep the
landing/serverless surface light (no network deps until a scan runs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit

_API_BASE = "https://api.github.com"
_PER_PAGE = 100
_DEFAULT_MAX_REPOS = 200


class OrgEnumerationError(Exception):
    """Base error for organization enumeration failures."""


class OrgNotFoundError(OrgEnumerationError):
    """The given name is neither a known organization nor a user."""


class OrgAccessError(OrgEnumerationError):
    """Authentication failed or the API rate limit was exhausted."""


@dataclass
class OrgEnumeration:
    """Result of enumerating an org/user: clonable repos plus skip counts."""

    owner: str
    repos: list[dict] = field(default_factory=list)
    total_seen: int = 0
    skipped_archived: int = 0
    skipped_forks: int = 0
    skipped_over_limit: int = 0


def parse_owner(value: str) -> str:
    """Extract the org/user name from a URL or bare name.

    Accepts ``https://github.com/acme``, ``github.com/acme/``, or ``acme``.
    If a full repo path is given (``github.com/acme/repo``) the owner
    (first path segment) is returned.
    """
    value = (value or "").strip()
    if not value:
        raise OrgEnumerationError("Organization name must be a non-empty string.")

    if "://" in value or value.lower().startswith("github.com"):
        # Normalize a scheme-less host so urlsplit populates the path.
        if "://" not in value:
            value = "https://" + value
        path = urlsplit(value).path
    else:
        path = value

    segments = [s for s in path.split("/") if s]
    if not segments:
        raise OrgEnumerationError(f"Could not parse an org/user name from: {value!r}")
    return segments[0]


def enumerate_org_repos(
    name: str,
    token: str | None = None,
    *,
    include_archived: bool = False,
    include_forks: bool = False,
    max_repos: int = _DEFAULT_MAX_REPOS,
) -> OrgEnumeration:
    """List repositories for a GitHub organization or user.

    Tries the ``/orgs/{name}`` endpoint first and falls back to
    ``/users/{name}`` (an account may be either). Returns clone URLs and
    default branches normalized into the dict shape the scan loop expects,
    plus counts of repos skipped by the active filters.

    Only repositories the token is authorized to see are returned; private
    repos outside the token's scope are invisible to the API and cannot be
    reported.
    """
    import httpx  # lazy: keep landing/serverless import light

    owner = parse_owner(name)
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    result = OrgEnumeration(owner=owner)
    with httpx.Client(timeout=30.0, headers=headers) as client:
        raw = _fetch_all_repos(client, owner)

    for repo in raw:
        result.total_seen += 1
        if not include_archived and repo.get("archived"):
            result.skipped_archived += 1
            continue
        if not include_forks and repo.get("fork"):
            result.skipped_forks += 1
            continue
        if len(result.repos) >= max_repos:
            result.skipped_over_limit += 1
            continue

        clone_url = repo.get("clone_url")
        if not clone_url:
            continue
        result.repos.append({
            "url": clone_url,
            "branch": repo.get("default_branch", "main"),
            "name": repo.get("name") or clone_url.rstrip("/").split("/")[-1].removesuffix(".git"),
            "token": token,
        })

    return result


def _fetch_all_repos(client, owner: str) -> list[dict]:
    """Fetch every repo page for an org, falling back to a user account."""
    import httpx

    for kind in ("orgs", "users"):
        url: str | None = (
            f"{_API_BASE}/{kind}/{owner}/repos?per_page={_PER_PAGE}&type=all"
        )
        repos: list[dict] = []
        first = True
        while url:
            resp = client.get(url)
            if first and resp.status_code == 404:
                # Not this kind of account; try the next endpoint.
                break
            _raise_for_status(resp, owner)
            payload = resp.json()
            if isinstance(payload, list):
                repos.extend(payload)
            url = _next_link(resp.headers.get("Link"))
            first = False
        else:
            # Loop completed without hitting the 404 break → this kind matched.
            return repos

    raise OrgNotFoundError(
        f"'{owner}' is not a known GitHub organization or user "
        f"(or no token was provided for a private account)."
    )


def _raise_for_status(resp, owner: str) -> None:
    if resp.status_code in (401, 403):
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            raise OrgAccessError(
                "GitHub API rate limit exhausted. Provide a token (--token) "
                "or wait for the limit to reset."
            )
        raise OrgAccessError(
            f"GitHub API denied access while enumerating '{owner}' "
            f"(HTTP {resp.status_code}). Check the token and its scope."
        )
    if resp.status_code >= 400:
        raise OrgEnumerationError(
            f"GitHub API error while enumerating '{owner}': HTTP {resp.status_code}."
        )


def _next_link(link_header: str | None) -> str | None:
    """Extract the ``rel="next"`` URL from a GitHub ``Link`` header."""
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.split(";")
        if len(section) < 2:
            continue
        url_part = section[0].strip().strip("<>")
        for rel in section[1:]:
            if rel.strip() == 'rel="next"':
                return url_part
    return None
