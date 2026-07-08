"""Repo character detector (Sprint 0.3, Spec v13 §3.4).

Classifies a scanned repository as ``production``, ``tutorial_example``,
``experiment`` or ``unknown`` so the aggregation layer can stop a
teaching repo with 120 lesson folders from reporting as 120 AI
solutions.

Discipline (datamodel spec §0-§1): the character is an *observable with
evidence*, never a verdict — the result carries the finite-vocabulary
signals that produced it plus a confidence level, and defaults to
``unknown`` when the evidence is weak. Misclassifying a production repo
as a tutorial is the expensive error, so tutorial/experiment need strong
positive evidence AND an absence of production signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

# ── Finite signal vocabulary (rendered in report evidence) ─────────────────

SIGNAL_SEQUENTIAL_NAMING = "sequential_dir_naming"
SIGNAL_LESSON_KEYWORDS = "lesson_keyword_dirs"
SIGNAL_MANY_SMALL_DIRS = "many_small_solution_dirs"
SIGNAL_NOTEBOOK_HEAVY = "notebook_heavy"
SIGNAL_README_COURSE = "readme_course_keywords"
SIGNAL_HAS_CI = "has_ci_pipeline"
SIGNAL_HAS_TESTS = "has_test_suite"
SIGNAL_HAS_CONTAINER = "has_container_deploy"
SIGNAL_HAS_LOCKFILE = "has_dependency_lockfile"

KIND_PRODUCTION = "production"
KIND_TUTORIAL = "tutorial_example"
KIND_EXPERIMENT = "experiment"
KIND_UNKNOWN = "unknown"

_LESSON_WORDS = (
    "lesson", "example", "examples", "demo", "demos", "tutorial",
    "tutorials", "exercise", "exercises", "sample", "samples", "chapter",
    "workshop", "bootcamp", "course", "day", "week", "lab", "labs",
)
_SEQ_DIR_RE = re.compile(r"^\d{1,3}[-_. ]")
_README_COURSE_RE = re.compile(
    r"\b(course|tutorial|workshop|bootcamp|curriculum|learning path|"
    r"lessons?|examples? (repo|collection)|step[- ]by[- ]step)\b",
    re.IGNORECASE,
)
_LOCKFILES = {
    "uv.lock", "poetry.lock", "package-lock.json", "yarn.lock",
    "pnpm-lock.yaml", "Pipfile.lock", "Cargo.lock", "go.sum",
}


@dataclass
class RepoCharacter:
    """Observable classification of a repository's nature."""

    kind: str = KIND_UNKNOWN
    confidence: str = "low"  # high | medium | low
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "confidence": self.confidence,
            "signals": self.signals,
        }


def detect_repo_character(
    file_paths: list[str],
    solution_dirs: list[str],
    readme_text: str = "",
) -> RepoCharacter:
    """Classify the repo from scanned file paths + solution layout.

    ``file_paths`` are repo-relative paths of every scanned file;
    ``solution_dirs`` are the directory-grouping keys the scanner
    produced (one per directory-level asset); ``readme_text`` is the
    head of the root README when available.
    """
    signals: list[str] = []
    paths = [PurePosixPath(p) for p in file_paths]
    n_files = len(paths)
    n_dirs = len(solution_dirs)

    # ── Tutorial-shaped evidence ────────────────────────────────────────
    top_dirs = sorted({
        d.split("/")[0] for d in solution_dirs if d not in (".", "")
    })
    seq_dirs = [d for d in top_dirs if _SEQ_DIR_RE.match(d)]
    if top_dirs and len(seq_dirs) >= max(3, round(0.3 * len(top_dirs))):
        signals.append(SIGNAL_SEQUENTIAL_NAMING)

    dir_words = " ".join(
        part.lower() for d in solution_dirs for part in re.split(r"[/\d_.-]+", d)
    ).split()
    lesson_hits = sum(1 for w in dir_words if w in _LESSON_WORDS)
    if lesson_hits >= 3:
        signals.append(SIGNAL_LESSON_KEYWORDS)

    if n_dirs >= 8 and n_files and (n_files / n_dirs) <= 3:
        signals.append(SIGNAL_MANY_SMALL_DIRS)

    notebooks = sum(1 for p in paths if p.suffix == ".ipynb")
    if n_files and notebooks / n_files >= 0.3:
        signals.append(SIGNAL_NOTEBOOK_HEAVY)

    if readme_text and _README_COURSE_RE.search(readme_text[:4000]):
        signals.append(SIGNAL_README_COURSE)

    # ── Production-shaped evidence — REPO-LEVEL locations only ──────────
    # Validated on real teaching repos (AI-developer-3, AI-Agents-2): the
    # lessons themselves ship lockfiles, Dockerfiles and test files as
    # teaching material. Deploy evidence only counts when it sits at the
    # repo root, in a root-level tests/ directory, or in CI config —
    # artifacts nested inside solution directories say nothing about the
    # repo being an operated application.
    root_names = {p.name for p in paths if len(p.parts) == 1}
    path_strs = [str(p) for p in paths]
    if any(
        s.startswith(".github/workflows/") or s.startswith(".gitlab-ci")
        for s in path_strs
    ):
        signals.append(SIGNAL_HAS_CI)
    if any(
        len(p.parts) > 1 and p.parts[0] in ("tests", "test") for p in paths
    ) or any(n.startswith("test_") for n in root_names):
        signals.append(SIGNAL_HAS_TESTS)
    if root_names & {"Dockerfile", "Containerfile", "docker-compose.yml",
                     "docker-compose.yaml", "compose.yml", "compose.yaml",
                     "Procfile", "fly.toml"}:
        signals.append(SIGNAL_HAS_CONTAINER)
    if root_names & _LOCKFILES:
        signals.append(SIGNAL_HAS_LOCKFILE)

    return _classify(signals, n_dirs, notebooks, n_files)


def _classify(
    signals: list[str], n_dirs: int, notebooks: int, n_files: int
) -> RepoCharacter:
    tutorial_signals = {
        SIGNAL_SEQUENTIAL_NAMING, SIGNAL_LESSON_KEYWORDS,
        SIGNAL_MANY_SMALL_DIRS, SIGNAL_NOTEBOOK_HEAVY, SIGNAL_README_COURSE,
    } & set(signals)
    # Semantic signals say "teaching"; shape signals (sequential naming,
    # many small dirs) are shared by real apps whose pipeline stages are
    # numbered folders — shape alone must never collapse a production
    # repo (validated on Fleurdin_AI).
    semantic_signals = {
        SIGNAL_LESSON_KEYWORDS, SIGNAL_README_COURSE,
    } & set(signals)
    production_signals = {
        SIGNAL_HAS_CI, SIGNAL_HAS_TESTS, SIGNAL_HAS_CONTAINER,
        SIGNAL_HAS_LOCKFILE,
    } & set(signals)

    # Tutorial: strong positive evidence including at least one semantic
    # signal, and no production counterweight. A course repo that *also*
    # ships repo-level CI/deploy stays unknown — a human decides.
    if (
        len(tutorial_signals) >= 3 and semantic_signals
        and not production_signals
    ):
        return RepoCharacter(KIND_TUTORIAL, "high", sorted(signals))
    if (
        len(tutorial_signals) == 2 and semantic_signals
        and not production_signals and n_dirs >= 8
    ):
        return RepoCharacter(KIND_TUTORIAL, "medium", sorted(signals))

    # Experiment: notebook-dominated scratch work, small, nothing deployable.
    if (
        SIGNAL_NOTEBOOK_HEAVY in signals
        and not production_signals
        and n_dirs < 8
    ):
        return RepoCharacter(KIND_EXPERIMENT, "medium", sorted(signals))

    if len(production_signals) >= 2:
        return RepoCharacter(KIND_PRODUCTION, "high", sorted(signals))
    if production_signals:
        return RepoCharacter(KIND_PRODUCTION, "medium", sorted(signals))

    return RepoCharacter(KIND_UNKNOWN, "low", sorted(signals))
