"""Git Repository Scanner — discovers AI assets in source code."""

from __future__ import annotations

import json
import re
import tempfile
import tomllib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import git

from aiscout.knowledge.providers import get_provider
from aiscout.models import (
    AIAsset,
    AssetType,
    Finding,
    FindingType,
    ProviderInfo,
    ScannerConfig,
    ScanResult,
)
from aiscout.scanners.base import BaseScanner

# ── Constants ──────────────────────────────────────────────────────────────

MAX_FILE_SIZE = 1_048_576  # 1 MB

SCAN_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    ".java", ".cs", ".go", ".rs", ".rb", ".php",
    ".yaml", ".yml", ".toml", ".json", ".env", ".ipynb",
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
}

DEPENDENCY_FILES = {
    "requirements.txt", "pyproject.toml", "setup.py", "package.json",
}

# ── AI Import Patterns ────────────────────────────────────────────────────
# provider -> list of regex patterns (compiled at module load)

_AI_IMPORT_PATTERNS: dict[str, list[str]] = {
    "openai": [
        r"^(?:import|from)\s+openai\b",
        r"require\(\s*['\"]openai['\"]\s*\)",
        r"from\s+['\"]openai['\"]",
    ],
    "anthropic": [
        r"^(?:import|from)\s+anthropic\b",
        r"require\(\s*['\"]@anthropic-ai/sdk['\"]\s*\)",
        r"from\s+['\"]@anthropic-ai/sdk['\"]",
    ],
    "langchain": [
        r"^(?:import|from)\s+langchain\b",
        r"^(?:import|from)\s+langchain_core\b",
        r"^(?:import|from)\s+langchain_community\b",
        r"^(?:import|from)\s+langchain_openai\b",
    ],
    "llamaindex": [
        r"^(?:import|from)\s+llama_index\b",
        r"^(?:import|from)\s+llamaindex\b",
    ],
    "huggingface": [
        r"^(?:import|from)\s+transformers\b",
        r"^(?:import|from)\s+huggingface_hub\b",
        r"require\(\s*['\"]@huggingface/['\"]\s*\)",
        r"from\s+['\"]@huggingface/",
    ],
    "mistral": [
        r"^(?:import|from)\s+mistralai\b",
        r"require\(\s*['\"]@mistralai/['\"]\s*\)",
        r"from\s+['\"]@mistralai/",
    ],
    "cohere": [
        r"^(?:import|from)\s+cohere\b",
        r"require\(\s*['\"]cohere-ai['\"]\s*\)",
        r"from\s+['\"]cohere-ai['\"]",
    ],
    "ollama": [
        r"^(?:import|from)\s+ollama\b",
        r"require\(\s*['\"]ollama['\"]\s*\)",
        r"from\s+['\"]ollama['\"]",
    ],
    "chromadb": [
        r"^(?:import|from)\s+chromadb\b",
    ],
    "pinecone": [
        r"^(?:import|from)\s+pinecone\b",
        r"require\(\s*['\"]@pinecone-database/pinecone['\"]\s*\)",
    ],
    "qdrant": [
        r"^(?:import|from)\s+qdrant_client\b",
    ],
    "weaviate": [
        r"^(?:import|from)\s+weaviate\b",
    ],
    "google_ai": [
        r"^(?:import|from)\s+google\.generativeai\b",
        r"^(?:import|from)\s+google\.ai\b",
        r"require\(\s*['\"]@google/generative-ai['\"]\s*\)",
        r"from\s+['\"]@google/generative-ai['\"]",
    ],
    "aws_bedrock": [
        r"^(?:import|from)\s+boto3\b.*bedrock",
        r"require\(\s*['\"]@aws-sdk/client-bedrock['\"]\s*\)",
        r"from\s+['\"]@aws-sdk/client-bedrock['\"]",
    ],
    "replicate": [
        r"^(?:import|from)\s+replicate\b",
    ],
    "together": [
        r"^(?:import|from)\s+together\b",
    ],
    "groq": [
        r"^(?:import|from)\s+groq\b",
        r"require\(\s*['\"]groq-sdk['\"]\s*\)",
        r"from\s+['\"]groq-sdk['\"]",
    ],
}

AI_IMPORT_PATTERNS: dict[str, list[re.Pattern]] = {
    provider: [re.compile(p, re.MULTILINE) for p in patterns]
    for provider, patterns in _AI_IMPORT_PATTERNS.items()
}

# ── API Key Patterns ──────────────────────────────────────────────────────

API_KEY_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("openai", re.compile(r"sk-[a-zA-Z0-9]{20,}"), "OpenAI API Key"),
    ("anthropic", re.compile(r"sk-ant-[a-zA-Z0-9]{20,}"), "Anthropic API Key"),
    ("google", re.compile(r"AIza[a-zA-Z0-9_-]{35}"), "Google AI API Key"),
    ("huggingface", re.compile(r"hf_[a-zA-Z0-9]{30,}"), "HuggingFace Token"),
    ("replicate", re.compile(r"r8_[a-zA-Z0-9]{20,}"), "Replicate Token"),
]

# ── AI Package Names ─────────────────────────────────────────────────────

AI_PYTHON_PACKAGES = {
    "openai", "anthropic", "langchain", "langchain-core", "langchain-community",
    "langchain-openai", "langchain-anthropic", "llama-index", "llamaindex",
    "transformers", "huggingface-hub", "diffusers", "accelerate", "datasets",
    "tokenizers", "safetensors", "sentence-transformers",
    "mistralai", "cohere", "ollama", "replicate", "together",
    "groq", "fireworks-ai",
    "chromadb", "pinecone-client", "pinecone", "qdrant-client", "weaviate-client",
    "faiss-cpu", "faiss-gpu",
    "auto-gptq", "bitsandbytes", "peft", "trl",
    "guidance", "dspy-ai", "instructor", "outlines",
    "crewai", "autogen", "semantic-kernel",
    "google-generativeai", "google-cloud-aiplatform",
    "boto3-bedrock",
}

AI_NPM_PACKAGES = {
    "openai", "@anthropic-ai/sdk", "langchain", "@langchain/core",
    "@langchain/openai", "@langchain/anthropic",
    "@huggingface/inference", "@huggingface/hub",
    "@mistralai/mistralai", "cohere-ai", "ollama",
    "replicate", "groq-sdk",
    "@pinecone-database/pinecone", "@qdrant/js-client-rest",
    "@google/generative-ai", "@aws-sdk/client-bedrock-runtime",
}


# ── Git Scanner ───────────────────────────────────────────────────────────


class GitScanner(BaseScanner):
    """Scans Git repositories for AI-related code, keys, and dependencies."""

    def __init__(
        self,
        repo_path: str | None = None,
        repo_url: str | None = None,
        branch: str = "main",
        token: str | None = None,
    ):
        self.repo_path = repo_path
        self.repo_url = repo_url
        self.branch = branch
        self.token = token
        self._cleanup = None

    def cleanup(self):
        """Clean up temporary files (cloned repos). Call after code analysis."""
        if self._cleanup:
            self._cleanup()
            self._cleanup = None

    def get_config(self) -> ScannerConfig:
        return ScannerConfig(
            name="git_scanner",
            required_credentials=["git_token (optional, for private repos)"],
            description="Scans Git repositories for AI imports, API keys, and dependencies",
        )

    def get_name(self) -> str:
        return "Git Repository Scanner"

    def scan(self, **kwargs) -> ScanResult:
        started_at = datetime.now(timezone.utc)
        all_findings: list[Finding] = []
        files_scanned = 0
        repo_name = ""

        try:
            root, cleanup, repo_name = self._resolve_repo()

            for file_path in self._walk_files(root):
                files_scanned += 1
                content = self._read_file(file_path)
                if content is None:
                    continue

                rel_path = str(file_path.relative_to(root))

                # For .ipynb files, extract source code from cells
                if file_path.suffix == ".ipynb":
                    content = self._extract_notebook_source(content)
                    if not content:
                        continue

                # Run detectors
                all_findings.extend(self._detect_imports(rel_path, content))
                all_findings.extend(self._detect_api_keys(rel_path, content))

                # Dependency files get special treatment
                if file_path.name in DEPENDENCY_FILES:
                    all_findings.extend(
                        self._scan_dependencies(file_path.name, rel_path, content)
                    )

            assets = self._group_findings_into_assets(all_findings, repo_name)

            # Extract git authors for each asset
            self._enrich_with_git_authors(root, assets)

            # Store cleanup for later (after code analysis reads the files)
            self._cleanup = cleanup

            return ScanResult(
                scanner="git_scanner",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                assets=assets,
                metadata={
                    "repository": repo_name,
                    "branch": self.branch,
                    "files_scanned": files_scanned,
                    "repo_root": str(root),
                },
            )

        except Exception as e:
            return ScanResult(
                scanner="git_scanner",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                errors=[f"Scan failed for {repo_name or self.repo_url or self.repo_path}: {e}"],
                metadata={"repository": repo_name},
            )

    # ── Private helpers ────────────────────────────────────────────────

    def _resolve_repo(self) -> tuple[Path, callable | None, str]:
        """Resolve repo source. Returns (root_path, cleanup_fn, repo_name)."""
        if self.repo_path:
            root = Path(self.repo_path).resolve()
            repo_name = root.name
            return root, None, repo_name

        if self.repo_url:
            tmpdir = tempfile.mkdtemp(prefix="aiscout_")
            url = self.repo_url
            if self.token and "://" in url:
                # Embed token: https://TOKEN@github.com/org/repo
                proto, rest = url.split("://", 1)
                url = f"{proto}://{self.token}@{rest}"

            git.Repo.clone_from(url, tmpdir, branch=self.branch, depth=10)
            repo_name = self.repo_url.rstrip("/").split("/")[-1].removesuffix(".git")

            def cleanup():
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)

            return Path(tmpdir), cleanup, repo_name

        raise ValueError("Either repo_path or repo_url must be provided")

    def _walk_files(self, root: Path):
        """Yield files matching scan criteria."""
        for path in root.rglob("*"):
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue
            if not path.is_file():
                continue
            if path.suffix not in SCAN_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            yield path

    def _read_file(self, path: Path) -> str | None:
        """Read file content, returning None on failure."""
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            return None

    def _extract_notebook_source(self, content: str) -> str:
        """Extract source code from Jupyter notebook JSON."""
        try:
            nb = json.loads(content)
            sources = []
            for cell in nb.get("cells", []):
                if cell.get("cell_type") in ("code", "markdown"):
                    source = cell.get("source", [])
                    if isinstance(source, list):
                        sources.append("".join(source))
                    elif isinstance(source, str):
                        sources.append(source)
            return "\n".join(sources)
        except (json.JSONDecodeError, KeyError):
            return ""

    def _enrich_with_git_authors(self, root: Path, assets: list[AIAsset]):
        """Extract git authors for files in each asset."""
        try:
            repo = git.Repo(root)
        except git.InvalidGitRepositoryError:
            return

        for asset in assets:
            authors: set[str] = set()
            file_paths = asset.file_path.split(", ")
            for fp in file_paths[:5]:  # limit to first 5 files for perf
                try:
                    commits = list(repo.iter_commits(paths=fp, max_count=3))
                    for commit in commits:
                        if commit.author:
                            authors.add(commit.author.name)
                except Exception:
                    continue

            if authors:
                asset.owner = ", ".join(sorted(authors))
                asset.users = sorted(authors)

    def _detect_imports(self, file_path: str, content: str) -> list[Finding]:
        """Detect AI-related imports in source code."""
        findings = []
        lines = content.splitlines()

        for provider, patterns in AI_IMPORT_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(content):
                    line_num = content[:match.start()].count("\n") + 1
                    line_text = lines[line_num - 1].strip() if line_num <= len(lines) else ""
                    findings.append(Finding(
                        type=FindingType.IMPORT_DETECTED,
                        file_path=file_path,
                        line_number=line_num,
                        content=line_text,
                        provider=provider,
                    ))
                    break  # One match per pattern set per provider is enough

        return findings

    def _detect_api_keys(self, file_path: str, content: str) -> list[Finding]:
        """Detect hardcoded API keys using regex patterns."""
        findings = []
        lines = content.splitlines()

        for provider, pattern, description in API_KEY_PATTERNS:
            for match in pattern.finditer(content):
                key = match.group()
                line_num = content[:match.start()].count("\n") + 1
                findings.append(Finding(
                    type=FindingType.API_KEY_DETECTED,
                    file_path=file_path,
                    line_number=line_num,
                    content=key,
                    redacted_content=_redact_key(key),
                    provider=provider,
                ))

        return findings

    def _scan_dependencies(
        self, filename: str, file_path: str, content: str
    ) -> list[Finding]:
        """Detect AI packages in dependency files."""
        if filename == "requirements.txt":
            return self._scan_requirements_txt(file_path, content)
        if filename == "pyproject.toml":
            return self._scan_pyproject_toml(file_path, content)
        if filename == "setup.py":
            return self._scan_setup_py(file_path, content)
        if filename == "package.json":
            return self._scan_package_json(file_path, content)
        return []

    def _scan_requirements_txt(self, file_path: str, content: str) -> list[Finding]:
        findings = []
        for i, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Strip version specifiers: package>=1.0 -> package
            pkg = re.split(r"[>=<!\[;@\s]", line)[0].strip().lower()
            if pkg in AI_PYTHON_PACKAGES:
                findings.append(Finding(
                    type=FindingType.DEPENDENCY_DETECTED,
                    file_path=file_path,
                    line_number=i,
                    content=line,
                    provider=_package_to_provider(pkg),
                ))
        return findings

    def _scan_pyproject_toml(self, file_path: str, content: str) -> list[Finding]:
        findings = []
        try:
            data = tomllib.loads(content)
        except Exception:
            return findings

        deps: list[str] = []
        # [project.dependencies]
        deps.extend(data.get("project", {}).get("dependencies", []))
        # [project.optional-dependencies]
        for group in data.get("project", {}).get("optional-dependencies", {}).values():
            deps.extend(group)
        # [dependency-groups]
        for group in data.get("dependency-groups", {}).values():
            if isinstance(group, list):
                deps.extend(item for item in group if isinstance(item, str))

        for dep in deps:
            pkg = re.split(r"[>=<!\[;@\s]", dep)[0].strip().lower()
            if pkg in AI_PYTHON_PACKAGES:
                findings.append(Finding(
                    type=FindingType.DEPENDENCY_DETECTED,
                    file_path=file_path,
                    content=dep.strip(),
                    provider=_package_to_provider(pkg),
                ))
        return findings

    def _scan_setup_py(self, file_path: str, content: str) -> list[Finding]:
        findings = []
        match = re.search(r"install_requires\s*=\s*\[(.*?)\]", content, re.DOTALL)
        if not match:
            return findings
        for item in re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)):
            pkg = re.split(r"[>=<!\[;@\s]", item)[0].strip().lower()
            if pkg in AI_PYTHON_PACKAGES:
                findings.append(Finding(
                    type=FindingType.DEPENDENCY_DETECTED,
                    file_path=file_path,
                    content=item,
                    provider=_package_to_provider(pkg),
                ))
        return findings

    def _scan_package_json(self, file_path: str, content: str) -> list[Finding]:
        findings = []
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return findings

        for section in ("dependencies", "devDependencies"):
            for pkg in data.get(section, {}):
                if pkg in AI_NPM_PACKAGES:
                    version = data[section][pkg]
                    findings.append(Finding(
                        type=FindingType.DEPENDENCY_DETECTED,
                        file_path=file_path,
                        content=f"{pkg}@{version}",
                        provider=_package_to_provider(pkg),
                    ))
        return findings

    def _group_findings_into_assets(
        self, findings: list[Finding], repo_name: str
    ) -> list[AIAsset]:
        """Group findings by solution directory, not by provider.

        Each leaf directory with AI code becomes one AI solution/asset.
        This way, different tools in the same directory are one solution,
        and same provider in different directories are separate solutions.
        """
        # Group by solution directory
        by_solution: dict[str, list[Finding]] = defaultdict(list)
        for f in findings:
            solution_dir = _get_solution_dir(f.file_path)
            by_solution[solution_dir].append(f)

        assets = []
        for solution_dir, dir_findings in by_solution.items():
            # Collect unique file paths
            file_paths = sorted({f.file_path for f in dir_findings})
            # Collect unique providers used in this solution
            providers = sorted({f.provider for f in dir_findings if f.provider})
            primary_provider = providers[0] if providers else ""
            # Collect dependencies
            deps = sorted({
                f.content for f in dir_findings
                if f.type == FindingType.DEPENDENCY_DETECTED
            })
            # Check for API keys
            has_api_keys = any(
                f.type == FindingType.API_KEY_DETECTED for f in dir_findings
            )

            risk = 0.3
            if has_api_keys:
                risk = 0.7

            solution_name = _derive_solution_name(file_paths, primary_provider, repo_name)

            # Provider info: use primary, but list all in dependencies
            provider_info = ProviderInfo(name=primary_provider) if primary_provider else None
            all_provider_names = [get_provider(p).display_name for p in providers]

            assets.append(AIAsset(
                name=solution_name,
                type=AssetType.CUSTOM_CODE,
                provider=provider_info,
                risk_score=risk,
                discovered_via=["git_scanner"],
                repository=repo_name,
                file_path=", ".join(file_paths),
                dependencies=deps,
                raw_findings=dir_findings,
                users=all_provider_names,  # reuse users field for provider list temporarily
            ))

        # Disambiguate duplicate names
        name_counts: dict[str, int] = defaultdict(int)
        for a in assets:
            name_counts[a.name] += 1
        for a in assets:
            if name_counts[a.name] > 1:
                # Append the solution directory for disambiguation
                dir_parts = a.file_path.split(", ")[0].rsplit("/", 1)
                if len(dir_parts) > 1:
                    suffix = dir_parts[0].split("/")[-1]
                    cleaned = _clean_dir_name(suffix)
                    if cleaned and cleaned != a.name:
                        a.name = f"{a.name} — {cleaned}"

        return sorted(assets, key=lambda a: a.risk_score, reverse=True)


# ── Module-level helpers ──────────────────────────────────────────────────


def _redact_key(key: str) -> str:
    """Redact an API key, showing only first 8 and last 4 characters."""
    if len(key) <= 12:
        return key[:4] + "..." + key[-2:]
    return key[:8] + "..." + key[-4:]


def _package_to_provider(pkg: str) -> str:
    """Map a package name to a provider name."""
    mapping = {
        "openai": "openai",
        "anthropic": "anthropic",
        "langchain": "langchain", "langchain-core": "langchain",
        "langchain-community": "langchain", "langchain-openai": "langchain",
        "langchain-anthropic": "langchain",
        "llama-index": "llamaindex", "llamaindex": "llamaindex",
        "transformers": "huggingface", "huggingface-hub": "huggingface",
        "diffusers": "huggingface", "accelerate": "huggingface",
        "datasets": "huggingface", "tokenizers": "huggingface",
        "safetensors": "huggingface", "sentence-transformers": "huggingface",
        "mistralai": "mistral",
        "cohere": "cohere", "cohere-ai": "cohere",
        "ollama": "ollama",
        "replicate": "replicate",
        "together": "together",
        "groq": "groq", "groq-sdk": "groq",
        "fireworks-ai": "fireworks",
        "chromadb": "chromadb",
        "pinecone-client": "pinecone", "pinecone": "pinecone",
        "@pinecone-database/pinecone": "pinecone",
        "qdrant-client": "qdrant", "@qdrant/js-client-rest": "qdrant",
        "weaviate-client": "weaviate",
        "faiss-cpu": "faiss", "faiss-gpu": "faiss",
        "auto-gptq": "huggingface", "bitsandbytes": "huggingface",
        "peft": "huggingface", "trl": "huggingface",
        "guidance": "guidance", "dspy-ai": "dspy",
        "instructor": "instructor", "outlines": "outlines",
        "crewai": "crewai", "autogen": "autogen",
        "semantic-kernel": "semantic-kernel",
        "google-generativeai": "google_ai",
        "google-cloud-aiplatform": "google_ai",
        "@google/generative-ai": "google_ai",
        "boto3-bedrock": "aws_bedrock",
        "@anthropic-ai/sdk": "anthropic",
        "@huggingface/inference": "huggingface",
        "@huggingface/hub": "huggingface",
        "@mistralai/mistralai": "mistral",
        "@aws-sdk/client-bedrock-runtime": "aws_bedrock",
        "@langchain/core": "langchain",
        "@langchain/openai": "langchain",
        "@langchain/anthropic": "langchain",
    }
    return mapping.get(pkg, pkg)


def _get_solution_dir(file_path: str) -> str:
    """Get the solution directory for a file path.

    The solution directory is the most specific meaningful parent directory.
    For 'a/b/c/main.py' → 'a/b/c'
    For 'main.py' → '.'
    """
    from pathlib import PurePosixPath
    parts = PurePosixPath(file_path).parts

    if len(parts) <= 1:
        return "."

    # Return parent directory (all parts except filename)
    return str(PurePosixPath(*parts[:-1]))


def _derive_solution_name(file_paths: list[str], provider: str, repo_name: str) -> str:
    """Derive a human-readable solution name from file paths and context.

    Prioritizes directory structure and file names over provider names.
    Examples:
        - files in "5-Backend/app.py" → "Backend API"
        - files in "4-RAG_Pipeline/" → "RAG Pipeline"
        - files in "3-Fine_tuning/" → "Fine-tuning"
        - single file "chat_bot.py" → "Chat Bot"
    """
    from pathlib import PurePosixPath

    provider_display = get_provider(provider).display_name

    # Find common parent directory (skip root-level)
    dirs: list[str] = []
    file_names: list[str] = []
    for fp in file_paths:
        parts = PurePosixPath(fp).parts
        if len(parts) > 1:
            # Take the most meaningful directory (skip archive-like dirs)
            for part in parts[:-1]:
                cleaned = part.strip("[]")
                if cleaned.upper() not in ("ARCHIVE", "OLD", "DEPRECATED", "BACKUP"):
                    dirs.append(part)
                    break
        file_names.append(PurePosixPath(fp).stem)

    # Find the most common meaningful directory
    if dirs:
        from collections import Counter
        dir_counts = Counter(dirs)
        best_dir = dir_counts.most_common(1)[0][0]

        # Clean up directory name into human-readable form
        name = _clean_dir_name(best_dir)
        if name:
            return name

    # Fallback: derive from file names
    if len(file_names) == 1:
        name = _clean_file_name(file_names[0])
        if name:
            return name

    # Last fallback: use provider display name
    return f"{provider_display} Integration"


def _clean_dir_name(name: str) -> str:
    """Convert directory name to human-readable solution name."""
    import re

    # Remove common prefixes like "0-", "1-", "5-"
    cleaned = re.sub(r"^\d+[-_.]?\s*", "", name)
    if not cleaned:
        return ""

    # Replace separators with spaces
    cleaned = cleaned.replace("_", " ").replace("-", " ")

    # Title case, preserving acronyms
    words = cleaned.split()
    result = []
    for w in words:
        upper = w.upper()
        if upper in ("AI", "ML", "API", "RAG", "LLM", "NLP", "DB", "SDK", "CLI"):
            result.append(upper)
        else:
            result.append(w.capitalize())

    return " ".join(result)


def _clean_file_name(name: str) -> str:
    """Convert a file stem to human-readable name."""
    import re

    # Remove common prefixes
    cleaned = re.sub(r"^(script[_-]?|test[_-]?)", "", name, flags=re.IGNORECASE)
    if not cleaned or len(cleaned) < 3:
        return ""

    cleaned = cleaned.replace("_", " ").replace("-", " ")

    words = cleaned.split()
    result = []
    for w in words:
        upper = w.upper()
        if upper in ("AI", "ML", "API", "RAG", "LLM", "NLP", "DB"):
            result.append(upper)
        else:
            result.append(w.capitalize())

    return " ".join(result)
