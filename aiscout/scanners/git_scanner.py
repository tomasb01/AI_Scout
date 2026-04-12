"""Git Repository Scanner — discovers AI assets in source code."""

from __future__ import annotations

import json
import os
import re
import stat
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

# Sprint 2 — local model weight/artifact extensions. Contents are never read
# (binary files up to GB in size); only path + size recorded.
LOCAL_MODEL_EXTENSIONS = {
    ".gguf", ".ggml", ".safetensors", ".onnx", ".bin",
    ".pt", ".pth", ".ckpt", ".tflite", ".mlmodel",
}

# Sprint 2 — config / manifest files that aren't in SCAN_EXTENSIONS via suffix
# alone (either they have no extension like "Dockerfile", or the name carries
# the signal such as "mcp.json").
SPECIAL_FILENAMES = {
    "Dockerfile", "Containerfile",
    "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml",
    "mcp.json", ".mcp.json", "claude_desktop_config.json",
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
}

DEPENDENCY_FILES = {
    "requirements.txt", "pyproject.toml", "setup.py", "package.json",
}

# Sprint 2 — Docker/Compose image patterns that signal a self-hosted AI
# runtime. Value is the provider key surfaced in findings.
_CONTAINER_IMAGE_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"ollama/ollama", re.IGNORECASE), "ollama", "Ollama runtime"),
    (re.compile(r"\bvllm/\w+", re.IGNORECASE), "vllm", "vLLM inference server"),
    (re.compile(r"vllm/vllm-openai", re.IGNORECASE), "vllm", "vLLM OpenAI-compatible server"),
    (re.compile(r"ghcr\.io/huggingface/text-generation-inference|huggingface/text-generation-inference", re.IGNORECASE), "huggingface", "HF Text Generation Inference (TGI)"),
    (re.compile(r"nvcr\.io/nvidia/tritonserver|tritonserver", re.IGNORECASE), "triton", "NVIDIA Triton"),
    (re.compile(r"localai/localai|quay\.io/go-skynet/local-ai", re.IGNORECASE), "localai", "LocalAI"),
    (re.compile(r"ghcr\.io/ggerganov/llama\.cpp|llama\.cpp", re.IGNORECASE), "llamacpp", "llama.cpp"),
    (re.compile(r"qdrant/qdrant", re.IGNORECASE), "qdrant", "Qdrant vector DB"),
    (re.compile(r"chromadb/chroma", re.IGNORECASE), "chromadb", "Chroma vector DB"),
    (re.compile(r"weaviate/weaviate", re.IGNORECASE), "weaviate", "Weaviate vector DB"),
    (re.compile(r"milvusdb/milvus", re.IGNORECASE), "milvus", "Milvus vector DB"),
    (re.compile(r"langfuse/langfuse", re.IGNORECASE), "langfuse", "Langfuse observability"),
    (re.compile(r"open-webui/open-webui", re.IGNORECASE), "open-webui", "Open WebUI"),
]

# Sprint 3 — YAML/TOML config model references. Looks for explicit
# model/deployment names in configuration files that aren't code. The
# value is captured so downstream enrichment can surface which model a
# project has pinned in its config. Pattern order matters — more
# specific keys first.
_CONFIG_MODEL_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Azure OpenAI: deployment_name / deployment
    (re.compile(r"""(?im)^\s*(?:deployment_?name|deployment)\s*[:=]\s*["']?([\w.\-]+)["']?"""), "azure_openai"),
    (re.compile(r"""(?im)^\s*azure_?endpoint\s*[:=]\s*["']?(https?://[^\s"']+)"""), "azure_openai"),
    # OpenAI / generic LLM: model name (any *_model key)
    (re.compile(r"""(?im)^\s*(?:[a-z_]*model(?:_name)?|llm|chat_model)\s*[:=]\s*["']?([\w./:\-]+)["']?"""), "openai"),
    # Ollama: pulled model
    (re.compile(r"""(?im)^\s*ollama_?model\s*[:=]\s*["']?([\w./:\-]+)["']?"""), "ollama"),
    # Bedrock: model_id
    (re.compile(r"""(?im)^\s*model_id\s*[:=]\s*["']?([\w.\-:]+)["']?"""), "aws_bedrock"),
]

_CONFIG_SCAN_SUFFIXES = {".yaml", ".yml", ".toml"}
_CONFIG_SKIP_NAMES = {
    # Already handled by dedicated detectors — don't double-scan
    "pyproject.toml", "package.json", "docker-compose.yml", "docker-compose.yaml",
    "compose.yml", "compose.yaml", "mcp.json", ".mcp.json",
    "claude_desktop_config.json",
}


# Sprint 3 — CI/CD pipelines that run AI code. Matches on:
#   * uses:/image: lines referencing AI SDKs, LLM code reviewers, etc.
#   * env vars that hand AI provider credentials to runners
#   * `run:` shell steps invoking python scripts that are themselves
#     AI fine-tuning / inference code (best-effort regex signal)
_CI_AI_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bANTHROPIC_API_KEY\b"), "anthropic", "Anthropic credential in CI"),
    (re.compile(r"\bOPENAI_API_KEY\b"), "openai", "OpenAI credential in CI"),
    (re.compile(r"\bAZURE_OPENAI_API_KEY\b"), "azure_openai", "Azure OpenAI credential in CI"),
    (re.compile(r"\bHUGGINGFACE(?:HUB)?_TOKEN\b|\bHF_TOKEN\b"), "huggingface", "HuggingFace token in CI"),
    (re.compile(r"\bGOOGLE_API_KEY\b|\bGEMINI_API_KEY\b"), "google_ai", "Google AI credential in CI"),
    (re.compile(r"\bCOHERE_API_KEY\b"), "cohere", "Cohere credential in CI"),
    (re.compile(r"\bREPLICATE_API_TOKEN\b"), "replicate", "Replicate token in CI"),
    (re.compile(r"\bMISTRAL_API_KEY\b"), "mistral", "Mistral credential in CI"),
    (re.compile(r"\bAWS_BEDROCK|bedrock-runtime\b", re.IGNORECASE), "aws_bedrock", "AWS Bedrock access in CI"),
    (re.compile(r"anthropics/claude-code-action|anthropic/.*-action"), "anthropic", "Claude GitHub Action"),
    (re.compile(r"openai/.*-action|coderabbitai/"), "openai", "OpenAI-based GitHub Action"),
    (re.compile(r"huggingface/.*-action|huggingface/transformers"), "huggingface", "HuggingFace Action/runner"),
    (re.compile(r"python\s+.*?(?:train|finetune|fine_tune)\w*\.py"), "huggingface", "CI runs a training script"),
    (re.compile(r"accelerate\s+launch|deepspeed\s|torchrun\s"), "huggingface", "CI launches distributed training"),
    (re.compile(r"modal\s+run|runpod"), "huggingface", "CI submits jobs to GPU cloud"),
]

# Sprint 2 — detect training vs inference from code body keywords.
_TRAINING_MARKERS = re.compile(
    r"\b("
    r"Trainer\s*\(|TrainingArguments|SFTTrainer|DPOTrainer|PPOTrainer|"
    r"model\.fit\(|model\.train\(\)|\.backward\(\)|\.step\(\)|"
    r"loss\.backward|optimizer\.step|"
    r"AdamW|torch\.optim|lr_scheduler|peft|LoraConfig|get_peft_model|"
    r"accelerate\.Accelerator|deepspeed|fsdp"
    r")\b"
)
_EVALUATION_MARKERS = re.compile(
    r"\b("
    r"model\.eval\(\)|evaluate\.load|sklearn\.metrics|classification_report|"
    r"confusion_matrix|compute_metrics|rouge_score|bleu_score|"
    r"Trainer\.evaluate|lm-eval|lm_eval"
    r")\b"
)

# ── AI Import Patterns ────────────────────────────────────────────────────
# provider -> list of regex patterns (compiled at module load)

_AI_IMPORT_PATTERNS: dict[str, list[str]] = {
    # Azure OpenAI listed BEFORE "openai" so the more specific match wins
    # during detection (see _detect_imports loop which returns the first hit).
    "azure_openai": [
        r"^(?:import|from)\s+openai\s+import\s+AzureOpenAI\b",
        r"^(?:import|from)\s+openai\s+import\s+AsyncAzureOpenAI\b",
        r"\bAzureOpenAI\s*\(",
        r"\bAsyncAzureOpenAI\s*\(",
        r"require\(\s*['\"]@azure/openai['\"]\s*\)",
        r"from\s+['\"]@azure/openai['\"]",
    ],
    "mcp": [
        r"^(?:import|from)\s+mcp\b",
        r"^(?:import|from)\s+mcp\.server\b",
        r"^(?:import|from)\s+mcp\.client\b",
        r"require\(\s*['\"]@modelcontextprotocol/sdk['\"]\s*\)",
        r"from\s+['\"]@modelcontextprotocol/sdk",
    ],
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
    "mcp", "fastmcp",
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
    "@modelcontextprotocol/sdk", "@azure/openai",
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
                rel_path = str(file_path.relative_to(root))

                # ── Local model artifacts — never read content ──
                if file_path.suffix.lower() in LOCAL_MODEL_EXTENSIONS:
                    all_findings.append(
                        self._local_model_finding(file_path, rel_path)
                    )
                    continue

                content = self._read_file(file_path)
                if content is None:
                    continue

                # For .ipynb files, extract source code from cells
                if file_path.suffix == ".ipynb":
                    content = self._extract_notebook_source(content)
                    if not content:
                        continue

                # ── Docker / compose manifests ──
                if file_path.name in {
                    "Dockerfile", "Containerfile",
                    "docker-compose.yml", "docker-compose.yaml",
                    "compose.yml", "compose.yaml",
                }:
                    all_findings.extend(
                        self._detect_containers(rel_path, content)
                    )
                    continue

                # ── MCP config files ──
                if file_path.name in {
                    "mcp.json", ".mcp.json", "claude_desktop_config.json",
                }:
                    all_findings.extend(
                        self._detect_mcp_config(rel_path, content)
                    )
                    continue

                # ── CI/CD pipelines ──
                if _looks_like_ci_file(rel_path, file_path.name):
                    all_findings.extend(
                        self._detect_ci_pipeline(rel_path, content)
                    )
                    # Intentionally fall through — a workflow file may also
                    # contain AI imports referenced via `uses:`/`script:`.

                # Run code detectors
                all_findings.extend(self._detect_imports(rel_path, content))
                all_findings.extend(self._detect_api_keys(rel_path, content))
                all_findings.extend(self._detect_azure_openai_config(rel_path, content))

                # Sprint 3 — YAML/TOML model/deployment config references.
                # We don't want to rescan pyproject.toml etc. which are
                # handled by _scan_dependencies.
                if (
                    file_path.suffix in _CONFIG_SCAN_SUFFIXES
                    and file_path.name not in _CONFIG_SKIP_NAMES
                ):
                    all_findings.extend(
                        self._detect_config_model_refs(rel_path, content)
                    )

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
                    "repo_url": self.repo_url or "",
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
        """Resolve repo source. Returns (root_path, cleanup_fn, repo_name).

        Security:
          * Remote clones use a ``TemporaryDirectory`` with 0700 perms so the
            working copy is never world-readable and is removed on interpreter
            exit even if the scanner crashes.
          * Git credentials are passed via ``GIT_ASKPASS`` instead of being
            embedded in the clone URL — tokens never appear in process args,
            GitPython error strings, or the cloned repo's ``.git/config``.
        """
        if self.repo_path:
            root = Path(self.repo_path).resolve()
            repo_name = root.name
            return root, None, repo_name

        if self.repo_url:
            repo_name = self.repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
            tmp = tempfile.TemporaryDirectory(prefix="aiscout_")
            try:
                os.chmod(tmp.name, 0o700)
            except OSError:
                pass

            env = {
                "GIT_TERMINAL_PROMPT": "0",  # never block on tty auth prompt
                "GIT_ASKPASS": "/bin/echo",  # safe default if token absent
            }
            askpass_path: str | None = None
            if self.token:
                askpass_path = _write_askpass_helper(tmp.name)
                env["GIT_ASKPASS"] = askpass_path
                env["AISCOUT_GIT_TOKEN"] = self.token

            try:
                git.Repo.clone_from(
                    self.repo_url,
                    tmp.name,
                    branch=self.branch,
                    depth=10,
                    env=env,
                )
            except Exception:
                tmp.cleanup()
                raise
            finally:
                if askpass_path:
                    try:
                        os.unlink(askpass_path)
                    except OSError:
                        pass

            def cleanup():
                tmp.cleanup()

            return Path(tmp.name).resolve(), cleanup, repo_name

        raise ValueError("Either repo_path or repo_url must be provided")

    def _walk_files(self, root: Path):
        """Yield files matching scan criteria.

        Security: symlinks are skipped entirely, and each candidate path is
        resolved and checked to be strictly inside ``root`` so a malicious
        repository cannot trick the scanner into reading ``/etc/passwd`` or
        files outside the working copy.
        """
        root_resolved = root.resolve()
        for dirpath, dirnames, filenames in os.walk(
            root, topdown=True, followlinks=False
        ):
            # Prune skipped dirs in-place so os.walk doesn't descend into them
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            dir_path = Path(dirpath)
            for name in filenames:
                path = dir_path / name
                if any(skip in path.parts for skip in SKIP_DIRS):
                    continue

                is_code = path.suffix in SCAN_EXTENSIONS
                is_special = name in SPECIAL_FILENAMES
                is_model = path.suffix.lower() in LOCAL_MODEL_EXTENSIONS
                is_dep = name in DEPENDENCY_FILES  # requirements.txt, setup.py
                if not (is_code or is_special or is_model or is_dep):
                    continue

                try:
                    if path.is_symlink():
                        continue
                    st = path.lstat()
                    if not stat.S_ISREG(st.st_mode):
                        continue
                    # Model weights are allowed to be huge — we never read them.
                    if not is_model and st.st_size > MAX_FILE_SIZE:
                        continue
                    # Guard against path-traversal via intermediate symlinks
                    resolved = path.resolve()
                    resolved.relative_to(root_resolved)
                except (OSError, ValueError):
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
        """Detect AI-related imports in source code.

        Per line, the first (most specific) provider listed in
        ``AI_IMPORT_PATTERNS`` wins. This lets e.g. ``azure_openai`` be
        detected on a line like ``from openai import AzureOpenAI`` without
        also emitting a generic ``openai`` finding for the same line.
        """
        findings = []
        lines = content.splitlines()
        claimed_lines: set[int] = set()
        # Track provider-per-file de-dup so we don't emit 40 findings for the
        # same package across a 400-line module.
        seen_provider_in_file: set[str] = set()

        for provider, patterns in AI_IMPORT_PATTERNS.items():
            matched_line = None
            for pattern in patterns:
                for match in pattern.finditer(content):
                    line_num = content[:match.start()].count("\n") + 1
                    if line_num in claimed_lines:
                        continue
                    matched_line = line_num
                    break
                if matched_line:
                    break
            if matched_line and provider not in seen_provider_in_file:
                line_text = (
                    lines[matched_line - 1].strip()
                    if matched_line <= len(lines) else ""
                )
                findings.append(Finding(
                    type=FindingType.IMPORT_DETECTED,
                    file_path=file_path,
                    line_number=matched_line,
                    content=line_text,
                    provider=provider,
                ))
                claimed_lines.add(matched_line)
                seen_provider_in_file.add(provider)

        return findings

    def _detect_api_keys(self, file_path: str, content: str) -> list[Finding]:
        """Detect hardcoded API keys using regex patterns.

        Security: raw keys are never stored in the Finding. Both `content` and
        `redacted_content` receive the redacted form so that downstream code
        (LLM prompts, report templates, logs) cannot accidentally leak secrets.
        """
        findings = []

        for provider, pattern, description in API_KEY_PATTERNS:
            for match in pattern.finditer(content):
                key = match.group()
                redacted = _redact_key(key)
                line_num = content[:match.start()].count("\n") + 1
                findings.append(Finding(
                    type=FindingType.API_KEY_DETECTED,
                    file_path=file_path,
                    line_number=line_num,
                    content=redacted,
                    redacted_content=redacted,
                    provider=provider,
                ))

        return findings

    # ── Sprint 2 detectors ─────────────────────────────────────────────

    def _local_model_finding(self, full_path: Path, rel_path: str) -> Finding:
        """Record the presence of a local model weight artifact.

        We never load the file. Only its relative path and size are
        stored, so scanning a 70 GB .gguf never spikes memory.
        """
        try:
            size = full_path.stat().st_size
        except OSError:
            size = 0
        return Finding(
            type=FindingType.LOCAL_MODEL_DETECTED,
            file_path=rel_path,
            content=f"{full_path.name} ({_human_size(size)})",
            provider=_model_ext_to_provider(full_path.suffix.lower()),
        )

    def _detect_containers(self, file_path: str, content: str) -> list[Finding]:
        """Detect AI runtime / vector DB images in Dockerfile & compose files."""
        findings: list[Finding] = []
        seen: set[str] = set()
        for pattern, provider, label in _CONTAINER_IMAGE_PATTERNS:
            match = pattern.search(content)
            if not match:
                continue
            if provider in seen:
                continue
            seen.add(provider)
            line_num = content[:match.start()].count("\n") + 1
            findings.append(Finding(
                type=FindingType.CONTAINER_DETECTED,
                file_path=file_path,
                line_number=line_num,
                content=label,
                provider=provider,
            ))
        return findings

    def _detect_mcp_config(self, file_path: str, content: str) -> list[Finding]:
        """Detect MCP server configuration files.

        Every ``mcpServers`` entry in the config becomes one finding. The
        server name goes into ``content`` so the asset card can list each
        configured tool explicitly.
        """
        findings: list[Finding] = []
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return findings
        servers = {}
        if isinstance(data, dict):
            servers = data.get("mcpServers") or data.get("mcp_servers") or {}
        if not isinstance(servers, dict):
            return findings
        for name in sorted(servers.keys()):
            findings.append(Finding(
                type=FindingType.CONFIG_DETECTED,
                file_path=file_path,
                content=f"mcp server: {name}",
                provider="mcp",
            ))
        if not findings and isinstance(data, dict):
            # Config file exists but has no server entries — still record it
            # so the report surfaces "MCP config present but empty".
            findings.append(Finding(
                type=FindingType.CONFIG_DETECTED,
                file_path=file_path,
                content="mcp config (no servers)",
                provider="mcp",
            ))
        return findings

    def _detect_config_model_refs(
        self, file_path: str, content: str
    ) -> list[Finding]:
        """Extract model / deployment references from plain YAML/TOML.

        Only strings that look like actual model IDs (contain a dash,
        slash or digit and aren't purely boolean/number values) are
        accepted. Deduped per file so a config that lists the same model
        in ``default_model`` and ``fallback_model`` emits one finding.
        """
        findings: list[Finding] = []
        seen: set[tuple[str, str]] = set()
        for pattern, provider in _CONFIG_MODEL_PATTERNS:
            for match in pattern.finditer(content):
                value = match.group(1).strip().strip("\"'")
                if not _is_plausible_model_ref(value):
                    continue
                key = (provider, value.lower())
                if key in seen:
                    continue
                seen.add(key)
                line_num = content[:match.start()].count("\n") + 1
                findings.append(Finding(
                    type=FindingType.CONFIG_DETECTED,
                    file_path=file_path,
                    line_number=line_num,
                    content=f"config: {value}",
                    provider=provider,
                ))
        return findings

    def _detect_ci_pipeline(
        self, file_path: str, content: str
    ) -> list[Finding]:
        """Detect AI workloads in CI/CD pipelines.

        Emits one CONFIG_DETECTED finding per distinct AI signal in the
        pipeline (credential, action/image reference, training command).
        De-duplicated by provider so a workflow that exports
        ``OPENAI_API_KEY`` twice produces a single finding.
        """
        findings: list[Finding] = []
        seen: set[tuple[str, str]] = set()
        for pattern, provider, label in _CI_AI_PATTERNS:
            match = pattern.search(content)
            if not match:
                continue
            key = (provider, label)
            if key in seen:
                continue
            seen.add(key)
            line_num = content[:match.start()].count("\n") + 1
            findings.append(Finding(
                type=FindingType.CONFIG_DETECTED,
                file_path=file_path,
                line_number=line_num,
                content=f"CI: {label}",
                provider=provider,
            ))
        return findings

    def _detect_azure_openai_config(
        self, file_path: str, content: str
    ) -> list[Finding]:
        """Detect Azure OpenAI usage that isn't caught by import patterns.

        The Python ``openai`` SDK is shared with Azure — many codebases use
        ``from openai import AzureOpenAI`` (caught by imports) but some
        configure Azure implicitly via env vars or kwargs. This scan
        surfaces those, producing an ``azure_openai`` import-type finding
        so the asset's provider is correctly attributed to Azure rather
        than OpenAI.
        """
        findings: list[Finding] = []
        markers = (
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "azure_endpoint=",
            "AZURE_OPENAI_DEPLOYMENT",
            "api_type='azure'",
            'api_type="azure"',
        )
        for marker in markers:
            idx = content.find(marker)
            if idx >= 0:
                line_num = content[:idx].count("\n") + 1
                findings.append(Finding(
                    type=FindingType.IMPORT_DETECTED,
                    file_path=file_path,
                    line_number=line_num,
                    content=marker,
                    provider="azure_openai",
                ))
                break  # one per file is enough
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
                for provider in _package_to_providers(pkg):
                    findings.append(Finding(
                        type=FindingType.DEPENDENCY_DETECTED,
                        file_path=file_path,
                        line_number=i,
                        content=line,
                        provider=provider,
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
                for provider in _package_to_providers(pkg):
                    findings.append(Finding(
                        type=FindingType.DEPENDENCY_DETECTED,
                        file_path=file_path,
                        content=dep.strip(),
                        provider=provider,
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
                for provider in _package_to_providers(pkg):
                    findings.append(Finding(
                        type=FindingType.DEPENDENCY_DETECTED,
                        file_path=file_path,
                        content=item,
                        provider=provider,
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
                    for provider in _package_to_providers(pkg):
                        findings.append(Finding(
                            type=FindingType.DEPENDENCY_DETECTED,
                            file_path=file_path,
                            content=f"{pkg}@{version}",
                            provider=provider,
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
            primary_provider = _pick_primary_provider(providers)
            # Collect dependencies
            deps = sorted({
                f.content for f in dir_findings
                if f.type == FindingType.DEPENDENCY_DETECTED
            })
            # Category bookkeeping
            has_api_keys = any(
                f.type == FindingType.API_KEY_DETECTED for f in dir_findings
            )
            has_mcp = any(f.provider == "mcp" for f in dir_findings)
            has_local_model = any(
                f.type == FindingType.LOCAL_MODEL_DETECTED for f in dir_findings
            )
            has_container = any(
                f.type == FindingType.CONTAINER_DETECTED for f in dir_findings
            )

            risk = 0.3
            if has_api_keys:
                risk = 0.7
            if has_mcp:
                risk = max(risk, 0.5)
            if has_local_model:
                risk = max(risk, 0.4)

            asset_type = AssetType.CUSTOM_CODE
            if has_mcp:
                asset_type = AssetType.MCP_SERVER
            elif has_local_model and not any(
                f.type == FindingType.IMPORT_DETECTED for f in dir_findings
            ):
                asset_type = AssetType.LOCAL_MODEL

            solution_name = _derive_solution_name(file_paths, primary_provider, repo_name)

            # Provider info: use primary, but list all in dependencies
            provider_info = ProviderInfo(name=primary_provider) if primary_provider else None
            all_provider_names = [get_provider(p).display_name for p in providers]

            assets.append(AIAsset(
                name=solution_name,
                type=asset_type,
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


def _write_askpass_helper(tmp_dir: str) -> str:
    """Write a short-lived GIT_ASKPASS helper script.

    The helper echoes ``$AISCOUT_GIT_TOKEN`` — the token itself is passed via
    the per-subprocess ``env`` dict handed to ``clone_from``, never touching
    the parent process environment, the clone URL, or the helper file on
    disk. The script is created with 0700 permissions in the scanner's temp
    dir (itself 0700) and is unlinked as soon as clone returns.
    """
    fd, path = tempfile.mkstemp(prefix="askpass_", suffix=".sh", dir=tmp_dir)
    try:
        with os.fdopen(fd, "w") as f:
            f.write("#!/bin/sh\nprintf '%s' \"$AISCOUT_GIT_TOKEN\"\n")
        os.chmod(path, 0o700)
    except Exception:
        os.unlink(path)
        raise
    return path


# More-specific providers supersede their generic parent so the asset card
# attributes data flow to the real destination (e.g. Azure tenant, not
# "OpenAI"). These entries only affect tie-breaking when BOTH providers
# show up in the same asset; they do not re-order unrelated providers.
_PROVIDER_SUPERSEDES = {
    "azure_openai": "openai",
}

_FRAMEWORK_PROVIDERS = {"langchain", "llamaindex", "semantic-kernel", "mcp"}


def _pick_primary_provider(providers: list[str]) -> str:
    """Pick the provider shown prominently on the asset card.

    Rules (in order):
    1. If a provider supersedes another present one (Azure OpenAI over
       OpenAI), drop the superseded provider.
    2. Prefer direct LLM/vector providers over generic frameworks
       (LangChain wraps the real destination — LangChain alone is
       uninformative for residency questions).
    3. Fall back to alphabetical order for deterministic ties.
    """
    if not providers:
        return ""
    pool = set(providers)
    for specific, general in _PROVIDER_SUPERSEDES.items():
        if specific in pool and general in pool:
            pool.discard(general)
    concrete = sorted(p for p in pool if p not in _FRAMEWORK_PROVIDERS)
    if concrete:
        return concrete[0]
    return sorted(pool)[0]


_KNOWN_MODEL_KEYWORDS = (
    "gpt", "claude", "gemini", "mistral", "llama", "qwen", "phi",
    "command", "cohere", "bedrock", "deepseek", "mixtral", "falcon",
    "whisper", "stable-diffusion", "dall-e", "gemma", "nova",
)


def _is_plausible_model_ref(value: str) -> bool:
    """Filter out obviously non-model strings extracted from YAML/TOML.

    YAML scanning is loose so a ``model: "primary"`` line would pass the
    regex. We keep only values that either contain a known model family
    substring, or match the shape of a deployment id (alphanumeric with
    at least one hyphen/dot/digit and length 4-80).
    """
    if not value or len(value) < 2 or len(value) > 80:
        return False
    low = value.lower()
    if low in {"true", "false", "null", "none", "default", "primary", "fallback"}:
        return False
    if any(kw in low for kw in _KNOWN_MODEL_KEYWORDS):
        return True
    # Looks like a deployment id: alphanum with hyphens AND at least one digit
    if re.fullmatch(r"[A-Za-z][\w.\-]{3,}", value) and any(c.isdigit() for c in value):
        return True
    return False


def _looks_like_ci_file(rel_path: str, filename: str) -> bool:
    """Heuristic — is this file a CI/CD pipeline manifest?"""
    rel_lower = rel_path.replace("\\", "/").lower()
    if ".github/workflows/" in rel_lower and filename.endswith((".yml", ".yaml")):
        return True
    if filename in (
        ".gitlab-ci.yml", ".gitlab-ci.yaml",
        "bitbucket-pipelines.yml", "azure-pipelines.yml", "cloudbuild.yaml",
        "Jenkinsfile",
    ):
        return True
    if rel_lower.startswith(".circleci/") and filename.endswith((".yml", ".yaml")):
        return True
    return False


def _human_size(num_bytes: int) -> str:
    """Format a byte count as a short human-readable string."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < step or unit == "TB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{int(num_bytes)} B"
        num_bytes /= step
    return f"{num_bytes:.1f} TB"


def _model_ext_to_provider(ext: str) -> str:
    """Map a local model extension to the most-likely runtime/framework."""
    return {
        ".gguf": "llamacpp",
        ".ggml": "llamacpp",
        ".safetensors": "huggingface",
        ".onnx": "onnx",
        ".bin": "huggingface",
        ".pt": "pytorch",
        ".pth": "pytorch",
        ".ckpt": "pytorch",
        ".tflite": "tflite",
        ".mlmodel": "coreml",
    }.get(ext, "local_model")


def _redact_key(key: str) -> str:
    """Redact an API key, showing only first 8 and last 4 characters."""
    if len(key) <= 12:
        return key[:4] + "..." + key[-2:]
    return key[:8] + "..." + key[-4:]


def _package_to_providers(pkg: str) -> list[str]:
    """Map a package name to one or more provider names.

    LangChain / llama-index integration packages declare both the wrapper
    framework **and** the real backend (e.g. ``langchain-openai`` → both
    ``openai`` and ``langchain``). Returning both keeps the framework in
    the tech stack while letting ``_pick_primary_provider`` attribute the
    asset to the real data destination.
    """
    backend = _package_to_provider(pkg)
    lc_framework_pkgs = {
        "langchain-openai", "langchain-anthropic",
        "langchain-google-genai", "langchain-google-vertexai",
        "langchain-mistralai", "langchain-cohere", "langchain-aws",
        "langchain-huggingface", "langchain-ollama",
        "langchain-chroma", "langchain-pinecone",
        "langchain-qdrant", "langchain-weaviate",
        "@langchain/openai", "@langchain/anthropic",
        "@langchain/google-genai",
    }
    if pkg in lc_framework_pkgs:
        return [backend, "langchain"]
    return [backend]


def _package_to_provider(pkg: str) -> str:
    """Map a package name to a provider name."""
    mapping = {
        "mcp": "mcp", "fastmcp": "mcp",
        "@modelcontextprotocol/sdk": "mcp",
        "@azure/openai": "azure_openai",
        "openai": "openai",
        "anthropic": "anthropic",
        # LangChain sub-packages declare the actual backend provider. We
        # attribute them to the backend so reports show the real data
        # destination — "LangChain" on its own is uninformative for
        # residency/training policy questions. The framework itself is
        # still surfaced via `langchain-core` / `langchain` imports.
        "langchain": "langchain", "langchain-core": "langchain",
        "langchain-community": "langchain",
        "langchain-openai": "openai",
        "langchain-anthropic": "anthropic",
        "langchain-google-genai": "google_ai",
        "langchain-google-vertexai": "google_ai",
        "langchain-mistralai": "mistral",
        "langchain-cohere": "cohere",
        "langchain-aws": "aws_bedrock",
        "langchain-huggingface": "huggingface",
        "langchain-ollama": "ollama",
        "langchain-chroma": "chromadb",
        "langchain-pinecone": "pinecone",
        "langchain-qdrant": "qdrant",
        "langchain-weaviate": "weaviate",
        "@langchain/openai": "openai",
        "@langchain/anthropic": "anthropic",
        "@langchain/google-genai": "google_ai",
        "@langchain/community": "langchain",
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
