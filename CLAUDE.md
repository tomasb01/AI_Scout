# AI Scout

Enterprise AI Discovery & Security Assessment Tool.
Self-hosted, open-source (BSL), CLI + Web UI.

**Current status: v0.7.0** — functional end-to-end product, 5 sprints completed, ~116 tests passing.
Full status: `03_Documentation/PROJECT_STATUS.md` · Sprint detail: `03_Documentation/SPRINT_LOG.md`

## What is this

AI Scout automatically discovers and maps all AI solutions used in an organization — commercial SaaS (ChatGPT, Copilot, Claude), custom scripts, local LLMs, AI agents. It runs entirely on customer infrastructure, no data leaves the perimeter.

Key differentiator: Uses the customer's own LLM (enterprise API or local Ollama) for analysis. Even analytical data stays inside the perimeter. Core analysis is rule-based and works without any LLM; the LLM is optional enrichment.

Pitch: *"Here are 47 AI solutions in your company. Leadership doesn't know about 30 of them. 12 work with sensitive data. 8 send data outside the EU. Here's the map, risk assessment, and recommendations."*

## Who it's for (personas)

| Persona | Need | Entry point |
|---------|------|-------------|
| **Individual developer** | Don't leak keys / send sensitive data to an LLM while building | `aiscout check` (pre-commit hook + GitHub Action) |
| **Small team / startup** | See all AI across our org | `aiscout scan --org <name> --token <pat>` |
| **Mid-size / scale-up** | Low-trust first scan, then expand | `aiscout scan --org <name> --manifests-only` → opt-in deeper scan |
| **Enterprise / regulated** | Least-privilege access, full inventory, audit | `--org` + org-owner token / GitHub App (planned) — see `03_Documentation/GITHUB_ACCESS_STRATEGY.md` |

## Tech stack

- Language: Python 3.11+ (Docker image: 3.12-slim)
- Deployment: Docker Compose (Scout + Ollama + model); also `pip`/`uv`; landing on Vercel
- CLI framework: Click + Rich
- Web UI: FastAPI + sse-starlette (SSE progress)
- Data model: Pydantic v2
- Git operations: GitPython (lazy-imported)
- HTTP client: httpx (Ollama / OpenAI-compatible API)
- Report templating: Jinja2
- Output: Self-contained HTML report (no CDN, all inline, works offline) + JSON export
- Package/dev tooling: `uv`, pytest

## CLI usage

```bash
uv sync

# Scan without LLM (rule-based only)
uv run aiscout scan --repo https://github.com/org/repo --no-llm --output report.html

# JSON export (auto-detected from .json extension)
uv run aiscout scan --repo https://github.com/org/repo --no-llm --output report.json

# With local Ollama
uv run aiscout scan --repo https://github.com/org/repo --llm-model qwen2.5-coder:7b --output report.html

# With OpenAI-compatible API (Azure OpenAI, vLLM, LocalAI, TGI, Groq, ...)
uv run aiscout scan --repo ... --llm-mode openai --llm-url https://api.openai.com --llm-key sk-... --llm-model gpt-4o-mini --output report.html

# Scan a whole GitHub organization / user (all token-visible repos)
uv run aiscout scan --org acme --token ghp_xxx --no-llm --output report.html

# Low-sensitivity first scan — dependency manifests only, never source
uv run aiscout scan --org acme --token ghp_xxx --manifests-only --output report.html

# Developer guardrail (pre-commit / CI): fail on leaked keys or sensitive LLM egress
uv run aiscout check --path .          # exit 1 on issues; --warn-only to never fail

# Web UI (3-step wizard)
uv run aiscout web --port 8080

# Tests
uv run pytest tests/ -q
```

Key CLI parameters:
- `aiscout scan` — `--repo`/`--local`/`--org` (source; multi-repo + YAML config supported), `--include-archived`/`--include-forks`/`--max-repos` (org filters), `--manifests-only`, `--llm-url`, `--llm-model`, `--llm-mode` (ollama|openai), `--llm-key`, `--no-llm`, `--output` (.html/.json auto-detect), `--branch`, `--token`.
- `aiscout check` — `--path` (default `.`), `--warn-only`. Rule-based, no network; exits 1 on hardcoded keys or sensitive data sent to an external LLM (local runtimes like Ollama exempt).
- `aiscout web` — `--host`, `--port`.

## Architecture (implemented)

```
INPUT      CLI (Click) · Web UI (FastAPI) · YAML config
   ↓
DISCOVERY  Git Scanner → Code Context Extractor
           (future: M365/Entra ID, Network/DNS, Endpoint, GWS)
   ↓
ANALYSIS   Data Flow Mapper (rule-based) · LLM Engine (optional enrichment)
           Enrichment (risk, categories, tech stack, overlap, tags, task types)
           Provider KB (30+) + Dependency Advisories
   ↓
OUTPUT     HTML Report (graph, data flow, analytics) · JSON Export
```

Architecture docs: `02_Architecture/` (`00_System_Overview.md`, `01_Git_Scanner_MVP.md`, `02_Data_Flow_Mapper.md`).

### Directory structure

```
AI_Scout/
├── pyproject.toml · Dockerfile · docker-compose.yml · vercel.json · uv.lock
├── .pre-commit-hooks.yaml              # `ai-scout-guardrail` hook for other repos
├── 01_Prod_specs/                      # Product spec (latest: v10), [Archive]/ older
├── 02_Architecture/                    # Architecture docs
├── 03_Documentation/                   # PROJECT_STATUS.md, SPRINT_LOG.md, GITHUB_ACCESS_STRATEGY.md
├── aiscout/
│   ├── cli.py                          # CLI: aiscout scan / check / web
│   ├── scanners/
│   │   ├── base.py                     # BaseScanner ABC
│   │   ├── git_scanner.py              # Git Repository Scanner (+ manifests-only mode)
│   │   └── github_org.py               # GitHub org/user repo enumeration (REST API)
│   ├── engine/
│   │   ├── code_analyzer.py            # Code Context Extractor (AST + regex)
│   │   ├── data_flow.py                # Data Flow Mapper (rule-based, ~706 lines)
│   │   ├── enrichment.py               # Risk, categories, overlap, tags, task types
│   │   └── llm.py                      # LLM Engine (Ollama / OpenAI-compatible)
│   ├── knowledge/
│   │   ├── providers.py                # Provider KB (30+ profiles)
│   │   └── dependency_advisories.py    # Offline vulnerable-version KB
│   ├── models/
│   │   └── assets.py                   # Pydantic models
│   ├── report/
│   │   ├── html.py                     # HTML report generator (+ GitHub Coverage section)
│   │   ├── json_export.py              # JSON export
│   │   └── templates/report.html.j2    # Dashboard template
│   └── web/
│       ├── app.py                      # FastAPI server
│       └── templates/index.html        # Scanner wizard
├── examples/ai-scout-guardrail.yml     # GitHub Action template for the guardrail
├── landing/                            # Landing page + screenshots
├── api/ · scripts/
└── tests/                              # ~139 tests across 11 files
```

### GitHub org scanning & developer guardrail

- **Org enumeration** (`scanners/github_org.py`): `--org <name>` resolves a GitHub org/user into all token-visible repos via REST (`/orgs` → `/users` fallback, `Link` pagination), skips archived/forks by default, caps with `--max-repos`, feeds the existing multi-repo loop. Only repos the token can see are returned.
- **Manifest-only** (`--manifests-only`): reads only dependency manifests, never source (skips code-context analysis too) — a low-sensitivity first scan for security sign-off.
- **GitHub Coverage** report section: repos found / scanned / skipped, with a token-visibility disclaimer.
- **Guardrail** (`aiscout check`): rule-based, network-free pre-commit/CI gate. Exits non-zero on hardcoded keys or sensitive data (personal/financial/medical/credentials) sent to an *external* LLM (local runtimes like Ollama exempt). Ships `.pre-commit-hooks.yaml` + `examples/ai-scout-guardrail.yml`.

### Scanner plugin interface

```python
class BaseScanner(ABC):
    def get_config(self) -> ScannerConfig
    def scan(self, **kwargs) -> ScanResult
    def get_name(self) -> str
```

### Git Scanner — three sub-modules

1. **Import Detector** — AI-related imports (18+ providers: OpenAI, Anthropic, LangChain, LlamaIndex, HuggingFace, Mistral, Cohere, Ollama, ChromaDB, Pinecone, etc.)
2. **API Key Detector** — regex patterns for hardcoded keys (OpenAI `sk-`, Anthropic `sk-ant-`, Google `AIza`, HuggingFace `hf_`, etc.). Keys redacted in findings/report.
3. **Dependency Scanner** — AI packages in `requirements.txt`, `pyproject.toml`, `setup.py`, `package.json`.

Plus: directory-based grouping (each dir with AI code = one solution), Git author extraction, MCP config detection.

Scanned extensions: `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.mjs`, `.cjs`, `.java`, `.cs`, `.go`, `.rs`, `.rb`, `.php`, `.yaml`, `.yml`, `.toml`, `.json`, `.env`, `.ipynb`
Skipped dirs: `.git`, `node_modules`, `__pycache__`, `.venv`, `venv`, `dist`, `build`. Max file size: 1 MB. Clone depth: 10.

### Code Context Extractor

Python AST parsing + regex fallback for JS/TS. Extracts functions, classes, API calls, system prompts, data sources/sinks, env vars, LLM model names. Reads all files in solution dir + README.md. Three layers: AST (Python) → regex (all langs) → heuristics.

### Data Flow Mapper (rule-based, no LLM required)

Builds `DataFlowMap` from `CodeContext`: sources (DB, files, API inputs) → processing steps → sinks (AI APIs, DB writes, file outputs). Data category detection, confidence scoring (high/medium/low), solution-purpose synthesis. Overlap detection via fingerprinting (same sinks + same data categories). LLM optionally enriches the rule-based output.

### LLM Engine

Two modes: (1) Enterprise / OpenAI-compatible API (Azure OpenAI, AWS Bedrock, vLLM, LocalAI, TGI, Groq), (2) Local Ollama. Default model: `qwen2.5-coder:7b` (fits 8 GB RAM). Pre-processing redacts API keys (`<REDACTED_API_KEY>`), prompt-injection defense via `<untrusted>` tags + sanitization. Health check, batch classify, rate limiting. Fallback: full pipeline works with `--no-llm`.

### Data model

Primary entity: `AIAsset` — id, name, type, owner/author, users, data_inputs/outputs, provider, risk_score, data_classification, discovered_via, file_path, repository, dependencies, raw_findings, plus `data_flow` (DataFlowMap), `task_types`, `tags`, `code_contexts`.
Helper models: `DataFlowMap`, `FlowSource`, `FlowSink`, `CodeContext`, `TaskType` enum, `Finding`, `ProviderInfo`, `ClassificationResult`, `ScanResult`. All Pydantic v2 in `aiscout/models/assets.py`.

## Security (implemented)

API keys never stored raw (redacted in `Finding`, `<REDACTED_API_KEY>` in prompts) · Git tokens via `GIT_ASKPASS` helper, never in URL · symlinks skipped · path traversal blocked · temp dirs `chmod 0700` · URL scheme whitelist + SSRF block · LLM prompt-injection defense.

## Core principles

- **Self-hosted only** — no SaaS component, no phone-home
- **Read-only access** — audit, not enforcement
- **Full transparency** — every operation logged, code auditable (BSL core)
- **AI-first focus** — exclusively AI tools and solutions
- **Your LLM, your data** — analysis via customer's LLM, data stays in perimeter
- **Rule-based first, LLM as enrichment** — no hard LLM dependency
- **Self-contained HTML report** — no CDN, opens offline

## Key decisions

1. Directory-based grouping (not provider-based)
2. Solution names from code purpose: README → prompts → function names → directory
3. Tech-stack deduplication: model name suppresses provider (GPT-4o → drops "OpenAI" tag)
4. Overlap via DataFlowMap fingerprinting
5. Rule-based first, LLM optional
6. Default model `qwen2.5-coder:7b` (8 GB RAM)
7. Lazy imports (GitPython only on scan → enables Vercel landing)
8. Security by default

## Project conventions

- Product spec: `01_Prod_specs/AI_Scout_Product_Spec_v10.docx` (latest)
- Architecture: `02_Architecture/`
- Status / sprint log: `03_Documentation/`
- Primary language for code/docs: English · Product spec language: Czech
- License: BSL (Business Source License, BUSL-1.1)

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| 0 MVP | ✅ done (v0.7.0) | Git scanner, Code Context, Data Flow Mapper, LLM Engine, Enrichment, HTML/JSON report, Web UI, CLI, Docker |
| — | ✅ recent | GitHub org/user enumeration (`--org`), manifest-only scan, GitHub Coverage report, developer guardrail (`aiscout check` + pre-commit + Action) |
| — | ⚠️ in progress | Report redesign (3 prototypes A/B/C), risk-scoring calibration on more repos |
| 1 Expand | ⏳ | GitHub API scanner (read via REST, not clone), M365/Entra ID, Power Platform, Network/DNS, Google Workspace, Endpoint, MCP/Agent scanners; GitHub App auth for least-privilege org access |
| 2 Analyze | ⏳ | Instrumented execution (sandboxed runtime analysis), Data Classification Modes (schema-only / sampled / customer-exec) |
| 3 Secure | ⏳ | Security Assessment module (API key mgmt, data exposure, input sanitization, access control, compliance, architectural review) |
| 4 Monitor | ⏳ | Continuous monitoring / Watch Mode, alerts, Remediation Roadmap, Scout Cloud Engine (Mode 3) |
| 5 Scale | ⏳ | Enterprise connectors, custom scanner SDK |
