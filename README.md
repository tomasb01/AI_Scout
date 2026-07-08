# AI Scout

**Enterprise AI Discovery & Security Assessment Tool**

AI Scout scans your Git repositories for AI-related code — imports, API keys, dependencies, data flows — and generates a self-contained HTML report with an executive summary, risk evidence, and a map of where your data goes. The core analysis is rule-based and deterministic; an LLM (your own endpoint or local Ollama) is optional enrichment.

Self-hosted. Transparent. Your LLM, your data. No data leaves your perimeter — not even to us.

## Quick Start (no LLM, ~2 minutes)

```bash
git clone https://github.com/tomasb01/AI_Scout && cd AI_Scout

# with uv (recommended)
uv sync
uv run aiscout scan --local /path/to/repo --no-llm --output report.html

# or with pip
pip install .
aiscout scan --local /path/to/repo --no-llm --output report.html
```

Open `report.html` — it is fully self-contained and works offline.

## With LLM enrichment (optional)

```bash
# Local Ollama — default model is qwen2.5-coder:7b (fits 8 GB RAM)
ollama pull qwen2.5-coder:7b
aiscout scan --local /path/to/repo --output report.html

# Better quality on 16 GB+ machines
ollama pull qwen2.5-coder:14b
aiscout scan --local /path/to/repo --llm-model qwen2.5-coder:14b --output report.html

# Any OpenAI-compatible endpoint (Azure OpenAI, vLLM, LocalAI, TGI, Groq, ...)
aiscout scan --local /path/to/repo --llm-mode openai --llm-url https://your-endpoint --llm-key KEY --llm-model MODEL --output report.html
```

## Common tasks

```bash
# Scan a remote repository
aiscout scan --repo https://github.com/org/repo --token YOUR_TOKEN --no-llm --output report.html

# Scan a whole GitHub organization (all token-visible repos)
aiscout scan --org acme --token ghp_xxx --no-llm --output report.html

# Low-sensitivity first scan — dependency manifests only, never source
aiscout scan --org acme --token ghp_xxx --manifests-only --output report.html

# Machine-readable output (auto-detected from extension)
aiscout scan --local /path/to/repo --no-llm --output report.json

# Developer guardrail for pre-commit / CI: fail on leaked keys or sensitive LLM egress
aiscout check --path .

# Web UI (3-step wizard)
aiscout web --port 8080
```

## YAML Config

For scanning multiple repositories:

```yaml
# repos.yaml
repositories:
  - url: https://github.com/org/backend-api
    branch: main
    token_env: GITHUB_TOKEN

  - url: https://github.com/org/data-pipeline
    branch: develop

  - path: /local/checkout/frontend

llm:
  mode: ollama
  url: http://localhost:11434
  model: qwen2.5-coder:7b

output:
  path: reports/company_ai_scan.html
```

## CLI Reference — `aiscout scan`

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--repo` / `-r` | Git repo URL (repeatable) | — |
| `--local` / `-l` | Local repo path (repeatable) | — |
| `--org` | GitHub org/user — scans all token-visible repos (repeatable) | — |
| `--config` / `-c` | YAML config file | — |
| `--token` / `-t` | Git access token | env `AISCOUT_GIT_TOKEN` |
| `--branch` / `-b` | Default branch | `main` |
| `--include-archived` / `--include-forks` | Widen `--org` scans | off |
| `--max-repos` | Cap on repos per `--org` | `200` |
| `--manifests-only` | Read only dependency manifests, never source | off |
| `--output` / `-o` | Output path (`.html` or `.json`) | `aiscout_report.html` |
| `--llm-url` | LLM API URL | `http://localhost:11434` |
| `--llm-model` | LLM model name | `qwen2.5-coder:7b` |
| `--llm-mode` | `ollama` or `openai` | `ollama` |
| `--llm-key` | Bearer token for OpenAI mode | env `AISCOUT_LLM_KEY` |
| `--no-llm` | Skip LLM classification | off |
| `--strict` | CI mode: non-zero exit if the report QA linter suppressed anything | off |

Other commands: `aiscout check --path . [--warn-only]` (guardrail), `aiscout web [--host] [--port]` (wizard UI).

## What It Detects

**AI Imports** — 18+ providers: OpenAI, Anthropic, LangChain, LlamaIndex, HuggingFace, Mistral, Cohere, Ollama, ChromaDB, Pinecone, Qdrant, Weaviate, Google AI, AWS Bedrock, and more.

**API Keys** — Hardcoded keys for OpenAI (`sk-`), Anthropic (`sk-ant-`), Google AI (`AIza`), HuggingFace (`hf_`), Replicate (`r8_`). Keys are redacted in the report.

**Dependencies** — AI packages in `requirements.txt`, `pyproject.toml`, `setup.py`, `package.json` — including known-vulnerable versions (offline advisory KB).

**Data flows** — sources → processing → sinks per solution, rule-based, no LLM required. Plus MCP configs, CI/CD pipelines, Docker manifests, local model artifacts.

**Repo character** — production / tutorial–example / experiment, so a course repo with 100 lesson folders reports as one teaching collection, not 100 AI solutions.

## Docker

```bash
docker compose up -d
docker exec ai-scout-ollama ollama pull qwen2.5-coder:7b
docker exec ai-scout aiscout scan --config /app/config/repos.yaml --output /app/reports/report.html
```

## Development

```bash
uv sync              # installs runtime + dev dependencies
uv run pytest -q     # run the test suite
```

## Architecture

```
aiscout/
├── cli.py                  # CLI: aiscout scan / check / web (Click + Rich)
├── scanners/
│   ├── base.py             # Scanner plugin interface
│   ├── git_scanner.py      # Git repository scanner (+ manifests-only)
│   └── github_org.py       # GitHub org/user repo enumeration
├── engine/
│   ├── code_analyzer.py    # Code Context Extractor (AST + regex)
│   ├── data_flow.py        # Data Flow Mapper (rule-based)
│   ├── enrichment.py       # Naming, categories, risk reasons, overlap
│   ├── aggregation.py      # Solution = application/service boundary
│   ├── repo_character.py   # production / tutorial / experiment detector
│   └── llm.py              # LLM engine (Ollama / OpenAI-compatible)
├── knowledge/
│   ├── providers.py        # Provider KB (30+ profiles, offline)
│   └── dependency_advisories.py
├── models/
│   └── assets.py           # Pydantic data models
├── report/
│   ├── html.py             # HTML report generator
│   ├── json_export.py      # JSON export
│   ├── insights.py         # Typed insight catalog I-01–I-10
│   ├── linter.py           # Report QA linter L-01–L-10
│   ├── qa.py               # QA pipeline (validate → render → lint → degrade)
│   └── templates/report.html.j2
└── web/
    └── app.py              # FastAPI wizard UI
```

## License

Business Source License 1.1 (BUSL-1.1)
