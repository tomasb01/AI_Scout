# AI Scout

**Enterprise AI Discovery & Security Assessment Tool**

AI Scout scans your Git repositories for AI-related code — imports, API keys, dependencies — classifies findings via LLM, and generates a self-contained HTML report.

Self-hosted. Transparent. Your LLM, your data.

## Quick Start

```bash
# Install
pip install -e .

# Scan a local repository (no LLM)
aiscout scan --local /path/to/repo --no-llm --output report.html

# Scan with LLM classification (requires Ollama)
ollama pull qwen2.5-coder:14b
aiscout scan --local /path/to/repo --output report.html

# Scan a remote repository
aiscout scan --repo https://github.com/org/repo --token YOUR_TOKEN --output report.html

# Scan multiple repositories
aiscout scan --local /repo1 --local /repo2 --repo https://github.com/org/repo3 --output report.html

# Scan from YAML config
aiscout scan --config repos.yaml --output report.html
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
  model: qwen2.5-coder:14b

output:
  path: reports/company_ai_scan.html
```

## CLI Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--repo` / `-r` | Git repo URL (repeatable) | — |
| `--local` / `-l` | Local repo path (repeatable) | — |
| `--config` / `-c` | YAML config file | — |
| `--token` / `-t` | Git access token | env `AISCOUT_GIT_TOKEN` |
| `--branch` / `-b` | Default branch | `main` |
| `--output` / `-o` | Output report path | `aiscout_report.html` |
| `--llm-url` | LLM API URL | `http://localhost:11434` |
| `--llm-model` | LLM model name | `qwen2.5-coder:14b` |
| `--llm-mode` | `ollama` or `openai` | `ollama` |
| `--llm-key` | API key for OpenAI mode | env `AISCOUT_LLM_KEY` |
| `--no-llm` | Skip LLM classification | `false` |

## What It Detects

**AI Imports** — 18+ providers: OpenAI, Anthropic, LangChain, LlamaIndex, HuggingFace, Mistral, Cohere, Ollama, ChromaDB, Pinecone, Qdrant, Weaviate, Google AI, AWS Bedrock, and more.

**API Keys** — Hardcoded keys for OpenAI (`sk-`), Anthropic (`sk-ant-`), Google AI (`AIza`), HuggingFace (`hf_`), Replicate (`r8_`). Keys are redacted in the report.

**Dependencies** — AI packages in `requirements.txt`, `pyproject.toml`, `setup.py`, `package.json`.

## Docker

```bash
docker compose up -d
docker exec ai-scout-ollama ollama pull qwen2.5-coder:14b
docker exec ai-scout aiscout scan --config /app/config/repos.yaml --output /app/reports/report.html
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# With uv
uv run pytest -v
```

## Architecture

```
aiscout/
├── cli.py              # CLI entry point (Click + Rich)
├── scanners/
│   ├── base.py         # Scanner plugin interface
│   └── git_scanner.py  # Git repository scanner
├── engine/
│   └── llm.py          # LLM classification engine
├── models/
│   └── assets.py       # Pydantic data models
└── report/
    ├── html.py         # Report generator
    └── templates/
        └── report.html.j2  # HTML template
```

## License

Business Source License 1.1 (BUSL-1.1)
