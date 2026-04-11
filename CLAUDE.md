# AI Scout

Enterprise AI Discovery & Security Assessment Tool.
Self-hosted, open-source (BSL), CLI-first.

## What is this

AI Scout automatically discovers and maps all AI solutions used in an organization — commercial SaaS (ChatGPT, Copilot, Claude), custom scripts, local LLMs, AI agents. It runs entirely on customer infrastructure, no data leaves the perimeter.

Key differentiator: Uses the customer's own LLM (enterprise API or local Ollama) for analysis. Even analytical data stays inside the perimeter.

## Tech stack

- Language: Python 3.11+
- Deployment: Docker Compose (Scout + Ollama + model)
- CLI framework: Click + Rich
- Data model: Pydantic
- Git operations: GitPython
- HTTP client: httpx (Ollama/OpenAI API)
- Report templating: Jinja2
- Output: Self-contained HTML report (no external dependencies, works offline)

## CLI usage

```bash
aiscout scan --local /path/to/repo --llm-model qwen2.5-coder:14b --output report.html
aiscout scan --repo https://github.com/org/repo --token TOKEN --no-llm --output report.html
```

Key CLI parameters: `--repo`/`--local` (source), `--llm-url`, `--llm-model`, `--llm-mode` (ollama|openai), `--llm-key`, `--no-llm`, `--output`, `--branch`, `--token`.

## Architecture (MVP)

Four layers: CLI → Git Scanner → LLM Engine → HTML Report.
Full architecture doc: `Architecture/AI_Scout_MVP_Architecture.md`

### Directory structure

```
ai-scout/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── aiscout/
│   ├── __init__.py
│   ├── cli.py                  # CLI (Click + Rich)
│   ├── scanners/
│   │   ├── base.py             # BaseScanner ABC
│   │   └── git_scanner.py      # Git Repository Scanner
│   ├── engine/
│   │   └── llm.py              # LLM Analysis Engine
│   ├── models/
│   │   └── assets.py           # AIAsset, ScanResult, DataFlow, ...
│   └── report/
│       └── html.py             # HTML report generator (Jinja2)
└── tests/
```

### Scanner plugin interface

```python
class BaseScanner(ABC):
    def get_config(self) -> ScannerConfig
    def scan(self, **kwargs) -> ScanResult
    def get_name(self) -> str
```

### Git Scanner — three sub-modules

1. **Import Detector** — AI-related imports in source code (18+ providers: OpenAI, Anthropic, LangChain, LlamaIndex, HuggingFace, Mistral, Cohere, Ollama, ChromaDB, Pinecone, etc.)
2. **API Key Detector** — regex patterns for hardcoded keys (OpenAI `sk-`, Anthropic `sk-ant-`, Google `AIza`, HuggingFace `hf_`, etc.). Keys are redacted in report.
3. **Dependency Scanner** — AI packages in `requirements.txt`, `pyproject.toml`, `setup.py`, `package.json`

Scanned extensions: `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.mjs`, `.cjs`, `.java`, `.cs`, `.go`, `.rs`, `.rb`, `.php`, `.yaml`, `.yml`, `.toml`, `.json`, `.env`, `.ipynb`
Skipped dirs: `.git`, `node_modules`, `__pycache__`, `.venv`, `venv`, `dist`, `build`
Max file size: 1 MB

### LLM Engine (MVP)

Two modes: (1) Enterprise API (OpenAI-compatible), (2) Local Ollama.
Pre-processing: format findings, limit to 20 findings per asset, redact API keys.
Output: `ClassificationResult` with categories, confidence, details, recommendations.
Fallback: scan works without LLM (`--no-llm`), report generated without classification data.

### Data model

Primary entity: `AIAsset` — id, name, type, owner, users, data_inputs/outputs, provider, risk_score, data_classification, discovered_via, file_path, repository, dependencies, raw_findings.
Helper models: `DataFlow`, `ProviderInfo`, `ClassificationResult`, `ScanResult`.
All in `aiscout/models/assets.py` using Pydantic.

## MVP scope (Phase 0)

- Scanner: Git Repository only (M365/Entra ID is Phase 1)
- LLM Engine: Mode 1 (Enterprise API) + Mode 2 (Local Ollama)
- Output: Self-contained HTML report with stats, risk heatmap, filters, asset cards
- Interface: CLI only
- Deployment: pip install + Ollama, or Docker Compose

## Implementation plan

1. Project init — `pyproject.toml`, directory structure, `.gitignore`
2. Data model — `models/assets.py`
3. Scanner interface — `scanners/base.py`
4. Git Scanner — `scanners/git_scanner.py`
5. LLM Engine — `engine/llm.py`
6. HTML Report — `report/html.py`
7. CLI — `cli.py`
8. Test on real repo — end-to-end demo
9. Dockerfile + docker-compose.yml
10. README + GitHub repo

## Core principles

- **Self-hosted only** — no SaaS component, no phone-home
- **Read-only access** — audit, not enforcement
- **Full transparency** — every operation logged, code auditable
- **AI-first focus** — exclusively AI tools and solutions
- **Your LLM, your data** — analysis via customer's LLM

## Project conventions

- Product spec: `Prod_specs/AI_Scout_Product_Spec_v6.docx`
- Architecture: `Architecture/AI_Scout_MVP_Architecture.md`
- Primary language for code/docs: English
- Product spec language: Czech
- License: BSL (Business Source License)

## Roadmap

| Phase | Timeframe | Focus |
|-------|-----------|-------|
| 0 MVP | Month 1-2 | Git scanner, LLM Engine, HTML report, CLI |
| 1 Expand | Month 3-4 | M365/Entra ID, Power Platform, Network/DNS, Google Workspace scanners, dashboard |
| 2 Analyze | Month 5-6 | Overlap & Gap, Endpoint/MCP scanners, sandboxed execution |
| 3 Secure | Month 7-8 | Security Assessment, risk scoring |
| 4 Monitor | Month 9-10 | Continuous monitoring, alerts, remediation roadmap |
| 5 Scale | Month 11+ | Enterprise connectors, custom scanner SDK |
