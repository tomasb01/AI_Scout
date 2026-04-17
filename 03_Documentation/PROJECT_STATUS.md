# AI Scout — Project Status & Documentation

**Last updated: April 17, 2026 | Version: 0.6.0 + Sprinty 1–4**

> Detailní log čtyř bezpečnostních a kvalitativních sprintů (12.–15. 4. 2026): viz **[SPRINT_LOG.md](SPRINT_LOG.md)** (107 testů, risk rework, LLM e2e, MCP server/client, CI/CD scanner, dep advisories).

---

## What AI Scout Is

AI Scout is a self-hosted, open-source tool that automatically discovers, maps, and assesses all AI solutions in an organization's Git repositories. It scans code for AI integrations (imports, API keys, dependencies), analyzes what each solution does through deep code analysis, and generates an interactive HTML report with executive summary, risk assessment, overlap detection, and visual analytics.

**Motto:** Visibility. Efficiency. Security.

---

## What's Been Built (v0.1 → v0.6)

### Core Engine

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| **Git Scanner** | `aiscout/scanners/git_scanner.py` | ✅ Done | Clones repos, detects AI imports (18+ providers), API keys (with redaction), dependencies. Groups findings by solution directory. Extracts git authors. |
| **Scanner Base** | `aiscout/scanners/base.py` | ✅ Done | Abstract base class for future scanners (M365, Network/DNS, etc.) |
| **Code Context Extractor** | `aiscout/engine/code_analyzer.py` | ✅ Done | Python AST parsing + regex fallback for JS/TS. Extracts functions, classes, API calls, system prompts, data sources/sinks, env vars, LLM model names. Reads ALL files in solution directory + README.md. |
| **LLM Engine** | `aiscout/engine/llm.py` | ✅ Done | Ollama + OpenAI-compatible API modes. Prompt includes full code context (functions, prompts, API calls, data flows). Health check, batch classify, rate limiting, graceful fallback. Default model: qwen2.5-coder:7b. |
| **Enrichment** | `aiscout/engine/enrichment.py` | ✅ Done | Solution naming (from code purpose, not framework), category classification (7 categories), tech stack extraction, data involved detection, risk reasoning with specific explanations, recommendations, overlap detection. |
| **Provider Knowledge Base** | `aiscout/knowledge/providers.py` | ✅ Done | 30+ AI provider profiles: display name, vendor, data residency, training policy, certifications, free tier risk, enterprise notes. |
| **Data Model** | `aiscout/models/assets.py` | ✅ Done | Pydantic models: AIAsset, CodeContext, Finding, DataFlow, ProviderInfo, ClassificationResult, ScanResult, ScannerConfig. |

### Report & Visualization

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| **HTML Report** | `aiscout/report/html.py` + `templates/report.html.j2` | ✅ Done | Self-contained dark-theme dashboard. |
| **Executive Summary** | In report | ✅ Done | Auto-generated bullet points: total solutions, overlaps, data egress to US, SPOF authors, tech concentration, sensitive data. |
| **AI Solutions Map** | In report (canvas JS) | ✅ Done | Force-directed graph with 3 views (Solutions/Tech Stack/People). Category clusters with background circles. Sidebar filter. Draggable nodes. Overlap edges (orange) + tech edges (purple). |
| **Analytics** | In report | ✅ Done | Collapsible section: Tech Stack radar (bar chart), Data Types Processed (sensitivity heatmap), Author Coverage (SPOF detection), Functional Overlap (expandable cards with file paths + consolidation recommendation). |
| **Solutions Table** | In report | ✅ Done | Columns: Solution, Repo, Built On, Author, Data Involved, Risk + compliance flags (PII, FIN, US). Click to expand detail: risk analysis, recommendations, provider info, findings, GitHub links. |

### Web UI & Deployment

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| **Web UI** | `aiscout/web/app.py` + `templates/index.html` | ✅ Done | FastAPI 3-step wizard: Repositories → LLM Config (No LLM / Ollama / API Key) → Scan with real-time SSE progress. Report rendered inline. |
| **Landing Page** | `landing/index.html` + `landing/screenshots/` | ✅ Done | Sales pitch: hero, screenshots, role-based benefits (CEO → CTO → DevOps → CISO), features, pricing (Free / Pro TBA), CTA. |
| **CLI** | `aiscout/cli.py` | ✅ Done | `aiscout scan` (multi-repo, YAML config, LLM options) + `aiscout web` (start web server). |
| **Docker** | `Dockerfile` + `docker-compose.yml` | ✅ Done | Python 3.12-slim + git. Default: `aiscout web --port 8080`. |
| **Vercel** | `api/index.py` + `vercel.json` | ⚠️ Partial | Landing page works, scan doesn't (no git binary on Vercel serverless). |

### Tests

| File | Tests | What it covers |
|------|-------|----------------|
| `tests/test_models.py` | 8 | UUID generation, merge, enum serialization |
| `tests/test_git_scanner.py` | 10 | Import/key/dependency detection, file walking, grouping |
| `tests/test_code_analyzer.py` | 9 | AST parsing, prompts, API calls, DB operations, JS analysis |
| `tests/test_llm_engine.py` | 7 | Ollama/OpenAI mock calls, parse failure, health check, truncation |
| `tests/test_enrichment.py` | 8 | Summary generation, risk scoring, categories, provider KB |
| `tests/test_report.py` | 5 | HTML generation, risk counts, overlap, empty scan |
| `tests/test_cli.py` | 4 | Version, no repos error, local scan, YAML config |
| **Total** | **51** | All passing |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  INPUT LAYER                         │
│  CLI (Click)  ·  Web UI (FastAPI)  ·  YAML Config   │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              DISCOVERY ENGINE                        │
│  Git Scanner → Code Context Extractor                │
│  (future: M365, Network/DNS, Endpoint scanners)      │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              ANALYSIS ENGINE                         │
│  LLM Engine (Ollama/OpenAI) → Enrichment             │
│  Provider Knowledge Base (30+ providers)              │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              OUTPUT ENGINE                           │
│  HTML Report (graph, analytics, exec summary)        │
│  (future: JSON/CSV export, dashboard)                │
└─────────────────────────────────────────────────────┘
```

Architecture documents: `02_Architecture/`
- `00_System_Overview.md` — high-level system view
- `01_Git_Scanner_MVP.md` — Git Scanner architecture
- `02_Data_Flow_Mapper.md` — Data Flow Mapper design (not yet implemented)

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v0.1.0 | Apr 11 | MVP: Git Scanner, LLM Engine, HTML Report, CLI, 42 tests |
| v0.2.0 | Apr 11 | Code Context Extractor (AST parsing, prompt extraction) |
| v0.3.0 | Apr 11 | Directory-based grouping, solution-focused dashboard, categories |
| v0.4.0 | Apr 11 | Graph visualization, exec summary, analytics, overlap detection |
| v0.4.1 | Apr 11 | LLM prompt with full code context, default qwen2.5-coder:7b |
| v0.5.0 | Apr 11 | Web UI (FastAPI wizard), repo column, graph improvements |
| v0.5.1 | Apr 11 | Docker deployment ready |
| v0.6.0 | Apr 11 | Landing page with screenshots, role-based benefits |

---

## How to Run

### Local (development)
```bash
# Install
uv sync

# CLI scan
uv run aiscout scan --repo https://github.com/org/repo --no-llm --output report.html

# CLI scan with LLM
uv run aiscout scan --repo https://github.com/org/repo --llm-model qwen2.5-coder:7b --output report.html

# Web UI
uv run aiscout web --port 8080
# Open http://localhost:8080

# Tests
uv run pytest tests/ -v
```

### Docker (production)
```bash
git clone https://github.com/tomasb01/AI_Scout.git
cd AI_Scout
docker compose up -d
# Open http://<server-ip>:8080
```

---

## What's NOT Built Yet — Next Steps

### Priority 1: LLM Integration Testing
- **What:** End-to-end test with real Ollama (qwen2.5-coder:7b)
- **Why:** Code is ready but Ollama was crashing during session. LLM will dramatically improve solution descriptions.
- **Effort:** Small — just needs working Ollama instance
- **Impact:** Highest — descriptions go from "Stock Price & Dividend Date" to detailed paragraph about what the solution does

### Priority 2: Data Flow Mapper
- **What:** Implement `aiscout/engine/data_flow.py` — builds source → processing → sink model for each solution
- **Why:** Currently we extract code context but don't visualize data flows
- **Architecture:** Already designed in `02_Architecture/02_Data_Flow_Mapper.md`
- **Effort:** Medium
- **Impact:** High — shows exactly what data goes where

### Priority 3: GitHub API Scanner (alternative to git clone)
- **What:** Scanner that uses GitHub REST API instead of `git clone`
- **Why:** Works on serverless (Vercel), faster for large repos, no git binary needed
- **Effort:** Medium
- **Impact:** Medium — enables cloud deployment without Docker

### Priority 4: JSON/CSV Export
- **What:** Machine-readable output alongside HTML report
- **Why:** Integration with SIEM, compliance tools, dashboards
- **Effort:** Small
- **Impact:** Medium

### Priority 5: M365 / Entra ID Scanner
- **What:** Scan OAuth app registrations, enterprise apps, AI-related consent grants
- **Why:** Covers commercial SaaS AI tools (ChatGPT, Copilot) that Git scanner can't see
- **Effort:** Large — requires Microsoft Graph API integration
- **Impact:** High — biggest coverage gap

### Priority 6: Continuous Monitoring (Watch Mode)
- **What:** Background service that re-scans periodically, detects new AI tools
- **Why:** One-time scan becomes outdated
- **Effort:** Medium
- **Impact:** Medium — Pro tier feature

### Priority 7: Security Assessment Module
- **What:** Deep security analysis — input sanitization, prompt injection checks, access control review
- **Why:** Pro tier value proposition
- **Effort:** Large
- **Impact:** High — Pro tier feature

---

## Key Files Reference

```
AI_Scout/
├── 01_Prod_specs/                    # Product specification (v7)
├── 02_Architecture/                  # Architecture documents
│   ├── 00_System_Overview.md
│   ├── 01_Git_Scanner_MVP.md
│   └── 02_Data_Flow_Mapper.md
├── 03_Documentation/
│   └── PROJECT_STATUS.md             # ← THIS FILE
├── aiscout/
│   ├── cli.py                        # CLI commands (scan, web)
│   ├── engine/
│   │   ├── code_analyzer.py          # AST parser + regex extractor
│   │   ├── enrichment.py             # Risk, summary, categories, overlap
│   │   └── llm.py                    # Ollama + OpenAI LLM client
│   ├── knowledge/
│   │   └── providers.py              # 30+ provider profiles
│   ├── models/
│   │   └── assets.py                 # Pydantic data models
│   ├── report/
│   │   ├── html.py                   # Report generator + analytics
│   │   └── templates/report.html.j2  # Dashboard template
│   ├── scanners/
│   │   ├── base.py                   # Scanner ABC
│   │   └── git_scanner.py            # Git repository scanner
│   └── web/
│       ├── app.py                    # FastAPI server
│       └── templates/index.html      # Scanner wizard UI
├── landing/
│   ├── index.html                    # Sales landing page
│   └── screenshots/                  # Demo screenshots
├── tests/                            # 107 tests (Sprinty 1–4)
├── Dockerfile                        # Docker deployment
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Key Decisions Made

1. **Directory-based grouping over provider-based** — each directory with AI code = one solution. Shows 136 individual solutions instead of 7 provider groups.

2. **Solution names from code, not framework** — "Stock Price & Dividend Date" instead of "OpenAI Integration". Uses README → prompts → function names → directory as fallback chain.

3. **Rule-based analysis first, LLM as enrichment** — works without LLM (--no-llm), LLM only improves descriptions. No hard dependency on external services.

4. **Self-contained HTML report** — no external dependencies, opens offline, all CSS/JS inline. Graph uses canvas, not D3.js.

5. **Default model qwen2.5-coder:7b** — fits 8 GB RAM, good code understanding, reliable JSON output.

6. **Lazy imports for serverless** — GitPython imported only when scanning, not at startup. Enables Vercel landing page.
