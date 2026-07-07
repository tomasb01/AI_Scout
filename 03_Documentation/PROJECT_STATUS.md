# AI Scout — Project Status & Documentation

**Last updated: May 21, 2026 | Version: 0.7.0 (+ org-scan / guardrail on branch) | Sprints 1–5 completed**

> Detailní log sprintů: viz **[SPRINT_LOG.md](SPRINT_LOG.md)** (security hardening, Data Flow Mapper, LLM e2e validace, risk rework, CI/CD scanner, dep advisories, report redesign prototypy). Currently **139 tests** passing (116 at v0.7.0 + org enumeration, manifest-only, GitHub Coverage, developer guardrail).

---

## What AI Scout Is

AI Scout is a self-hosted, open-source tool that automatically discovers, maps, and assesses all AI solutions in an organization's Git repositories. It scans code for AI integrations (imports, API keys, dependencies), analyzes what each solution does through deep code analysis, builds data flow maps (sources → processing → sinks), and generates an interactive HTML report with executive summary, risk assessment, overlap detection, and visual analytics.

**Motto:** Visibility. Efficiency. Security.

---

## Who It's For (at a glance)

One tool, four audiences — pick the entry point that matches the trust you can get.

| Persona | Their need | What AI Scout gives them | How they run it |
|---------|-----------|--------------------------|-----------------|
| **Individual developer** | Don't leak keys or send sensitive data to an LLM while building | Fast local guardrail, fully rule-based, no network | `aiscout check` · pre-commit hook · GitHub Action (`examples/ai-scout-guardrail.yml`) |
| **Small team / startup** | See every AI solution across our org | Whole-org discovery in one report | `aiscout scan --org <name> --token <pat>` |
| **Mid-size / scale-up** | A low-trust first scan, then expand | Manifest-first scan (reads only dependency files, never source) | `aiscout scan --org <name> --manifests-only` → opt-in deeper scan |
| **Enterprise / regulated** | Least-privilege access, full inventory, audit | Org scan + "GitHub Coverage" inventory in report; tiered access model | `--org` + org-owner token / GitHub App (planned). See **[GITHUB_ACCESS_STRATEGY.md](GITHUB_ACCESS_STRATEGY.md)** |

Access approaches per organization size (individual → enterprise) are detailed in
**[GITHUB_ACCESS_STRATEGY.md](GITHUB_ACCESS_STRATEGY.md)**.

---

## What's Been Built (v0.1 → v0.7)

### Core Engine

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| **Git Scanner** | `scanners/git_scanner.py` | ✅ | Clones repos (depth=10), detects AI imports (18+ providers), API keys (redacted), dependencies. Groups by solution directory. Git author extraction. Security: GIT_ASKPASS, symlink guard, path traversal protection. |
| **Scanner Base** | `scanners/base.py` | ✅ | Abstract base class for future scanners (M365, Network/DNS, etc.) |
| **Code Context Extractor** | `engine/code_analyzer.py` | ✅ | Python AST parsing + regex fallback for JS/TS. Extracts functions, classes, API calls, system prompts, data sources/sinks, env vars, LLM model names. Reads ALL files in solution dir + README.md. |
| **Data Flow Mapper** | `engine/data_flow.py` | ✅ Sprint 5 | Rule-based DataFlowMap builder (706 lines). Sources → processing steps → sinks from CodeContext. Data category detection, confidence scoring, solution purpose synthesis. No LLM required. |
| **LLM Engine** | `engine/llm.py` | ✅ | Ollama + any OpenAI-compatible endpoint (vLLM, LocalAI, TGI, Groq). Prompt injection defense (`<untrusted>` tags). API keys replaced with `<REDACTED_API_KEY>`. Health check, batch classify, rate limiting. Default: qwen2.5-coder:7b. |
| **Enrichment** | `engine/enrichment.py` | ✅ | Solution naming from code purpose. Category classification (7 categories). Tech stack extraction with deduplication (model suppresses provider). Data involved detection. Risk reasoning. Task type detection (inference/training/fine_tuning). Tags. Overlap detection via DataFlowMap fingerprinting. |
| **Provider KB** | `knowledge/providers.py` | ✅ | 30+ provider profiles: OpenAI, Anthropic, Google, Mistral, Cohere, HuggingFace, LangChain, LlamaIndex, Ollama, ChromaDB, Pinecone, Qdrant, Weaviate, AWS Bedrock, Azure OpenAI, CrewAI, AutoGen, Semantic Kernel, DSPy, Instructor, Outlines, FAISS, Fireworks, Guidance, Replicate, Together, Groq. |
| **Dependency Advisories** | `knowledge/dependency_advisories.py` | ✅ Sprint 3 | Offline KB for known vulnerable AI package versions. |
| **Data Model** | `models/assets.py` | ✅ | AIAsset (with data_flow, task_types, tags, code_contexts), DataFlowMap, FlowSource, FlowSink, CodeContext, TaskType enum, Finding, ProviderInfo, ClassificationResult, ScanResult. |

### Report & Visualization

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| **HTML Report** | `report/html.py` + `templates/report.html.j2` | ✅ | Self-contained dark-theme dashboard. No external dependencies. |
| **Executive Summary** | In report | ✅ | Auto-generated: total solutions, overlaps, data egress, SPOF authors, tech concentration, sensitive data. |
| **AI Solutions Map** | In report (canvas) | ✅ | Force-directed graph, 3 views (Solutions/Tech Stack/People). Category clusters with backgrounds. Sidebar filter. Draggable. Overlap edges (orange) + tech edges (purple). |
| **Data Flow** | In report detail | ✅ Sprint 5 | Per-solution: Sources (green), Processing Steps (blue), Destinations (red). |
| **Analytics** | In report | ✅ | Collapsible: Tech Stack radar, Data Types Processed, Author Coverage (SPOF), Functional Overlap (expandable). |
| **Solutions Table** | In report | ✅ | Solution, Repo, Built On, Author, Data Involved, Risk + compliance flags (PII, FIN, US). Click to expand detail with GitHub links. |
| **JSON Export** | `report/json_export.py` | ✅ | Auto-detected from .json extension. Full structured output with DataFlowMap fingerprint-based overlap detection. |

### Web UI & Deployment

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| **Web UI** | `web/app.py` + `templates/index.html` | ✅ | FastAPI 3-step wizard: Repositories → LLM Config (No LLM / Ollama / API Key) → Scan with SSE progress. Data Flow integrated in pipeline. |
| **Landing Page** | `landing/index.html` + screenshots/ | ✅ | Sales pitch with screenshots, role-based benefits (CEO → CTO → DevOps → CISO), pricing (Free / Pro TBA). |
| **CLI** | `cli.py` | ✅ | `aiscout scan` (multi-repo, YAML config, LLM options, .html/.json auto-detect) + `aiscout check` + `aiscout web`. |
| **Docker** | `Dockerfile` + `docker-compose.yml` | ✅ | Python 3.12-slim + git + landing. Default: `aiscout web --port 8080`. |

### GitHub Org Scanning & Developer Guardrail (latest)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| **Org/User Enumeration** | `scanners/github_org.py` | ✅ | Resolves a GitHub org or user (URL or bare name) into all token-visible repos via the REST API. Tries `/orgs` then falls back to `/users`, follows `Link` pagination, skips archived/forks by default, caps with `--max-repos`. Feeds the existing multi-repo scan loop. |
| **`--org` CLI flag** | `cli.py` | ✅ | `aiscout scan --org <name>` (repeatable) + `--include-archived` / `--include-forks` / `--max-repos`. API clone URLs revalidated through the SSRF/scheme guard. |
| **Manifest-only mode** | `cli.py` + `git_scanner.py` | ✅ | `--manifests-only` reads only dependency manifests (requirements.txt, package.json, pyproject.toml), never source — including skipping code-context analysis. Low-sensitivity first scan for security sign-off. |
| **GitHub Coverage in report** | `report/html.py` + template | ✅ | "GitHub Coverage" section shows repos found / scanned / skipped (archived, forks, over-limit, blocked) with a token-visibility disclaimer. |
| **Guardrail (`aiscout check`)** | `cli.py` | ✅ | Rule-based, network-free pre-commit/CI check. Exits non-zero on hardcoded LLM API keys or sensitive data (personal/financial/medical/credentials) sent to an external LLM. Local runtimes (Ollama) exempt. `--warn-only` to report without failing. |
| **Pre-commit hook** | `.pre-commit-hooks.yaml` | ✅ | `ai-scout-guardrail` hook id, consumable by any repo via the pre-commit framework. |
| **GitHub Action template** | `examples/ai-scout-guardrail.yml` | ✅ | Copy-paste workflow to gate pull requests with the guardrail. |

### Security (Sprint 1)

- API keys never stored raw — always redacted in Finding, `<REDACTED_API_KEY>` in LLM prompts
- Git tokens via GIT_ASKPASS helper, not URL embedding
- Symlinks skipped, path traversal blocked (`resolved.relative_to(root)`)
- Temp dirs: `TemporaryDirectory` + `chmod 0700`
- CLI input validation: URL scheme whitelist, SSRF block, system path block
- LLM prompt injection defense: `<untrusted>` tags + sanitization

### Tests

| File | Count | Coverage |
|------|-------|----------|
| `test_models.py` | 8 | Models, UUID, merge, enums |
| `test_git_scanner.py` | ~14 | Imports, keys, deps, grouping, symlink guard, path traversal |
| `test_code_analyzer.py` | 9 | AST, prompts, API calls, DB ops, JS, end-to-end |
| `test_llm_engine.py` | ~10 | Ollama/OpenAI mock, parse failure, sanitization, prompt injection |
| `test_enrichment.py` | ~14 | Summary, risk, categories, task_type, tags, MCP |
| `test_data_flow.py` | 9 | DataFlowMap construction, sources, sinks, steps |
| `test_report.py` | 5 | HTML/JSON generation, risk counts, overlap |
| `test_cli.py` | ~8 | Version, scan, YAML config, URL/path validation |
| `test_dependency_advisories.py` | ~4 | Advisory matching |
| `test_regression.py` | ~7 | Golden snapshot baselines (stable + volatile) |
| **Total** | **116** | All passing |

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
│  Data Flow Mapper (rule-based, Sprint 5)             │
│  LLM Engine (Ollama/OpenAI, optional enrichment)     │
│  Enrichment (risk, categories, tech stack, overlap)  │
│  Provider KB (30+ profiles) + Dep Advisories         │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              OUTPUT ENGINE                           │
│  HTML Report (graph, data flow, analytics)           │
│  JSON Export (structured, DataFlowMap fingerprints)   │
└─────────────────────────────────────────────────────┘
```

Architecture documents: `02_Architecture/`
- `00_System_Overview.md` — high-level system view
- `01_Git_Scanner_MVP.md` — Git Scanner architecture
- `02_Data_Flow_Mapper.md` — Data Flow Mapper design

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v0.1.0 | Apr 11 | MVP: Git Scanner, LLM Engine, HTML Report, CLI |
| v0.2.0 | Apr 11 | Code Context Extractor (AST parsing, prompt extraction) |
| v0.3.0 | Apr 11 | Directory-based grouping, solution-focused dashboard |
| v0.4.0 | Apr 11 | Graph visualization, exec summary, analytics, overlap |
| v0.4.1 | Apr 11 | LLM prompt with full code context |
| v0.5.0 | Apr 11 | Web UI (FastAPI wizard), repo column |
| v0.5.1 | Apr 11 | Docker deployment ready |
| v0.6.0 | Apr 11 | Landing page with screenshots |
| v0.6.x | Apr 12-15 | Sprints 1-4: security, task_types, MCP, dep advisories, CI/CD, LLM e2e |
| **v0.7.0** | **Apr 20** | **Sprint 5: Data Flow Mapper, DataFlowMap models, overlap fingerprinting, 116 tests** |
| _(branch)_ | May 21 | GitHub org/user enumeration (`--org`), manifest-only mode, GitHub Coverage report section, developer guardrail (`aiscout check` + pre-commit hook + Action template). 139 tests |
| **v0.8.0** | **Jul 7** | **Sprint 0.1: stable IDs (sol-/f- hashes), severity × confidence + categorical risk_status (weighted risk_score removed), "No findings" wording, Scout+KB version in header, Scope & Limitations section, schema_version 1.1.0 in JSON, bit-identical output via AISCOUT_TIMESTAMP. 141 tests** |

---

## How to Run

### Local (development)
```bash
uv sync

# CLI scan (no LLM)
uv run aiscout scan --repo https://github.com/org/repo --no-llm --output report.html

# CLI scan (JSON export)
uv run aiscout scan --repo https://github.com/org/repo --no-llm --output report.json

# CLI scan (with Ollama LLM)
uv run aiscout scan --repo https://github.com/org/repo --llm-model qwen2.5-coder:7b --output report.html

# CLI scan (with OpenAI-compatible API)
uv run aiscout scan --repo https://github.com/org/repo --llm-mode openai --llm-url https://api.openai.com --llm-key sk-... --llm-model gpt-4o-mini --output report.html

# Scan a whole GitHub organization (all token-visible repos)
uv run aiscout scan --org acme --token ghp_xxx --no-llm --output report.html

# Low-sensitivity first scan — dependency manifests only, no source read
uv run aiscout scan --org acme --token ghp_xxx --manifests-only --output report.html

# Developer guardrail (pre-commit / CI) — fails on leaked keys or sensitive LLM egress
uv run aiscout check --path .

# Web UI
uv run aiscout web --port 8080

# Tests
uv run pytest tests/ -q
```

### Docker (production)
```bash
git clone https://github.com/tomasb01/AI_Scout.git
cd AI_Scout
docker compose up -d
# Landing: http://<server-ip>:8080
# Scanner: http://<server-ip>:8080/app
```

---

## What's NOT Built Yet — Next Steps

### Priority 0: Report redesign
- 3 HTML prototypes in `prototypes/` (A: Executive Dashboard, B: Data Flow First, C: Risk-Action Focused)
- Current report doesn't fully reflect Sprint 5 Data Flow capabilities
- Awaiting variant selection

### Priority 1: Risk scoring calibration
- Sprint 3 reworked scoring, tested on 2 repos, needs 3-5 more diverse repos

### Priority 2: Instrumentovaná exekuce (Phase 2 per prod spec)
- LLM instruments code (WRITE→log), Docker sandbox, classifies actual data
- Prerequisite: DataFlowMap (done) identifies READ/WRITE ops
- For enterprises without Docker: Customer-executed script fallback (vrstva 3)

### Priority 3: GitHub API Scanner
- REST API instead of git clone, works on serverless (Vercel)

### Priority 4: Docker deployment on external server
- Dockerfile ready, untested on remote

### Priority 5: Enterprise scanners (M365/Entra ID, Network/DNS, Endpoint)

### Priority 6: Pro tier (Continuous Monitoring, Security Assessment, Remediation Roadmap)

---

## Key Files Reference

```
AI_Scout/
├── 01_Prod_specs/                    # Product specification (v8)
├── 02_Architecture/
│   ├── 00_System_Overview.md
│   ├── 01_Git_Scanner_MVP.md
│   └── 02_Data_Flow_Mapper.md
├── 03_Documentation/
│   ├── PROJECT_STATUS.md             # ← THIS FILE
│   └── SPRINT_LOG.md                 # Sprint-by-sprint detail
├── aiscout/
│   ├── cli.py                        # CLI (scan, web)
│   ├── engine/
│   │   ├── code_analyzer.py          # AST + regex extractor
│   │   ├── data_flow.py              # Data Flow Mapper (Sprint 5)
│   │   ├── enrichment.py             # Risk, summary, categories, overlap
│   │   └── llm.py                    # LLM client (Ollama/OpenAI)
│   ├── knowledge/
│   │   ├── providers.py              # 30+ provider profiles
│   │   └── dependency_advisories.py  # Vulnerable version KB
│   ├── models/
│   │   └── assets.py                 # Pydantic models (AIAsset, DataFlowMap, ...)
│   ├── report/
│   │   ├── html.py                   # HTML report generator
│   │   ├── json_export.py            # JSON export
│   │   └── templates/report.html.j2  # Dashboard template
│   ├── scanners/
│   │   ├── base.py                   # Scanner ABC
│   │   └── git_scanner.py            # Git scanner
│   └── web/
│       ├── app.py                    # FastAPI server
│       └── templates/index.html      # Scanner wizard
├── landing/
│   ├── index.html                    # Landing page
│   └── screenshots/                  # Demo screenshots
├── prototypes/                       # Report redesign variants A/B/C
├── scripts/                          # Helper scripts
├── tests/                            # 116 tests
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Key Decisions Made

1. **Directory-based grouping** — each directory with AI code = one solution (not provider-based)
2. **Solution names from code purpose** — README → prompts → function names → directory
3. **Tech stack deduplication** — model name suppresses provider (GPT-4o → removes "OpenAI" tag)
4. **Overlap via DataFlowMap fingerprinting** — same sinks + same data categories = overlap
5. **Rule-based first, LLM as enrichment** — works without LLM, no hard dependency
6. **Self-contained HTML report** — no CDN, all inline, opens offline
7. **Default model qwen2.5-coder:7b** — fits 8 GB RAM
8. **Lazy imports** — GitPython only on scan, enables Vercel landing
9. **Security by default** — key redaction, prompt injection defense, symlink guard, SSRF protection
