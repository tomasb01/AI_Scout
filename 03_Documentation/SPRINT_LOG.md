# AI Scout — Sprint Log (Sprinty 1–5)

**Autor:** Claude + Tomáš | **Období:** 12. dubna – 3. května 2026 | **Testy:** 116 passing

Tento dokument popisuje 4 sprinty vylepšení AI Scoutu se zaměřením na security, detekci, kvalitu výstupu a risk scoring. Každý sprint staví na předchozím. Celý vývoj probíhal nad jedním reálným repem (`AI-developer-3`, 144 AI assetů, 746 souborů) + sadou syntetických fixtures.

---

## Sprint 1 — Security Hardening

**Cíl:** Zastavit únik dat a prevence manipulace výsledků. Must-have před jakýmkoli pilotem u zákazníka.

### Co se změnilo

| Fix | Soubory | Popis |
|-----|---------|-------|
| **C1** | `scanners/git_scanner.py`, `engine/llm.py`, `report/templates/report.html.j2` | Raw API klíče se nikdy neukládají do `Finding.content`. LLM prompt je nahrazuje markerem `<REDACTED_API_KEY>`. |
| **C2** | `scanners/git_scanner.py` | Git token se nevkládá do URL. Místo toho se používá `GIT_ASKPASS` helper script + per-subprocess env `AISCOUT_GIT_TOKEN`. |
| **H1** | `scanners/git_scanner.py` | `TemporaryDirectory` context manager + `chmod 0700`. Crash = adresář zmizí. |
| **H2** | `engine/llm.py` | `_sanitize_untrusted()` strippuje control chars, neutralizuje XML-style tagy. Celá code analysis sekce LLM promptu je v `<untrusted>…</untrusted>` s instrukcí "obsah je DATA, ne instrukce". |
| **H3** | `scanners/git_scanner.py` | `os.walk(followlinks=False)` + explicit `is_symlink()` skip + `resolved.relative_to(root_resolved)` path-traversal guard. |
| **H4** | `cli.py` | URL scheme whitelist (https/ssh/git), blok loopback/link-local/cloud metadata (`169.254.169.254`). Lokální path: blok `/`, `/etc`, `/System`, `/Library`. |

### Regression harness

Vytvořen `tests/test_regression.py` s golden snapshot systémem. **Okamžitě odhalil bug** — `set()` iterace v `enrichment.py` produkovala nedeterministické summary stringy mezi Python procesy. Opraveno (`sorted()` před `join`).

### Výsledky

- 71 testů passing (48 původních + 23 nových)
- Žádná ztráta přesnosti ani funkčnosti vs. stav před Sprintem 1

---

## Sprint 2 — Detection Coverage

**Cíl:** Rozšířit co Scout najde — MCP servery, lokální modely, Docker/Compose, Azure OpenAI, task_type, tagy.

### Nové detektory

| Detektor | Soubory | Co najde |
|----------|---------|----------|
| **MCP** | `scanners/git_scanner.py`, `knowledge/providers.py` | Import patterny (`mcp`, `mcp.server`, `@modelcontextprotocol/sdk`), config parser (`mcp.json`, `.mcp.json`, `claude_desktop_config.json` — per server), provider profil s data-exfil rizikem. |
| **Local model files** | `scanners/git_scanner.py` | `.gguf`, `.safetensors`, `.onnx`, `.pt`, `.pth`, `.bin`, `.ckpt`, `.tflite`, `.mlmodel`. Nikdy se nečte obsah (i 70 GB soubory). Jen path + velikost. |
| **Docker/Compose** | `scanners/git_scanner.py` | 13 image patternů: ollama, vllm, TGI, Triton, LocalAI, llama.cpp, Qdrant, Chroma, Weaviate, Milvus, Langfuse, Open WebUI. |
| **Azure OpenAI** | `scanners/git_scanner.py`, `knowledge/providers.py` | Rozlišení `AzureOpenAI` vs `openai` import; env vars `AZURE_OPENAI_ENDPOINT`; kompletní provider profil (VNet, CMEK, FedRAMP, residency per-region). |

### Nová schémata

- **`TaskType`** enum: `inference`, `training`, `fine_tuning`, `evaluation`, `unknown`
- **`tags: list[str]`** na `AIAsset`: 10 pravidel (chatbot, rag, agent, training, fine_tuning, evaluation, transcription, image_generation, local_model, mcp)
- **Finding types:** `CONFIG_DETECTED`, `LOCAL_MODEL_DETECTED`, `CONTAINER_DETECTED`
- **Asset types:** `MCP_SERVER`, `LOCAL_MODEL` (k existujícímu `CUSTOM_CODE`)
- **Provider priority:** `_pick_primary_provider` — Azure OpenAI superseduje plain OpenAI; konkrétní LLM API providery supersedují frameworky

### Tagy v HTML reportu

Barevné chipy na kartě assetu (`.task-tag` CSS) — chatbot=modrý, rag=fialový, agent=růžový, training=červený, local_model=zelený, mcp=purple, atd.

### Výsledky

- 85 testů passing (+14 nových Sprint 2 testů)
- Sprint 2 fixture tree: `tests/fixtures/sprint2/` (mcp.json, docker-compose.yml, azure_chat.py, finetune_lora.py, tiny-llama.gguf)
- Druhý golden snapshot (`golden_sprint2.json`)

---

## Sprint 2.1 — Hotfix (kvality výstupu)

**Cíl:** Reagovat na reálná data — 47% assetů mělo `langchain` jako primary provider (neříká nic o residency), noisy summary ("Run in terminal commands."), task_type nedetekoval většinu training assetů.

### 3 fixes + summary rework

| Fix | Popis | Impact |
|-----|-------|--------|
| **LangChain sub-package mapping** | `langchain-openai` → `openai`, `langchain-anthropic` → `anthropic`, atd. (20+ sub-packages). `_package_to_providers()` emituje backend + framework. | langchain primary: 68 → 32 (−53%), openai primary: 11 → 46 (+318%) |
| **Training task_type** | Deps (`peft`, `trl`, `bitsandbytes`) + API calls (`client.fine_tuning.jobs`) + leaf-dir path heuristika. | training task_type: 4 → 8 (skutečné, bez false positive na `basics_azure/`) |
| **Tag fallback** | Když keyword detekce vrátí prázdno → fallback z providers, deps, path. | empty tags: 37 → 18 (−51%) |
| **Synthesized purpose** | `_synthesize_purpose()` — deterministická jedna věta z task_type+tags+model_names+provider. Má přednost před README noise. | noisy summaries: 6 → 1; 126/144 assetů má synth-led summary |
| **README noise filter** | `_is_descriptive_line()` rejektuje `pip install`, `Run in terminal`, `cd`, `git clone`, imperative prefixes. | "Run in terminal commands." → "Model training pipeline for Mistral." |

### Před / po příklady

```
BEFORE: Run (Runpod.io) — "Run in terminal commands."
AFTER:  Fine-tuning & Training — Tools — "Model training pipeline for Mistral."

BEFORE: Model & Inference — Run Models — "Run in terminal commands."
AFTER:  Model & Inference — Run Models — "Local inference on Mistral."

BEFORE: Browser Automation — provider=langchain, risk=0.80
AFTER:  Browser Automation — provider=openai, risk=0.14, "AI agent that calls tools via MCP servers."
```

---

## Sprint 3 — Risk Scoring + CI/CD + Dep Advisories

**Cíl:** Risk = akce + kontext, ne pouhá existence integrace. Přidat CI/CD scanner, YAML config parser, offline dep advisory KB.

### S3.1 Risk scoring rework

**Princip:** "Existence integrace ≠ riziko."

| Severity | Co trigguje | Dříve | Po Sprint 3 |
|----------|-------------|-------|-------------|
| **critical** | Hardcoded key; PII + free tier provider; LLM flagged ≥ 0.8 | Jen keys | + PII+provider combo |
| **warning** | Training pipeline; MCP server; deprecated dep; PII + external API; embeddings v cloudu | *Everything*: data leaves EU, external API, training policy (reflexivní) | Jen akční signály |
| **info** | Provider context (residency, training policy); framework; local runtime; MCP client | Jen frameworks/local | + provider context (jedna řádka místo 3 reflexivních warningů) |

**Skórovací floors:**
- critical → ≥ 0.70, warning → ≥ 0.40, info ≤ 0.25
- Score nikdy nesumuje fraktionální váhy → nemůže se stát, že 5× info = warning

**Impact na reálný repo:** 216 warningů → **13 warningů** (−94%). Distribuce: 129 OK / 10 warning / 5 critical.

### MCP server vs client rozlišení

- `_asset_is_mcp_server()` — kontroluje import findings + code context dekorátory
- MCP server (exposes tools) = **warning**
- MCP client (uses servers) = **info**

### S3.2 CI/CD pipeline scanner

`_detect_ci_pipeline()` — parsuje `.github/workflows/*.yml`, `.gitlab-ci.yml`, `.circleci/config.yml`, `Jenkinsfile`, `azure-pipelines.yml`, `bitbucket-pipelines.yml`. 15 patternů:
- AI credentials: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`, …
- Actions: `anthropics/claude-code-action`, `coderabbitai/…`
- Training: `python finetune_*.py`, `accelerate launch`, `deepspeed`, `modal run`

### S3.3 YAML/TOML config model parser

`_detect_config_model_refs()` — `deployment_name:`, `model:`, `fallback_model:`, `azure_endpoint:`, `model_id:`. Plausibility filter `_is_plausible_model_ref()` eliminuje false positives.

### S3.4 Offline dependency advisory KB

`aiscout/knowledge/dependency_advisories.py` — 9 high-signal entries:
- `openai < 1.0` (legacy API, warning)
- `langchain < 0.1` (pre-split, CVEs, warning)
- `transformers < 4.36` (RCE risks, warning)
- `llama-index < 0.10` (SSRF/path-traversal, warning)
- `chromadb < 0.4` (unsupported schema, warning)
- `gradio < 4.0` (XSS advisories, warning)
- `pydantic < 2.0` (incompatibility, info)
- `langchain 0.1.x` (superseded, info)

### S3.5 Summary edge cases

- `_looks_like_runtime_log()` filtruje emoji/print statementy z prompt extrakce
- `_is_descriptive()` — "Run in terminal" / "pip install" / shell commands → rejected z README

### BONUS: Bug fix

`requirements.txt` a `setup.py` nebyly procházeny v `_walk_files` (suffix `.txt` / `.py` je v `SCAN_EXTENSIONS`, ale `requirements.txt` specificky neprošel přes `is_dep = name in DEPENDENCY_FILES` check, který chyběl). Přidáno.

### Výsledky

- 102 testů passing
- Sprint 3 fixtures: `.github/workflows/llm-review.yml`, `llm_config.yaml`, `requirements.txt` s legacy deps
- Třetí golden snapshot

---

## Sprint 4 — LLM Integration + E2E Validace

**Cíl:** Regression harness pro LLM, MCP classifier fix, end-to-end LLM scan, CI detector validace na reálných repech.

### S4.1 Regression harness split

`_normalise_stable()` — strict diff (provider, tags, task_types, risk_score, reason severity+title, tech_stack, data_involved)
`_normalise_volatile()` — smoke floor (summary nonempty, ≥ 15 chars, ≥ 1 reason)

Summary text se neukládá do goldenu → LLM runs mohou přepisovat summary bez golden driftů.

**Ověřeno 2 záměrnými testy:** prázdné summary → fail ✓; jiný summary text → pass ✓.

### S4.2 MCP server/client classifier fix

Problém: Browser Automation Operator byl chybně klasifikován jako MCP server (substring match "server" v text blobu).

Fix: Strukturní signály místo substring matchu:
1. `raw_findings[IMPORT_DETECTED]` pro `mcp.server` / `mcp.client` import paths
2. `code_contexts.functions[].decorators` pro `@mcp.tool` / `@server.tool`
3. `api_calls` pro `FastMCP(` / `MultiServerMCPClient(`
4. Path tie-breaker (soubor `server.py` nebo `/servers/`) jako poslední instance

**Validováno na 3 repech:**
- `AI-developer-3`: Browser Operator → client (info) ✓, Chatbot Server → server (warning) ✓
- `langchain-mcp-adapters`: 4 klienti + 4 servery (včetně 2 dříve false-negative) ✓
- `claude-code-action`: MCP test server ✓

### S4.3 LLM end-to-end

**Runtime:** 57 minut / 144 assetů / qwen2.5-coder:7b / Ollama / M-series Mac.

**Risk score + reasons synchronizace:** LLM risk_score se NENAHRAZUJE rule-based score. Místo toho se přidává viditelný `LLM review flagged elevated risk` reason → score a reasons jsou synchronní.

**OpenAI-compatible backend compat:**
- `temperature=0.1` pro determinismus
- Retry bez `response_format` pokud backend vrátí 400 (TGI, starší vLLM)
- System message pro JSON output
- CLI help + docstring explicitně jmenují: **vLLM, LocalAI, LM Studio, llama.cpp, TGI, Together, Groq, Mistral La Plateforme, Fireworks, DeepInfra, OpenRouter, Azure OpenAI**

**LLM správně identifikoval finanční data** v Stock Price tools → propagace přes `_asset_processes_pii` → critical reason "Personal data flows to provider with training-on-data risk" → 9 nových legitních criticals.

**Summary kvalita — příklady:**
```
NO-LLM: "Model training pipeline for Mistral."
W/ LLM: "fine-tuning mistralai/Mistral-7B-Instruct-v0.3 using HuggingFace's
          transformers library. It processes data from an SQL database..."

NO-LLM: "LLM application that consumes tools from MCP servers."
W/ LLM: "Web scraping solution that uses Playwright to navigate Google and take
          screenshots of search results for a specified query."
```

### S4.4 CI/CD detektor validace na reálných repech

- `anthropics/claude-code-action`: **14 CI findings** (Anthropic credentials + Claude GitHub Action)
- `langchain-mcp-adapters`: 0 CI findings (žádné workflows s AI creds)
- `openai-cookbook`: 0 CI findings (no workflows)
- YAML config detektor: 0 hits na produkčních repech — patterny jsou úzké, Sprint 5 kandidát

### E2E validace — 7 kroků

| # | Test | ✓ |
|---|------|---|
| 1 | 107 testů passing | ✓ |
| 2 | Determinismus 3× s PYTHONHASHSEED=random | ✓ |
| 3 | CLI help dokumentuje vLLM/LocalAI/etc. | ✓ |
| 4 | HTML + JSON generace | ✓ |
| 5 | HTML integrity (autoescape, no XSS, task-tag CSS, redacted keys) | ✓ |
| 6 | CLI security (file://, 127.0.0.1, / → rejected) | ✓ |
| 7 | Full metrics evolution Sprint 2→3→4 | ✓ |

---

## Kumulativní metriky (reálný repo, 144 assetů)

| metrika | Sprint 2 hotfix | Sprint 3 | Sprint 4 no-LLM | Sprint 4 w/ LLM |
|---|---:|---:|---:|---:|
| Critical | 10 | 5 | 5 | 14 |
| Warning | 56 | 10 | 9 | 11 |
| OK | 78 | 129 | 130 | 119 |
| Warning reasons (total) | 216 | 13 | 12 | 14 |
| Noisy summaries | 6 | 1 | 1 | 0 |
| Synth-led summaries | 0 | 126 | 126 | — (LLM) |
| Empty tags | 18 | 18 | 18 | 18 |
| `langchain` as primary | 32 | 32 | 32 | 32 |
| `openai` as primary | 46 | 46 | 46 | 46 |
| Test count | 85 | 102 | 107 | 107 |

---

## Soubory změněné/vytvořené (všechny Sprinty)

### Změněné
- `aiscout/scanners/git_scanner.py` — detektory, walk_files, symlink guard, GIT_ASKPASS, dep parsing, CI/CD, YAML config, container, MCP, model files, Azure OpenAI, primary provider selection
- `aiscout/engine/enrichment.py` — risk scoring rework, tag/task_type derivation, MCP server/client classifier, synth purpose, README noise filter, dep advisory integration, LLM reason sync
- `aiscout/engine/llm.py` — prompt sanitization, `<untrusted>` wrapping, OpenAI-compat retry, temperature, system message, docstring
- `aiscout/engine/code_analyzer.py` — print/emoji noise filter v prompt extrakci
- `aiscout/models/assets.py` — `TaskType` enum, `tags`/`task_types` fieldy, nové `FindingType` hodnoty
- `aiscout/models/__init__.py` — export `TaskType`
- `aiscout/cli.py` — input validace (URL scheme, loopback, path), CLI help text pro LLM backendy
- `aiscout/knowledge/providers.py` — `azure_openai` + `mcp` provider profily
- `aiscout/report/templates/report.html.j2` — tag chipy CSS, API key defenzivní rendering

### Vytvořené
- `aiscout/knowledge/dependency_advisories.py` — offline dep advisory KB
- `tests/test_regression.py` — golden snapshot harness (stable + volatile split)
- `tests/test_dependency_advisories.py` — advisory KB testy
- `tests/fixtures/sprint2/*` — MCP, Docker, Azure, fine-tune, model file fixtures
- `tests/fixtures/sprint3/*` — CI workflow, YAML config, legacy requirements
- `tests/regression/golden*.json` — 3 golden snapshots

---

## Sprint 5 — Data Flow Mapper (20. dubna 2026)

**Cíl:** Sestavit chybějící Step 2 z architektury — rule-based Data Flow Mapper, který z extrahovaného CodeContextu (Step 1) konstruuje strukturovaný DataFlowMap (sources → processing steps → sinks). Žádný LLM potřeba.

**Pozadí:** Audit product spec v8 odhalil, že Scout extrahuje bohatá data z kódu (funkce, API calls, prompty, data sources/sinks) ale 80 % z nich zahazuje — místo strukturovaného flow mapy produkuje generický string "Conversational chatbot powered by OpenAI."

### Co přibylo

| # | Změna | Soubor |
|---|-------|--------|
| **S5.1** | `DataFlowMap`, `FlowSource`, `FlowSink` modely; `data_flow` field na `AIAsset` | `models/assets.py` |
| **S5.2** | `engine/data_flow.py` — rule-based flow: `_identify_sources`, `_identify_sinks`, `_infer_processing_steps`, `_compose_purpose`, `_classify_data_categories`, `_assess_confidence`. Filtry pro noise (cursor setup, fetchall, INSERT jako source). | nový soubor |
| **S5.3** | Pipeline wiring: `cli.py` + `web/app.py` volají `build_data_flows()`. Summary se generuje Z DataFlowMap. | `cli.py`, `web/app.py`, `enrichment.py` |
| **S5.4** | HTML report: Data Flow sekce (Sources zelené, Processing Steps modré, Destinations červené) | `report/templates/report.html.j2` |
| **S5.5** | Overlap detekce přes DataFlowMap fingerprint. MCP display_name normalizace. Tech stack synonym dedup. JSON exporter přepsán. | `report/html.py`, `report/json_export.py`, `knowledge/providers.py`, `enrichment.py` |
| **S5.6** | 9 DataFlowMap testů, 4 regression goldens | `tests/test_data_flow.py` |

### Srovnání: Architektura (spec) vs Scout výstup po Sprint 5

```
ARCHITEKTURA:                        SCOUT PRODUKUJE:
sources:                             sources:
  POST /chat                           [user_input] /chat              ✓
  get_history(session_id)               [database] SELECT...messages   ✓

sinks:                               sinks:
  Claude API                            [ai_api] Claude (model)        ✓
  save_to_db                            [database] Database write      ✓
  HTTP response                         [http_response] /chat          ✓

steps:                               steps:
  1. Receive message                    1. Receive user input          ✓
  2. Load history                       2. Query data from database    ✓
  3. Send to Claude                     3. Load conversation history   ✓
  4. Store in DB                        4. Send prompt to LLM API      ✓
  5. Return response                    5. Store results in database   ✓
                                        6. Return response to client   ✓
```

### Výsledky

- 116 testů passing (+9 DataFlowMap testů)
- Overlaps: "8 solutions (?)" → "4× MCP Client Pattern [MCP & Integration]"
- Tech stack dedup: "MCP" 44 + "Model Context Protocol" 38 → "MCP" 51 + 0
- DataFlowMap na 100 % assetů (118/118)

---

## Report redesign — prototypy (20. dubna 2026)

Audit aktuálního reportu odhalil, že vizuální design neodráží Sprint 5 capabilities. Analytics sekce ("Data Types Processed") je matoucí. Vytvořeny 3 HTML prototypy v `prototypes/`:

- **Varianta A** (`variant_a.html`): Executive Dashboard — KPI → heatmap → tech/flow → overlaps → solutions tabulka
- **Varianta B** (`variant_b.html`): Data Flow First — agregátní Sankey flow jako centrální vizuál
- **Varianta C** (`variant_c.html`): Risk-Action Focused — "Where does your data go?" + exit points + action checklist

Čeká na feedback a iteraci.

---

## Sprint 0.2 — QA vrstva reportu (8. července 2026, v0.9.0)

Zadání: `01_Prod_specs/specs/AI_Scout_QA_spec.md`. Cíl: v no-LLM režimu report nikdy neukáže rozbitou, nesmyslnou nebo z kódu prosáklou větu.

### Co vzniklo

- **`aiscout/report/qa_vocab.py`** — kontrolované slovníky (P-1): zkratky pro L-07, počitatelná substantiva pro L-10, stop-list pro L-05, labely fact stripu (sources/sinks/pattern), konečný slovník důvodů pro I-02, labely citlivých kategorií.
- **`aiscout/report/insights.py`** — typovaný insight katalog **I-01–I-10** (datový model dle QA spec §1.1), ICU MessageFormat šablony + vlastní deterministický mini-ICU renderer (plural =N/one/other, select, `#`, date medium). *Rozhodnutí: PyICU nepřidán — C extenze proti systémové ICU knihovně jde proti „10minutové offline instalaci"; renderer má ~100 řádků a plné pokrytí testy.* Jediná pct funkce s explicitním jmenovatelem; `validate_invariants` (§3) běží před renderem a padá tvrdě.
- **`aiscout/report/linter.py`** — pravidla **L-01–L-10** nad finálním vyrenderovaným textem: zdvojená slova, nepárové závorky/uvozovky, nevyřešené placeholdery, osiřelá interpunkce, useknuté věty, >100 %, prosáknutí kódu do prózy (se safe-token maskováním KB labelů a git identit), duplicitní summary (L-08, cross-entity), délkové meze, nesoulad plurálu.
- **`aiscout/report/qa.py`** — pipeline `prepare_qa`: data → invarianty → render → linter → degradace. **Fact strip** (Sources/Sinks/Pattern/Tech z kontrolovaného slovníku) jako default detail v no-LLM režimu (P-3); LLM próza jen unikátní a lint-čistá, jinak degradace (P-4). LLM próza je vyjmuta z L-07 (legitimně odkazuje na kód, nese štítek LLM).
- **HTML report** — exec summary z typovaných insightů (severity tečky), detail řešení: fact strip `RULE` / LLM próza `LLM` (provenance štítky), recommendations přes linter, **QA appendix** („Suppressed by QA linter", sbalený) + QA counts ve footeru.
- **JSON export** — `schema_version` 1.1.0 → **1.2.0** (aditivně): `insights` (typovaný katalog) + `qa` (suppressed/warnings). Stejná pipeline jako HTML — identické věty i suprese.
- **CLI** — `aiscout scan --strict`: nenulový exit (2) při jakékoli supresi; produkční default zůstává degradace, ne pád (P-4).

### QA vrstva chytila vlastní chyby už při vývoji

Edge-case testy + linter odhalily 3 reálné kompoziční chyby šablon ještě před mergem: „all **1 solutions**" (I-04 při total=1), „used by 1 of **1 solutions**" (I-05), „**1 files** scanned" (I-01). Fix: plural větev pro files (šablona `T-INVENTORY-v2`) a trigger `total ≥ 2` pro koncentrační insighty I-04/I-05.

### Výsledky

- **248 testů passing** (+105: edge-case matice všech 10 šablon vč. povinného pct=100, kartézský součin kategorií I-07, všechna pravidla linteru, degradační pipeline, determinismus HTML, --strict gate)
- Nový golden: `tests/regression/golden_qa.json` (insight věty + QA counts + fact strips nad fixtures)
- Akceptační kritéria QA spec §5: všechna splněna — fixture „1 přispěvatel" renderuje „A single contributor … created all N solutions" (žádné „over 100%"), duplicitní summaries degradují na fact strip, dvojí běh = bitově shodný výstup, `--strict` vrací nenulový exit code

---

## Sprint 0.3 — Agregační hranice + detektor charakteru repa + install (8. července 2026, v0.10.0)

Zadání: Spec v13 §3.4 + QA spec. Cíl: report počítá aplikace, ne adresáře — a čerstvá instalace skutečně funguje za 10 minut.

### Co vzniklo

- **`aiscout/engine/repo_character.py`** — detektor charakteru repa: `production │ tutorial_example │ experiment │ unknown` + confidence + signály z konečného slovníku (sequential_dir_naming, lesson_keyword_dirs, many_small_solution_dirs, notebook_heavy, readme_course_keywords vs. has_ci/tests/container/lockfile). Observable s evidencí, ne verdikt; produkční signály vetují tutorial klasifikaci (dražší chyba). Výsledek v `ScanResult.metadata["repo_character"]`.
- **`aiscout/engine/aggregation.py`** — agregační hranice: **řešení = aplikace/služba**, directory grouping zůstává mechanismus. Dvě deterministická pravidla: (1) **tutorial collapse** — tutorial repo s ≥ 8 komponentami se složí do jednoho řešení „`<repo>` teaching collection (N examples)" s tagem `tutorial_collection`, všechny findingy si drží file:line evidenci; (2) **manifest roots** — komponenta se složí pod nejbližšího předka s dependency manifestem (odvozeno z findingů, žádný přístup na disk); bez manifestu se nic nemerguje (konzervativní default → ID beze změny).
- **`AIAsset.component_dirs`** — evidence složených adresářů (aditivní pole).
- **Výstupy** — HTML: badge „tutorial / example repo" u repa (se signály v tooltipu), štítek „N components" u složených řešení; JSON `schema_version` **1.3.0**: `repositories[].character` + `solutions[].components`.
- **Re-baseline `sol-` ID** — dle plánu (README_BUNDLE): ID hash root povýšen z adresáře na agregační root; golden snapshoty jednorázově přegenerovány (fixtures 4 → 3 řešení). Finding `f-` ID nedotčena.
- **Pročištěný install:** nalezena a opravena reálná díra — **wheel neobsahoval žádné šablony** (chybělo `package-data` v pyproject) → `aiscout scan` i `aiscout web` byly na čerstvé (needitovatelné) instalaci rozbité; fungovalo jen `pip install -e .`. Ověřeno buildem + instalací do čistého venv + e2e scanem. README přepsán: Quick Start bez LLM (~2 min), konzistentní default model 7b (dřív README naváděl stáhnout 14b, ale scan zkoušel 7b → pád), doplněny `--org`, `check`, `web`, `--strict`, aktuální architektura; Development sekce opravena (`uv sync` místo nefunkčního `pip install -e ".[dev]"`).

### Výsledky

- **261 testů passing** (+12: detektor, manifest-root folding, sibling manifests, worst-risk merge, tutorial collapse e2e + determinismus, ID stabilita nemergovaných řešení)
- Nová fixture `tests/fixtures_tutorial/` (10 lesson adresářů + course README) — v reportu **1 řešení místo 10**, I-01 říká „Found 1 AI solution"
- Enrichment fix: tagy se sjednocují se scannerovými (dřív by keyword derivace zahodila `tutorial_collection`)

---

## Validační milník po Sprintu 0 (8. července 2026)

Scan tří reálných rep (AI-developer-3, AI-Agents-2 — výuková; Fleurdin_AI — reálná aplikace), celkem 1 633 souborů. První běh odhalil dvě kalibrační chyby detektoru:

1. **Veto přebily artefakty uvnitř lekcí** — výuková repa nesou lockfily/Dockerfile/testy jako výukový materiál (`1-Intro/3_RNN/uv.lock`, `3_N8N/Dockerfile`). Fix: produkční signály se počítají jen z repo-level umístění (root, root `tests/`, `.github/workflows`; přidán Procfile/fly.toml).
2. **Tvar ≠ výuka** — Fleurdin_AI (reálná appka) má číslované adresáře jako fáze pipeline (`4-RAG_Pipeline`, `5-Backend`); tvarové signály samy o sobě by ji složily. Fix: tutorial klasifikace vyžaduje aspoň jeden sémantický signál (lesson slova v adresářích / course README).

3. **Repo-wide kolaps zahazoval informaci** (feedback Tomáše) — kurz probírá ~11 různých témat s různými stacky; „1 teaching collection (144 examples)" neříká, co v repu reálně je. Uživatelský scénář: „když má organizace 30 řešení, která jsou různé formy Copilot instrukcí, chci vědět, o co v konkrétním řešení jde a kam co volá." Fix: **kapitolový kolaps** — pod-příklady se skládají do top-level adresáře (kapitoly/tématu), ne do jednoho repo-wide blobu. Každá kapitola si drží vlastní tech stack, data flow, findingy a risk status.

**Výsledek po fixech: 258 → 27 řešení** (a každé něco říká).

| Repo | Před | Po | Charakter |
|------|------|-----|-----------|
| AI-developer-3 | 141 | **10 kapitol** — např. „Hugging Face (12 examples)" critical (klíče), „Langgraph (21 examples)", „Web Operator (3 examples)" s Playwright+MCP | tutorial_example / high |
| AI-Agents-2 | 107 | **7 kapitol** — „MCP (29 examples)", „DB (15 examples)" s ChromaDB/Elastic, „LLM API (29 examples)" | tutorial_example / high |
| Fleurdin_AI | 10 | **10** (reálná aplikace — beze změny) | unknown / low |

Critical status je lokalizovaný do konkrétní kapitoly (uniklé HF klíče = kapitola Hugging Face + archiv Old), ne rozmazaný přes celé repo. Exec summary: „Found 27 AI solutions across 3 repositories" — důvěryhodné a informativní. +2 regresní testy (repo-level produkční signály, sémantická podmínka), fixture rozšířena o vnořené pod-příklady (10 lekcí × 2 → 10 kapitol).

4. **Granularita > konsolidace** (feedback Tomáše, druhá iterace) — přidaná hodnota Scouta je granulární scouting UVNITŘ repa: v monorepu může v každé podvětvi žít jiné AI řešení (hook v určité fázi procesu spouští agenta) a struktura sama je informace („kde v repu agenti jsou"). Fix: (a) **root manifest už neskládá** — foldují jen pod-adresářové manifesty (svc/handlers → svc = jedna služba; root requirements.txt nesmí spolknout celé monorepo); (b) každé řešení nese **`root_path`** — cesta v repu viditelná v tabulce reportu (monospace pod jménem) i v JSON (`solutions[].path`). Fleurdin: 10 → **12 granulárních řešení**, každé s cestou (`5-Backend`, `4-RAG_Pipeline`, `[ARCHIVE]/3-Fine_tuning`…), slepenec tří složek pod root manifestem zmizel. Vedlejší efekt: CI workflow s AI voláním (`.github/workflows`) je viditelné jako samostatné řešení — přesně hook scénář. +1 monorepo regresní test.

---

## Refactor: účel jako identita řešení (8. července 2026, v0.11.0)

Feedback Tomáše po validaci: identita řešení musí být **účel → tech stack → struktura** — 10 LangGraph agentů s 10 účely je 10 řešení; totéž řešení v jiném frameworku je samostatné řešení viditelné v overlapu. Kolaps kapitol tuhle informaci zahazoval a kategorie se vařily ze slepené hromady.

### Změny

- **Pipeline přeskládána**: scan (directory grouping) → code context → data flow → **agregace** (`aggregate_scan_result`, volá CLI/web/testy) → enrichment. Agregace už neběží ve scanneru — identita se odvozuje z pochopených komponent, ne z adresářového mišmaše.
- **Merge kritérium = strukturální hranice + funkční otisk**: komponenty se slučují jen když sdílejí hranici (kapitola tutorial repa / podstrom podadresářového manifestu) **a** identický DataFlowMap fingerprint (sinky + kroky + kategorie dat) — „varianty téhož" prokázané tokem. Bez toku se nemerguje nic; stejný tok ve dvou větvích repa = overlap insight, ne merge. ID variant group = hash(repo, hranice, fingerprint).
- **Jména z účelu**: „A web researcher (3 variants)", „Browser Automation Operator" místo „Hugging Face (12 examples)"; kolize display jmen řeší přípona s cestou („Calculate Length — 0-AI_Dev_Scripts/1_joe").
- **Kategorie z tagů/task_types** místo text-first heuristiky — nalezen substring bug (`"train" in text` chytá „constraint", „dataset" je v každém RAG kódu → 62 % všeho bylo Fine-tuning). Teď: AI Agents 94, Chatbot 46, MCP 29, Model & Inference 24, RAG 23, Fine-tuning 13.
- **Dependency evidence**: manifest bez kódu není řešení („uses Hugging Face" je mechanismus použití) — vlastní kategorie „Dependency Evidence", jméno „Dependency manifest — repo root", **vyloučeno ze všech počtů** (I-01, dlaždice, JSON summary — konzistentně; `summary.dependency_evidence` počítadlo).

### Validace (3 reálná repa)

AI-developer-3: 127 řešení (z 141 dir-assetů; merge jen prokázané varianty), AI-Agents-2: 102, Fleurdin: 12 + 4× dependency evidence. **Každé řešení má účelové jméno, smysluplnou kategorii a cestu.** Čísla jsou vyšší než u kapitolového kolapsu — vědomě: výukové repo reálně obsahuje ~100 různých toků a granularita je hodnota produktu; badge „tutorial/example repo" dává kontext. 267 testů passing.

### Přehled + granularita: strukturální skupiny v tabulce (`_build_table_groups`)

Čistě prezentační vrstva nad granulárními řešeními: u rep s ≥ 10 řešeními se řádky tabulky seskupí podle top-level adresáře (skupina od 3 členů), header nese cestu + počet + critical/review badge. Skupiny bez critical jsou defaultně sbalené; klik rozbaluje; **aktivní filtr kolaps ignoruje** (výsledek filtru není nikdy schovaný). Identity, ID a počty nedotčené. Validace: 3 repa → 18 skupin, 15 sbalených — 241 řádků čte jako seznam kapitol/větví, každé řešení jeden klik daleko. 268 testů.

---

## AIBOM groundwork v JSON exportu (9. července 2026, schema 1.4.0)

Rozhodnutí: malá příprava na AIBOM (Sprint 5b) provedena hned — teplý kontext po refactoru, princip „sbírat teď, zobrazit později" (pilotní scany = budoucí diff baseline; zpětné dosbírání = přescanovat vše) a aditivní schema změna bez rizika. Tvrdá hranice scope: jen 3 pole, žádný PURL/pinning/podpisy (zůstávají v 5b dle plánu).

- `solutions[].model_refs` — normalizované reference modelů: `{model, resolution: code│config, evidence: [soubory]}`. Deduplikováno, deterministické řazení. Plní AIBOM Models cluster (mapping spec §3) **a** Sprint 4 observable zároveň. KB obohacení (tier, lifecycle, PURL) přijde aditivně v 5/5b.
- `solutions[].provenance` — rule/llm per odvozené pole (name, category, summary, data_involved) — AIBOM zásada č. 2 (`scout:provenance` na každém odvozeném poli).
- `solutions[].role` — `application` │ `dependency_manifest` — CycloneDX typování komponent připravené (aplikace vs. klasická SBOM library vrstva).

270 testů passing. Ověřeno na Fleurdin_AI: model_refs zachytává `google/gemma-2-2b-it(code)`, `gpt-4o(code)`, `mistralai/Mistral-7B-Instruct-v0.3(code)`.

---

## Sprint 1 — SARIF export (10. července 2026, v0.12.0)

Cíl: distribuce zadarmo — nálezy tam, kde žijí vývojáři (GitHub code scanning, CI security taby).

### Co vzniklo

- **`aiscout/report/sarif_export.py`** — SARIF 2.1.0 exporter, `--output *.sarif` autodetekce v CLI. Jeden `run` per repo (multi-repo scan nese `automationDetails.id`).
- **Security tab dostává security nálezy**: default exportuje findingy severity ≥ medium (dnes `SEC-KEY-001` hardcoded klíče jako `error` + GitHub `security-severity: 9.1`). Discovery inventář (importy/deps/configy) jen s `--sarif-include-discovery` jako `note` — tutorial repo nevysype 400 poznámek do security tabu.
- **Fingerprinty ze stabilních ID** (Sprint 0.1): `partialFingerprints.aiScoutFindingId/v1 = f-...` — alerty se mezi scany neduplikují. Zprávy nikdy neobsahují surový klíč (jen redakci) a jmenují řešení.
- **Rules katalog** s lidskými popisy per rule_id (SEC-KEY-001 + 5× DISC-*), tagy, helpUri.
- **Deterministický výstup** — bez timestampů, explicitní řazení; dva běhy = bitově shodný soubor.
- **CI šablony**: `examples/ai-scout-sarif.yml` (GitHub Action s `codeql-action/upload-sarif`), `examples/ai-scout-gitlab-ci.yml` (SARIF artefakt + poznámka ke konverzi do GitLab SAST formátu — GitLab SARIF nativně neingestuje). README rozšířeno.

### Validace

- Výstup **validuje proti oficiálnímu SARIF 2.1.0 JSON schématu** (fixtures i AI-developer-3 — 10 výsledků = 10 uniklých HF klíčů s file:line).
- 279 testů passing (+8: struktura/GitHub požadavky, redakce zpráv, fingerprinty, discovery flag, multi-repo, determinismus, CLI e2e).
- **Živé ověření hotovo (dogfood, 10. 7. 2026):** workflow `.github/workflows/ai-scout-sarif.yml` nasazen na AI_Scout repo — Action prošla na první pokus, SARIF upload bez warningů, v Security tabu se objevilo přesně 9 očekávaných alertů (falešné klíče v test fixtures, rule SEC-KEY-001, critical severity). Všech 9 zavřeno jako „used in tests" — stabilní fingerprinty drží dismissal napříč scany, tab od teď hlídá jen nové nálezy. **Akceptační kritérium sprintu splněno.** Bonus: Scout od teď trvale skenuje sám sebe při každém pushi/PR + týdně.

---

## Sprint 2 — Diff / trend režim (10. července 2026, v0.13.0)

Cíl: z one-shot nástroje opakovaně používaný; artefakt do change managementu. Stojí na stabilních ID ze Sprintu 0.1 — a ta drží: **diff dvou nezávislých scanů AI-developer-3 (127 řešení) = nulový drift.**

### Co vzniklo

- **`aiscout diff <old.json> <new.json>`** (`engine/diff.py`) — porovnání dvou exportů přes `sol-`/`f-` ID: řešení added/removed/changed (risk_status, findingy, model_refs, provider), noví provideři, nové/vyřešené klíče. `--output delta.json`, `--fail-on-new-critical` (exit 3) jako CI gate. Dependency evidence mimo počty (konzistentně). `resolved` = automatické pozorování (nález už není detekován), ne verdikt.
- **`aiscout scan --baseline old.json`** — report dostane **delta box** (dlaždice added/removed/changed/new keys/resolved keys/new providers + seznamy) a **insight I-09 SCAN_DELTA** v exec summary (šablona čekala připravená od Sprintu 0.2 — teď dostala živá data). JSON: top-level `delta` blok.
- **Finding workflow stavy** (`engine/findings_state.py`): `open │ accepted_risk │ resolved`, persistence v lokálním souboru `.aiscout/findings.json` (žádný server — self-hosted disciplína). CLI: `aiscout findings accept <id> --note`, `reopen`, `list`. Scan stavy razítkuje před enrichmentem: **accepted klíč přestává řídit critical status** (řešení → review, I-02 počítá jen open), ale zůstává viditelný — badge ACCEPTED v reportu + warning reason s audit stopou. K tomu automatický `first_seen` tracking („tenhle klíč tu je už 2 scany").
- JSON `schema_version` **1.5.0** (aditivně): `findings[].status`, `findings[].first_seen`, top-level `delta`.

### Akceptační kritéria (dev plan)

- ✅ Dva scany téže org → korektní delta report (reálné repo: 0 driftu při identickém stavu; fixtures: +10 při přidání tutorial stromu; risk_status/model_refs/finding změny detekovány)
- ✅ `accepted_risk` přežívá mezi scany (nový proces, nový scan, stejná stabilní ID → status drží, first_seen se nerazítkuje znovu)

294 testů passing (+15).

---

## Sprint 3 — MCP & Agent scanner (10. července 2026, v0.14.0) — LAUNCH FEATURE

Cíl: první nástroj na trhu mapující MCP/agent landscape. Kategorii-definující featura.

### Co vzniklo

- **`aiscout/scanners/agent_detect.py`** — MCP/agent surface detekce: MCP server konfigurace (`.mcp.json`, `claude_desktop_config.json`, + `mcpServers` blok vnořený v libovolném settings.json) se stdio/remote rozlišením; agent-instruction soubory (`CLAUDE.md`, `.cursorrules`, `.clinerules`, `copilot-instructions.md`, `AGENTS.md`, …); IDE surface markery (`.claude`/`.cursor`/`.aider`/`.windsurf`); tool definitions (`@tool`, `@mcp.tool`, `tools=[`); agent frameworky (LangGraph, CrewAI, AutoGen, Semantic Kernel, OpenAI Agents SDK, LlamaIndex, PydanticAI, smolagents, Google ADK, Strands).
- **Klasifikace autonomie** — `tool_calling │ supervised_agent │ autonomous_loop │ none` + confidence. Human-in-the-loop signál (`input()`, `approval`, `interrupt`) **stropuje** autonomii na supervised i při přítomnosti smyčky — konzervativní, nepřestřeluje. Inventurní atribut, ne verdikt (G7 to zvažuje pro příští revizi guidance). Uloženo v `AIAsset.autonomy/autonomy_confidence/agent_frameworks`, zapojeno do enrichmentu.
- **`aiscout mcp`** (`scanners/mcp_env.py`) — sken živého MCP prostředí stroje: Claude Desktop, Claude Code, Cursor, VS Code, Windsurf (dokumentované cesty per platforma). Read-only, offline, nikdy nespustí server. Redakce: u remote serverů jen host, nikdy query/token. `--path` pro custom config, `-o` JSON.
- **Výstupy**: autonomy badge v tabulce (barevně dle úrovně) + sekce Agent v detailu (autonomy + frameworks); JSON `schema_version` **1.6.0** — `solutions[].autonomy {level, confidence, frameworks}`.

### Živá validace (kritérium „hotovo když")

- **Vlastní setup:** `aiscout mcp` na dev stroji našel `playwright` MCP server v Claude Code i Claude Desktop configu (2 servery, 3 config soubory).
- **Cizí prostředí (AI-Agents-2):** autonomy distribuce **19 autonomous_loop / 30 tool_calling / 13 supervised / 41 none**; frameworky LangGraph 15, LangChain Agents 9, AutoGen 1; MCP zachyceno jako kód (desítky `mcp.server`/`mcp.client`/`@mcp.tool` s file:line) — config detekce správně 0, protože repo je kurz *jak psát* MCP servery, ne jejich nasazení. Kalibrace autonomie sedí: Plan-Execute agent high, ReAct agenti medium.

311 testů passing (+17). **Launch trojice SARIF + diff + MCP kompletní** → dle specu §17 následuje veřejné repo + threat model + pilot outreach.

---

## Otevřené body

1. **Report redesign** — implementovat vybranou variantu (nebo mix) jako nový `report.html.j2`
2. **Risk scoring kalibrace** — validace na 3–5 dalších reálných repech
3. **Summary quality edge cases** — "Conversational chatbot" pro API tutorials; overlaps naming
4. **Instrumentovaná exekuce** (spec 3.5) — LLM generuje instrumentovaný kód, Docker sandbox, klasifikace reálných dat. Phase 2.
5. **GitHub API Scanner** — REST API místo git clone, serverless support
6. **Enterprise scanners** — M365/Entra ID, Network/DNS, Endpoint
