# AI Scout — MVP Architecture

**Version 0.2 | April 2026**

---

## Přehled

MVP je end-to-end flow v pěti vrstvách: Vstupní vrstva (CLI + Web UI) → Git Scanner → LLM Engine → HTML Report. Cílem je funkční demo, které naskenuje jeden nebo více Git repozitářů, najde AI assety, klasifikuje je přes LLM a vygeneruje self-contained HTML report.

Scout je přístupný dvěma způsoby: přes CLI (pro DevOps a technické uživatele, automatizace, CI/CD) a přes jednoduché webové rozhraní (pro CTO, CISO, compliance — kliknou, zadají repo, dostanou report).

---

## Dva přístupové režimy

### CLI (primární, MVP den 1)

Pro technické uživatele, scripting, CI/CD pipeline integrace.

```bash
# Jeden repozitář
aiscout scan --local /path/to/repo --output report.html

# Více repozitářů přes parametry
aiscout scan --repo https://github.com/org/repo-1 --repo https://github.com/org/repo-2 --output report.html

# Více repozitářů přes konfigurační soubor
aiscout scan --config repos.yaml --output report.html

# Bez LLM klasifikace (rychlejší)
aiscout scan --local /path/to/repo --no-llm --output report.html
```

### Web UI (MVP fáze 1.5 — thin wrapper nad CLI logikou)

Pro netechnické uživatele (CTO, CISO, compliance). Jednoduchá webová stránka:

- Formulář: repo URL(s), token, branch, výběr LLM modelu
- Tlačítko "Spustit scan"
- Progress indikátor
- Výsledek: zobrazí / nabídne ke stažení ten samý HTML report

Technicky: FastAPI server s jedním POST endpointem, který volá identický kód jako CLI. Žádná databáze, žádný login, žádný state — čistý stateless scan. Frontend je single-page HTML s formulářem.

**Soubory:** `aiscout/web/app.py` (FastAPI server), `aiscout/web/templates/index.html` (formulář)

---

## Multi-repo podpora

### Příkazová řádka

Parametr `--repo` je opakovatelný:

```bash
aiscout scan --repo URL1 --repo URL2 --repo URL3 --output report.html
```

Kombinovatelný s `--local`:

```bash
aiscout scan --local /repo-1 --local /repo-2 --repo https://github.com/org/repo-3 --output report.html
```

### Konfigurační soubor (YAML)

Pro firemní nasazení s desítkami repozitářů:

```yaml
# repos.yaml
repositories:
  - url: https://github.com/org/backend-api
    branch: main
    token_env: GITHUB_TOKEN          # odkaz na env proměnnou

  - url: https://github.com/org/data-pipeline
    branch: develop

  - url: https://gitlab.company.com/team/ml-models
    token_env: GITLAB_TOKEN

  - path: /local/checkout/frontend
    branch: main

llm:
  mode: ollama
  url: http://localhost:11434
  model: qwen2.5-coder:14b

output:
  path: reports/company_ai_scan.html
```

Spuštění: `aiscout scan --config repos.yaml`

### Agregace výsledků

Scout projde každý repozitář samostatně (stejný Git scanner), výsledné `ScanResult` objekty se mergnou do jednoho agregovaného reportu. Report pak ukazuje:

- Přehled across all repos (celkový počet assetů, kritických nálezů)
- Filtr podle repozitáře
- Cross-repo overlap detekce (3 repa používají stejný OpenAI import = potenciální duplicita)

Každý `AIAsset` nese pole `repository`, takže v reportu je jasné, odkud co pochází.

---

## Vrstva 1: CLI entry point

**Soubor:** `aiscout/cli.py`

Vstupní bod celé aplikace. Orchestruje scan → klasifikaci → report.

**Parametry:**

| Parametr | Popis | Default |
|----------|-------|---------|
| `--repo` / `-r` | URL Git repozitáře (opakovatelný pro multi-repo) | — |
| `--local` / `-l` | Cesta k lokálnímu repozitáři (opakovatelný) | — |
| `--config` / `-c` | Cesta ke konfiguračnímu YAML souboru | — |
| `--token` / `-t` | Git access token (private repos) | env `AISCOUT_GIT_TOKEN` |
| `--branch` / `-b` | Branch ke scanu (default pro všechna repa) | `main` |
| `--output` / `-o` | Cesta k výstupnímu reportu | `aiscout_report.html` |
| `--llm-url` | URL LLM API | `http://localhost:11434` |
| `--llm-model` | Název modelu | `qwen2.5-coder:14b` |
| `--llm-mode` | Mód LLM: `ollama` nebo `openai` | `ollama` |
| `--llm-key` | API klíč pro OpenAI-compatible mód | env `AISCOUT_LLM_KEY` |
| `--no-llm` | Přeskočit LLM klasifikaci | `false` |

**Flow:**

1. Parsování CLI argumentů (Click) nebo načtení YAML configu
2. Sestavení seznamu repozitářů ke scanu
3. Pro každý repozitář: spuštění Git scanneru → `ScanResult`
4. Agregace všech `ScanResult` do jednoho
5. (volitelně) LLM klasifikace každého assetu → `ClassificationResult`
6. Generování HTML reportu
7. Výpis summary do konzole (Rich)

---

## Vrstva 2: Git Scanner (plugin #1)

**Soubor:** `aiscout/scanners/git_scanner.py`
**Interface:** `aiscout/scanners/base.py`

### Scanner Plugin Interface

Každý scanner implementuje toto rozhraní (pro budoucí M365, Power Platform, Network scannery):

```python
class BaseScanner(ABC):
    def get_config(self) -> ScannerConfig      # Jaké credentials potřebuje
    def scan(self, **kwargs) -> ScanResult      # Spustit discovery
    def get_name(self) -> str                   # Lidský název
```

### Git Scanner — tři sub-moduly

**Import Detector** — hledá AI-related importy ve zdrojovém kódu:

- Python: `import openai`, `from anthropic import ...`, `from langchain ...`
- JS/TS: `require('openai')`, `import { ... } from '@anthropic-ai/sdk'`
- Pokrytí: 18+ AI providerů/frameworků (OpenAI, Anthropic, LangChain, LlamaIndex, HuggingFace, Mistral, Cohere, Ollama, ChromaDB, Pinecone, Qdrant, Weaviate, ...)

**API Key Detector** — regex patterny na hardcoded klíče:

- `sk-[a-zA-Z0-9]{20,}` → OpenAI API key
- `sk-ant-[a-zA-Z0-9]{20,}` → Anthropic API key
- `AIza[a-zA-Z0-9_-]{35}` → Google AI API key
- `hf_[a-zA-Z0-9]{30,}` → HuggingFace token
- + další (Replicate, Pinecone)
- Klíče jsou v reportu redaktované (prvních 8 znaků + `...` + poslední 4)

**Dependency Scanner** — hledá AI balíčky v dependency souborech:

- `requirements.txt`, `pyproject.toml`, `setup.py` → PyPI balíčky (40+ AI packages)
- `package.json` → NPM balíčky (15+ AI packages)

### Jaké soubory scanner prochází

- Rozšíření: `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.mjs`, `.cjs`, `.java`, `.cs`, `.go`, `.rs`, `.rb`, `.php`, `.yaml`, `.yml`, `.toml`, `.json`, `.env`, `.ipynb`
- Přeskakuje: `.git`, `node_modules`, `__pycache__`, `.venv`, `venv`, `dist`, `build`
- Max velikost souboru: 1 MB

### Co Git scanner NEZACHYTÍ (a co to pokrývá)

| Slepé místo | Proč | Který budoucí scanner to řeší |
|-------------|------|-------------------------------|
| Komerční SaaS (ChatGPT, Copilot subscriptions) | Neprochází přes Git, ale přes OAuth grants | M365 / Entra ID scanner (fáze 1) |
| Lokální modely bez commitnutého kódu (Ollama na notebooku) | Není v žádném repu | Endpoint scanner (fáze 2) |
| Shadow AI (osobní ChatGPT přes prohlížeč) | Žádný kód, jen browser traffic | Network / DNS scanner (fáze 1) |
| Power Platform automatizace s AI konektory | Low-code, není v Gitu | Power Platform scanner (fáze 1) |
| AI agenti a MCP servery (Claude Desktop, Cursor) | Konfigurace mimo Git | MCP & Agent scanner (fáze 2) |

Git scanner pokrývá "co vývojáři postavili v kódu." Další scannery postupně pokryjí zbývající slepá místa.

### Výstup

`ScanResult` obsahující `List[AIAsset]`, kde každý asset reprezentuje jednu AI integraci (seskupeno podle providera/modulu).

---

## Vrstva 3: LLM Analysis Engine

**Soubor:** `aiscout/engine/llm.py`

### Dva módy provozu (MVP)

**Mód 1 — Enterprise API:** Scout se napojí na firemní LLM endpoint (Azure OpenAI, AWS Bedrock, jakýkoli OpenAI-kompatibilní API). Konfigurace: `--llm-mode openai --llm-url https://your-endpoint --llm-key KEY`.

**Mód 2 — Local Ollama:** Scout volá lokální Ollama REST API. Doporučené modely: `qwen2.5-coder:14b` nebo `mistral:7b`. Konfigurace: `--llm-mode ollama --llm-url http://localhost:11434`.

### Pre-processing vrstva

Před odesláním do LLM proběhne:

1. Formátování raw findings do čitelného textu
2. Omezení na max 20 findings per asset (token budget)
3. Redakce API klíčů (už provedena scannerem)

### LLM prompt

Scout pošle strukturovaný prompt s:

- Název a typ assetu
- Repozitář
- Seznam findings (importy, dependencies, API klíče)
- Instrukce vrátit JSON s: `data_categories`, `confidence`, `risk_score`, `summary`, `recommendations`

### Výstup

`ClassificationResult` připojený ke každému `AIAsset`:

- `categories`: list kategorie citlivosti dat (public, internal, confidential, pii, financial, source_code, unknown)
- `confidence`: high / medium / low
- `details`: textový souhrn + doporučení od LLM

### Fallback

Pokud LLM není dostupný (Ollama neběží, API timeout), scan proběhne bez klasifikace. Uživatel je informován v konzoli. Report se vygeneruje i bez LLM dat.

---

## Vrstva 4: HTML Report

**Soubor:** `aiscout/report/html.py`

Self-contained HTML soubor (žádné externí závislosti, otevřitelný offline):

- **Statistiky:** počet AI assetů, kritické nálezy, soubory s AI kódem, celkem proskenovaných souborů, počet skenovaných repozitářů
- **Risk heatmapa:** vizuální pruh zobrazující risk score všech assetů
- **Filtry:** All / Critical / Warning / OK + filtr podle repozitáře (při multi-repo scanu)
- **Asset karty:** název, provider, repozitář, risk badge, seznam findings, LLM klasifikace s tagy citlivosti, LLM analýza text
- **Cross-repo přehled:** pokud více repozitářů používá stejného AI providera, report to zvýrazní jako potenciální overlap
- **Metadata:** seznam repozitářů, branch, datum scanu

---

## Datový model

**Soubor:** `aiscout/models/assets.py`

### AIAsset (hlavní entita)

| Atribut | Typ | Popis |
|---------|-----|-------|
| `id` | UUID | Unikátní identifikátor |
| `name` | string | Název AI řešení |
| `type` | enum | commercial_saas / custom_code / local_model / automation / agent / mcp_server |
| `owner` | string | Osoba/tým (default: unknown) |
| `users` | List[string] | Kdo řešení používá |
| `data_inputs` | List[DataFlow] | Odkud data přicházejí |
| `data_outputs` | List[DataFlow] | Kam data odcházejí |
| `provider` | ProviderInfo | AI provider + metadata |
| `risk_score` | float (0–1) | Vážené skóre rizika |
| `data_classification` | ClassificationResult | Výsledek LLM klasifikace |
| `discovered_via` | List[string] | Které scannery asset nalezly |
| `last_activity` | datetime | Poslední známá aktivita |
| `documentation` | enum | none / partial / full |
| `file_path` | string | Cesta k souboru (Git-specific) |
| `repository` | string | Název repozitáře |
| `dependencies` | List[string] | AI dependencies |
| `raw_findings` | List[dict] | Raw scanner findings |

### Pomocné modely

- **DataFlow** — source, destination, data_types, description
- **ProviderInfo** — name, region, training_policy, certifications
- **ClassificationResult** — layer, categories, confidence, details
- **ScanResult** — scan_id, scanner, started_at, completed_at, assets, errors, metadata

---

## Adresářová struktura

```
ai-scout/
├── pyproject.toml              # Projekt konfigurace, dependencies, CLI entry point
├── Dockerfile                  # Docker image pro Scout
├── docker-compose.yml          # Scout + Ollama stack
├── README.md                   # Dokumentace
├── .gitignore
├── aiscout/
│   ├── __init__.py             # __version__
│   ├── cli.py                  # CLI (Click + Rich)
│   ├── scanners/
│   │   ├── __init__.py
│   │   ├── base.py             # BaseScanner interface
│   │   └── git_scanner.py      # Git Repository Scanner
│   ├── engine/
│   │   ├── __init__.py
│   │   └── llm.py              # LLM Analysis Engine
│   ├── models/
│   │   ├── __init__.py
│   │   └── assets.py           # AIAsset, ScanResult, DataFlow, ...
│   ├── report/
│   │   ├── __init__.py
│   │   └── html.py             # HTML report generátor (Jinja2)
│   └── web/                    # Web UI (fáze 1.5)
│       ├── __init__.py
│       ├── app.py              # FastAPI server
│       └── templates/
│           └── index.html      # Scan formulář
└── tests/
```

---

## Dependencies

| Balíček | Účel | MVP fáze |
|---------|------|----------|
| `click` | CLI framework | den 1 |
| `rich` | Konzolový výstup (barvy, panely, progress) | den 1 |
| `pydantic` | Datový model, validace | den 1 |
| `gitpython` | Git operace (klonování, procházení souborů) | den 1 |
| `httpx` | HTTP klient pro Ollama/OpenAI API | den 1 |
| `jinja2` | HTML šablona pro report | den 1 |
| `pyyaml` | Konfigurace (multi-repo YAML) | den 1 |
| `fastapi` | Web UI server | fáze 1.5 |
| `uvicorn` | ASGI server pro FastAPI | fáze 1.5 |

---

## Deployment

### Lokální (bez Dockeru)

```bash
pip install -e .
ollama pull qwen2.5-coder:14b
aiscout scan --local /path/to/repo --output report.html
```

### Multi-repo s YAML konfigurem

```bash
aiscout scan --config repos.yaml
```

### Docker Compose (Scout + Ollama)

```bash
docker compose up -d
docker exec ai-scout-ollama ollama pull qwen2.5-coder:14b
docker exec ai-scout aiscout scan --config /app/config/repos.yaml --output /app/reports/report.html
```

### Web UI (fáze 1.5)

```bash
aiscout web --port 8080
# Otevřít http://localhost:8080
```

---

## Implementační plán (krok po kroku)

| Krok | Co | Výstup |
|------|----|--------|
| 1 | Inicializace projektu — `pyproject.toml`, adresářová struktura, `.gitignore` | Prázdný projekt |
| 2 | Datový model — `models/assets.py` | AIAsset, ScanResult, pomocné modely |
| 3 | Scanner interface — `scanners/base.py` | BaseScanner ABC |
| 4 | Git Scanner — `scanners/git_scanner.py` | Import/key/dependency detekce |
| 5 | LLM Engine — `engine/llm.py` | Ollama + OpenAI-compatible integrace |
| 6 | HTML Report — `report/html.py` | Jinja2 template + generátor |
| 7 | CLI — `cli.py` | Click commands, multi-repo, YAML config |
| 8 | Test na reálném repu | Funkční end-to-end demo (single + multi repo) |
| 9 | Dockerfile + docker-compose.yml | Kontejnerizace |
| 10 | README + GitHub repo | Publikace |
| 11 | Web UI — `web/app.py` + `templates/index.html` | FastAPI + formulář (fáze 1.5) |

---

## Budoucí rozšíření (mimo MVP)

- M365 / Entra ID Scanner (fáze 1)
- Network / DNS Scanner (fáze 1)
- Power Platform Scanner (fáze 1)
- Google Workspace Scanner (fáze 1)
- Data Classification Modes — schema-only, sampled, customer-executed (fáze 1–2)
- Endpoint Scanner (fáze 2)
- MCP & Agent Scanner (fáze 2)
- Instrumentovaná exekuce v sandboxu (fáze 2)
- Security Assessment modul (fáze 3)
- Continuous Monitoring / Watch Mode (fáze 4)
- Enterprise konektory — SIEM, Splunk, ServiceNow (fáze 5)
