# AI Scout — Data Flow Mapper Architecture

**Version 0.1 | April 2026**

---

## Problém

MVP (v0.1) najde AI řešení v kódu — ví, že v souboru `app.py` je `import anthropic`. Ale neví:

- **Co to řešení dělá** — je to chatbot? překladač? analyzátor dokumentů?
- **Jaká data zpracovává** — uživatelské zprávy? finanční data? osobní údaje?
- **Odkud data přicházejí** — z databáze? z API? ze souboru? z uživatelského vstupu?
- **Kam data odcházejí** — do Claude API? do souboru? do jiné služby?

Bez těchto informací je report jen seznam providerů, ne mapa AI řešení.

---

## Řešení: dvě nové komponenty

### 1. Code Context Extractor

Čte obsah souborů, kde byly nalezeny AI importy, a extrahuje strukturovaný kontext:

- Jaké funkce/třídy soubor obsahuje
- Jaká API volání se dělají (HTTP requesty, DB dotazy, file I/O)
- Jaké proměnné a stringy naznačují účel (prompty, system messages, názvy endpointů)
- Jaké datové zdroje se čtou (soubory, DB, env vars, API)
- Jaké výstupy se generují (response, file write, DB write, API call)

### 2. Data Flow Mapper

Sestaví z extrahovaného kontextu datový model:

```
Data Sources → AI Processing → Data Sinks
(odkud)        (co se děje)     (kam to jde)
```

---

## Code Context Extractor — detail

**Soubor:** `aiscout/engine/code_analyzer.py`

### Co extrahuje

Pro každý soubor s AI findings extrahuje:

```python
@dataclass
class CodeContext:
    """Structured context extracted from source code."""
    file_path: str
    language: str                    # python, javascript, typescript
    functions: list[FunctionInfo]    # název, parametry, docstring, tělo (zkrácené)
    classes: list[ClassInfo]         # název, metody, docstring
    api_calls: list[APICall]         # HTTP requesty, SDK volání
    data_sources: list[DataSource]   # DB, soubory, env vars, API inputs
    data_sinks: list[DataSink]      # kam data odcházejí
    prompts: list[PromptInfo]       # system prompty, user prompty, šablony
    env_vars: list[str]             # použité environment variables
    comments: list[str]             # relevantní komentáře a docstringy
```

### Jak extrahuje — tři vrstvy

**Vrstva 1: AST parsing (Python)**

Pro Python soubory použije stdlib `ast` modul:

```python
import ast

tree = ast.parse(source_code)
```

Z AST extrahuje:
- **Funkce a třídy** — `ast.FunctionDef`, `ast.ClassDef` → název, argumenty, docstring, dekorátory
- **Volání funkcí** — `ast.Call` → identifikace API volání (`openai.chat.completions.create()`, `requests.post()`, `cursor.execute()`)
- **String literály** — `ast.Constant` kde `isinstance(value, str)` → hledá prompty, SQL dotazy, URL, connection stringy
- **Přiřazení** — `ast.Assign` → sleduje proměnné jako `system_prompt`, `api_key`, `db_url`
- **Import kontexty** — co se importuje kromě AI (flask, fastapi, psycopg2, boto3 → naznačuje web API, DB, AWS)

**Vrstva 2: Regex patterns (všechny jazyky)**

Pro JS/TS a jazyky bez AST parseru:

```python
PATTERNS = {
    # Prompty a system messages
    "system_prompt": [
        r'system[_\s]?(?:prompt|message|instruction)\s*[:=]\s*["\'](.+?)["\']',
        r'role["\']:\s*["\']system["\'].*?content["\']:\s*["\'](.+?)["\']',
    ],
    # Database operace
    "db_query": [
        r'(?:execute|query|run)\s*\(\s*["\'](.+?)["\']',
        r'SELECT\s+.+?\s+FROM\s+(\w+)',
        r'INSERT\s+INTO\s+(\w+)',
    ],
    # HTTP requesty
    "http_call": [
        r'(?:requests|httpx|fetch|axios)\s*\.\s*(get|post|put|delete)\s*\(\s*["\'](.+?)["\']',
    ],
    # File I/O
    "file_io": [
        r'open\s*\(\s*["\'](.+?)["\']',
        r'(?:read|write|load|save|dump)(?:_file|_json|_csv)?\s*\(',
    ],
    # Environment variables
    "env_var": [
        r'os\.environ\s*\[\s*["\'](\w+)["\']',
        r'os\.getenv\s*\(\s*["\'](\w+)["\']',
        r'process\.env\.(\w+)',
    ],
}
```

**Vrstva 3: Heuristiky z kontextu**

Odvozuje účel z kombinace signálů:

| Signály v kódu | Odvozený účel |
|---|---|
| `whisper`, `transcribe`, audio file I/O | Voice/audio transkripce |
| `embeddings`, vector store, `similarity_search` | RAG / sémantické vyhledávání |
| `chat.completions`, `system_prompt`, conversation history | Chatbot / konverzační AI |
| `fine_tune`, `trainer`, `training_args`, model upload | Fine-tuning pipeline |
| `translate`, `target_language` | Překlad |
| `summarize`, `summary` | Sumarizace textu |
| `classify`, `label`, `category` | Klasifikace / kategorizace |
| `generate`, `image`, `diffusion` | Generování obrázků |
| SQL, `cursor`, `connection`, tabulky | Práce s databází |
| `stock`, `price`, `ticker`, `exchange` | Finanční data / stock analysis |
| `scrape`, `crawl`, `beautifulsoup` | Web scraping |
| `email`, `smtp`, `send_message` | E-mailová integrace |

---

## Data Flow Mapper — detail

**Soubor:** `aiscout/engine/data_flow.py`

### Datový model

```python
@dataclass
class DataSource:
    """Odkud data přicházejí."""
    type: str          # "database", "file", "api", "user_input", "env_var"
    name: str          # "PostgreSQL customers table", "input.csv", "user message"
    detail: str        # SQL query, file path, API endpoint
    data_types: list[str]  # "text", "audio", "json", "csv"

@dataclass
class DataSink:
    """Kam data odcházejí."""
    type: str          # "ai_api", "database", "file", "http_response", "webhook"
    name: str          # "OpenAI GPT-4", "results.json", "PostgreSQL"
    detail: str        # API endpoint, file path, table name
    provider: str      # "openai", "anthropic", "" for non-AI

@dataclass
class DataFlowMap:
    """Kompletní datový tok jednoho AI řešení."""
    solution_purpose: str      # "Voice transcription pipeline for customer calls"
    sources: list[DataSource]
    sinks: list[DataSink]
    processing_steps: list[str]  # ["Load audio file", "Transcribe via Whisper", "Save to DB"]
    data_categories: list[str]   # ["audio", "personal_data", "transcripts"]
    confidence: str              # "high", "medium", "low"
```

### Jak Data Flow Mapper pracuje

**Krok 1 — Statická extrakce (bez LLM)**

Code Context Extractor projde soubory a vytáhne strukturovaný kontext. Tohle funguje vždy, i v `--no-llm` režimu:

```python
def extract_context(file_path: str, content: str, language: str) -> CodeContext:
    if language == "python":
        return _extract_python_ast(file_path, content)
    else:
        return _extract_regex(file_path, content, language)
```

**Krok 2 — Rule-based Data Flow (bez LLM)**

Z extrahovaného kontextu sestaví základní data flow mapu pomocí heuristik:

```python
def build_data_flow(asset: AIAsset, contexts: list[CodeContext]) -> DataFlowMap:
    sources = _identify_sources(contexts)      # DB, files, API inputs
    sinks = _identify_sinks(contexts)          # AI APIs, DB writes, file outputs
    purpose = _infer_purpose(contexts, asset)  # heuristiky z tabulky výše
    steps = _infer_steps(contexts)             # sekvence operací
    return DataFlowMap(...)
```

Tohle dá základní popis i bez LLM. Výstup bude typu:
> "Python script that reads audio files, uses Whisper for transcription, and saves results. Data sources: local audio files. Data sinks: OpenAI Whisper API, local JSON output."

**Krok 3 — LLM enrichment (volitelné)**

Pokud je LLM k dispozici, pošle se mu extrahovaný kontext s promptem:

```
Analyzuj tento kód a odpověz:
1. Co přesně tento kód/řešení dělá? (1-2 věty, konkrétně)
2. Jaká data zpracovává? (typy dat, ne konkrétní hodnoty)
3. Odkud data přicházejí? (databáze, soubory, API, uživatelský vstup)
4. Kam data odcházejí? (AI API, databáze, soubory, HTTP response)
5. Jsou v datech potenciálně citlivé informace? (PII, finanční, zdravotní)

Kontext kódu:
- Soubor: {file_path}
- Funkce: {functions}
- API volání: {api_calls}
- Prompty: {prompts}
- Datové zdroje: {data_sources}
```

LLM vrátí JSON, který obohatí rule-based výstup o přesnější popis.

Výstup s LLM bude typu:
> "Voice transcription pipeline for processing customer support calls. Loads WAV audio files from /recordings/, transcribes them using OpenAI Whisper API, extracts key topics and sentiment, and stores structured results in PostgreSQL table 'call_transcripts'. Processes audio recordings that may contain personal customer information (names, account numbers mentioned in calls)."

---

## Integrace do stávajícího kódu

### Kam se napojí

```
Git Scanner (existující)
    │
    │  scan() → ScanResult s List[AIAsset]
    │  (každý asset má file_path a raw_findings)
    │
    ▼
Code Context Extractor (NOVÝ)                    ← aiscout/engine/code_analyzer.py
    │
    │  Přečte obsah souborů z asset.file_path
    │  Extrahuje CodeContext pro každý soubor
    │  Připojí ke každému AIAsset
    │
    ▼
Data Flow Mapper (NOVÝ)                          ← aiscout/engine/data_flow.py
    │
    │  Z CodeContext sestaví DataFlowMap
    │  Rule-based (vždy) + LLM enrichment (volitelně)
    │  Aktualizuje asset.data_inputs, asset.data_outputs
    │  Generuje solution_purpose → nahradí starý _infer_purpose()
    │
    ▼
Enrichment (existující, upravený)                ← aiscout/engine/enrichment.py
    │
    │  Použije DataFlowMap místo _infer_purpose()
    │  Risk reasoning bere v potaz data_categories z flow mapy
    │
    ▼
Report Generator (existující, upravený)
    │
    │  Zobrazí: účel řešení, data flow diagram, data sources/sinks
```

### Změny v CLI flow

```python
# cli.py — scan command (aktualizovaný flow)

# 1. Scan repos (beze změny)
scanner = GitScanner(...)
result = scanner.scan()

# 2. NOVÉ: Extract code context
from aiscout.engine.code_analyzer import extract_contexts
for asset in result.assets:
    asset.code_contexts = extract_contexts(asset, repo_root)

# 3. NOVÉ: Build data flow maps
from aiscout.engine.data_flow import build_data_flows
for asset in result.assets:
    asset.data_flow = build_data_flows(asset)

# 4. LLM enrichment (volitelné) — teď dostane code context
if not no_llm:
    engine.classify_with_context(asset)  # nový prompt s code context

# 5. Enrichment + Report (upravené, aby použily data flow)
```

### Změny v datovém modelu

```python
# models/assets.py — nové modely

class CodeContext(BaseModel):
    file_path: str
    language: str
    functions: list[dict]      # {name, args, docstring, body_preview}
    api_calls: list[dict]      # {target, method, args_preview}
    data_sources: list[dict]   # {type, name, detail}
    data_sinks: list[dict]     # {type, name, detail}
    prompts: list[str]         # system/user prompt texty
    env_vars: list[str]

class DataFlowMap(BaseModel):
    solution_purpose: str
    sources: list[DataSource]
    sinks: list[DataSink]
    processing_steps: list[str]
    data_categories: list[str]
    confidence: str = "medium"

# AIAsset — nová pole:
class AIAsset(BaseModel):
    ...
    code_contexts: list[CodeContext] = []
    data_flow: DataFlowMap | None = None
```

---

## Omezení a hranice

### Co Code Context Extractor zvládne
- Python: plný AST parsing (funkce, třídy, volání, stringy, assignmenty)
- JS/TS: regex-based extrakce (méně přesné, ale funkční)
- Jupyter notebooks: extrakce z cells (už implementováno v Git Scanneru)
- Jakýkoli jazyk: regex patterny na prompty, SQL, HTTP, env vars

### Co nezvládne (a kam jít dál)
- **Runtime chování** — nevidí, co se stane když kód běží (→ Instrumentovaná exekuce, fáze 2)
- **Dynamické hodnoty** — nevidí obsah env vars, DB dat (→ Data Classification Modes, fáze 1-2)
- **Obfuskovaný kód** — pokud je kód záměrně nečitelný
- **Multi-file flow** — pokud data flow prochází přes 5+ souborů (částečně řešitelné přes import tracking)

### Bezpečnost
- Code Context Extractor **čte** kód, **nespouští** ho
- Před odesláním do LLM se redaktují: API klíče, connection stringy, hardcoded credentials
- LLM dostane strukturovaný kontext, ne raw soubory

---

## Implementační plán

| Krok | Co | Soubor |
|---|---|---|
| 1 | Datový model — CodeContext, DataFlowMap, DataSource, DataSink | `models/assets.py` |
| 2 | Python AST Extractor | `engine/code_analyzer.py` |
| 3 | Regex Extractor (JS/TS + fallback) | `engine/code_analyzer.py` |
| 4 | Heuristiky pro účel řešení | `engine/code_analyzer.py` |
| 5 | Data Flow Mapper — rule-based | `engine/data_flow.py` |
| 6 | LLM prompt s code context | `engine/llm.py` (update) |
| 7 | Napojení do CLI flow | `cli.py` (update) |
| 8 | Napojení do enrichment | `engine/enrichment.py` (update) |
| 9 | Report — zobrazení data flow | `report/html.py` + template (update) |
| 10 | Testy | `tests/test_code_analyzer.py`, `tests/test_data_flow.py` |

---

## Příklad výstupu

### Vstup: soubor `app.py`

```python
import anthropic
from flask import Flask, request, jsonify

app = Flask(__name__)
client = anthropic.Anthropic()

SYSTEM_PROMPT = "You are Fleurdin, a helpful florist assistant..."

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    history = get_history(request.json["session_id"])
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        system=SYSTEM_PROMPT,
        messages=history + [{"role": "user", "content": user_message}],
    )
    
    save_to_db(user_message, response.content[0].text)
    return jsonify({"response": response.content[0].text})
```

### Výstup: DataFlowMap

```json
{
  "solution_purpose": "Backend chatbot API for 'Fleurdin' florist assistant. Accepts user messages via REST endpoint, sends conversation with history to Claude API, stores messages in database, returns AI response.",
  "sources": [
    {"type": "user_input", "name": "Chat message", "detail": "POST /chat — request.json['message']"},
    {"type": "database", "name": "Conversation history", "detail": "get_history(session_id)"}
  ],
  "sinks": [
    {"type": "ai_api", "name": "Anthropic Claude API", "detail": "claude-sonnet-4-20250514 via messages.create()", "provider": "anthropic"},
    {"type": "database", "name": "Message storage", "detail": "save_to_db(user_message, response)"},
    {"type": "http_response", "name": "API response", "detail": "JSON response to client"}
  ],
  "processing_steps": [
    "Receive user message via POST /chat",
    "Load conversation history from database",
    "Send message + history + system prompt to Claude API",
    "Store user message and AI response in database",
    "Return AI response to client"
  ],
  "data_categories": ["user_messages", "conversation_history", "ai_generated_text"],
  "confidence": "high"
}
```

### Jak to vypadá v reportu

```
┌─────────────────────────────────────────────────────┐
│ Fleurdin Chatbot                           WARNING  │
│                                                     │
│ Backend chatbot API for 'Fleurdin' florist          │
│ assistant. Accepts user messages, sends them to     │
│ Claude API with conversation history, stores in DB. │
│                                                     │
│ 🏗️ Anthropic (Claude)  👤 Tomas Bohm  📁 1 file   │
│                                                     │
│ Data Flow:                                          │
│ User input ──→ Claude API ──→ Database              │
│      ↑              │                               │
│ Conv. history ──────┘                               │
│                                                     │
│ ▸ View details                                      │
└─────────────────────────────────────────────────────┘
```
