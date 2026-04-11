# AI Scout — System Architecture Overview

**Version 0.2 | April 2026**

---

## Systém na vysoké úrovni

AI Scout je modulární pipeline se třemi hlavními vrstvami:

```
┌─────────────────────────────────────────────────────────────────┐
│                        VSTUPNÍ VRSTVA                           │
│  CLI (Click)  ·  Web UI (FastAPI, fáze 1.5)  ·  YAML Config    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DISCOVERY ENGINE                           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Git Scanner  │  │ M365 Scanner │  │ Network/DNS  │  ...     │
│  │   (v0.1) ✅  │  │   (fáze 1)   │  │  (fáze 1)    │          │
│  └──────┬───────┘  └──────────────┘  └──────────────┘          │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────┐          │
│  │           Code Context Extractor                  │          │
│  │  AST parsing · funkce · volání API · data zdroje  │          │
│  │               (v0.2) 🔧 ← TEĎ                    │          │
│  └──────────────────────┬───────────────────────────┘          │
│                         │                                       │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSIS ENGINE                               │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │   LLM Engine     │  │   Enrichment     │                    │
│  │ Ollama / OpenAI  │  │ Risk · Summary   │                    │
│  │   (v0.1) ✅      │  │   (v0.1) ✅      │                    │
│  └────────┬─────────┘  └────────┬─────────┘                    │
│           │                     │                               │
│           ▼                     ▼                               │
│  ┌──────────────────────────────────────────────────┐          │
│  │           Data Flow Mapper                        │          │
│  │  Co řešení dělá · jaká data · odkud kam           │          │
│  │               (v0.2) 🔧 ← TEĎ                    │          │
│  └──────────────────────┬───────────────────────────┘          │
│                         │                                       │
│  ┌──────────────────────────────────────────────────┐          │
│  │        Provider Knowledge Base                    │          │
│  │  18 providerů · region · training policy          │          │
│  │               (v0.1) ✅                           │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OUTPUT ENGINE                                │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ HTML Report  │  │  JSON Export │  │  Dashboard   │          │
│  │   (v0.1) ✅  │  │   (fáze 1)   │  │  (fáze 1.5)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Moduly a jejich architektury

| Modul | Arch dokument | Stav |
|-------|--------------|------|
| Git Scanner | [01_Git_Scanner.md](01_Git_Scanner.md) | ✅ v0.1 |
| Data Flow Mapper | [02_Data_Flow_Mapper.md](02_Data_Flow_Mapper.md) | 🔧 v0.2 |
| LLM Engine | součást MVP Architecture | ✅ v0.1 |
| Report Engine | součást MVP Architecture | ✅ v0.1 |
| Enrichment (Risk/Summary) | součást MVP Architecture | ✅ v0.1 |
| Provider Knowledge Base | součást MVP Architecture | ✅ v0.1 |

---

## Datový tok

```
Git Repo(s)
    │
    ▼
Git Scanner
    │  najde: importy, API klíče, dependencies
    │  výstup: List[AIAsset] s raw_findings
    ▼
Code Context Extractor (NOVÉ v0.2)
    │  čte obsah souborů s AI kódem
    │  extrahuje: funkce, třídy, API volání, datové zdroje, prompty
    │  výstup: CodeContext připojený ke každému AIAsset
    ▼
LLM Engine (volitelné)
    │  dostane: CodeContext + findings
    │  vrátí: popis co řešení dělá, jaká data zpracovává,
    │         klasifikace dat, risk assessment
    ▼
Data Flow Mapper (NOVÉ v0.2)
    │  sestaví: data flow model (vstupy → zpracování → výstupy)
    │  identifikuje: data sources, data sinks, data types
    ▼
Enrichment + Provider KB
    │  přidá: risk reasons, recommendations, provider info
    ▼
Report Generator
    │  výstup: HTML report s kompletním přehledem
    ▼
aiscout_report.html
```

---

## Sdílený datový model

Všechny moduly komunikují přes Pydantic modely v `aiscout/models/`:

- **AIAsset** — hlavní entita (jedno AI řešení)
- **Finding** — jeden nález (import, API klíč, dependency)
- **CodeContext** — extrahovaný kontext kódu (NOVÉ v0.2)
- **DataFlowMap** — mapa datových toků (NOVÉ v0.2)
- **ScanResult** — výsledek scanu (seznam assetů + metadata)
- **ClassificationResult** — výsledek LLM klasifikace

---

## Adresářová struktura

```
aiscout/
├── cli.py                     # CLI entry point
├── models/
│   └── assets.py              # Sdílený datový model
├── scanners/
│   ├── base.py                # Scanner plugin interface
│   └── git_scanner.py         # Git Repository Scanner
├── engine/
│   ├── llm.py                 # LLM Analysis Engine
│   ├── enrichment.py          # Risk reasoning + summary
│   ├── code_analyzer.py       # Code Context Extractor (NOVÉ)
│   └── data_flow.py           # Data Flow Mapper (NOVÉ)
├── knowledge/
│   └── providers.py           # Provider Knowledge Base
└── report/
    ├── html.py                # Report generator
    └── templates/
        └── report.html.j2     # HTML šablona
```
