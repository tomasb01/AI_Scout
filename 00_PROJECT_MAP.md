# AI Scout — Mapa projektu (začni tady)

> Jeden dokument, který říká, co je kde, co je zdroj pravdy, co je hotové a co dělat dál.
> Aktualizuj při merge každého sprintu.

**Stav: v0.14.0 · větev `main` · 311 testů · dokončen Sprint 3 (MCP & Agent scanner — launch feature); launch trojice SARIF + diff + MCP kompletní**

---

## 1. Hierarchie dokumentů (co je co a co platí)

Dokumenty mají tři úrovně. Když si odporují, platí vyšší úroveň:

| Úroveň | Dokument | Role |
|--------|----------|------|
| **1. STRATEGIE** | `01_Prod_specs/AI_Scout_Product_Spec_v13.docx` | **Jediný platný produktový spec.** Scope, descopy (Appendix A), pricing, sprintová roadmapa (§15). Starší verze (v6–v10) jsou v `[Archive]/` a NEPLATÍ. |
| **2. OPERATIVA** | `01_Prod_specs/AI_Scout_Vyvojovy_plan.docx` | Companion ke Spec v13: rozpis sprintů — co přesně, v jakém pořadí, kritéria „hotovo když", odkazy na zadání. |
| **3. IMPLEMENTAČNÍ ZADÁNÍ** | `01_Prod_specs/specs/*.md` | Detailní zadání pro konkrétní sprinty (viz §2). |

Pomocné: `README_BUNDLE.md` (mapa balíčku podkladů + poznámka k AIAsset.id re-baseline) · `02_Architecture/` (technické popisy hotových komponent) · `03_Documentation/` (stav projektu, sprint log, access strategie).

## 2. Podkladové dokumenty → sprinty

| Dokument | Slouží pro | Stav sprintu |
|----------|-----------|--------------|
| `01_Prod_specs/specs/AI_Scout_datamodel_org_cost.md` | **Sprint 0.1** (stabilní ID, risk_status) + **Sprint 4** (org dimenze, cost observables) | 0.1 ✅ hotový · 4 ⏳ |
| `01_Prod_specs/specs/AI_Scout_QA_spec.md` | **Sprint 0.2** — insight katalog I-01–I-10, linter L-01–L-10, fact strip, degradace | ✅ hotový (v0.9.0) |
| `01_Prod_specs/specs/AI_Scout_AIBOM_mapping.md` | **Sprint 5b** — CycloneDX ML-BOM export dle G7 | ⏳ |
| `01_Prod_specs/specs/AI_Scout_product_plan.md` | Širší kontext (AIBOM strategie, rizika); při konfliktu má přednost Spec v13 | referenční |
| `prototypes/ai_scout_report_design.html` | Cílová podoba reportu (Sprint 0.2) | referenční |
| `prototypes/ai_scout_mode_comparison.html` | Web: srovnání static vs. LLM režimu | referenční |
| `prototypes/ai_scout_descent_concept.html` | Koncept „Sestup" (L0–L3) — cílová IA reportu po cost datech | referenční |

## 3. Co je implementované (kód, větev `main`)

**Základ v0.7.0:** Git Scanner → Code Context Extractor → Data Flow Mapper (rule-based) → LLM Engine (Ollama/OpenAI-compat, volitelný) → Enrichment → HTML report + JSON export · Web UI · CLI · Docker · security hardening.

**Navíc od v0.7.0:**
1. `--org` sken celé GitHub organizace (`aiscout/scanners/github_org.py`) + filtry + sekce „GitHub Coverage" v reportu
2. `--manifests-only` nízkocitlivostní sken (jen dependency manifesty)
3. `aiscout check` — pre-commit/CI guardrail (klíče + citlivý egress) + `.pre-commit-hooks.yaml` + `examples/ai-scout-guardrail.yml`
4. `03_Documentation/GITHUB_ACCESS_STRATEGY.md` — přístupy od jednotlivce po enterprise
5. **Sprint 0.1:** stabilní ID (`sol-`/`f-` hashe), dvouosý model severity × confidence + `risk_status` (vážené skóre odstraněno), „No findings" místo „OK", verze Scout+KB v hlavičce reportu, Scope & Limitations, JSON `schema_version 1.1.0`, deterministický výstup (`AISCOUT_TIMESTAMP`)
6. **Sprint 0.2 (v0.9.0):** QA vrstva reportu — typované insighty I-01–I-10 (ICU šablony, mini-ICU renderer bez PyICU), report linter L-01–L-10 s degradací na fact strip, fact strip z kontrolovaného slovníku jako default detail v no-LLM režimu, QA appendix, provenance štítky RULE/LLM, JSON `schema_version 1.2.0` (`insights` + `qa`), `aiscout scan --strict` (moduly `report/qa_vocab.py`, `report/insights.py`, `report/linter.py`, `report/qa.py`)
7. **Sprint 0.3 (v0.10.0):** agregační hranice (řešení = aplikace/služba; tutorial collapse + manifest roots, `engine/aggregation.py`) + detektor charakteru repa (production│tutorial_example│experiment│unknown, `engine/repo_character.py`) + pročištěný install (fix: wheel bez šablon = rozbitá čerstvá instalace; README přepsán). JSON `schema_version 1.3.0`, proveden plánovaný re-baseline `sol-` ID

Detailní stav: `03_Documentation/PROJECT_STATUS.md` · pro AI asistenty: `CLAUDE.md`.

## 4. Sprintová roadmapa a kde v ní jsme

| Sprint | Obsah | Zadání | Stav |
|--------|-------|--------|------|
| 0.1 | Stabilní ID + risk status model | datamodel spec §1–§2 | ✅ **hotový** (v0.8.0) |
| 0.2 | QA vrstva reportu (insighty, linter, fact strip) | QA spec | ✅ **hotový** (v0.9.0) |
| 0.3 | Agregační hranice + detektor tutorial/produkce + install | QA spec + Spec v13 §3.4 | ✅ **hotový** (v0.10.0) |
| → | Validační milník: reálná repa + purpose-first refactor identity (v0.11.0) + AIBOM groundwork (schema 1.4.0) | feedback z validace | ✅ hotový (viz SPRINT_LOG) |
| 1 | SARIF export | Spec v13 §15 | ✅ **hotový** (v0.12.0; živě ověřeno dogfood workflow na AI_Scout repu — Security tab bez warningů) |
| 2 | Diff / trend režim | datamodel spec (finding stavy) | ✅ **hotový** (v0.13.0) |
| 3 | MCP & Agent scanner (launch feature) | Spec v13 §15 | ✅ **hotový** (v0.14.0; validováno živě) |
| **→** | **Launch: veřejné repo + threat model + pilot outreach** | Spec v13 §17 | ⏳ **← jsme tady** (mimo kód) |
| 4 | SCM abstrakce + GitLab + org/cost observables | datamodel spec §2–§3 | ⏳ |
| 1 | SARIF export | Spec v13 §15 | ⏳ (odemčeno — stabilní ID hotové) |
| 2 | Diff / trend režim | datamodel spec (finding stavy) | ⏳ |
| 3 | MCP & Agent scanner (launch feature) | Spec v13 §15 | ⏳ |
| 4 | SCM abstrakce + GitLab + org/cost observables | datamodel spec §2–§3 | ⏳ |
| 5 / 5b | KB Premium Feed / AIBOM export | AIBOM mapping | ⏳ |
| 6 | Compliance reporting | Spec v13 §15 | ⏳ |

Paralelně (mimo kód): threat model dokument, veřejné repo+README po Sprintu 0, pilot outreach (RBCZ/V1).

## 5. Otevřené věci (vyžadují akci)

1. ~~Chybí `landing/index.html` a `aiscout/web/templates/index.html`~~ — ✅ commitnuto (a5b0387, 8. 7. 2026).
2. **Spec v0.14** — zapsat drift do specu (org sken, guardrail, Sprint 0.1, Sprint 0.2 / QA vrstva, 248 testů); podle domluvy připraví strategická konverzace.
3. ~~Merge větve do `main`~~ — ✅ větev je v main (19591f7).
4. Netrackované lokální soubory: `01_Prod_specs/[Archive]/v11+v12.docx`, `prototypes/variant_a/b/c.html` — rozhodnout commit vs. ignore.

## 6. Jak pokračovat v terminálu

```bash
git checkout main && git pull
uv sync
uv run pytest tests/ -q          # očekávej: 261 passed
uv run aiscout scan --local tests/fixtures --no-llm -o report.html   # ukázka výstupu
uv run aiscout scan --local tests/fixtures_tutorial --no-llm -o t.html  # tutorial collapse demo
```

Launch trojice **SARIF + diff + MCP je hotová** (Sprinty 1–3). Dle specu §17 teď následuje **launch mimo kód**: veřejné repo + README (kvalita reportu je obhajitelná), threat model dokument (před prvním pilotem — v prodejním rozhovoru udělá víc než tři featury), pilot outreach 1 firma z RBCZ/V1 („vyzkoušej a řekni, za co by sis zaplatil"). Další kódový sprint = **Sprint 4: SCM abstrakce + GitLab + org/cost observables** (první placený scanner; pozor: začíná ukládat team/model_refs/call_sites — „sbírat teď, zobrazit později").
