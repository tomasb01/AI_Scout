# AI Scout — podkladové dokumenty (bundle pro repo)

Struktura odpovídá layoutu repa (Product Spec §9.4). Zkopírovat obsah do kořene repa.

## Obsah a role dokumentů

**01_Prod_specs/**
- `AI_Scout_Product_Spec_v13.docx` — zastřešující spec (strategie, scope, sprintová roadmapa, Appendix A descopy). Zdroj pravdy; aktualizovat při merge každého sprintu.
- `AI_Scout_Vyvojovy_plan.docx` — operativní companion: sprinty s rozsahem, kritérii „hotovo když" a odkazy na podklady.

**01_Prod_specs/specs/** — implementační zadání:
- `AI_Scout_QA_spec.md` — Sprint 0.2: insight katalog I-01–I-10 (ICU šablony + edge-case testy), report linter L-01–L-10, degradace na fact strip, invarianty datové vrstvy, CI pojistky, akceptační kritéria.
- `AI_Scout_datamodel_org_cost.md` — Sprint 0.1 (stabilní ID, risk_status, provenance/confidence) a Sprint 4 (org dimenze, cost observables). Obsahuje závazný princip §0: observables, ne judgments.
- `AI_Scout_AIBOM_mapping.md` — Sprint 5b: mapování všech 50 G7 elementů → CycloneDX → Scout, statusy AUTO/EXTRACT/KB/CONFIG/UNKNOWN, akceptační kritéria MVP.
- `AI_Scout_product_plan.md` — širší produktový kontext (AIBOM strategie, validační milníky, rizika). Sprintová čísla viz Spec v13/Vývojový plán, které mají přednost.

**prototypes/**
- `ai_scout_report_design.html` — cílová podoba reportu (no-LLM režim: fact strip, provenance štítky, QA appendix, Scope & Limitations).
- `ai_scout_mode_comparison.html` — web srovnání static vs. LLM režimu.
- `ai_scout_descent_concept.html` — koncept „Sestup" (L0–L3, role lens) — cílová IA reportu po cost datech; teď slouží jako severka pro datový model.

## Poznámka pro Claude Code k plánu 0.1/0.2/0.3

Rozpad Sprintu 0 na tři plátky schválen. Jedno doplnění k 0.1: `AIAsset.id` odvozené z `repo + solution dir` se změní, až Sprint 0.3 (agregační hranice) předefinuje „řešení". Vědomě přijímáme jednorázový re-baseline golden snapshotů po 0.3 (diff přichází až ve Sprintu 2, tedy po něm). Finding ID (`hash(repo + rule_id + normalizovaná lokace)`) je vůči 0.3 stabilní. Přidat poznámku do kódu u definice AIAsset.id.

Drift dokumentace vs. větev (`--org`, `--manifests-only`, `aiscout check`, GitHub Coverage, 139 testů) bude zapsán do Spec v0.14 při příští revizi.
