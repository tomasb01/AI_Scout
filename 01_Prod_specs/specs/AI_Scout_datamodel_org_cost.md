# AI Scout — Rozšíření datového modelu: org dimenze + cost observables

> **Účel:** Zadání pro rozšíření scanneru a datového modelu tak, aby budoucí report (vč. cost mapy pro IT Governance a org-scale pohledů) byl jen renderováním již sebraných dat. Navazuje na `AI_Scout_QA_spec.md` (insight katalog, linter, invarianty).
>
> **Status modelu:** očekává se evoluce. Proto platí evoluční pravidla v §1 — model se rozšiřuje aditivně, nikdy se nemění význam existujících polí.

---

## 0. Zastřešující princip: observables, ne judgments

Datový model (a z něj generované insighty) smí obsahovat výhradně **pozorovatelné skutečnosti** a jejich deterministické agregace. Nikdy ne soudy o přiměřenosti, efektivitě nebo oprávněnosti.

| ✅ Povolené (fakta) | ❌ Zakázané (soudy) |
|---|---|
| „Call site volá model cenové třídy *premium* (KB)" | „Model je dražší, než úloha vyžaduje" |
| „Volání je uvnitř smyčky nad kolekcí" | „Řešení plýtvá tokeny" |
| „6 call sites kombinuje premium třídu a loop pattern" | „Doporučujeme downgrade na levnější model" |
| „Provider X: 14 řešení, 31 call sites" | „Náklady na providera X jsou příliš vysoké" |
| „Kontextový profil: RAG retrieval, k=20" | „RAG je nadbytečně velký" |

Pravidlo pro insight katalog: **cost insighty popisují koncentraci a strukturu nákladových míst, nikdy jejich oprávněnost.** Interpretace patří člověku (FinOps, governance, tým). Toto pravidlo je závazné i pro budoucí LLM-enrichment vrstvu — LLM smí popsat *co kód dělá*, ne *zda je to rozumné utrácení*.

---

## 1. Evoluční pravidla modelu (protože se bude vyvíjet)

1. **Aditivní evoluce:** nová pole a nové enum hodnoty ano; změna významu nebo odebrání existujícího pole jen s major verzí schématu.
2. **`schema_version`** na rootu (semver). Renderer odmítne major verzi, kterou nezná.
3. **Enumy vždy s `unknown`** a parser je forward-compatible: neznámou hodnotu mapuje na `unknown` + log, nikdy nepadá.
4. **Provenance + confidence na každém odvozeném poli:** `{ "value": ..., "provenance": "rule|llm|kb|manual", "confidence": 0.0–1.0, "rule_id": "..." }`. Deterministická fakta mají confidence 1.0.
5. **Stabilní ID všude** (řešení, nález, call site): deterministický hash, přežívá mezi scany — předpoklad diffu a trendů.
6. **Žádné surové řetězce z repozitáře** v polích určených k renderování do prózy (vazba na QA spec, princip P-1). Surová evidence žije jen v `evidence.*` polích.

---

## 2. Org dimenze

### 2.1 Hierarchie

```
org → repos[] → solutions[] → call_sites[] / findings[]
       ↑
     teams[] (mapování na repa/cesty, ne na lidi)
```

### 2.2 Schéma (anotovaný příklad)

```jsonc
{
  "schema_version": "1.1.0",
  "scan": {
    "scan_id": "s-2026-06-12-a41f",
    "previous_scan_id": "s-2026-05-12-9c2e",        // pro diff; null pokud první
    "timestamp": "2026-06-12T15:28:00Z",
    "scout_version": "0.8.2",
    "kb_version": "2026.23",
    "mode": "static",                                // static | llm_enriched
    "scope": {
      "org": "acme-bank",
      "repos_requested": 43, "repos_scanned": 41,
      "repos_skipped": [ { "repo": "legacy-x", "reason": "clone_failed" } ],
      "files_scanned": 18240, "files_skipped": 312
    }
  },

  "teams": [
    {
      "team_id": "t-payments",
      "name": "Payments",                            // z mapovacího souboru, ne z gitu
      "mapping_source": "codeowners",                // codeowners | manual_file | repo_topic | unknown
      "repos": ["r-pay-core", "r-pay-api"]
    }
  ],

  "repos": [
    {
      "repo_id": "r-pay-core",
      "name": "pay-core",
      "default_branch": "main",
      "team_id": "t-payments",                       // null = nezařazeno (validní stav!)
      "character": {                                  // tutorial/produkce detektor
        "value": "production",                        // production | tutorial_or_example | experiment | unknown
        "provenance": "rule", "confidence": 0.9, "rule_id": "CHAR-001"
      }
    }
  ],

  "solutions": [
    {
      "solution_id": "sol-7f3a91",                   // hash(repo + normalizovaná root cesta + pattern)
      "repo_id": "r-pay-core",
      "team_id": "t-payments",                       // denormalizováno pro filtry
      "name": { "value": "fine-tuning pipeline", "provenance": "rule", "confidence": 1.0 },
      "owner": {                                      // vlastnictví řešení, ne aktivita osob
        "identity": "lukaskellerstein",
        "method": "dominant_committer",               // dominant_committer | last_committer | codeowners
        "confidence": 0.95
      },
      "category": "fine_tuning",
      "pattern": "fine_tuning_pipeline",              // konečný slovník (QA spec §1.4)
      "sources": ["database_sql"],
      "sinks":   [ { "provider_id": "huggingface", "endpoint_class": "inference_api", "region": "US" } ],
      "data_categories": [
        { "value": "database_records", "provenance": "rule", "confidence": 1.0 },
        { "value": "pii", "provenance": "rule", "confidence": 0.6 }   // heuristika; <0.6 → jen manual review
      ],
      "risk_status": "critical",                      // critical | review | no_findings  (nikdy "ok")
      "finding_ids": ["f-8c41a2", "f-8c41a3"],
      "cost_observables": { "...": "viz §3" }
    }
  ],

  "findings": [
    {
      "finding_id": "f-8c41a2",                       // hash(repo + rule_id + normalizovaná lokace)
      "solution_id": "sol-7f3a91",
      "rule": { "id": "SEC-KEY-001", "version": 3 },
      "type": "hardcoded_api_key",
      "severity": "critical",
      "confidence": 1.0,
      "first_seen_scan_id": "s-2026-05-12-9c2e",      // → "tu je to už 2 scany"
      "status": "open",                               // open | accepted_risk | resolved (workflow stav)
      "evidence": {                                    // JEDINÉ místo pro surové řetězce; vždy redigované
        "file": "2_tools/mistral-v03.ipynb", "line": 47,
        "snippet_redacted": "login(token=\"hf_qhzLG…LnAX\")",
        "commit": "4f9c21d"
      }
    }
  ]
}
```

### 2.3 Poznámky k org dimenzi

- **Tým je atribut repa/cesty, ne osoby.** Mapování CODEOWNERS → manuální YAML → repo topic, v tomto pořadí priority; `team_id: null` je legitimní a report ho zobrazuje jako „Unassigned" (samo o sobě užitečný governance nález).
- **Owner = vlastnictví řešení** (dominantní committer dané cesty). Per-person aktivita (kolik kdo commituje, kdo používá AI) se **nesbírá** — hranice z diskuse o 5.3 platí i v modelu, ne jen v UI.
- Denormalizace `team_id` na řešení je záměrná: L1 osy (tým × provider × kategorie × cost) musí být jeden průchod, ne join.

---

## 3. Cost observables

### 3.1 Co scanner vidí už dnes (a začne ukládat)

```jsonc
"cost_observables": {
  "model_refs": [
    {
      "provider_id": "openai",
      "model": { "value": "gpt-4o", "provenance": "rule", "confidence": 1.0 },
      "model_resolution": "literal",                  // literal | config_key | env_var | dynamic_unknown
      "kb_price_tier": "premium",                     // z KB: economy | standard | premium | unknown
      "kb_lifecycle": "supported"                     // supported | deprecated | retired | unknown  (vazba na 5.7)
    }
  ],
  "call_sites": [
    {
      "call_site_id": "cs-19b2c4",                    // stabilní hash
      "model_ref_index": 0,
      "evidence": { "file": "pipeline/train.py", "line": 112 },
      "invocation_pattern": "loop_over_collection",   // per_request | loop_over_collection | batch_job |
                                                      // scheduled | startup_once | unknown
      "context_profile": "large_context",             // minimal | conversational_history | rag_retrieval |
                                                      // large_context | multimodal | unknown
      "context_evidence": ["history_accumulation"],   // konečný slovník signálů: history_accumulation,
                                                      // rag_k_gt_10, file_payload, image_payload, system_prompt_gt_4k
      "streaming": false,
      "retry_wrapper": true                           // retry/backoff kolem volání = multiplikátor volání
    }
  ]
}
```

### 3.2 Co se z toho smí odvodit (deterministické agregace — fakta)

- **Cost surface per provider/tým/repo:** počty řešení, call sites, modelů; rozpad podle `kb_price_tier`.
- **Koncentrační flagy (kombinace faktů, bez soudu):** `premium × loop_over_collection`, `premium × large_context`, `retry_wrapper × loop`. Renderují se jako „místa koncentrace nákladů k ručnímu posouzení", nikdy jako „plýtvání".
- **Insight I-11 · COST_CONCENTRATION** (doplnění QA katalogu):
  - Trigger: `concentration_sites > 0`.
  - Šablona: `{concentration_sites, plural, one {# call site combines} other {# call sites combine}} a premium-tier model with a high-volume invocation pattern — review for cost concentration.`
  - Testy: sites ∈ {1, 6}; invarianty: `concentration_sites ≤ total_call_sites`.
- **Insight I-12 · PROVIDER_COST_SURFACE** — `info`:
  - Šablona: `{provider_count, plural, one {# paid provider} other {# paid providers}} in use; the largest, {top_provider}, is called from {top_sites} call sites across {top_solutions} solutions.`
  - Čistě popisné; žádné částky, žádná přiměřenost.

### 3.3 Co se odvodit nesmí (zákazy — vazba na §0)

- Žádné odhady částek v měně ze statiky (neznáme objemy volání ani skutečné token counts). Pokud někdy, pak jedině z importovaných billing dat zákazníka jako samostatná, jasně označená vrstva — mimo scope scanneru.
- Žádná doporučení záměny modelu („use gpt-4o-mini instead").
- Žádné hodnocení přiměřenosti modelu vůči úloze.
- Žádné per-person cost metriky (průnik se zákazem z §2.3).

### 3.4 KB rozšíření

Provider KB doplnit o `price_tier` a `lifecycle` per model (jedna datová sada slouží 5.4, 5.7 i cost mapě). Tier je záměrně kategorie, ne částka — částky se mění týdně, tiery jsou stabilní a pro „kde se koncentrují náklady" stačí.

---

## 4. Dopady do existujících artefaktů

- **QA spec:** přidat I-11, I-12 do katalogu (šablony výše); linter beze změny.
- **Report (Sestup):** L0 governance fakt z I-12; L1 nová osa „Cost surface" (provider × tým × tier × pattern); L2 fact strip + řádek `Cost: gpt-4o (premium) · loop pattern · large context`; L1.5 inventura dostává filtry team/repo/tier.
- **Scanner:** ukládat `model_refs` a `call_sites` od příští verze i v případě, že je report zatím nezobrazuje — zpětné dosbírání = přescanování všeho.

## 5. Akceptační kritéria

- [ ] Schéma validuje (JSON Schema draft 2020-12 k doplnění z tohoto vzoru), `schema_version` na rootu, enumy s `unknown`.
- [ ] Scan org se 2+ repy produkuje korektní team mapping vč. `null` týmu.
- [ ] Cost observables se ukládají pro všechny detekované call sites; `model_resolution: dynamic_unknown` pokrývá modely volené za běhu.
- [ ] I-11/I-12 mají testy dle QA spec vzoru; žádný cost insight neobsahuje měnu, doporučení modelu ani hodnotící adjektiva (test na zakázaný slovník: `expensive, wasteful, unnecessary, overkill, cheaper, downgrade`).
- [ ] Dvojí scan = identická `solution_id`, `finding_id`, `call_site_id`.
