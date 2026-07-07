# AI Scout — AIBOM export: mapování G7 „SBOM for AI – Minimum Elements" → CycloneDX → Scout datový model

> **Účel:** Zadání pro implementaci exportní vrstvy `scout export --format cyclonedx-aibom`. Mapuje všech 7 klastrů a jejich elementy z G7 guidance (BSI/ACN/ANSSI/CSE/CISA/NCSC/NCO + EU Komise, květen 2026) na CycloneDX struktury a na Scout datový model (`AI_Scout_datamodel_org_cost.md`).
>
> **Cílový formát:** CycloneDX ≥ 1.6 (ML-BOM: typ komponenty `machine-learning-model`, `data`, objekt `modelCard`). SPDX 3.0 AI profil jako druhá fáze. Přesné názvy polí validovat proti aktuálnímu CycloneDX JSON schématu — mapování níže je na úrovni struktur.

---

## 0. Statusová legenda

| Status | Význam | Odhad práce |
|---|---|---|
| ✅ AUTO | Už je ve Scout datovém modelu — jen serializace | nízká |
| 🔧 EXTRACT | Scanner to vidí, ale zatím neukládá — nová extrakce | střední |
| 📚 KB | Doplní Provider Knowledge Base (rozšíření KB dat) | průběžná údržba |
| 👤 CONFIG | Jednorázový vstup zákazníka (org identita, klíče, odkazy) — nový konfigurační soubor `aibom.yaml` | nízká |
| ⬜ UNKNOWN | Mimo dosah statiky — exportuje se **explicitně jako `unknown`** (G7 to výslovně připouští a preferuje před vynecháním) | žádná |

**Zásady exportu (závazné):**
1. Jeden AIBOM per solution + agregovaný per repo a per org (CycloneDX `dependencies` graf skládá hierarchii).
2. Každé odvozené pole nese provenance/confidence přes namespaced properties: `scout:provenance`, `scout:confidence`, `scout:rule_id` — konzistentní s datovým modelem §1.4.
3. Neznámé hodnoty se zapisují explicitně, nikdy se pole nevynechává tam, kde ho G7 očekává.
4. Export je deterministický: dva běhy nad stejným scanem = bitově shodný AIBOM (řazení explicitní), aby fungoval diff a podpis.
5. Žádné soudy — platí §0 z datamodel spec i pro AIBOM (žádná „adequacy" pole).

---

## 1. Metadata Cluster (10 elementů) → CycloneDX `metadata`

| G7 element | CycloneDX | Scout zdroj | Status |
|---|---|---|---|
| SBOM author | `metadata.authors` / `metadata.organization` | Entita provozující Scout = **zákazník** (G7: author je ten, kdo tool spouští, ne tool). Z `aibom.yaml` | 👤 CONFIG |
| SBOM version | `metadata` + `version` (BOM serial + verze) | Per component-name/version pár; semver, major = 1; nová verze při změně obsahu. Odvodit ze scan sekvence + diffu | 🔧 EXTRACT |
| SBOM data format name | `bomFormat: "CycloneDX"` | konstanta | ✅ AUTO |
| SBOM data format version | `specVersion` | konstanta (nepoužívat deprecated verze formátu) | ✅ AUTO |
| SBOM author signature | JSF podpis / detached signature | Nová schopnost: podepsané exporty (algoritmus dle NIST DSS / ISO 14888-4 / ENISA doporučení). Klíč zákazníka z `aibom.yaml` | 🔧 EXTRACT + 👤 CONFIG |
| SBOM tool name | `metadata.tools[].name = "AI Scout"` | konstanta | ✅ AUTO |
| SBOM tool version | `metadata.tools[].version` | `scan.scout_version` | ✅ AUTO |
| SBOM generation context | `metadata.lifecycles` | Statická analýza zdrojů = **"pre-build"** (G7 příklad: „SBOM generated from source code could be identified as before build") | ✅ AUTO |
| SBOM timestamp | `metadata.timestamp` (RFC 9557) | `scan.timestamp` | ✅ AUTO |
| SBOM dependency relationship | `dependencies[]` graf | solution → model refs, datasety, komponenty; „derived from" pro fine-tuned modely (lineage z KB, je-li známa) | ✅ AUTO (graf) / 📚 KB (lineage) |

## 2. System Level Properties (9) → CycloneDX root komponenta typu `application` + `services`

| G7 element | Scout zdroj | Status |
|---|---|---|
| System name | `solution.name` (rule = pattern label; LLM režim = narativní název s provenance) | ✅ AUTO |
| System components | `solutions.sinks` + `model_refs` + tech + DB komponenty → CycloneDX `components[]` | ✅ AUTO |
| System producer | `solution.owner` + `team` (interní producent) nebo org z configu | ✅ AUTO + 👤 CONFIG |
| System version | Commit hash jako proxy verze (dokumentovat v `scout:version_scheme = "git-commit"`); nemá-li řešení release verzi → G7 připouští unknown | 🔧 EXTRACT |
| System timestamp | Poslední commit dotčených cest | 🔧 EXTRACT |
| System data flow | **Jádro Scouta**: Data Flow Mapper sources→sinks, API externích služeb, MCP protokoly (G7 explicitně: „multi-agent communication protocols", „bidirectional data flow towards external services") → `services[]` + `scout:dataflow` properties | ✅ AUTO; MCP část po dodání MCP scanneru |
| System data usage | Odkaz na KB provider policy (training-on-data, logging) per sink; vlastní zpracování = unknown/CONFIG | 📚 KB + ⬜ |
| System input/output properties | Modalita odvoditelná z kódu (text/image/audio payloady = `context_evidence` signály); tokenizer apod. = unknown | 🔧 EXTRACT (modalita) / ⬜ (zbytek) |
| Intended application area | Static = ⬜ unknown; LLM režim = purpose s confidence (`scout:provenance = "llm"`) | ⬜ / LLM |

## 3. Models Cluster (13) → CycloneDX `component` typu `machine-learning-model` + `modelCard`

| G7 element | Scout zdroj | Status |
|---|---|---|
| Model name | `model_refs[].model` (literal z kódu/konfigurace) | ✅ AUTO |
| Model identifier | PURL: `pkg:huggingface/org/model@rev` pro HF; `pkg:generic/openai/gpt-4o` pro API modely; + `model_resolution` (literal/config/env/dynamic_unknown) | 🔧 EXTRACT (PURL builder) |
| Model version | API modely: alias vs. pinned verze (`gpt-4o` vs. `gpt-4o-2024-11-20`) — rozlišit; nepinned = fakticky „latest", zapsat alias + `scout:version_pinned = false`; unknown kde nelze | 🔧 EXTRACT |
| Model timestamp | Release datum modelu | 📚 KB |
| Model producer | Vendor z Provider KB (vč. multi-producer u fine-tuned: base producer + interní tým) | 📚 KB ✅ |
| Model description | Capabilities, limitations, lineage (base model u fine-tuningu), dependencies | 📚 KB (API modely) / 🔧 EXTRACT (lokální fine-tuning: base model z kódu) |
| Model hash value | API modely: **unknown** (nemáme artefakt — G7 explicitně: pokud author nemá přístup k artefaktu, uvede unknown). Lokální weights v dosahu scanneru: spočítat | 🔧 EXTRACT (lokální) / ⬜ (API) |
| Model hash algorithm | SHA-256 (IANA textual name) tam, kde hash počítáme | ✅ AUTO |
| Model properties | Architektura, počet parametrů, typ (transformer…) → `modelCard.modelParameters` | 📚 KB |
| Model input-output properties | Modalita, context length → `modelCard` | 📚 KB |
| Model training properties | Odkaz na model card / dokumentaci producenta | 📚 KB (link) / ⬜ (detaily) |
| Model license | Licence modelu; open-weight/open-data flagy | 📚 KB |
| Model external references | Model card URL, dokumentace, paper → `externalReferences[]` | 📚 KB |

**Důsledek pro KB:** Models cluster dělá z Provider KB motor AIBOM exportu. Rozšíření KB per model: `release_date, producer, purl_template, architecture_family, params, context_length, modalities, license, model_card_url, lineage`. To je stejná datová sada, která slouží 5.4 (typy modelů), 5.7 (lifecycle) a cost mapě (tier) — jedna investice, čtyři features, a další argument pro KB jako subscription produkt.

## 4. Datasets Properties (10) → CycloneDX `component` typu `data`

Relevantní primárně pro fine-tuning/training řešení; u čistě inferenčních řešení pokrývá datové vstupy System data flow.

| G7 element | Scout zdroj | Status |
|---|---|---|
| Dataset name | Referencovaná cesta/URL normalizovaná na label (surová cesta jen v evidence) | 🔧 EXTRACT |
| Dataset description | Účel odvozený z pattern (fine-tuning/eval) — jen kde je deterministický | 🔧 EXTRACT (částečně) / ⬜ |
| Dataset content | Kategorie z Data Flow Mapperu (financial, records…) + formát (JSON/CSV z přípony/parseru) | ✅ AUTO (kategorie) / 🔧 (formát) |
| Dataset identifier | Cesta/URL/HF dataset ID | 🔧 EXTRACT |
| Dataset hash | Soubor v dosahu: spočítat; externí/runtime: unknown | 🔧 / ⬜ |
| Dataset provenance | Sběr, curation, labeling — z kódu nevyčitatelné | ⬜ UNKNOWN |
| Dataset statistical properties | Mimo statiku | ⬜ UNKNOWN |
| Dataset sensitivity | **Mapuje se přímo na `data_categories`** vč. PII klasifikace s confidence (LLM režim: element-level klasifikace bez extrakce) | ✅ AUTO / LLM |
| Dataset dependency relationship | Labeling/filtering tooly viditelné v dependencies | 🔧 EXTRACT (částečně) |
| Dataset license | Z kódu nevyčitatelné (HF datasety: z KB/API metadat) | ⬜ / 📚 |

## 5. Infrastructure (2) → CycloneDX `components` + externí reference

| G7 element | Scout zdroj | Status |
|---|---|---|
| Infrastructure software | Frameworks, runtime, third-party knihovny z dependency manifestů — klasické SBOM území; embed nebo reference na standardní SBOM (syft apod.), Scout přidává AI vrstvu | ✅ AUTO (manifesty) / 🔧 (kompozice s existujícím SBOM tooling) |
| Infrastructure hardware | Link na HBOM | 👤 CONFIG / ⬜ |

## 6. Security Properties (4)

| G7 element | Scout zdroj | Status |
|---|---|---|
| Security controls | **Jen detekované kontroly, nikdy tvrzení o absenci**: API authentication, input filtry, prompt-injection ochrany viditelné v kódu → seznam s evidence. (Absence kontrol patří do findings jako risk, ne do SBOM jako „chybí X".) | 🔧 EXTRACT (nová detekční pravidla „positive controls") |
| Security compliance | Vendor certifikace (SOC 2…) z KB per provider komponenta; systémová compliance zákazníka = CONFIG | 📚 KB + 👤 |
| Cybersecurity policy information | Link na security.txt organizace | 👤 CONFIG |
| Vulnerability referencing | Dependencies: OSV/advisory linky (integrace, fáze 2); modely: AI-vuln databáze zatím nezralé → unknown + KB advisory link per vendor | 🔧 (fáze 2) / ⬜ |

## 7. KPI (2)

| G7 element | Scout zdroj | Status |
|---|---|---|
| Security metrics | Mimo statiku (benchmarky, robustness) | ⬜ UNKNOWN / importovatelné |
| Operational performance KPIs | Runtime metriky — mimo scope | ⬜ UNKNOWN / importovatelné |

Export zapíše klastr s `unknown` + `scout:note = "not derivable from static analysis; importable from monitoring"` — validní dle G7, poctivé, a nechává prostor pro budoucí import vrstvu.

---

## 8. Souhrn pokrytí a plán

**Bilance (50 elementů):** ✅ AUTO ~14 · 🔧 EXTRACT ~15 · 📚 KB ~12 · 👤 CONFIG ~6 · ⬜ čistě UNKNOWN ~8 (překryvy: některé elementy kombinují statusy).

**Fáze 1 (MVP exportu):** serializace AUTO polí + KB polí, jež už existují (vendor, certifikace, lifecycle) + `aibom.yaml` config + explicitní unknown všude jinde → **validní CycloneDX ML-BOM bez nové analytické práce.**
**Fáze 2:** EXTRACT položky s nejlepším poměrem (PURL builder, verze pinning, commit-proxy verze, dataset identifikátory, positive-controls detekce) + podpis exportu.
**Fáze 3:** KB rozšíření per model (release, licence, architektura, lineage, model card URL) — průběžně, jako součást KB feedu.
**Fáze 4:** SPDX 3.0 AI profil; OSV integrace; import KPI/billing vrstev.

**Akceptační kritéria MVP:**
- [ ] Výstup validuje proti CycloneDX schématu (ML-BOM komponenty), `bomFormat/specVersion` korektní.
- [ ] Každý G7 element je v exportu přítomen: hodnotou, KB referencí, config hodnotou, nebo explicitním `unknown` — žádné tiché vynechání.
- [ ] Deterministický výstup (dva běhy = shodný soubor), stabilní BOM serial odvozený ze scan_id.
- [ ] `scout:provenance`/`scout:confidence` na všech odvozených polích.
- [ ] Žádné surové řetězce z repozitáře mimo evidence-typu pole; žádná hodnotící/adequacy pole.
- [ ] Per-solution i agregovaný org export; dependency graf skládá hierarchii org → repo → solution → komponenty.
