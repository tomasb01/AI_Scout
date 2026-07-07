# AI Scout — Specifikace QA vrstvy pro no-LLM generování reportu

> **Účel:** Zadání pro implementaci. Definuje (1) katalog typovaných insightů s šablonami a edge-case testy, (2) report linter s pravidly a chováním při degradaci, (3) CI pojistky. Cíl: v no-LLM režimu nesmí report nikdy obsahovat rozbitou, nesmyslnou nebo z kódu prosáklou větu.

---

## 0. Architektonické principy (závazné)

**P-1 — Próza jen z kontrolovaného slovníku.** Věty v reportu se skládají výhradně ze slov pocházejících z: (a) šablon napsaných člověkem, (b) labelů z Provider Knowledge Base, (c) konečného slovníku vzorů/kategorií. Jakýkoli řetězec extrahovaný z repozitáře zákazníka (název proměnné, fragment SQL, cesta, název složky) se **nikdy nevkládá do věty** — patří výhradně do evidence polí renderovaných jako kód (`monospace`).

**P-2 — Šablony nepočítají.** Šablona pouze renderuje hodnoty z datového modelu. Veškerá aritmetika (procenta, součty, podíly) probíhá v datové vrstvě, která má vlastní invarianty (sekce 3). Tím se eliminuje třída chyb „1 developer created over 100%" (vznikla nesouhlasnými jmenovateli: 144 řešení vs. 86 unikátních).

**P-3 — Fakta jako default, věty jako nadstavba.** Detail řešení v no-LLM režimu zobrazuje strukturovaný *fact strip* (Sources / Sinks / Pattern / Tech), nikoli pseudo-shrnutí. Věty existují pouze v Executive Summary a Recommended Actions, kde vznikají výhradně z typovaných insightů (sekce 1).

**P-4 — Degradace, ne pád.** Pokud linter (sekce 2) zachytí chybu ve vyrenderované větě, věta se nevypíše a nahradí ji fakt-fallback. Scan nikdy neselže kvůli textové chybě; každá suprese se loguje a objeví se v QA appendixu reportu.

---

## 1. Katalog typovaných insightů

### 1.1 Datový model insightu

```json
{
  "id": "I-04",
  "type": "AUTHOR_CONCENTRATION",
  "severity": "warning",            // info | warning | critical
  "metrics": { "top_author_pct": 100, "top_author_count": 144, "total": 144 },
  "entities": { "author": "lukaskellerstein" },
  "threshold": { "top_author_pct_gte": 50 },
  "provenance": "deterministic",    // deterministic | llm
  "template_id": "T-AUTHOR-CONC-v2"
}
```

Pravidla:
- `entities.*` smí obsahovat pouze hodnoty z bezpečných domén: identity autorů (git), labely z Provider KB, labely kategorií. Nikdy volný text z kódu.
- `metrics.*` jsou vždy čísla z datové vrstvy, již zvalidovaná invarianty (sekce 3).
- `template_id` je versováno — změna textu šablony = nová verze = vědomé schválení v golden-file diffu (sekce 4).

### 1.2 Šablony — formát

Šablony používají **ICU MessageFormat** (plurály, výběry) — žádná ruční konkatenace stringů. Implementace: `intl-messageformat` (JS) nebo `PyICU`/`babel` (Python). Vedlejší přínos: lokalizace (cs) zdarma.

### 1.3 Katalog

U každého insightu: **Trigger** (kdy se generuje), **Šablona** (EN, ICU), **Edge-case testy** (povinné unit testy), **Invarianty**.

---

**I-01 · INVENTORY_TOTAL** — `info`
- Trigger: vždy (i při 0 nálezech — viz P-coverage).
- Šablona: `Found {total, plural, =0 {no AI solutions} one {# AI solution} other {# AI solutions}} across {repos, plural, one {# repository} other {# repositories}} ({files} files scanned).`
- Testy: total ∈ {0, 1, 2, 144}; repos ∈ {1, 2}; total=0 musí vyrenderovat smysluplnou větu (coverage statement), ne prázdný report.
- Invarianty: `total ≥ 0`, `repos ≥ 1`, `files ≥ repos`.

**I-02 · CRITICAL_FINDINGS** — `critical`
- Trigger: `critical_count > 0`.
- Šablona: `{critical_count, plural, one {# solution requires} other {# solutions require}} immediate attention: {reasons}.` kde `{reasons}` je výčet z konečného slovníku (`hardcoded API keys`, `secrets in configuration`, …) spojený Oxford-comma joinerem na úrovni dat, ne šablony.
- Testy: count ∈ {1, 2}; reasons ∈ {1 položka, 2 položky, 3 položky}; reasons nesmí být prázdné při count > 0.
- Invarianty: `critical_count ≤ total`; `len(reasons) ≥ 1`.

**I-03 · DATA_EGRESS_REGION** — `warning`
- Trigger: `egress_count > 0` pro region mimo whitelist regionů zákazníka (default: mimo EU/local).
- Šablona: `{egress_count, plural, one {# solution sends} other {# solutions send}} data to {region}-based providers ({provider_list}) — verify DPA and data-residency requirements.`
- `provider_list`: max 3 jména z Provider KB + `and {n} more` při překročení (řeší datová vrstva).
- Testy: count ∈ {1, 56}; providers ∈ {1, 3, 5 → „and 2 more"}; region ∈ {US, „outside EU"}.
- Invarianty: `egress_count ≤ total`; každý provider existuje v KB.

**I-04 · AUTHOR_CONCENTRATION (SPOF)** — `warning`
- Trigger: `top_author_pct ≥ 50`.
- Šablona: `{top_author_pct, select, 100 {A single contributor ({author}) created all {total} solutions} other {One contributor ({author}) created {top_author_count} of {total} solutions ({top_author_pct}%)}} — single-point-of-failure risk.`
- Pozn.: speciální větev pro 100 % — „over 100%" třída chyb je touto šablonou + invariantem vyloučena; formulace „all N" je navíc čtivější.
- Testy: pct ∈ {50, 99, 100}; **pct = 100 povinný test**; pct > 100 musí selhat ve validaci dat, nikdy nedojít k šabloně.
- Invarianty: `0 ≤ top_author_pct ≤ 100`; `top_author_count ≤ total`; `pct == round(100 * count / total)` — jeden jmenovatel, definovaný v datové vrstvě.

**I-05 · DEPENDENCY_CONCENTRATION** — `info`
- Trigger: `top_tech_pct ≥ 40`.
- Šablona: `Highest dependency: {tech}, used by {tech_count} of {total} solutions ({tech_pct}%).`
- Testy: pct ∈ {40, 51, 100}; tech label vždy z KB.
- Invarianty: jako I-04.

**I-06 · OVERLAP_GROUPS** — `info`
- Trigger: `overlap_solutions ≥ 2` na úrovni **schopnosti** (capability), nikoli sdílené závislosti. „Using LangChain" není překryv; „RAG nad interní dokumentací" je.
- Šablona: `{overlap_solutions} solutions functionally overlap in {group_count, plural, one {# capability area} other {# capability areas}} — consolidation opportunity.`
- Testy: solutions ∈ {2, 58}; groups ∈ {1, 25}; skupina s 1 řešením se nesmí vygenerovat.
- Invarianty: `overlap_solutions ≥ 2 * group_count` je nepravda obecně, ale platí `overlap_solutions ≥ group_count + 1` pro každou skupinu ≥ 2; `overlap_solutions ≤ total`.

**I-07 · DATA_CATEGORY_VOLUME** — `warning`
- Trigger: `cat_count > 0` pro citlivé kategorie (PII, Financial data, Credentials/Secrets, Health).
- Šablona: `{cat_count, plural, one {# solution processes} other {# solutions process}} {category} — elevated compliance attention recommended.`
- Pozn.: `{category}` je label z konečného slovníku kategorií; tím zaniká chyba „Financial data data" (slovo *data* je součástí labelu, šablona ho nepřidává).
- Testy: count ∈ {1, 18}; všechny labely slovníku projít šablonou (test kartézského součinu) a zkontrolovat absencí zdvojení.
- Invarianty: `cat_count ≤ total`; category ∈ slovník.

**I-08 · UNKNOWN_PROVIDER_CANDIDATES** — `info`
- Trigger: `candidate_count > 0` (nálezy z heuristické záchranné sítě s nízkou confidence).
- Šablona: `{candidate_count, plural, one {# possible AI integration} other {# possible AI integrations}} could not be matched to a known provider — listed under “Manual review”.`
- Testy: count ∈ {1, 7}.
- Invarianty: kandidáti se nikdy nezapočítávají do `total` deterministických nálezů.

**I-09 · SCAN_DELTA** — `info`
- Trigger: existuje předchozí scan se shodným scope.
- Šablona: `Since {prev_date, date, medium}: {added, plural, =0 {no new solutions} one {# new solution} other {# new solutions}}, {removed, plural, =0 {none removed} one {# removed} other {# removed}}{new_providers, plural, =0 {} one {, # new provider} other {, # new providers}}.`
- Testy: (0,0,0) — věta musí dávat smysl; (1,0,0); (5,2,1); chybějící předchozí scan → insight se negeneruje.
- Invarianty: `added, removed, new_providers ≥ 0`. Vyžaduje stabilní ID nálezů (hash repo+pravidlo+normalizovaná lokace) — předpoklad pro celý diff.

**I-10 · LOCAL_ONLY_SHARE** — `info` (pozitivní insight)
- Trigger: `local_count > 0`.
- Šablona: `{local_count, plural, one {# solution runs} other {# solutions run}} fully local (no data egress).`
- Testy: count ∈ {1, 10}.
- Invarianty: `local_count ≤ total`; množiny local a egress jsou disjunktní na úrovni řešení.

### 1.4 Fact strip (P-3) — specifikace fallbacku

Renderuje se vždy v detailu řešení v no-LLM režimu a jako náhrada za supresovanou větu:

```
Sources:  {source_labels}        // konečný slovník: database (SQL), REST endpoint, file input, web search
Sinks:    {sink_labels}          // z Provider KB, vč. regionu: Hugging Face Inference API (US)
Pattern:  {pattern_label}        // konečný slovník: fine-tuning pipeline, RAG, agent loop, chat completion, MCP server
Tech:     {tech_labels}          // z KB: transformers, PEFT, LangChain
```

Žádné pole nesmí obsahovat surový extrahovaný řetězec; pokud pro zdroj/sink neexistuje label, renderuje se `unclassified` + odkaz do evidence.

---

## 2. Report linter

### 2.1 Umístění v pipeline

```
datový model → [validace invariantů §3] → render šablon → [LINTER] → výstup (HTML/PDF/JSON)
```

Linter běží nad **finálním vyrenderovaným textem** každého textového pole (insight věty, action věty, labely). Čistě deterministický (regex + slovníky), žádné LLM, žádná síť — běží u zákazníka.

### 2.2 Pravidla

| ID | Pravidlo | Detekce | Severita |
|----|----------|---------|----------|
| L-01 | Zdvojené slovo | `\b(\w+)\s+\1\b` (case-insensitive; whitelist legitimních dublet, např. „had had" — pro EN reporty prakticky prázdný) | ERROR |
| L-02 | Nepárové závorky/uvozovky | counter přes `()[]{}""''` | ERROR |
| L-03 | Nevyřešený placeholder | výskyt `{`, `}`, `undefined`, `null`, `None`, `NaN`, `[object Object]` | ERROR |
| L-04 | Prázdné závorky / osiřelá interpunkce | `\(\s*\)`, `\s[,;:]\s*$`, `^\s*[,;:.]` | ERROR |
| L-05 | Useknutá věta | nekončí `.?!`, končí spojkou/předložkou ze stop-listu, nebo končí uprostřed slova (poslední token bez mezery před limitem) | ERROR |
| L-06 | Číselná nesmyslnost v textu | `(\d{3,})\s*%` s hodnotou > 100; záporné počty; `over 100%` | ERROR |
| L-07 | Prosáknutí zdrojového kódu do prózy | heuristiky: `snake_case`/`camelCase` identifikátor mimo monospace pole; SQL klíčová slova (`SELECT`, `FROM`, `WHERE`) v próze; ALL-CAPS token délky ≥ 3, který není ve slovníku zkratek (PII, GDPR, DPA, API, MCP, RAG, US, EU, SQL…); cesta (`/` + přípona souboru) v próze | ERROR |
| L-08 | Identická věta napříč řešeními | shodný summary string u ≥ 2 různých řešení → degradace na fact strip u všech výskytů (duplicitní „shrnutí" nenese informaci) | WARN→ERROR při ≥ 3 |
| L-09 | Délkové meze | insight věta 20–280 znaků; action title ≤ 90; mimo meze | WARN |
| L-10 | Nesoulad plurálu | `\b1\s+\w+s\b` pro spočetná substantiva ze slovníku (`1 developers`, `1 solutions`) | ERROR |

Slovníky (zkratky pro L-07, substantiva pro L-10, stop-list pro L-05) jsou konfigurační soubory v repu, verzované spolu se šablonami.

### 2.3 Chování při nálezu (degradace)

- **ERROR:** věta se **nevypíše**. Místo ní se renderuje fact strip (§1.4) příslušné entity; u exec summary se insight vypustí celý. Do logu jde záznam `{rule, template_id, rendered_text, entity_id}` a věta se objeví v **QA appendixu reportu** (skrytá sekce „Suppressed by QA linter", default sbalená) — vývojář/auditor vidí, co a proč bylo potlačeno; běžný čtenář nikdy nevidí rozbitou větu.
- **WARN:** věta se vypíše, jde do logu a QA appendixu.
- **Report-level flag:** výstupní JSON nese `qa: { suppressed: n, warnings: m }`; nenulové `suppressed` vrací nenulový exit code v CI režimu (`--strict`), v produkčním režimu nikoli (P-4: degradace, ne pád).

### 2.4 Co linter neřeší

Gramatickou správnost obecné angličtiny — tu garantuje konstrukce (P-1: člověkem psané šablony + KB labely). Linter je pojistka proti chybám *skládání*, ne korektor.

---

## 3. Invarianty datové vrstvy (validace před renderem)

Spouští se po dokončení analýzy, před renderem. Porušení = chyba scanu (tvrdá, loguje se), protože znamená bug v analýze, ne v textu:

1. `0 ≤ pct ≤ 100` pro všechna procentuální pole; všechna procenta počítá jediná funkce s explicitním jmenovatelem.
2. Σ počtů per kategorie == `total` (každé řešení právě v jedné primární kategorii).
3. Σ(no_findings, review, critical) == `total`.
4. Všechny počty ≥ 0; `subset_count ≤ total` pro každé podmnožinové číslo.
5. Každý provider/tech/kategorie label existuje v KB/slovníku (cizí klíč).
6. Stabilní ID nálezu je deterministické: dva běhy nad shodným commitem ⇒ shodná množina ID (test v CI).
7. Disjunktnost: local-only ∩ egress = ∅.

---

## 4. CI pojistky

1. **Unit testy šablon:** každá šablona × všechny edge-cases z katalogu (0/1/mnoho, 100 %, prázdné seznamy, dlouhé seznamy). Generovat kartézský součin automaticky.
2. **Golden files:** sada fixture repozitářů (prázdné repo; 1 soubor; 1 přispěvatel/100 %; tutorial repo s 1000 notebooky; repo bez AI; repo s neznámým providerem). Scan v CI → snapshot vyrenderovaného reportu → diff vyžaduje schválení.
3. **Determinismus:** dvojí běh nad shodným commitem, diff výstupů musí být prázdný (vč. pořadí — řazení všude explicitní).
4. **Korpusový jazykový test (interní QA, ne runtime):** jednou per release prohnat všechny golden reporty lokálním LanguageTool; nálezy jsou podnět k úpravě šablon, nikdy runtime závislost u zákazníka.

---

## 5. Akceptační kritéria

- [ ] Žádné textové pole reportu neobsahuje řetězec extrahovaný ze skenovaného repozitáře (ověřitelné L-07 + code review render vrstvy).
- [ ] Všech 10 insight typů má šablonu, edge-case testy a invarianty; testy zelené vč. povinného `pct = 100`.
- [ ] Linter implementuje L-01–L-10, degradace na fact strip funguje, QA appendix se renderuje.
- [ ] Fixture „1 přispěvatel" produkuje větu „A single contributor … created all N solutions" (nikoli `over 100%`).
- [ ] Fixture „tutorial repo" neprodukuje identická summary u různých řešení (L-08).
- [ ] Dvojí běh = bitově shodný výstup.
- [ ] `--strict` režim vrací nenulový exit code při jakékoli supresi.
