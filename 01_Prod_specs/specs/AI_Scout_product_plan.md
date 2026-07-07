# AI Scout — Produktový plán a roadmapa vývoje

> **Účel dokumentu:** Konsolidovaný plán navazující na produktová review a specifikace. Shrnuje strategická rozhodnutí, vyzdvihuje AIBOM jako novou klíčovou schopnost, a definuje fázovaný plán vývoje respektující reálnou kapacitu (jednotky hodin týdně vedle dvou pracovních rolí).
>
> **Navazující specifikace (samostatné dokumenty):**
> 1. `AI_Scout_QA_spec.md` — insight katalog, report linter, degradace, CI pojistky
> 2. `AI_Scout_datamodel_org_cost.md` — org dimenze, cost observables, evoluční pravidla
> 3. `AI_Scout_AIBOM_mapping.md` — mapování G7 minimum elements → CycloneDX → Scout
> 4. `ai_scout_report_design.html` — referenční design reportu (no-LLM režim s QA vrstvou)
> 5. `ai_scout_mode_comparison.html` — srovnání static vs. LLM režimu (web)
> 6. `ai_scout_descent_concept.html` — koncept „Sestup" pro budoucí report (severka pro datový model)

---

## 1. Pozice produktu

**AI Scout je self-hosted nástroj pro evidence-based AI discovery na úrovni kódu.** V trhu „AI discovery" existují tři kategorie: síťové/SaaS discovery (Zscaler, Netskope — vidí provoz), AI-SPM (Wiz, Prisma — vidí cloudovou infrastrukturu) a code-level discovery — kde je prostor prakticky prázdný a kde Scout stojí. Primární segment: **regulované evropské organizace** (banky, pojišťovny, utility). Jednotliví vývojáři a startupy nejsou samostatný segment, ale distribuční kanál (free CLI → adopce zespodu).

**Neporušitelné jádro identity:**
- Self-hosted, nic neopouští perimetr zákazníka — v žádném režimu.
- Evidence, ne verdikt: každé tvrzení má stopu k důkazu (file:line, commit, pravidlo, confidence); nástroj nikdy nevynáší soudy (compliance ✓/✗, risk score, přiměřenost nákladů).
- Deterministické jádro: rule-based analýza, auditovatelné a opakovatelné výsledky; LLM je vrstva porozumění nad identickými fakty, nikdy zdroj faktů.
- Žádný monitoring lidí: vlastnictví řešení ano, per-person aktivita ne.
- Observables, ne judgments: model smí obsahovat jen pozorovatelné skutečnosti (viz datamodel §0).

Tyto principy jsou konkurenční výhoda, ne omezení — jsou to přesně vlastnosti, které cloud-based konkurence nemůže nabídnout a regulovaný segment vyžaduje.

---

## 2. AIBOM: nová klíčová schopnost

### 2.1 Co to je

Export **AI Bill of Materials** podle G7 guidance „Software Bill of Materials for AI – Minimum Elements" (BSI, ACN, ANSSI, CSE, CISA, NCSC, NCO + EU Komise; květen 2026) ve formátu CycloneDX ML-BOM (SPDX 3.0 AI profil ve druhé fázi). Sedm klastrů: Metadata, System Level Properties, Models, Dataset Properties, Infrastructure, Security Properties, KPI. Kompletní mapování elementů: `AI_Scout_AIBOM_mapping.md`.

### 2.2 Proč je to pro Scout strategické

**1. Mimořádný poměr hodnota/úsilí.** Scout už dnes sbírá ~80 % AIBOM dat (data flow, modely, provideři, findings, dependencies). MVP exportu je serializační vrstva nad existujícím datovým modelem + KB — validní CycloneDX ML-BOM bez nové analytické práce.

**2. Řeší verifikační mezeru, kterou vidí celý trh.** Vendor-deklarovaný SBOM je slib; security týmy potřebují ověřit, že odpovídá produkci a zůstává aktuální. Scout generuje AIBOM **z kódu, s evidencí a stabilními ID** — a díky diff režimu umí i „zůstává aktuální" (AIBOM per scan, delta mezi scany). Pozice: *„Nedeklarujeme. Dokazujeme."* Žádný dashboard-konkurent to ze svých dat (síťové logy) nesloží.

**3. Regulatorní vítr do zad.** G7 guidance je dobrovolná, ale stává se referenčním bodem pro AI governance, due diligence, vendor contracting a procurement; pomáhá i s dokumentačními povinnostmi EU AI Act. Regulované organizace dostanou otázku „máte AI SBOM?" od auditorů a procurementu — Scout je odpověď. To vytváří konkrétní, srozumitelný nákupní důvod (mnohem prodejnější než abstraktní „AI discovery").

**4. Posiluje KB jako jádro hodnoty.** Models cluster (producer, licence, architektura, lifecycle, model card, lineage) plní Provider KB — tatáž datová sada slouží typologii modelů, deprecation trackingu, cost tierům i AIBOM. Jedna investice, čtyři features, a nejsilnější argument pro KB subscription.

**5. Uzavírá příběh důvěryhodnosti.** Nástroj, který generuje AI SBOMy, publikuje svůj vlastní SBOM/AIBOM + threat model + podepsané buildy. Podpis exportu (G7 element „SBOM author signature") povyšuje dřívější doporučení „podepsané artefakty" z nice-to-have na funkční požadavek.

### 2.3 K čemu to zákazníkům je (use cases)

- **Governance/compliance:** dokumentační podklad pro AI Act, DORA ICT risk, interní AI inventory povinnosti; příloha k ROPA a vendor assessmentům.
- **Procurement a vendor management:** porovnání vendor-deklarovaného SBOM s realitou v kódu; podklad pro contracting.
- **Security operations:** napojení AI komponent na vulnerability management (dependencies → OSV; modely → vendor advisories); incident response („kde všude běží model X?").
- **Change management:** AIBOM diff mezi scany = auditovatelný záznam změn AI landscape („co přibylo, kdo to schválil").
- **Interní katalog:** strojově čitelná inventura pro navazující systémy (CMDB, GRC tooling).

### 2.4 Další směry (po MVP)

- Podpis exportů (JSF/detached, algoritmy dle NIST/ENISA doporučení).
- PURL identifikátory modelů; detekce pinned vs. nepinned verzí (`gpt-4o` vs. `gpt-4o-2024-11-20`) — nepinned model = plovoucí závislost, samostatně cenný governance signál.
- SPDX 3.0 AI profil; OSV integrace pro dependencies.
- **Klasifikace autonomie agentních systémů** (tool-calling / řízený agent / autonomní smyčka): G7 ji v Discussion sekci zvažovala a zatím nezařadila kvůli vývoji agentic AI — pokud se v revizi guidance objeví, Scout ji díky MCP & Agent scanneru naplní jako první nástroj na trhu; i bez toho je to užitečný atribut inventury.
- Import vrstvy pro KPI/billing data (jasně oddělené od statické evidence).

---

## 3. Roadmapa vývoje

Fáze jsou řazené tak, aby každá stála na předchozí, dala se dokončit v malých krocích a končila použitelným výstupem. U každé: cíl, obsah, kritérium hotovosti. Odhady záměrně nejsou v hodinách — kapacita kolísá; fáze jsou definované výstupem, ne časem.

### Fáze 0 — Základy kvality *(předpoklad všeho ostatního)*
**Cíl:** report, který nikdy neobsahuje rozbitou větu, a datový tvar, který škáluje.
- Implementace QA spec: typované insighty + ICU šablony + testy, report linter L-01–L-10, degradace na fact strip, QA appendix, golden files v CI.
- **Agregační hranice:** řešení = aplikace/služba, ne soubor; tutorial/experiment detektor (`repo.character`).
- Přejmenování „OK" → „No findings"; verze nástroje + KB do hlavičky; Scope & Limitations sekce; stabilní ID nálezů (hash repo+pravidlo+lokace).
**Hotovo, když:** fixture „1 přispěvatel" a „tutorial repo" projdou akceptačními kritérii QA spec; dvojí běh = bitově shodný výstup.

### Fáze 1 — Datový model pro budoucnost *(sbírat teď, zobrazit později)*
**Cíl:** scanner ukládá vše, co budoucí features potřebují, i když to report zatím nezobrazuje (zpětné dosbírání = přescanovat všechno).
- Org dimenze: repos, teams (CODEOWNERS → manual YAML → topic), denormalizované team_id, owner na úrovni řešení.
- Cost observables: model_refs (tier z KB, lifecycle), call_sites (invocation pattern, context profile, retry).
- `schema_version`, enumy s `unknown`, provenance/confidence všude.
**Hotovo, když:** scan reálné Git organizace (viz Validace níže) produkuje validní data dle datamodel spec vč. `null` týmů.

### Fáze 2 — AIBOM MVP *(první velký prodejní artefakt)*
**Cíl:** `scout export --format cyclonedx-aibom` per solution + per org.
- Serializace AUTO polí, existujících KB polí, `aibom.yaml` config (org identita, security.txt, HBOM link), explicitní `unknown` všude jinde.
- Deterministický výstup, `scout:provenance/confidence` properties, dependency graf org → repo → solution → komponenty.
**Hotovo, když:** výstup validuje proti CycloneDX schématu a projde akceptačními kritérii mapping spec §8. 
**Marketing moment:** „první evidence-based AIBOM generátor podle G7 guidance" — publikovatelný milestone.

### Fáze 3 — Diff & continuity *(z one-shot nástroje opakovaně používaný)*
**Cíl:** čas jako dimenze produktu.
- `scout diff`: nová/zmizelá řešení, provideři, klíče, toky; insight I-09; delta box v reportu.
- AIBOM verzování (SBOM version per component-name/version pár dle G7).
- Finding workflow stavy (open / accepted_risk / resolved).
**Hotovo, když:** dva scany téže org produkují korektní delta report a AIBOM verze.

### Fáze 4 — Dosah: SCM abstrakce + SARIF *(distribuce)*
**Cíl:** Scout tam, kde žije enterprise kód, a výstupy tam, kde žijí vývojáři.
- SCM abstrakce (discovery vrstva: projekty, skupiny, auth; scan jádro platform-agnostické) → GitLab scanner vč. `.gitlab-ci.yml` parsingu; Azure DevOps jako druhý za zlomek ceny.
- SARIF export → nálezy v GitHub/GitLab security tabech a CI (nejlevnější distribuce do dev segmentu).
**Hotovo, když:** scan self-managed GitLab org end-to-end; SARIF se zobrazuje v GitLab security tabu.

### Fáze 5 — MCP & Agent scanner *(kategorie-definující feature)*
**Cíl:** první nástroj na trhu, který mapuje MCP/agent landscape.
- MCP server konfigurace, `.claude/` adresáře, tool definitions, agent frameworky; klasifikace autonomie (viz 2.4).
- Plní SLP cluster AIBOM („multi-agent communication protocols" — explicitní G7 element).
**Hotovo, když:** MCP nálezy procházejí celou pipeline (scan → report → AIBOM) s evidencí.
**Marketing moment:** launch feature — „governance pro agentic AI".

### Fáze 6 — LLM enrichment *(porozumění nad fakty)*
**Cíl:** narativní vrstva tam, kde pravidla produkují artefakty.
- Pořadí podle viditelnosti rozdílu: (1) summary/název/účel řešení, (2) element-level klasifikace dat (kategorie + confidence, nikdy hodnoty), (3) identifikace neznámých providerů (→ podněty do KB), (4) sémantické překryvy.
- Režimy: lokální Ollama / privátní endpoint zákazníka (pro banky realističtější). Scout Cloud odděleně, s možností buildu, kde cesta fyzicky neexistuje.
- Provenance LLM/RULE všude; invarianty a linter platí i pro LLM výstupy.
**Hotovo, když:** srovnávací stránka (mode comparison) jde naplnit skutečnými, neupravenými výstupy.

### Fáze 7 — Report „Sestup" + cost mapa *(prezentační vrstva)*
**Cíl:** finální report, až jsou data kompletní (vč. cost) — render, ne přestavba.
- L0 briefing (role lens, territory strip, print/PDF) → L1 přehledy s přepínatelnými osami (tým × provider × kategorie × cost tier) → L1.5 inventura (search, filtry, group-by) → L2 detail → L3 evidence.
- Cost surface: insighty I-11/I-12, exit points s tier dimenzí — vždy fakta o koncentraci, nikdy soudy o přiměřenosti.
**Hotovo, když:** report unese 100+ řešení napříč org bez ztráty čitelnosti (test na reálných datech z Validace).

### Vědomě odloženo / nedělat
- **M365 / Power Platform / Entra ID scanner** — jiná disciplína, Microsoft tlačí vlastní tooling; odloženo na neurčito.
- **Network/DNS a Endpoint scanner** — hřiště Zscaleru; místo toho (později) *import* proxy logů a korelace s code-level nálezy.
- **Remediation Roadmap Generator** — předčasné; generický výstup by podkopal důvěryhodnost.
- **Egress proxy / runtime gateway režim** — Scout je assessment nástroj, ne DLP.
- **Risk score / compliance verdikty / adequacy soudy** — trvale mimo produkt (principy §1).

---

## 4. Průběžné pilíře (nejsou fáze, běží stále)

- **Provider KB jako produkt:** rozšiřování per model (tier, lifecycle, licence, architektura, model card, lineage); podepsaný KB update feed = základ subscription; potvrzené LLM identifikace neznámých providerů jako vstupní pipeline KB.
- **Důkazy důvěryhodnosti:** publikovaný threat model (levné, účinné hned), SBOM+AIBOM Scouta samotného, podepsané buildy, SBOM signing; později nezávislý audit. Vstupenky do regulovaného segmentu.
- **Licencování:** zvážit řez jádro + Git scanner permisivně (adopce) vs. enterprise scannery/reporting/KB feed komerčně — řez vést tam, kde je skutečná hodnota (KB a její údržba, ne kód scanneru).

## 5. Validace (nejlevnější test správnosti směru)

1. **Po Fázi 0–1: scan reálné Git organizace (100+ řešení)** — ověří agregační hranici, tvar org dat, čitelnost inventury. Jediný způsob, jak zjistit skutečný tvar dat před investicí do prezentace.
2. **Po Fázi 2: AIBOM před skutečného security/governance čtenáře** (interní security tým, ideálně design partner v regulované organizaci) — ověří, zda export odpovídá tomu, co auditor/procurement reálně chce vidět.
3. **Po Fázi 5: MCP scan reálného prostředí** — MCP landscape se vyvíjí rychle; scanner kalibrovat na živých konfiguracích.
4. Každou fázi uzavřít golden-file snapshotem — regresní ochrana i dokumentace vývoje výstupů.

## 6. Rizika a protiopatření

| Riziko | Protiopatření |
|---|---|
| Scope creep vs. kapacita jednoho člověka | Fáze definované výstupem; sekce „nedělat" je závazná; každý nový nápad se testuje větou „je to assessment s evidencí, nebo monitoring/verdikt?" |
| Kvalita generovaného textu podkope důvěru | Fáze 0 před vším ostatním; linter blokuje regrese trvale |
| KB údržba neuzvedne tempo trhu | LLM identifikační pipeline plní KB ze scanů; KB feed jako placený produkt financuje údržbu |
| AIBOM standard se vyvine (G7 revize, CycloneDX verze) | Evoluční pravidla schématu; mapping spec verzovat proti verzi guidance; sledovat autonomy element |
| Konkurence vstoupí do code-level segmentu | Rychlost na MCP scanner + AIBOM first-mover; KB a evidence trail jako nejhůř replikovatelné části |

---

*Dokument navazuje na produktová review a specifikace vzniklé v červnu–červenci 2026. Při změně strategických rozhodnutí aktualizovat nejprve tento plán, poté navazující spec dokumenty.*
