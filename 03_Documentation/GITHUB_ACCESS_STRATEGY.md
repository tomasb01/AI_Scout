# GitHub Access Strategy — od jednotlivce po enterprise

> Jak dostat AI Scout k repozitářům zákazníka tak, aby to prošlo přes jeho security.
> Pro každý typ organizace jedna doporučená cesta + konkrétní postup.

**Stav implementace značíme:** ✅ dnes funguje · 🟡 částečně / nutné dodělat · ⏳ navrhované rozšíření (roadmapa)

---

## Proč to řešíme

AI Scout potřebuje číst kód, aby v něm našel AI řešení. Jenže přístup ke zdrojovému kódu celé firmy je citlivá věc — a čím větší organizace, tím méně ochotně dá někdo „token na všechno". Org-wide Personal Access Token (PAT) s read právy ke všem repozitářům je pro CISO těžko stravitelný: dělá ze samotného Scoutu high-value target.

**Princip:** vchod nesmí být jen „dej nám klíče od všeho". Pro každou velikost a vyspělost organizace má existovat cesta s odpovídající mírou důvěry a citlivosti dat. Níže je jich pět, seřazených od nejnižší po nejvyšší laťku.

Klíčová výhoda Scoutu napříč všemi cestami: **běží u zákazníka (self-hosted), token a data nikdy neopouštějí perimetr.** Tohle je třeba vždy explicitně říct — odlišuje nás od SaaS konkurence.

---

## Přehled přístupových metod

| Metoda | Co odemkne | Citlivost / footprint | Stav |
|--------|-----------|----------------------|------|
| **A. Veřejná URL bez auth** | Jen public repa | Nulová — veřejná data | ✅ |
| **B. Personal Access Token (PAT)** | Repa, na která má daný účet práva | Vysoká, váže se na osobu, klasický PAT vidí moc | ✅ |
| **C. Fine-grained PAT** | Vybraná repa, granulární práva | Střední — scopovatelné na repo + read-only | 🟡 |
| **D. GitHub App (org-installed)** | Repa vybraná adminem org, krátkodobé tokeny | Nízká — least privilege, revokovatelné, auditovatelné | ⏳ |
| **E. GitHub Action (ephemerální token)** | Per-repo/per-run, auto-expiruje | Nejnižší — žádný dlouhodobý token | ⏳ |

A dvě ortogonální „úsporné" varianty, které se kombinují s kteroukoli metodou výše:

| Varianta | Princip | Stav |
|----------|---------|------|
| **Manifest-only sken** | Čte jen `requirements.txt`/`package.json`/`pyproject.toml` — ne zdroják | ⏳ |
| **GitHub API scanner** | Místo `git clone` čte přes REST API jen AI-relevantní soubory | ⏳ |
| **Customer-executed enumerace** | Scout vygeneruje skript, zákazník ho spustí sám, vrátí jen manifest repo | ⏳ |

---

## Cesta podle typu organizace

### 1. Jednotlivec / Solo developer

**Situace:** Vlastní repa pod osobním účtem, žádná security funkce, chce rychlý výsledek.

**Doporučená cesta:** A (public) nebo B (PAT) — co nejjednodušší.

**Postup (dnes funguje):**
```bash
# Veřejné repo
uv run aiscout scan --repo https://github.com/me/my-repo --no-llm --output report.html

# Privátní repo — fine-grained PAT s read-only na Contents
uv run aiscout scan --repo https://github.com/me/my-repo \
  --token github_pat_xxx --no-llm --output report.html
```
- PAT vytvoř na GitHubu: *Settings → Developer settings → Personal access tokens → Fine-grained → Repository access: jen daná repa, Permissions: Contents = Read-only.*
- Token jde do Scoutu přes `GIT_ASKPASS`, nikdy se neobjeví v URL ani v logu.

**Co Scout uvidí:** repa, která tokenu povolíš. **Footprint:** minimální, sám sobě admin.

---

### 2. Malý tým / Startup (1 organizace, pár lidí)

**Situace:** Jedna GitHub org, pár desítek repo, důvěra je vysoká, security ještě neformalizovaná. Chtějí přehled „co všechno tu máme".

**Doporučená cesta:** B/C s org-level enumerací (⏳ navrhované) — zadat URL organizace + token člena s org read.

**Postup (cílový stav, ⏳):**
```bash
# Projde všechna repa organizace, na která token vidí
uv run aiscout scan --org https://github.com/acme-startup \
  --token github_pat_xxx --output report.html
```
- Scout zavolá `GET /orgs/{org}/repos`, vyjmenuje repa a každé prožene existující pipeline.
- Token = fine-grained PAT člena org s read na vybraná (nebo všechna) repa.

**Postup (dnes, jako náhrada):** vyjmenovat repa ručně do YAML configu a spustit multi-repo sken — to ✅ funguje už teď:
```yaml
# repos.yaml
repositories:
  - url: https://github.com/acme-startup/api
  - url: https://github.com/acme-startup/web
  - url: https://github.com/acme-startup/ml-pipeline
token: github_pat_xxx
```
```bash
uv run aiscout scan --config repos.yaml --output report.html
```

**Co Scout uvidí:** repa viditelná tokenu. Privátní mimo scope tokenu zůstanou neviditelná. **Footprint:** přijatelný — malá firma, vysoká důvěra.

---

### 3. Střední firma / Scale-up (víc týmů, formující se security)

**Situace:** Několik týmů, stovky repo, někdo už klade otázky „kdo a proč k tomu má přístup". Org-wide PAT je problém, ale jsou flexibilní.

**Doporučená cesta:** **fázovaný sken** — nejdřív manifest-only, pak opt-in kód; ideálně přes fine-grained PAT scopovaný na vybrané týmy/repa.

**Postup:**
1. **Fáze 1 — manifest-only (⏳):** první sken čte jen dependency manifesty (`requirements.txt`, `package.json`, `pyproject.toml`). Tyto soubory **nejsou citlivé jako zdroják** → nízká laťka pro schválení. Prokáže hodnotu (seznam AI závislostí napříč firmou).
2. **Fáze 2 — analýza kódu na opt-in repech:** týmy, které vidí hodnotu, povolí hlubší sken svých repo (Code Context, Data Flow). Rozšiřuje se po krocích, ne plošně.

**Token:** fine-grained PAT s read-only na Contents, scopovaný na konkrétní repa/týmy. Žádný klasický PAT s plným `repo` scope.

**Co Scout uvidí:** přesně to, co každá fáze povolí. **Footprint:** roste řízeně, security má kontrolu nad rozsahem.

---

### 4. Enterprise / Regulovaný sektor (CISO, vault, audit)

**Situace:** Banka, energetika, zdravotnictví, státní správa. Žádný org-wide PAT neprojde. Vyžadují least privilege, auditovatelnost, revokovatelnost, žádný dlouhodobý osobní token.

**Doporučená cesta:** **D — GitHub App nainstalovaná adminem organizace** (⏳, hlavní enterprise cesta).

**Proč to security schválí:**
- Instaluje ji **admin org**, ne jednotlivec — nevisí na osobním účtu.
- Granulární práva: typicky jen `Contents: Read-only` + `Metadata: Read`.
- Admin vybere, **která repa** App vidí (klidně podmnožinu).
- Tokeny jsou **krátkodobé** (installation token ~1 h), ne dlouhodobý PAT.
- Revokovatelné jedním klikem, plně auditovatelné.
- Přesně takhle fungují Snyk, Dependabot, Mend — security tento vzor **už zná a důvěřuje mu**.

**Postup (cílový stav, ⏳):**
1. Admin organizace nainstaluje „AI Scout" GitHub App, vybere repa, schválí read-only práva.
2. Scout (self-hosted u zákazníka) si vymění installation token a projde povolená repa.
3. Token i data zůstávají v perimetru zákazníka.

**Alternativy pro nejpřísnější prostředí:**
- **E — GitHub Action v jejich CI (⏳):** Scout běží jako Action uvnitř firemního CI s ephemerálním `GITHUB_TOKEN` scopovaným per-run. Žádný token k uložení vůbec.
- **Customer-executed enumerace (⏳):** Scout vygeneruje skript, zákazník ho spustí sám se svými credentials, vrátí jen manifest repozitářů (jména, metadata, seznam AI-relevantních souborů). Scout credentials nikdy nevidí — analogie „Vrstvy 3" z Data Classification.
- **Credential storage přes vault:** integrace s HashiCorp Vault / Azure Key Vault (viz prod spec 3.2).

**Co Scout uvidí:** repa schválená adminem/Appkou. **Footprint:** nejnižší možný, plně auditovatelný.

---

## Rozhodovací shrnutí

| Typ organizace | Doporučená cesta | Klíčový argument pro security |
|----------------|------------------|-------------------------------|
| Jednotlivec | A / B — public nebo fine-grained PAT | Sám sobě admin |
| Malý tým / startup | Org URL + member PAT (⏳) / dnes YAML multi-repo | Self-hosted, token neopouští perimetr |
| Střední firma | Manifest-first → opt-in kód, fine-grained PAT | Fázovaný rozsah, security drží kontrolu |
| Enterprise / regulace | GitHub App (org-installed) ± GitHub Action | Least privilege, krátkodobé tokeny, audit, revokace |

**Univerzální pravidla:**
- Nikdy nepožadovat víc práv, než je na daný sken potřeba (least privilege).
- Vždy nabídnout manifest-only jako „nízkorizikový první krok".
- Vždy zdůraznit: **Scout běží u zákazníka, token ani kód neodcházejí ven.**
- Org-wide PAT nabízet jen tam, kde si o něj zákazník sám řekne — nikdy jako jedinou cestu.

---

## Co je potřeba doimplementovat

Pořadí podle poměru hodnota/úsilí pro odemčení těchto cest:

1. **`--org` enumerace** (⏳) — `GET /orgs|users/{name}/repos`, paginace, detekce org vs. user, filtry (skip archived/forks, limit souběhu, per-repo timeout). Odemyká cesty 2 a 3.
2. **Manifest-only mód** (⏳) — `--manifests-only`, čte jen dependency soubory. Nízká laťka pro střední firmy.
3. **GitHub API scanner** (⏳, roadmapa P3) — číst strom + jen AI-relevantní soubory přes REST místo `git clone`. Menší footprint, běží serverless.
4. **GitHub App integrace** (⏳) — installation flow, výměna krátkodobých tokenů. Hlavní enterprise odemčení.
5. **GitHub Action balíček** (⏳) — reusable workflow s `GITHUB_TOKEN`. Pro nejpřísnější prostředí.
