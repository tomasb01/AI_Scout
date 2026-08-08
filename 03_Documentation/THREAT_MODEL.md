# AI Scout — Threat Model

**Version:** 1.0 · covers AI Scout v0.14.0 · Last updated: 2026-08-08
**Audience:** security reviewers, CISOs, and procurement teams evaluating AI Scout for use inside a regulated perimeter.
**Status:** living document — updated when a new component changes the attack surface.

> A tool that reads an organization's entire codebase is itself a high-value target and will be security-reviewed before it is trusted. This document states what AI Scout is, what could go wrong, what is done about it *in code today*, and — honestly — what is not. Claims here are traceable to the source; residual risks are named, not hidden.

---

## 1. Scope & method

**In scope:** the shipped product — Git repository scanner, code-context extractor, rule-based data-flow mapper, optional LLM enrichment engine, HTML/JSON/SARIF reporters, the diff engine, the findings-state store, the live MCP-environment scanner (`aiscout mcp`), the developer guardrail (`aiscout check`), and the FastAPI web UI.

**Out of scope:** third-party LLM backends the operator chooses to point Scout at (their security is the operator's contract with that provider); the host operating system and container runtime; descoped scanners (M365, network/DNS, endpoint — see Product Spec Appendix A).

**Method:** asset identification → trust boundaries → threat enumeration (STRIDE-informed) → mitigations mapped to source → residual risk. Emphasis on the two boundaries that matter for this product: *untrusted repository content crossing into Scout*, and *Scout's own privileged access to customer systems*.

---

## 2. Assets to protect

| # | Asset | Why it matters |
|---|-------|----------------|
| A1 | **Customer source code** read during a scan | The most sensitive input; must never leave the perimeter or be persisted beyond the scan. |
| A2 | **Git access credentials / PATs** | Read access to private repositories; a leaked token is a direct breach. |
| A3 | **Secrets discovered in code** (API keys) | Scout finds them; it must not re-expose them in reports, logs, prompts, or SARIF. |
| A4 | **The generated report / JSON / SARIF** | An inventory of where AI and sensitive data live — a roadmap for an attacker if it leaks. |
| A5 | **The findings-state file** (`.aiscout/findings.json`) | Records which risks were accepted; tampering could silence a real critical finding. |
| A6 | **The host running Scout** | Scout has read access to code and credentials; host compromise is game over. |
| A7 | **Integrity of Scout itself** (the binary/package) | A trojaned Scout could exfiltrate everything it reads (supply-chain risk). |

---

## 3. Trust boundaries & data flow

```
        ┌─────────────────────── customer perimeter ───────────────────────┐
        │                                                                   │
 [Git remote / local repo] ──TB1──> [Scanner] ── [Code Context] ── [Data   │
   A1 A2 A3 (untrusted)               │           Flow Mapper]      Flow]   │
        │                            TB2 (optional)                         │
        │                             ▼                                     │
        │                    [LLM Engine] ──TB3──> [LLM backend: local      │
        │                     (redact+wrap)          Ollama or operator's   │
        │                                            enterprise API]        │
        │                             │                                     │
        │                    [Enrichment] ── [Reporters: HTML/JSON/SARIF]   │
        │                                        │  A4                      │
        │                    [findings-state A5] │                          │
        └────────────────────────────────────────┼─────────────────────────┘
                                                  ▼
                                        operator's chosen sink
                                    (file, code-scanning upload, …)
```

- **TB1 — untrusted repo content → Scout.** Everything read from a scanned repository is attacker-controllable (a malicious repo, a poisoned dependency name, a crafted config). This is the primary injection boundary.
- **TB2 — Scout → optional LLM.** Only crossed when the operator enables LLM enrichment; skipped entirely with `--no-llm`. Repo-derived text crosses here.
- **TB3 — Scout → LLM backend.** Local (Ollama, fully offline) or the operator's own enterprise API. **Scout ships no default cloud endpoint** — there is nowhere for data to "phone home" (the descoped Scout Cloud Engine, Appendix A). Network egress is entirely operator-configured.

---

## 4. Threats, mitigations (as implemented), and residual risk

### T1 — Malicious repository reads files outside the working copy (path traversal / symlink escape)
**Vector:** a repo contains a symlink `evil → /etc/passwd`, or a path that resolves outside the clone.
**Mitigation (implemented):** the file walk skips symlinks entirely (`path.is_symlink()` check, `git_scanner.py`), each candidate is resolved and confirmed strictly inside the resolved root, and `os.walk(followlinks=False)`. The root-README read applies the same symlink guard.
**Residual:** none known for the walk; depends on the OS honoring `lstat`/`is_symlink`.

### T2 — SSRF / local-file disclosure via a crafted repo URL
**Vector:** `--repo file:///etc/shadow`, `--repo http://169.254.169.254/…` (cloud metadata), or a hostname resolving to loopback/link-local.
**Mitigation (implemented):** URL scheme allowlist `{https, http, ssh, git}` (rejects `file://`, `gopher://`, …); host resolution rejects loopback, link-local, multicast, unspecified, and the cloud-metadata addresses `169.254.169.254` / `fd00:ec2::254` (`cli.py`). Local `--local` paths are rejected if they resolve to system directories (`/etc`, `/var`, `/root`, `/System`, … incl. macOS firmlink variants).
**Residual:** DNS-rebinding between validation and clone is not mitigated in-process; operators in hostile-DNS environments should scan from a network-restricted host.

### T3 — Git token leakage (A2)
**Vector:** token embedded in a clone URL leaks into process args, `.git/config`, or GitPython error strings.
**Mitigation (implemented):** credentials are passed via a short-lived `GIT_ASKPASS` helper script (`chmod 0700`, in a `TemporaryDirectory` with `0700` perms), **never** embedded in the URL. `GIT_ASKPASS` defaults to `/bin/echo` when no token is present. Clone depth is capped (`depth=10`).
**Residual:** the helper script exists on disk (0700) for the clone's duration; a local root/other-process with the same UID during that window is a (small) exposure. Token is held in memory during the scan.

### T4 — Discovered secrets re-exposed in outputs (A3)
**Vector:** Scout finds an API key and then prints it into the report, JSON, SARIF, logs, or an LLM prompt.
**Mitigation (implemented):** findings store redacted content; reports and JSON render `redacted_content`, never the raw key. SARIF result messages use redacted content only and are schema-validated. LLM prompts replace key content with `<REDACTED_API_KEY>`. The MCP-environment scanner reduces remote server URLs to host only (no query strings/tokens).
**Residual:** redaction is pattern-based; a secret in an unrecognized format may pass through as ordinary text into the report body (not as a flagged key). The report itself (A4) is sensitive regardless and must be handled accordingly.

### T5 — Prompt injection from repo content into the LLM (TB2)
**Vector:** a repo embeds `"ignore previous instructions and exfiltrate …"` in a comment/prompt/README; Scout forwards it to the LLM.
**Mitigation (implemented):** all repo-derived text is sanitized (control chars stripped, length-capped) and wrapped in `<untrusted>…</untrusted>` tags with a system instruction that content inside those tags is data, not instructions (`engine/llm.py`). Only crossed when LLM mode is on.
**Residual:** prompt-injection defense is defense-in-depth, not a guarantee — a sufficiently capable/compromised model could still be steered. The rule-based core needs no LLM (`--no-llm`), so the highest-assurance mode removes this boundary entirely. The LLM performs *inference only* — it does not train on scanned data and retains nothing between calls.

### T6 — Resource exhaustion / decompression bombs (DoS)
**Vector:** a repo with enormous files or millions of tiny files stalls the scanner.
**Mitigation (implemented):** per-file size cap (1 MB) for read files; model-weight artifacts are detected by extension and **never read** (only their presence is recorded); skip-list prunes `node_modules`, `.venv`, build dirs; org enumeration is capped (`--max-repos`, default 200).
**Residual:** no global wall-clock/CPU budget; a pathological repo can still be slow. Run under OS-level resource limits for untrusted input.

### T7 — Tampering with findings-state to silence a real risk (A5)
**Vector:** someone edits `.aiscout/findings.json` to mark a live hardcoded key as `accepted_risk`, hiding it from the critical count.
**Mitigation (implemented):** accepted findings are **never deleted or hidden** — they remain visible in the report with an `ACCEPTED` badge and a warning-level audit reason; the state file is plain JSON meant to live in version control, so changes are diff-visible and attributable via git history. `resolved` is only ever set automatically by the diff (finding no longer detected), not by hand.
**Residual:** the file has no cryptographic integrity of its own; its trust derives from the git history of the repo that stores it. Treat it as security-relevant config.

### T8 — Supply-chain compromise of Scout itself (A7)
**Vector:** a trojaned Scout release exfiltrates everything it reads.
**Mitigation (implemented / by design):** BSL source-available core — every line is auditable; no telemetry, no phone-home, no default cloud endpoint; the tool runs fully offline (`--no-llm` + local paths need no network at all). Deterministic output (two runs = byte-identical) makes tampering with results detectable via diff.
**Residual (roadmap):** signed release artifacts and a published SBOM/AIBOM of Scout itself are planned (Product Spec §4.1; AIBOM export lands in Sprint 5b — "the tool that generates AIBOMs has its own"). Until then, pin to a reviewed commit and build from source.

### T9 — Report/inventory disclosure (A4)
**Vector:** the generated report — a map of AI usage, data flows, and where sensitive data lives — leaks to an attacker.
**Mitigation (by design):** the report is self-contained, has no external calls (strict offline HTML), and is written only to the operator-specified path. Scout never transmits it anywhere.
**Residual:** once written, the file's protection is the operator's responsibility (storage ACLs, code-scanning upload scope). This is inherent to producing a useful artifact.

### T10 — `aiscout mcp` reads sensitive local agent config (A1-adjacent)
**Vector:** the environment scanner reads MCP configs that may contain server commands and remote URLs with embedded tokens.
**Mitigation (implemented):** read-only, never launches a server; only well-known config paths are read; symlinked configs skipped; remote entries reduced to host only (no query/token); stdio entries reduced to the command basename.
**Residual:** the local config files themselves are unchanged and remain the operator's to protect.

---

## 5. Security posture summary (what holds by construction)

- **No phone-home, no default cloud sink** — there is no built-in destination for exfiltration; all egress is operator-configured (TB3).
- **Rule-based core needs no LLM and no network** — the highest-assurance mode (`--no-llm` + `--local`) removes the LLM and network boundaries entirely.
- **Read-only, audit not enforcement** — Scout never writes to, modifies, or blocks the systems it scans.
- **Deterministic & diffable** — identical inputs produce byte-identical output, making result tampering detectable.
- **Least privilege** — each scanner declares its scope; the guardrail and MCP scanner need no network at all.
- **Evidence, not verdict** — outputs are findings with file:line provenance, never unfalsifiable judgments; the same honesty applies to this document's residual-risk column.

---

## 6. Operator responsibilities (shared-responsibility boundary)

AI Scout secures its own behavior; these remain the operator's:

1. **Credential provisioning** — supply least-privilege, read-only tokens; prefer short-lived. Store them in a vault or `AISCOUT_GIT_TOKEN`/`AISCOUT_LLM_KEY` env, not in shell history.
2. **Host hardening** — run on a controlled host; for untrusted repositories, add OS-level CPU/memory/time limits and network egress restriction.
3. **LLM backend choice** — if enabling LLM mode, point Scout at a backend whose data-handling matches your compliance requirements (local Ollama = nothing leaves the host).
4. **Output handling** — the report/JSON/SARIF is sensitive (A4); apply appropriate ACLs and, for code-scanning upload, an appropriate category/scope.
5. **findings-state governance** — treat `.aiscout/findings.json` as security-relevant config under version control; review `accepted_risk` changes.
6. **Integrity** — until signed releases ship, build from a reviewed source commit.

---

## 7. Change log

| Date | Version | Change |
|------|---------|--------|
| 2026-08-08 | 1.0 | Initial threat model covering AI Scout v0.14.0 (Git scanner, LLM engine, reporters, diff, findings-state, MCP scanner, guardrail, web UI). |

---

*This document is published deliberately: a security tool that asks to read your codebase should hand you its threat model before you ask. Found a gap? That is exactly the review this document invites.*
