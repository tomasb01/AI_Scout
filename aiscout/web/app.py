"""AI Scout Web UI — FastAPI server with scan API and wizard interface."""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from aiscout import __version__

app = FastAPI(title="AI Scout", version=__version__)

# In-memory scan state
_scans: dict[str, dict] = {}


# Mount static files for landing page screenshots
_landing_dir = Path(__file__).parent.parent.parent / "landing"
if _landing_dir.exists():
    _ss_dir = _landing_dir / "screenshots"
    if _ss_dir.exists():
        app.mount("/screenshots", StaticFiles(directory=str(_ss_dir)), name="screenshots")

# ── Static UI ──────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def landing():
    """Serve the landing page."""
    landing_path = Path(__file__).parent.parent.parent / "landing" / "index.html"
    if landing_path.exists():
        return HTMLResponse(landing_path.read_text(encoding="utf-8"))
    # Fallback to app if no landing page
    return await app_ui()


@app.get("/app", response_class=HTMLResponse)
async def app_ui():
    """Serve the wizard UI."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(template_path.read_text(encoding="utf-8"))


# ── API: Ollama models ────────────────────────────────────────────────────


@app.get("/api/ollama/models")
async def list_ollama_models(url: str = "http://localhost:11434"):
    """List available Ollama models."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                return {"status": "ok", "models": models}
            return {"status": "error", "models": [], "detail": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "models": [], "detail": str(e)}


# ── API: Scan ──────────────────────────────────────────────────────────────


@app.post("/api/scan")
async def start_scan(request: Request):
    """Start a new scan. Returns scan_id for progress tracking."""
    body = await request.json()

    scan_id = str(uuid.uuid4())[:8]
    _scans[scan_id] = {
        "status": "running",
        "progress": [],
        "report_html": None,
        "error": None,
        "config": body,
    }

    # Run scan in background
    asyncio.create_task(_run_scan(scan_id, body))

    return {"scan_id": scan_id}


@app.get("/api/scan/{scan_id}/progress")
async def scan_progress(scan_id: str):
    """SSE stream of scan progress."""
    async def event_generator():
        seen = 0
        while True:
            scan = _scans.get(scan_id)
            if not scan:
                yield {"event": "error", "data": json.dumps({"message": "Scan not found"})}
                break

            # Send new progress messages
            messages = scan["progress"]
            while seen < len(messages):
                yield {"event": "progress", "data": json.dumps(messages[seen])}
                seen += 1

            if scan["status"] == "done":
                yield {"event": "done", "data": json.dumps({"report_available": True})}
                break
            elif scan["status"] == "error":
                yield {"event": "error", "data": json.dumps({"message": scan["error"]})}
                break

            await asyncio.sleep(0.3)

    return EventSourceResponse(event_generator())


@app.get("/api/scan/{scan_id}/report", response_class=HTMLResponse)
async def get_report(scan_id: str):
    """Get the generated HTML report."""
    scan = _scans.get(scan_id)
    if not scan or not scan["report_html"]:
        return HTMLResponse("<p>Report not available</p>", status_code=404)
    return HTMLResponse(scan["report_html"])


# ── Scan runner ────────────────────────────────────────────────────────────


async def _run_scan(scan_id: str, config: dict):
    """Run the full scan pipeline in background."""
    scan = _scans[scan_id]

    def log(msg: str, level: str = "info"):
        scan["progress"].append({"message": msg, "level": level})

    try:
        # Lazy imports — GitPython needs git executable, only import when scanning
        from aiscout.engine.code_analyzer import analyze_assets
        from aiscout.engine.enrichment import enrich_assets
        from aiscout.engine.llm import LLMEngine
        from aiscout.report.html import ReportGenerator
        from aiscout.scanners.git_scanner import GitScanner

        repos = config.get("repositories", [])
        llm_config = config.get("llm", {"mode": "none"})

        if not repos:
            raise ValueError("No repositories specified")

        log(f"Starting scan of {len(repos)} repository(ies)...")

        # ── Phase 1: Scan ──
        scan_results = []
        scanners = []

        log(f"Repositories to scan: {len(repos)}")
        for idx, repo_entry in enumerate(repos, 1):
            name = repo_entry.get("url", repo_entry.get("path", "unknown"))
            log(f"[{idx}/{len(repos)}] Scanning {name}...")

            try:
                scanner = GitScanner(
                    repo_path=repo_entry.get("path"),
                    repo_url=repo_entry.get("url"),
                    branch=repo_entry.get("branch", "main"),
                    token=repo_entry.get("token"),
                )

                result = await asyncio.to_thread(scanner.scan)
                scan_results.append(result)
                scanners.append(scanner)

                if result.errors:
                    for err in result.errors:
                        log(f"Error in {name}: {err}", "error")
                else:
                    log(f"Found {len(result.assets)} AI solution(s) in {result.metadata.get('files_scanned', 0)} files")
            except Exception as e:
                log(f"Failed to scan {name}: {e}", "error")

        # ── Phase 2: Code Analysis ──
        log("Analyzing code context...")
        for result in scan_results:
            repo_root = result.metadata.get("repo_root")
            if repo_root and result.assets:
                await asyncio.to_thread(analyze_assets, result.assets, repo_root)

        # Cleanup cloned repos
        for scanner in scanners:
            scanner.cleanup()

        all_assets = [a for r in scan_results for a in r.assets]

        # ── Phase 3: LLM Classification ──
        if llm_config.get("mode") != "none" and all_assets:
            mode = llm_config.get("mode", "ollama")
            url = llm_config.get("url", "http://localhost:11434")
            model = llm_config.get("model", "qwen2.5-coder:7b")
            api_key = llm_config.get("api_key")

            engine = LLMEngine(mode=mode, url=url, model=model, api_key=api_key)

            if engine.check_health():
                log(f"Classifying {len(all_assets)} solution(s) via {mode} ({model})...")
                for i, asset in enumerate(all_assets):
                    try:
                        result = await asyncio.to_thread(engine.classify, asset)
                        asset.data_classification = result
                        if result.risk_score > 0:
                            asset.risk_score = max(asset.risk_score, result.risk_score)
                    except Exception as e:
                        log(f"LLM failed for '{asset.name}': {e}", "warning")

                    if (i + 1) % 5 == 0:
                        log(f"Classified {i + 1}/{len(all_assets)}...")
            else:
                log(f"LLM not available at {url}, skipping classification", "warning")

        # ── Phase 4: Enrichment + Report ──
        log("Generating report...")
        insights = enrich_assets(all_assets)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        gen = ReportGenerator(scan_results, output_path=output_path, insights=insights)
        gen.generate()

        scan["report_html"] = Path(output_path).read_text(encoding="utf-8")
        Path(output_path).unlink(missing_ok=True)

        log(f"Done! Found {len(all_assets)} AI solutions.", "success")
        scan["status"] = "done"

    except Exception as e:
        log(f"Scan failed: {e}", "error")
        scan["status"] = "error"
        scan["error"] = str(e)


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Start the web server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
