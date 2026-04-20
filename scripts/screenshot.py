"""Playwright-based visual inspector for AI Scout web UI.

Used during development to let Claude see the running application,
verify UI changes, and catch visual regressions.

Usage:
    # Landing page (full page)
    .venv/bin/python scripts/screenshot.py

    # Scanner wizard
    .venv/bin/python scripts/screenshot.py http://localhost:8080/app

    # Custom viewport (mobile)
    .venv/bin/python scripts/screenshot.py --width 390 --height 844

    # Click a button, wait, then capture
    .venv/bin/python scripts/screenshot.py --click "text=Start Scanning"

    # Scroll to a section first
    .venv/bin/python scripts/screenshot.py --scroll "#pricing"

    # Open a local HTML report file
    .venv/bin/python scripts/screenshot.py file:///tmp/report.html

    # Viewport-only screenshot (no full page scroll)
    .venv/bin/python scripts/screenshot.py --no-full-page

    # Run a full scan through the wizard and screenshot the report
    .venv/bin/python scripts/screenshot.py --scan https://github.com/org/repo
"""

from __future__ import annotations

import argparse
import sys

from playwright.sync_api import sync_playwright


def screenshot(
    url: str = "http://localhost:8080",
    output: str = "/tmp/scout_screenshot.png",
    width: int = 1440,
    height: int = 900,
    full_page: bool = True,
    wait_ms: int = 1000,
    click: str | None = None,
    scroll_to: str | None = None,
    timeout: int = 15000,
) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(url, wait_until="networkidle", timeout=timeout)

        if scroll_to:
            try:
                page.locator(scroll_to).scroll_into_view_if_needed(timeout=5000)
                page.wait_for_timeout(500)
            except Exception as e:
                print(f"scroll_to '{scroll_to}' failed: {e}", file=sys.stderr)

        if click:
            try:
                page.locator(click).first.click(timeout=5000)
                page.wait_for_timeout(1000)
            except Exception as e:
                print(f"click '{click}' failed: {e}", file=sys.stderr)

        if wait_ms:
            page.wait_for_timeout(wait_ms)

        page.screenshot(path=output, full_page=full_page)
        browser.close()
    return output


def run_scan_and_screenshot(
    repo_url: str,
    base_url: str = "http://localhost:8080",
    output: str = "/tmp/scout_scan_result.png",
    width: int = 1440,
    height: int = 900,
    scan_timeout: int = 300000,
    no_llm: bool = True,
) -> str:
    """Walk through the entire wizard: fill repo URL, configure LLM,
    start scan, wait for completion, click View Report, screenshot.

    This replicates what the user does manually so Claude can see the
    same result page.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height})

        print(f"1/5  Loading wizard...", file=sys.stderr)
        page.goto(f"{base_url}/app", wait_until="networkidle", timeout=15000)

        # Step 1: Fill in repository URL
        print(f"2/5  Filling repo: {repo_url}", file=sys.stderr)
        repo_input = page.locator("input[placeholder*='github.com']").first
        repo_input.fill(repo_url)
        page.wait_for_timeout(300)

        # Click "Next: LLM Config"
        page.locator("text=Next").first.click()
        page.wait_for_timeout(500)

        # Step 2: LLM Configuration — skip LLM if requested
        if no_llm:
            print("3/5  Skipping LLM (no-llm mode)...", file=sys.stderr)
            skip_checkbox = page.locator(
                "input[type='checkbox'], label:has-text('Skip'), "
                "label:has-text('skip'), label:has-text('No LLM'), "
                "text=Skip LLM"
            ).first
            try:
                skip_checkbox.click(timeout=3000)
            except Exception:
                # Maybe it's a different UI — try checking checkbox by id
                try:
                    page.locator("#skip-llm, #noLlm, [name='skip_llm']").first.check(timeout=2000)
                except Exception:
                    print("  Could not find skip-LLM toggle, proceeding anyway", file=sys.stderr)
        page.wait_for_timeout(300)

        # Click "Next: Scan" or "Start Scan"
        try:
            page.locator("text=Start Scan").first.click(timeout=3000)
        except Exception:
            try:
                page.locator("text=Next").first.click(timeout=3000)
                page.wait_for_timeout(500)
                # Now we're on step 3, scan starts automatically
            except Exception:
                print("  Could not advance to scan step", file=sys.stderr)

        # Step 3: Wait for scan to complete
        print(f"4/5  Waiting for scan (timeout {scan_timeout//1000}s)...", file=sys.stderr)
        try:
            page.locator("text=View Report").wait_for(
                state="visible", timeout=scan_timeout
            )
            print("     Scan complete!", file=sys.stderr)
        except Exception as e:
            print(f"     Scan may not have completed: {e}", file=sys.stderr)
            # Screenshot whatever we see
            page.screenshot(path=output, full_page=True)
            browser.close()
            return output

        # Click "View Report"
        page.wait_for_timeout(1000)
        page.locator("text=View Report").first.click()
        page.wait_for_timeout(2000)

        # The report loads in an iframe — wait for it
        print("5/5  Capturing report...", file=sys.stderr)
        iframe = page.frame_locator("#reportFrame")
        try:
            iframe.locator("body").wait_for(state="visible", timeout=10000)
            page.wait_for_timeout(2000)
        except Exception:
            print("  iframe not ready, capturing anyway", file=sys.stderr)

        page.screenshot(path=output, full_page=True)
        browser.close()
    return output


def main():
    parser = argparse.ArgumentParser(description="Screenshot AI Scout web UI")
    parser.add_argument("url", nargs="?", default="http://localhost:8080")
    parser.add_argument("-o", "--output", default="/tmp/scout_screenshot.png")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--no-full-page", action="store_true")
    parser.add_argument("--click", help="Playwright selector to click before capture")
    parser.add_argument("--scroll", dest="scroll_to", help="CSS selector to scroll into view")
    parser.add_argument("--wait", type=int, default=1000, help="Extra wait in ms after load")
    parser.add_argument(
        "--scan",
        metavar="REPO_URL",
        help="Run full wizard: fill repo URL, scan, screenshot report",
    )
    parser.add_argument(
        "--scan-timeout",
        type=int,
        default=300,
        help="Max seconds to wait for scan completion (default 300)",
    )
    parser.add_argument("--with-llm", action="store_true", help="Enable LLM during --scan")
    args = parser.parse_args()

    if args.scan:
        path = run_scan_and_screenshot(
            repo_url=args.scan,
            base_url=args.url,
            output=args.output,
            width=args.width,
            height=args.height,
            scan_timeout=args.scan_timeout * 1000,
            no_llm=not args.with_llm,
        )
    else:
        path = screenshot(
            url=args.url,
            output=args.output,
            width=args.width,
            height=args.height,
            full_page=not args.no_full_page,
            wait_ms=args.wait,
            click=args.click,
            scroll_to=args.scroll_to,
        )
    print(f"Screenshot saved: {path}")


if __name__ == "__main__":
    main()
