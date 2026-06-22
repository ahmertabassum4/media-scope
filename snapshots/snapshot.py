#!/usr/bin/env python3
"""
snapshot.py — Take a snapshot (screenshot) of a website and save it to disk.

Usage:
    python snapshot.py https://www.bbc.com
    python snapshot.py https://www.reuters.com --output shots --full-page
    python snapshot.py https://example.com --width 1440 --height 900 --format jpeg

Requires:
    pip install playwright
    playwright install chromium
"""

import argparse
import sys
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


def slugify_url(url: str) -> str:
    """Turn a URL into a safe-ish filename fragment."""
    parsed = urlparse(url)
    base = (parsed.netloc + parsed.path).strip("/")
    if not base:
        base = "snapshot"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return slug[:80].strip("_") or "snapshot"


def slugify_name(name: str) -> str:
    """Turn a media name into a safe filename fragment (e.g. 'BBC News' -> 'BBC_News')."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return slug[:80].strip("_") or "snapshot"


def auto_scroll(page, step: int = 400, delay_ms: int = 500, max_scrolls: int = 400):
    """Scroll the page from top to bottom to trigger lazy-loaded images/content,
    then scroll back to the top. Many media sites only fetch images once they're
    about to enter the viewport, so a full-page screenshot taken immediately
    misses them."""
    page.evaluate(
        """
        async ({ step, delay, maxScrolls }) => {
            const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
            let last = -1;
            let count = 0;
            while (count < maxScrolls) {
                window.scrollBy(0, step);
                await sleep(delay);
                count += 1;
                const pos = window.scrollY + window.innerHeight;
                const height = document.body.scrollHeight;
                if (pos >= height) {
                    // Give the bottom-most content a moment, then confirm
                    // the page hasn't grown (infinite-scroll guard).
                    await sleep(delay);
                    if (document.body.scrollHeight === height && height === last) break;
                    last = height;
                }
            }
            window.scrollTo(0, 0);
        }
        """,
        {"step": step, "delay": delay_ms, "maxScrolls": max_scrolls},
    )


def wait_for_images(page, timeout_ms: int = 10000):
    """Best-effort wait until all <img> elements have finished loading."""
    try:
        page.wait_for_function(
            "() => Array.from(document.images).every(img => img.complete && img.naturalHeight > 0)",
            timeout=timeout_ms,
        )
    except PlaywrightTimeoutError:
        # Some images may be broken or never resolve; capture what we have.
        pass


def take_snapshot(
    url: str,
    output_dir: Path,
    label: str = None,
    full_page: bool = False,
    width: int = 1366,
    height: int = 768,
    img_format: str = "png",
    timeout_ms: int = 45000,
    scroll: bool = True,
    settle_ms: int = 2000,
) -> Path:
    """Open the URL in a headless browser and save a screenshot. Returns the file path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify_name(label) if label else slugify_url(url)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{slug}_{timestamp}.{img_format}"
    filepath = output_dir / filename

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                viewport={"width": width, "height": height},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()
            page.set_default_timeout(timeout_ms)

            # Go to the page and wait until the network settles so dynamic
            # content (images, lazy-loaded media) has a chance to appear.
            try:
                page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            except PlaywrightTimeoutError:
                # networkidle can never fire on sites with persistent polling;
                # fall back to "load" so we still capture something useful.
                page.goto(url, wait_until="load", timeout=timeout_ms)

            # Trigger lazy-loaded media by scrolling through the whole page,
            # then wait for images to finish and let layout settle.
            if scroll:
                auto_scroll(page)
            # Wait for network to go idle after scrolling triggers lazy loads,
            # then do a final image-complete check and a settle pause.
            try:
                page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 15000))
            except PlaywrightTimeoutError:
                pass
            wait_for_images(page, timeout_ms=min(timeout_ms, 10000))
            if settle_ms > 0:
                page.wait_for_timeout(settle_ms)

            screenshot_kwargs = {"path": str(filepath), "full_page": full_page}
            if img_format == "jpeg":
                screenshot_kwargs["quality"] = 85
            page.screenshot(**screenshot_kwargs)
        finally:
            context.close()
            browser.close()

    return filepath


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Take a snapshot of a website and save it to an output directory."
    )
    parser.add_argument("url", help="The website URL to capture (e.g. https://www.bbc.com)")
    parser.add_argument(
        "-o", "--output", default="output",
        help="Output directory for the screenshot (default: ./output)",
    )
    parser.add_argument(
        "--full-page", action="store_true",
        help="Capture the entire scrollable page instead of just the viewport.",
    )
    parser.add_argument("--width", type=int, default=1366, help="Viewport width (default: 1366)")
    parser.add_argument("--height", type=int, default=768, help="Viewport height (default: 768)")
    parser.add_argument(
        "--format", dest="img_format", choices=["png", "jpeg"], default="png",
        help="Image format (default: png)",
    )
    parser.add_argument(
        "--no-scroll", dest="scroll", action="store_false",
        help="Disable pre-capture scrolling (faster, but lazy-loaded images may be blank).",
    )
    parser.add_argument(
        "--settle", type=int, default=1000,
        help="Extra wait in ms after scrolling, before capture (default: 1000).",
    )
    parser.add_argument(
        "--timeout", type=int, default=30000,
        help="Page load timeout in milliseconds (default: 30000)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    url = args.url
    if not urlparse(url).scheme:
        url = "https://" + url  # be forgiving if the user omits the scheme

    try:
        path = take_snapshot(
            url=url,
            output_dir=Path(args.output),
            full_page=args.full_page,
            width=args.width,
            height=args.height,
            img_format=args.img_format,
            timeout_ms=args.timeout,
            scroll=args.scroll,
            settle_ms=args.settle,
        )
    except Exception as e:
        print(f"Failed to capture {url!r}: {e}", file=sys.stderr)
        return 1

    print(f"Saved snapshot to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())