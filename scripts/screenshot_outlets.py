import asyncio
import json
import re
from pathlib import Path

from playwright.async_api import async_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRS = {
    "high_factuality": PROJECT_ROOT / "data" / "factuality_splits" / "high_factuality",
    "low_factuality": PROJECT_ROOT / "data" / "factuality_splits" / "low_factuality",
}
OUTPUT_BASE = PROJECT_ROOT / "data" / "screenshots" / "factuality"
VIEWPORT = {"width": 1280, "height": 800}
TIMEOUT_MS = 20_000
CONCURRENCY = 5


def safe_filename(name: str) -> str:
    return re.sub(r'[^\w\-. ]', '_', name).strip()


async def screenshot_one(sem, browser, url, out_path):
    if out_path.exists():
        return "skipped"
    async with sem:
        page = await browser.new_page(viewport=VIEWPORT)
        try:
            await page.goto(url, timeout=TIMEOUT_MS, wait_until="domcontentloaded")
            await page.screenshot(path=str(out_path), full_page=False)
            return "ok"
        except Exception as e:
            return f"error: {e}"
        finally:
            await page.close()


async def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    tasks_meta = []

    for label, src_dir in INPUT_DIRS.items():
        out_dir = OUTPUT_BASE / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(src_dir.glob("*.json")):
            with open(path, encoding="utf-8") as f:
                data = json.loads(f.read())
            url = data.get("media link", "").strip()
            name = data.get("media name", path.stem)
            if not url:
                continue
            out_path = out_dir / f"{safe_filename(name)}.png"
            tasks_meta.append((name, url, out_path))

    total = len(tasks_meta)
    print(f"Outlets to process: {total}")

    sem = asyncio.Semaphore(CONCURRENCY)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        results = await asyncio.gather(*[
            screenshot_one(sem, browser, url, out_path)
            for name, url, out_path in tasks_meta
        ])
        await browser.close()

    ok = results.count("ok")
    skipped = results.count("skipped")
    errors = [(tasks_meta[i][0], r) for i, r in enumerate(results) if r.startswith("error")]

    print(f"\nDone: {ok} captured, {skipped} skipped, {len(errors)} errors")
    if errors:
        print("Errors:")
        for name, err in errors[:20]:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    asyncio.run(main())
