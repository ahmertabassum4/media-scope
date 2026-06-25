from urllib.parse import urlparse

from playwright.async_api import async_playwright

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


async def capture(url):
    """Full-page homepage capture. Returns (png_bytes, html, final_url, host).

    Uses Playwright's Async API so it runs natively inside FastAPI's asyncio loop.
    """
    if "://" not in url:
        url = "https://" + url
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1366, "height": 768}, user_agent=UA
        )
        page = await context.new_page()
        page.set_default_timeout(30000)
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
        except Exception:
            await page.goto(url, wait_until="load", timeout=30000)
        final_url = page.url
        html = await page.content()
        png = await page.screenshot(full_page=True)
        await browser.close()
    host = urlparse(final_url).netloc
    return png, html, final_url, host
