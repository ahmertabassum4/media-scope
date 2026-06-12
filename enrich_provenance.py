#!/usr/bin/env python3
"""
enrich_provenance.py — Extend each outlet JSON in json_data/ with provenance,
transparency, page-composition and content/structure features pulled live from
the homepage (media link).

HTML is fetched with a headless Chromium browser via Playwright's ASYNC API,
driven by an asyncio event loop with a bounded semaphore for concurrency. This
avoids the "Sync API inside the asyncio loop / socket connection refused" error
that occurs when the sync Playwright API is driven from worker threads under a
conda/Jupyter-style runtime that already owns an event loop. A site that the
browser cannot render falls back to a plain requests.get. WHOIS and TLS are
blocking lookups, so they run in a thread executor off the event loop.

Fetch reliability (for flaky networks / transient Playwright driver-socket
failures) is handled by a BrowserSupervisor that owns the Chromium instance,
performs per-page retries with exponential backoff + jitter, and recycles the
browser both periodically and on detection of a driver-socket disruption. A
driver-socket error (e.g. "[Errno 61] Connection refused", "closing socket -
timed out") indicates the local Python<->Node pipe is wounded and the shared
browser object may be poisoned for all subsequent contexts; the supervisor
drains in-flight pages (bounded) then relaunches. Per-site network errors are
distinguished from driver errors and from deterministic errors (NXDOMAIN, 404,
cert failure) so that only transient failures consume retries.

Each source file gains a single new "provenance" object; nothing existing is
touched. Re-runs skip files that already carry the block unless --force is set.

Setup (once):
    pip install playwright python-whois tldextract beautifulsoup4 requests
    playwright install chromium

Usage:
    python enrich_provenance.py                       # enrich every *.json in json_data/
    python enrich_provenance.py --concurrency 3       # simultaneous pages (default 3)
    python enrich_provenance.py --timeout 25          # per-page timeout in seconds
    python enrich_provenance.py --retries 3           # browser attempts per page (default 3)
    python enrich_provenance.py --recycle-every 40    # relaunch browser every N pages
    python enrich_provenance.py --force               # re-fetch even if already enriched
    python enrich_provenance.py --retry-errors        # only re-process files whose last fetch failed
    python enrich_provenance.py --no-archive          # skip storing raw html/css
    python enrich_provenance.py --no-browser          # requests-only (no Chromium)
    python enrich_provenance.py --limit 5             # process only the first N (smoke test)
    python enrich_provenance.py --preflight           # launch browser, fetch one URL, report, exit
"""

import argparse
import asyncio
import json
import random
import re
import socket
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import tldextract
import whois
from bs4 import BeautifulSoup

DATA_DIR = Path("json_data")
ARCHIVE_DIR = Path("html_archive")

# Use the snapshot of the public suffix list bundled with tldextract rather
# than fetching it live (the live fetch can 403 or be blocked on some networks).
_tld = tldextract.TLDExtract(suffix_list_urls=())

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# TLDs commonly associated with low-cost / disposable registration. Flagged,
# not judged — the model decides what to do with the signal.
SUSPECT_TLDS = {
    "news", "info", "biz", "click", "online", "site", "website", "live",
    "xyz", "top", "club", "buzz", "today", "world", "press", "report", "wtf",
}

# Trackers / ad networks we can spot by substring in script src or inline code.
TRACKER_PATTERNS = [
    "google-analytics", "googletagmanager", "googlesyndication", "doubleclick",
    "adservice", "/gtag/", "gtag(", "ga(", "fbevents", "connect.facebook",
    "facebook.net", "amazon-adsystem", "adsbygoogle", "taboola", "outbrain",
    "criteo", "scorecardresearch", "quantserve", "hotjar", "mixpanel",
    "segment.com", "cdn.segment", "matomo", "piwik", "chartbeat", "parsely",
    "moatads", "adnxs", "pubmatic", "rubiconproject", "openx", "media.net",
    "newrelic", "bugsnag", "sentry", "cloudflareinsights", "bing.com/bat",
]

# Page-type detection: regex over href + anchor text, scored per category.
PAGE_SIGNALS = {
    "about": [r"\babout\b", r"who[-\s]?we[-\s]?are", r"\bmasthead\b", r"our[-\s]?story",
              r"our[-\s]?mission", r"about[-\s]?us"],
    "ownership": [r"\bownership\b", r"\bimprint\b", r"\bimpressum\b", r"who[-\s]?owns",
                  r"\bfunding\b", r"\bownership[-\s]?and[-\s]?funding\b"],
    "corrections": [r"\bcorrections?\b", r"editorial[-\s]?standards", r"editorial[-\s]?policy",
                    r"editorial[-\s]?guidelines", r"\bethics\b", r"ethics[-\s]?policy",
                    r"fact[-\s]?check[-\s]?policy", r"standards[-\s]?and[-\s]?practices"],
    "authors": [r"\bauthor[s]?\b", r"\bcontributor[s]?\b", r"/staff\b", r"our[-\s]?team",
                r"/writers?\b", r"\bcolumnists?\b", r"/people\b", r"/byline"],
    "contact": [r"\bcontact\b", r"contact[-\s]?us", r"get[-\s]?in[-\s]?touch"],
}

CRED_TLDS = {"gov", "edu", "mil", "int"}
CRED_DOMAINS = {
    "reuters.com", "apnews.com", "ap.org", "bbc.com", "bbc.co.uk", "npr.org",
    "pbs.org", "nytimes.com", "washingtonpost.com", "theguardian.com",
    "wsj.com", "economist.com", "nature.com", "science.org", "nih.gov",
    "cdc.gov", "who.int", "nasa.gov", "noaa.gov", "factcheck.org",
    "politifact.com", "snopes.com", "pubmed.ncbi.nlm.nih.gov",
    "scholar.google.com", "jstor.org", "doi.org", "afp.com", "bloomberg.com",
}


def reg_domain(url_or_host: str) -> str:
    ext = _tld(url_or_host)
    if not ext.domain:
        return ""
    return ".".join(p for p in (ext.domain, ext.suffix) if p)


# ---- error classification ---------------------------------------------------

# Substrings that mark a wounded local Python<->Node driver pipe. These poison
# the shared browser object: the cure is to recycle the browser, not to retry
# the page against the same dead instance.
DRIVER_SOCKET_MARKERS = (
    "connect to socket",
    "closing socket",
    "errno 61",
    "connection refused",
    "target closed",
    "browser has been closed",
    "browser closed",
    "connection closed",
    "pipe closed",
    "websocket",
    "transport closed",
    "page crashed",
    "session closed",
)

# Substrings that mark a deterministic, won't-change-on-retry failure for a
# given site. Retrying these just burns the network on a flaky link.
PERMANENT_MARKERS = (
    "err_name_not_resolved",
    "err_name_resolution_failed",
    "name or service not known",
    "nodename nor servname",
    "no address associated",
    "err_cert_",            # cert errors: authority invalid, date invalid, etc.
    "certificate verify failed",
    "ssl:",
    "err_bad_ssl",
    "err_invalid_url",
    "err_unknown_url_scheme",
    "err_aborted",
    "404 client error",
    "410 client error",
    "err_blocked_by",
)


def is_driver_socket_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in DRIVER_SOCKET_MARKERS)


def is_permanent_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in PERMANENT_MARKERS)


# ---- provenance / infrastructure (blocking; run in executor) ----------------

def whois_age(host: str) -> dict:
    out = {"registration_date": None, "domain_age_days": None, "registrar": None}
    try:
        w = whois.whois(host)
    except Exception:
        return out
    created = w.creation_date
    if isinstance(created, list):
        created = next((c for c in created if c), None)
    if isinstance(created, datetime):
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        out["registration_date"] = created.date().isoformat()
        out["domain_age_days"] = (datetime.now(timezone.utc) - created).days
    reg = w.registrar
    if isinstance(reg, list):
        reg = reg[0] if reg else None
    out["registrar"] = reg
    return out


def tls_info(host: str, timeout: int) -> dict:
    out = {"https": False, "cert_issuer": None, "cert_subject_cn": None,
           "cert_is_dv": None, "cert_not_after": None}
    ctx = ssl.create_default_context()
    try:
        with socket.create_connection((host, 443), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
        out["https"] = True
    except Exception:
        return out

    def flatten(field):
        d = {}
        for tup in cert.get(field, ()):
            for k, v in tup:
                d[k] = v
        return d

    issuer = flatten("issuer")
    subject = flatten("subject")
    out["cert_issuer"] = issuer.get("organizationName") or issuer.get("commonName")
    out["cert_subject_cn"] = subject.get("commonName")
    # DV certs carry no organizationName in the subject; OV/EV do.
    out["cert_is_dv"] = "organizationName" not in subject
    out["cert_not_after"] = cert.get("notAfter")
    return out


def fetch_requests(url: str, timeout: int):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout,
                     allow_redirects=True)
    r.raise_for_status()
    return r.url, r.text


# ---- page composition / content (pure; sync) --------------------------------

def categorise_links(soup: BeautifulSoup, home_host: str):
    """Return per-category page presence, outbound counts, and credible ratio."""
    home_reg = reg_domain(home_host)
    found = {k: None for k in PAGE_SIGNALS}
    outbound = 0
    outbound_credible = 0

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        text = a.get_text(" ", strip=True).lower()
        haystack = f"{href.lower()} {text}"

        for cat, patterns in PAGE_SIGNALS.items():
            if found[cat] is None and any(re.search(p, haystack) for p in patterns):
                found[cat] = urljoin(f"https://{home_host}", href)

        parsed = urlparse(href if "://" in href else urljoin(f"https://{home_host}", href))
        link_reg = reg_domain(parsed.netloc)
        if link_reg and link_reg != home_reg:
            outbound += 1
            ext = _tld(parsed.netloc)
            if link_reg in CRED_DOMAINS or ext.suffix in CRED_TLDS:
                outbound_credible += 1

    ratio = round(outbound_credible / outbound, 4) if outbound else 0.0
    return found, outbound, outbound_credible, ratio


def count_trackers(html: str, soup: BeautifulSoup) -> dict:
    srcs = [s.get("src", "") for s in soup.find_all("script")]
    src_blob = " ".join(srcs).lower()
    inline_blob = " ".join(
        s.get_text() for s in soup.find_all("script") if not s.get("src")
    ).lower()
    blob = src_blob + " " + inline_blob
    hits = sorted({p for p in TRACKER_PATTERNS if p in blob})
    return {"tracker_script_count": len(hits), "tracker_hits": hits,
            "total_script_count": len(srcs)}


def detect_bylines(html: str, soup: BeautifulSoup) -> dict:
    patterns = [r'rel=["\']author', r'class=["\'][^"\']*byline', r'itemprop=["\']author',
                r'\bby\s+[A-Z][a-z]+\s+[A-Z][a-z]+', r'property=["\']article:author']
    has_byline = any(re.search(p, html) for p in patterns)
    has_dateline = bool(
        soup.find("time") or re.search(r'datetime=["\']', html)
        or re.search(r'property=["\']article:published_time', html)
    )
    return {"byline_in_html": has_byline, "dateline_in_html": has_dateline}


def head_meta(soup: BeautifulSoup) -> dict:
    title = soup.title.get_text(strip=True) if soup.title else None

    def meta(attr, val):
        tag = soup.find("meta", attrs={attr: val})
        return tag.get("content", "").strip() if tag and tag.get("content") else None

    og = {}
    for t in soup.find_all("meta", property=re.compile(r"^og:")):
        if t.get("content"):
            og[t["property"]] = t["content"].strip()

    return {
        "title": title,
        "title_len": len(title) if title else 0,
        "meta_description": meta("name", "description"),
        "og_tag_count": len(og),
        "og_tags": og,
        "has_og": bool(og),
    }


def extract_css(soup: BeautifulSoup) -> str:
    blocks = [s.get_text() for s in soup.find_all("style")]
    return "\n\n".join(b for b in blocks if b.strip())


def parse_html(html: str, host: str) -> dict:
    """All synchronous HTML parsing for one page, bundled so it can be handed to
    a thread executor in one shot."""
    soup = BeautifulSoup(html, "html.parser")
    pages, outbound, out_cred, ratio = categorise_links(soup, host)
    result = {
        "transparency": {
            "about_page": pages["about"],
            "ownership_page": pages["ownership"],
            "corrections_or_standards_page": pages["corrections"],
            "author_or_contributor_page": pages["authors"],
            "contact_page": pages["contact"],
            "has_about": pages["about"] is not None,
            "has_ownership": pages["ownership"] is not None,
            "has_corrections_or_standards": pages["corrections"] is not None,
            "has_author_pages": pages["authors"] is not None,
        },
        "composition": {
            "outbound_link_count": outbound,
            "outbound_credible_count": out_cred,
            "credible_outbound_ratio": ratio,
            **count_trackers(html, soup),
            **detect_bylines(html, soup),
        },
        "content": head_meta(soup),
        "_css": extract_css(soup),
    }
    return result


# ---- async fetch ------------------------------------------------------------

async def fetch_browser(browser, url: str, timeout: int):
    """Render the page in a shared async Chromium browser and return
    (final_url, html). Each call gets its own context, opened and closed inside
    the single event loop — no threads, so no sync-API/loop conflict."""
    from playwright.async_api import TimeoutError as PWTimeout
    timeout_ms = timeout * 1000
    context = await browser.new_context(
        viewport={"width": 1366, "height": 768}, user_agent=UA
    )
    try:
        page = await context.new_page()
        page.set_default_timeout(timeout_ms)
        try:
            await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        except PWTimeout:
            # networkidle never fires on sites that poll forever; "load" is enough.
            await page.goto(url, wait_until="load", timeout=timeout_ms)
        final_url = page.url
        html = await page.content()
        return final_url, html
    finally:
        await context.close()


# ---- browser supervisor -----------------------------------------------------

class BrowserSupervisor:
    """Owns the Chromium instance and hands out the *current* browser per fetch
    (never a captured reference), so the browser can be recycled safely under
    concurrency.

    Recycling triggers:
      * periodic — every `recycle_every` successful page fetches;
      * on-demand — when a driver-socket error is observed, indicating the
        Python<->Node pipe is wounded and the shared browser may be poisoned.

    On a recycle, the supervisor stops issuing new fetches (recycle lock),
    drains in-flight fetches up to `timeout` seconds (bounded so one hung page
    cannot hold the relaunch hostage), tears down the old browser, and relaunches
    via _launch_with_retry. Fetches always go through `run_fetch`, which acquires
    the current browser after the recycle gate, so they bind to the live instance.
    """

    def __init__(self, timeout: int, recycle_every: int):
        self.timeout = timeout
        self.recycle_every = recycle_every
        self._pw = None
        self._browser = None
        self._loop = None
        # Coordination primitives.
        self._recycle_lock = asyncio.Lock()   # held while a relaunch is in progress
        self._gate = asyncio.Event()          # set => fetches may proceed
        self._inflight = 0                     # number of fetches currently on the browser
        self._inflight_zero = asyncio.Event()  # set when _inflight == 0
        self._inflight_zero.set()
        self._since_recycle = 0
        self._generation = 0                   # bumps each relaunch; for logging
        self.recycle_count = 0

    async def start(self):
        from playwright.async_api import async_playwright
        self._loop = asyncio.get_running_loop()
        self._pw = await async_playwright().start()
        self._browser = await _launch_with_retry(self._pw)
        self._gate.set()

    async def close(self):
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:
                pass
            self._pw = None

    def _enter(self):
        self._inflight += 1
        self._inflight_zero.clear()

    def _leave(self):
        self._inflight -= 1
        if self._inflight <= 0:
            self._inflight = 0
            self._inflight_zero.set()

    async def _recycle(self, reason: str):
        """Relaunch the browser. Drains in-flight fetches (bounded) first so we
        never tear the instance out from under a live page."""
        async with self._recycle_lock:
            # Another task may have already recycled while we waited on the lock.
            self._gate.clear()  # stop new fetches from starting
            self._generation += 1
            gen = self._generation
            print(f"[recycle #{gen}] {reason}; draining {self._inflight} in-flight "
                  f"(<= {self.timeout}s)…", file=sys.stderr)
            try:
                await asyncio.wait_for(self._inflight_zero.wait(), timeout=self.timeout)
            except asyncio.TimeoutError:
                print(f"[recycle #{gen}] drain timed out; relaunching anyway "
                      f"({self._inflight} still in-flight)", file=sys.stderr)
            old = self._browser
            self._browser = None
            if old is not None:
                try:
                    await old.close()
                except Exception:
                    pass
            try:
                self._browser = await _launch_with_retry(self._pw)
            except Exception as e:
                # Couldn't relaunch in the same Playwright session; restart it.
                print(f"[recycle #{gen}] relaunch failed ({e}); restarting "
                      f"playwright session", file=sys.stderr)
                try:
                    await self._pw.stop()
                except Exception:
                    pass
                from playwright.async_api import async_playwright
                self._pw = await async_playwright().start()
                self._browser = await _launch_with_retry(self._pw)
            self._since_recycle = 0
            self.recycle_count += 1
            self._gate.set()
            print(f"[recycle #{gen}] browser back up", file=sys.stderr)

    async def run_fetch(self, url: str, timeout: int):
        """Fetch one URL on the current browser. Waits behind any in-progress
        recycle, marks itself in-flight, and on a driver-socket error requests a
        recycle then re-raises so the caller's retry loop can try the new browser.

        In-flight accounting uses a `left` guard: branches that must recycle call
        _leave() early (so the drain can complete), and the finally only leaves if
        that hasn't already happened — preventing a double-decrement."""
        # Wait until fetches are permitted (no recycle in progress).
        await self._gate.wait()
        browser = self._browser
        if browser is None:
            # Lost a race with a recycle; trigger/await one and retry binding.
            await self._recycle("browser missing at fetch bind")
            await self._gate.wait()
            browser = self._browser
        self._enter()
        left = False
        try:
            result = await fetch_browser(browser, url, timeout)
            # Periodic recycle bookkeeping on success.
            self._since_recycle += 1
            if self._since_recycle >= self.recycle_every:
                # Leave in-flight before recycling so the drain can complete.
                self._leave(); left = True
                await self._recycle(f"periodic ({self.recycle_every} pages)")
            return result
        except Exception as e:
            if is_driver_socket_error(e):
                # The shared browser may be poisoned. Leave in-flight first so the
                # drain can complete, then recycle, then re-raise so the caller's
                # retry loop tries the fresh browser.
                self._leave(); left = True
                await self._recycle(f"driver-socket error: {str(e)[:80]}")
            raise
        finally:
            if not left:
                self._leave()


async def get_html(supervisor, url: str, timeout: int, use_browser: bool,
                   retries: int, loop):
    """Return (final_url, html, method). Browser-with-retries first, then a
    single requests fallback. Transient errors consume retries with exponential
    backoff + jitter; permanent errors (NXDOMAIN, cert, 404) short-circuit to the
    fallback immediately. The blocking requests call runs in the thread executor."""
    errors = []
    if use_browser and supervisor is not None:
        for attempt in range(1, retries + 1):
            try:
                final_url, html = await supervisor.run_fetch(url, timeout)
                if html and len(html) > 200:
                    return final_url, html, "browser"
                errors.append(f"browser attempt {attempt}: empty/short content")
            except Exception as e:
                errors.append(f"browser attempt {attempt}: {e}")
                if is_permanent_error(e):
                    break  # won't change on retry; go straight to requests
                if attempt < retries:
                    # Exponential backoff with jitter; driver recycles already
                    # happened inside run_fetch, so the next attempt hits a fresh
                    # browser. Backoff also throttles a flaky link.
                    delay = min(2 ** (attempt - 1), 8) + random.uniform(0, 0.75)
                    await asyncio.sleep(delay)
    try:
        final_url, html = await loop.run_in_executor(None, fetch_requests, url, timeout)
        return final_url, html, "requests"
    except Exception as e:
        errors.append(f"requests: {e}")
    raise RuntimeError("; ".join(errors) or "no html")


# ---- per-outlet driver ------------------------------------------------------

async def enrich_one(path: Path, supervisor, sem, timeout: int, archive: bool,
                     use_browser: bool, retries: int, loop) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    media_link = (data.get("media link") or "").strip()
    if not media_link:
        return {"file": path.name, "status": "skip", "reason": "no media link"}
    if "://" not in media_link:
        media_link = "https://" + media_link

    host = urlparse(media_link).netloc
    ext = _tld(host)

    prov = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "final_url": None,
        "fetch_ok": False,
        "fetch_method": None,
        "fetch_error": None,
        "tld": ext.suffix,
        "tld_suspect": ext.suffix.split(".")[-1] in SUSPECT_TLDS,
        "registered_domain": reg_domain(host),
    }

    # WHOIS + TLS are blocking; run them in the thread executor concurrently.
    whois_fut = loop.run_in_executor(None, whois_age, host)
    tls_fut = loop.run_in_executor(None, tls_info, host, timeout)

    async with sem:
        try:
            final_url, html, method = await get_html(
                supervisor, media_link, timeout, use_browser, retries, loop
            )
            prov["final_url"] = final_url
            prov["fetch_ok"] = True
            prov["fetch_method"] = method
            parsed = await loop.run_in_executor(None, parse_html, html, host)
            prov["transparency"] = parsed["transparency"]
            prov["composition"] = parsed["composition"]
            prov["content"] = parsed["content"]

            if archive:
                ARCHIVE_DIR.mkdir(exist_ok=True)
                stem = path.stem
                (ARCHIVE_DIR / f"{stem}.html").write_text(html, encoding="utf-8")
                css = parsed["_css"]
                if css:
                    (ARCHIVE_DIR / f"{stem}.css").write_text(css, encoding="utf-8")
                prov["archive"] = {
                    "html_path": f"{ARCHIVE_DIR}/{stem}.html",
                    "css_path": f"{ARCHIVE_DIR}/{stem}.css" if css else None,
                    "html_bytes": len(html.encode("utf-8")),
                }
        except Exception as e:
            prov["fetch_error"] = str(e)

    prov.update(await whois_fut)
    prov["tls"] = await tls_fut

    data["provenance"] = prov
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    status = "ok" if prov["fetch_ok"] else "fetch_error"
    return {"file": path.name, "status": status, "host": host,
            "method": prov["fetch_method"], "error": prov["fetch_error"]}


def _already_failed(path: Path) -> bool:
    """True if the file has a provenance block whose last fetch did not succeed."""
    try:
        prov = json.loads(path.read_text(encoding="utf-8")).get("provenance")
    except Exception:
        return False
    return bool(prov) and not prov.get("fetch_ok", False)


def select_files(data_dir: Path, force: bool, retry_errors: bool, limit):
    files = sorted(data_dir.glob("*.json"))
    if not files:
        return files
    if retry_errors:
        before = len(files)
        files = [f for f in files if _already_failed(f)]
        print(f"Retry mode: {len(files)} of {before} files had a failed fetch")
    elif not force:
        before = len(files)
        kept = []
        for f in files:
            try:
                if "provenance" in json.loads(f.read_text(encoding="utf-8")):
                    continue
            except Exception:
                pass
            kept.append(f)
        files = kept
        print(f"Skipping {before - len(files)} already-enriched, {len(files)} remaining")
    if limit:
        files = files[:limit]
    return files


async def _launch_with_retry(pw, attempts: int = 3):
    """Launch headless Chromium, retrying on the transient 'connect to socket'
    failures that the Playwright Node driver occasionally throws on macOS. Falls
    back to the system Chrome channel if the bundled headless shell won't start."""
    last = None
    for i in range(1, attempts + 1):
        try:
            return await pw.chromium.launch(headless=True)
        except Exception as e:
            last = e
            print(f"[launch] attempt {i}/{attempts} failed: {e}", file=sys.stderr)
            await asyncio.sleep(2 * i)
    # Last resort: try the full Chrome build (channel='chrome') if installed.
    try:
        print("[launch] trying channel='chrome' fallback", file=sys.stderr)
        return await pw.chromium.launch(headless=True, channel="chrome")
    except Exception as e:
        raise RuntimeError(
            f"Could not launch a browser after {attempts} attempts "
            f"(last error: {last}; chrome-channel error: {e}). "
            f"Run `playwright install chromium` in THIS environment, or use "
            f"--no-browser for a requests-only pass."
        )


async def preflight(timeout: int, retries: int):
    """Launch a supervisor and fetch one known-good URL, printing a plain result.
    Use this to confirm the browser actually works before a long run."""
    url = "https://example.com"
    loop = asyncio.get_running_loop()
    sup = BrowserSupervisor(timeout=timeout, recycle_every=10_000)
    await sup.start()
    try:
        final_url, html, method = await get_html(sup, url, timeout, True, retries, loop)
        ok = bool(html) and "Example Domain" in html
        print(f"preflight: fetched {url} via {method}, "
              f"{len(html)} bytes, sentinel {'found' if ok else 'MISSING'}")
        return ok
    finally:
        await sup.close()


async def run(files, concurrency, timeout, archive, use_browser, retries, recycle_every):
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)

    supervisor = None
    if use_browser:
        supervisor = BrowserSupervisor(timeout=timeout, recycle_every=recycle_every)
        await supervisor.start()

    ok = err = skip = 0
    by_method = {"browser": 0, "requests": 0}
    failed_files = []
    start = time.time()
    done = 0
    total = len(files)

    # A hard ceiling on any single outlet so a hung page cannot occupy a
    # semaphore slot forever. Generous headroom over the per-page timeout to
    # allow for retries + backoff + the requests fallback.
    per_task_ceiling = timeout * (retries + 2) + 30

    async def guarded(f):
        try:
            return await asyncio.wait_for(
                enrich_one(f, supervisor, sem, timeout, archive,
                           use_browser, retries, loop),
                timeout=per_task_ceiling,
            )
        except asyncio.TimeoutError:
            # Persist a failure so --retry-errors can pick it up next pass.
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                data.setdefault("provenance", {})
                data["provenance"].update({
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "fetch_ok": False,
                    "fetch_error": f"task ceiling exceeded ({per_task_ceiling}s)",
                })
                f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            return {"file": f.name, "status": "fetch_error",
                    "error": f"task ceiling exceeded ({per_task_ceiling}s)"}
        except Exception as e:
            return {"file": f.name, "status": "fetch_error", "error": str(e)}

    try:
        tasks = [asyncio.create_task(guarded(f)) for f in files]
        for fut in asyncio.as_completed(tasks):
            r = await fut
            done += 1
            if r["status"] == "ok":
                ok += 1
                if r.get("method") in by_method:
                    by_method[r["method"]] += 1
                print(f"[{done}/{total}] {r['status']:11s} {r['file']}")
            elif r["status"] == "skip":
                skip += 1
                print(f"[{done}/{total}] {r['status']:11s} {r['file']}  "
                      f"-- {r.get('reason','')}")
            else:
                err += 1
                failed_files.append(r["file"])
                msg = (r.get("error") or "")[:90]
                print(f"[{done}/{total}] {r['status']:11s} {r['file']}  -- {msg}")
    finally:
        if supervisor is not None:
            await supervisor.close()

    elapsed = time.time() - start
    print(f"\nDone. {ok} fetched ({by_method['browser']} browser / "
          f"{by_method['requests']} requests), {skip} skipped, {err} errors "
          f"in {elapsed:.0f}s.")
    if supervisor is not None:
        print(f"Browser recycled {supervisor.recycle_count} time(s).")
    if failed_files:
        # Write the still-failing set so a targeted re-run is trivial.
        out = Path("still_failing.txt")
        out.write_text("\n".join(failed_files), encoding="utf-8")
        print(f"{len(failed_files)} still failing — names written to {out}. "
              f"Re-run with: python enrich_provenance.py --retry-errors "
              f"--concurrency {max(1, concurrency - 1)}")


def main():
    ap = argparse.ArgumentParser(description="Enrich outlet JSONs with provenance features.")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--concurrency", type=int, default=3,
                    help="Simultaneous pages in the single browser (default 3)")
    ap.add_argument("--timeout", type=int, default=25, help="Per-page timeout, seconds")
    ap.add_argument("--retries", type=int, default=3,
                    help="Browser attempts per page before requests fallback (default 3)")
    ap.add_argument("--recycle-every", type=int, default=40,
                    help="Relaunch the browser every N successful pages (default 40)")
    ap.add_argument("--force", action="store_true",
                    help="Re-fetch even if a provenance block already exists")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Only re-process files whose last fetch failed")
    ap.add_argument("--no-archive", dest="archive", action="store_false",
                    help="Do not store raw html/css")
    ap.add_argument("--no-browser", dest="use_browser", action="store_false",
                    help="Use requests only (no Chromium)")
    ap.add_argument("--preflight", action="store_true",
                    help="Launch the browser, fetch one known URL, report, and exit")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.preflight:
        ok = asyncio.run(preflight(args.timeout, args.retries))
        return 0 if ok else 1

    data_dir = Path(args.data_dir)
    files = select_files(data_dir, args.force, args.retry_errors, args.limit)
    if not files:
        print("Nothing to do." if data_dir.exists() else f"No JSON files in {data_dir}")
        return 0

    asyncio.run(run(files, args.concurrency, args.timeout, args.archive,
                    args.use_browser, args.retries, args.recycle_every))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())