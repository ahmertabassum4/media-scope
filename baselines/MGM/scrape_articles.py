"""Scrape a bag of article texts for every outlet in the train/test split.

For each outlet URL we fetch the homepage, discover internal article links
(links longer than 65 characters are preferred, following Baly et al. 2020),
optionally boost the candidate pool from sitemap/RSS, extract article bodies
with trafilatura, and write one JSON line per outlet to
data/articles/articles.jsonl. Each article inherits the outlet's split,
label_3class, and bias_rating (distant supervision). The run is resumable:
outlets already in the output file are skipped on restart.
"""

import argparse
import hashlib
import json
import re
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import tldextract
import trafilatura
import urllib3
from bs4 import BeautifulSoup
from tqdm import tqdm

# reuse the project's split files and key normalization from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dataset import (  # noqa: E402
    CLASS_ORDER,
    RAW_TEST_CSV,
    RAW_TRAIN_CSV,
    ROOT,
    normalize_key,
)

OUT_DEFAULT = ROOT / "data" / "articles" / "articles.jsonl"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

SKIP_PATH = re.compile(
    r"/(tag|tags|category|categories|topic|topics|author|authors|about|contact|contact-us|"
    r"privacy|privacy-policy|terms|login|register|signup|subscribe|subscription|account|"
    r"wp-login|wp-admin|feed|rss|page|shop|store|cart|search|gallery|photo|photos|video|videos|"
    r"advertise|advertising|newsletter|newsletters|donate|careers|jobs|classifieds|events|"
    r"weather|horoscope|puzzles|games|staff|masthead|sitemap|directory)(/|$)", re.I)
SKIP_EXT = re.compile(r"\.(jpe?g|png|gif|svg|webp|pdf|zip|gz|mp3|mp4|avi|mov|wmv|css|js|ico|xml|json)(\?.*)?$", re.I)

# platforms that can never yield the outlet's own articles (used for redirect guard)
SOCIAL_PLATFORMS = {"youtube.com", "facebook.com", "twitter.com", "x.com", "instagram.com",
                    "t.me", "telegram.me", "tiktok.com", "linkedin.com", "google.com"}

_robots_cache = {}
_robots_lock = threading.Lock()


def build_worklist():
    outlets, seen = [], set()
    for split, path in (("train", RAW_TRAIN_CSV), ("test", RAW_TEST_CSV)):
        for rec in pd.read_csv(path).to_dict("records"):
            key = normalize_key(Path(str(rec.get("image_path", ""))).name)
            label = str(rec.get("label_3class", "")).upper()
            bias_rating = str(rec.get("bias_rating", "")).strip().upper()
            url = str(rec.get("url", "")).strip()
            if not key or key in seen or label not in CLASS_ORDER or not url.startswith("http"):
                continue
            seen.add(key)
            outlets.append({"key": key, "media_name": str(rec.get("media_name", "")),
                            "url": url, "split": split, "label_3class": label,
                            "bias_rating": bias_rating})
    return outlets


def fetch(session, url, timeout, notes=None, allow_insecure_ssl=False):
    """GET a URL; SSL verification remains enabled unless explicitly opted out."""
    for attempt in (1, 2):
        try:
            r = session.get(url, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and r.text:
                return r.text, str(r.url)
            return None, None
        except requests.exceptions.SSLError:
            if not allow_insecure_ssl:
                if notes is not None:
                    notes.add("ssl_error")
                return None, None
            try:
                # Kept behind an explicit CLI flag for exceptional legacy sites.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)
                    r = session.get(url, timeout=timeout, allow_redirects=True, verify=False)
                if r.status_code == 200 and r.text:
                    if notes is not None:
                        notes.add("ssl_insecure")
                    return r.text, str(r.url)
            except requests.RequestException:
                pass
            return None, None
        except requests.RequestException:
            if attempt == 1:
                time.sleep(2)
    return None, None


def robots_for(session, base_url):
    host = urlparse(base_url).netloc
    with _robots_lock:
        if host in _robots_cache:
            return _robots_cache[host]
    rp = None
    try:
        r = session.get(urljoin(base_url, "/robots.txt"), timeout=8)
        if r.status_code == 200 and r.text:
            rp = robotparser.RobotFileParser()
            rp.parse(r.text.splitlines())
    except requests.RequestException:
        rp = None
    with _robots_lock:
        _robots_cache[host] = rp
    return rp


def robots_allows(rp, url):
    if rp is None:
        return True
    try:
        return rp.can_fetch("*", url)
    except Exception:
        return True


def clean_link(href, base_url):
    href = urljoin(base_url, href.strip())
    p = urlparse(href)
    if p.scheme not in ("http", "https"):
        return None, None
    return p._replace(fragment="").geturl(), p


def is_candidate(p, domains):
    if tldextract.extract(p.geturl()).registered_domain not in domains:
        return False
    if p.path in ("", "/") or SKIP_PATH.search(p.path) or SKIP_EXT.search(p.path):
        return False
    return True


def discover_links(html, base_url, domains, pool):
    """Internal links from the homepage; >65-char links first (Baly et al. rule)."""
    soup = BeautifulSoup(html, "lxml")
    seen, long_links, short_links = set(), [], []
    for a in soup.find_all("a", href=True):
        href, p = clean_link(a["href"], base_url)
        if not href or href in seen:
            continue
        seen.add(href)
        if not is_candidate(p, domains):
            continue
        anchor = a.get_text(" ", strip=True)
        if len(href) > 65 or len(anchor) > 65:
            long_links.append(href)
        else:
            short_links.append(href)
    short_links.sort(key=lambda u: -len(urlparse(u).path))
    return (long_links + short_links)[:pool]


def booster_links(session, base_url, domains, pool, notes, allow_insecure_ssl=False):
    """Extra candidates from sitemap.xml / RSS for link-sparse homepages."""
    found = []
    for suffix in ("/sitemap.xml", "/feed", "/rss", "/rss.xml"):
        text, _ = fetch(session, base_url.rstrip("/") + suffix, timeout=10, notes=notes,
                        allow_insecure_ssl=allow_insecure_ssl)
        if not text:
            continue
        locs = re.findall(r"<loc>\s*(https?://[^<\s]+)\s*</loc>", text)
        locs += re.findall(r"<link>\s*(https?://[^<\s]+)\s*</link>", text)
        locs += re.findall(r'<link[^>]+href="(https?://[^"]+)"', text)
        # one level of sitemap-index indirection
        subs = [u for u in locs if u.lower().endswith(".xml")][:2]
        for sub in subs:
            sub_url, sub_parsed = clean_link(sub, base_url)
            if (not sub_url or tldextract.extract(sub_parsed.geturl()).registered_domain not in domains):
                continue
            sub_text, _ = fetch(session, sub_url, timeout=10, notes=notes,
                                allow_insecure_ssl=allow_insecure_ssl)
            if sub_text:
                locs += re.findall(r"<loc>\s*(https?://[^<\s]+)\s*</loc>", sub_text)
        found.extend(locs)
        if len(found) >= pool:
            break
    keep, seen = [], set()
    for href in found:
        href, p = clean_link(href, base_url)
        if href and href not in seen and is_candidate(p, domains):
            seen.add(href)
            keep.append(href)
    return keep[:pool]


def extract_body(html, url):
    text = None
    try:
        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
    except Exception:
        text = None
    if text:
        return text
    try:
        soup = BeautifulSoup(html, "lxml")
        node = soup.find("article") or soup.body
        if node:
            text = "\n".join(t for t in (p.get_text(" ", strip=True) for p in node.find_all("p")) if t)
    except Exception:
        text = None
    return text or None


def looks_like_prose(text, min_words):
    tokens = text.split()
    if len(tokens) < min_words:
        return False
    # listings/directories are number-heavy; articles are not
    if sum(1 for t in tokens if any(c.isdigit() for c in t)) / len(tokens) > 0.20:
        return False
    sentences = [s for s in re.split(r"[.!?]+", text) if len(s.split()) >= 8]
    return len(sentences) >= 3


def scrape_outlet(outlet, args):
    result = {**outlet, "status": "error:unknown", "n_articles": 0, "articles": [],
              "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    notes = set()
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        domain = tldextract.extract(outlet["url"]).registered_domain

        html, final_url = fetch(session, outlet["url"], args.timeout, notes,
                                allow_insecure_ssl=args.allow_insecure_ssl)
        if html is None and outlet["url"].startswith("https://"):
            html, final_url = fetch(
                session, "http://" + outlet["url"][len("https://"):], args.timeout, notes,
                allow_insecure_ssl=args.allow_insecure_ssl,
            )
        if html is None:
            result["status"] = "no_homepage"
            result["notes"] = sorted(notes)
            return result
        base_url = final_url or outlet["url"]

        # sites often redirect to a rebranded domain; follow it, but never to a platform
        final_domain = tldextract.extract(base_url).registered_domain
        if final_domain in SOCIAL_PLATFORMS:
            result["status"] = "social_redirect"
            result["notes"] = sorted(notes)
            return result
        domains = {domain, final_domain}

        rp = robots_for(session, base_url)
        candidates = discover_links(html, base_url, domains, args.pool)
        if len(candidates) < args.target * 2:
            extra = booster_links(session, base_url, domains, args.pool, notes,
                                  allow_insecure_ssl=args.allow_insecure_ssl)
            seen = set(candidates)
            candidates += [u for u in extra if u not in seen]

        want = min(args.target, args.cap)
        seen_hashes = set()
        for cand in candidates:
            if len(result["articles"]) >= want:
                break
            if not robots_allows(rp, cand):
                continue
            time.sleep(args.delay)
            page, page_url = fetch(session, cand, args.timeout, notes,
                                   allow_insecure_ssl=args.allow_insecure_ssl)
            if not page:
                continue
            text = extract_body(page, page_url or cand)
            if not text or not looks_like_prose(text, args.min_words):
                continue
            digest = hashlib.sha256(text[:2000].encode("utf-8", "ignore")).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            result["articles"].append({"url": cand, "n_words": len(text.split()), "text": text})

        n = len(result["articles"])
        result["n_articles"] = n
        result["status"] = "ok" if n >= args.min_articles else ("low_coverage" if n else "no_articles")
    except Exception as e:  # noqa: BLE001 - one bad outlet must never kill the run
        result["status"] = f"error:{type(e).__name__}"
    result["notes"] = sorted(notes)
    return result


def canonical_outlet_rows(out_path):
    """Read one canonical result per outlet, matching pipeline.load_outlets."""
    if not out_path.exists():
        return [], 0
    rows, order, raw_count = {}, [], 0
    for line_number, line in enumerate(out_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        raw_count += 1
        try:
            row = json.loads(line)
            key = row["key"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise SystemExit(f"invalid article cache row at {out_path}:{line_number}") from exc
        if key not in rows:
            rows[key] = row
            order.append(key)
        elif int(row.get("n_articles", 0)) > int(rows[key].get("n_articles", 0)):
            rows[key] = row
    return [rows[key] for key in order], raw_count


def compact_outlet_records(out_path):
    """Remove stale duplicate outlet rows while retaining the richest scrape."""
    rows, raw_count = canonical_outlet_rows(out_path)
    duplicates_removed = raw_count - len(rows)
    if duplicates_removed:
        temp_path = out_path.with_name(f".{out_path.name}.{threading.get_native_id()}.tmp")
        temp_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        temp_path.replace(out_path)
    return duplicates_removed


def load_done_keys(out_path):
    return {row["key"] for row in canonical_outlet_rows(out_path)[0]}


def backfill_bias_ratings(out_path, outlets):
    """Add source bias ratings to records written by older scraper versions."""
    if not out_path.exists():
        return 0

    bias_by_key = {outlet["key"]: outlet["bias_rating"] for outlet in outlets}
    lines = out_path.read_text().splitlines()
    updated_lines = []
    n_updated = 0
    for line in lines:
        if not line.strip():
            updated_lines.append(line)
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            updated_lines.append(line)
            continue
        bias_rating = bias_by_key.get(row.get("key"))
        if bias_rating is not None and row.get("bias_rating") != bias_rating:
            row["bias_rating"] = bias_rating
            line = json.dumps(row, ensure_ascii=False)
            n_updated += 1
        updated_lines.append(line)

    if n_updated:
        temp_path = out_path.with_name(f".{out_path.name}.{threading.get_native_id()}.tmp")
        temp_path.write_text("\n".join(updated_lines) + "\n")
        temp_path.replace(out_path)
    return n_updated


def report(out_path):
    if not out_path.exists():
        print(f"no output yet at {out_path}")
        return
    rows, raw_count = canonical_outlet_rows(out_path)
    df = pd.DataFrame([{k: r.get(k) for k in
                       ("key", "split", "label_3class", "bias_rating", "status", "n_articles")}
                       for r in rows])
    total = len(build_worklist())
    print(f"\nscraped {len(df)}/{total} outlets  ->  {out_path}")
    if raw_count != len(rows):
        print(f"collapsed {raw_count - len(rows)} stale duplicate cache rows in this report")
    df["ok"] = df["status"] == "ok"
    df["zero"] = df["n_articles"] == 0
    summary = df.groupby(["split", "label_3class"]).agg(
        outlets=("key", "count"), ok=("ok", "sum"),
        mean_articles=("n_articles", "mean"), zero=("zero", "sum")).round(1)
    print(summary.to_string())
    print("\nstatus counts:")
    print(df["status"].value_counts().to_string())
    print("\nbias rating counts:")
    print(df["bias_rating"].fillna("<missing>").value_counts().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--target", type=int, default=25, help="articles to aim for per outlet")
    ap.add_argument("--cap", type=int, default=30, help="hard maximum articles stored per outlet")
    ap.add_argument("--min-words", type=int, default=100, help="minimum words for a kept article")
    ap.add_argument("--min-articles", type=int, default=5, help="minimum articles for status=ok")
    ap.add_argument("--pool", type=int, default=60, help="max candidate links per outlet")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests to one host")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--allow-insecure-ssl", action="store_true",
                    help="allow certificate-verification bypass for legacy sites (unsafe)")
    ap.add_argument("--limit", type=int, default=0, help="scrape only the first N pending outlets")
    ap.add_argument("--report", action="store_true", help="print coverage summary of --out and exit")
    ap.add_argument("--backfill-bias-only", action="store_true",
                    help="add bias_rating to existing records and exit without scraping")
    args = ap.parse_args()

    if args.report:
        report(args.out)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_compacted = compact_outlet_records(args.out)
    if n_compacted:
        print(f"compacted {n_compacted} stale duplicate outlet records")
    outlets = build_worklist()
    n_backfilled = backfill_bias_ratings(args.out, outlets)
    if n_backfilled:
        print(f"backfilled bias_rating in {n_backfilled} existing records")
    if args.backfill_bias_only:
        return

    done = load_done_keys(args.out)
    work = [o for o in outlets if o["key"] not in done]
    if args.limit:
        work = work[:args.limit]
    print(f"outlets: {len(done)} already scraped, {len(work)} to go "
          f"(target {args.target}/outlet, {args.workers} workers)")
    if not work:
        report(args.out)
        return

    write_lock = threading.Lock()
    with args.out.open("a") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(scrape_outlet, outlet, args) for outlet in work]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="outlet"):
            res = fut.result()
            with write_lock:
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()

    report(args.out)


if __name__ == "__main__":
    main()
