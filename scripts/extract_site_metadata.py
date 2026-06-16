import argparse
import json
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_DIR = PROJECT_ROOT / "data" / "metadata" / "media_metadata"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "metadata" / "raw_site_metadata.jsonl"
USER_AGENT = "Mozilla/5.0 (compatible; UGRIPMetadataBot/1.0; +https://example.org/research)"
MAX_HTML_BYTES = 2_000_000


class RedirectCounter(urllib.request.HTTPRedirectHandler):
    def __init__(self):
        self.count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.count += 1
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = []
        self.meta = {}
        self.meta_properties = []
        self.links = []
        self.link_rels = []
        self.feed_links = []
        self.scripts = []
        self.stylesheets = []
        self.images = []
        self.headings = []
        self.schema_types = []
        self.jsonld_blocks = []
        self.tag_counts = {}
        self.form_count = 0
        self.search_seen = False
        self.amp_seen = False
        self._in_title = False
        self._in_script = False
        self._script_type = ""
        self._script_text = []
        self._in_style = False
        self._heading_tag = ""
        self._heading_text = []
        self._text_words = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs = {k.lower(): (v or "") for k, v in attrs}
        self.tag_counts[tag] = self.tag_counts.get(tag, 0) + 1

        if tag == "html" and ("amp" in attrs or "⚡" in attrs):
            self.amp_seen = True
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            name = (attrs.get("name") or attrs.get("property") or attrs.get("http-equiv") or "").lower()
            content = attrs.get("content", "").strip()
            if name:
                self.meta[name] = content
                if name.startswith("og:") or name.startswith("twitter:"):
                    self.meta_properties.append(name)
        elif tag == "link":
            rel = attrs.get("rel", "").lower()
            kind = attrs.get("type", "").lower()
            href = attrs.get("href", "").strip()
            if rel:
                self.link_rels.append(rel)
            if href:
                self.links.append(href)
                if "stylesheet" in rel:
                    self.stylesheets.append(href)
                if "rss" in kind or "atom" in kind or "feed" in rel:
                    self.feed_links.append(href)
                if "amphtml" in rel:
                    self.amp_seen = True
        elif tag == "a":
            href = attrs.get("href", "").strip()
            if href:
                self.links.append(href)
        elif tag == "script":
            src = attrs.get("src", "").strip()
            self._in_script = True
            self._script_type = attrs.get("type", "").lower()
            self._script_text = []
            if src:
                self.scripts.append(src)
        elif tag == "style":
            self._in_style = True
        elif tag == "img":
            src = attrs.get("src", "").strip()
            if src:
                self.images.append(src)
        elif tag in {"h1", "h2", "h3"}:
            self._heading_tag = tag
            self._heading_text = []
        elif tag == "form":
            self.form_count += 1
            action = attrs.get("action", "").lower()
            if "search" in action:
                self.search_seen = True
        elif tag == "input" and attrs.get("type", "").lower() == "search":
            self.search_seen = True

        role = attrs.get("role", "").lower()
        if role == "search":
            self.search_seen = True
        itemtype = attrs.get("itemtype", "")
        if "schema.org" in itemtype:
            self.schema_types.append(itemtype)

    def handle_data(self, data):
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self._in_title:
            self.title.append(text)
        elif self._in_script:
            if "ld+json" in self._script_type:
                self._script_text.append(text)
        elif self._in_style:
            return
        elif self._heading_tag:
            self._heading_text.append(text)
        else:
            self._text_words += len(re.findall(r"\b\w+\b", text))

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        elif tag == "script":
            if self._script_text:
                self.jsonld_blocks.append(" ".join(self._script_text))
            self._in_script = False
            self._script_type = ""
            self._script_text = []
        elif tag == "style":
            self._in_style = False
        elif tag == self._heading_tag:
            text = re.sub(r"\s+", " ", " ".join(self._heading_text)).strip()
            if text:
                self.headings.append({"tag": self._heading_tag, "text": text})
            self._heading_tag = ""
            self._heading_text = []

    @property
    def word_count(self):
        return self._text_words


def normalize_url(url):
    url = str(url or "").strip()
    if not url:
        return ""
    if not re.match(r"^https?://", url, re.I):
        return "https://" + url
    return url


def clean_host(url):
    host = urllib.parse.urlparse(url).hostname or ""
    return host.lower().removeprefix("www.")


def base_domain(host):
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    return ".".join(parts[-2:])


def host_of(url, base_url):
    try:
        absolute = urllib.parse.urljoin(base_url, url)
        return clean_host(absolute)
    except Exception:
        return ""


def is_internal(host, root_domain):
    return host == root_domain or host.endswith("." + root_domain)


def fetch_url(url, timeout):
    redirect = RedirectCounter()
    opener = urllib.request.build_opener(redirect)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    start = time.monotonic()
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read(MAX_HTML_BYTES + 1)
            return {
                "ok": True,
                "status_code": getattr(resp, "status", resp.getcode()),
                "final_url": resp.geturl(),
                "headers": dict(resp.headers.items()),
                "body": body[:MAX_HTML_BYTES],
                "truncated": len(body) > MAX_HTML_BYTES,
                "redirect_count": redirect.count,
                "elapsed_ms": round((time.monotonic() - start) * 1000),
                "error": "",
            }
    except urllib.error.HTTPError as e:
        body = e.read(MAX_HTML_BYTES + 1)
        return {
            "ok": False,
            "status_code": e.code,
            "final_url": e.url,
            "headers": dict(e.headers.items()) if e.headers else {},
            "body": body[:MAX_HTML_BYTES],
            "truncated": len(body) > MAX_HTML_BYTES,
            "redirect_count": redirect.count,
            "elapsed_ms": round((time.monotonic() - start) * 1000),
            "error": str(e),
        }
    except Exception as e:
        return {
            "ok": False,
            "status_code": 0,
            "final_url": url,
            "headers": {},
            "body": b"",
            "truncated": False,
            "redirect_count": redirect.count,
            "elapsed_ms": round((time.monotonic() - start) * 1000),
            "error": str(e),
        }


def decode_body(body, headers):
    content_type = headers.get("Content-Type") or headers.get("content-type") or ""
    match = re.search(r"charset=([^;\s]+)", content_type, re.I)
    encodings = [match.group(1)] if match else []
    encodings += ["utf-8", "latin-1"]
    for encoding in encodings:
        try:
            return body.decode(encoding, errors="replace")
        except Exception:
            continue
    return body.decode("utf-8", errors="replace")


def cert_text(name_parts):
    chunks = []
    for group in name_parts or []:
        for key, value in group:
            chunks.append(f"{key}={value}")
    return ", ".join(chunks)


def ssl_metadata(url, timeout):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return {"checked": False, "valid": False, "days_left": None, "issuer": "", "subject": "", "tls_version": "", "error": ""}
    port = parsed.port or 443
    try:
        context = ssl.create_default_context()
        with socket.create_connection((parsed.hostname, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=parsed.hostname) as ssock:
                cert = ssock.getpeercert()
                expires = datetime.strptime(cert.get("notAfter", ""), "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
                days_left = (expires - datetime.now(timezone.utc)).days
                return {
                    "checked": True,
                    "valid": True,
                    "days_left": days_left,
                    "issuer": cert_text(cert.get("issuer")),
                    "subject": cert_text(cert.get("subject")),
                    "tls_version": ssock.version() or "",
                    "error": "",
                }
    except Exception as e:
        return {"checked": True, "valid": False, "days_left": None, "issuer": "", "subject": "", "tls_version": "", "error": str(e)}


def linked_page_flags(links):
    joined = " ".join(links).lower()
    return {
        "about": bool(re.search(r"/(about|about-us|who-we-are)(/|$|\?|#)", joined)),
        "contact": bool(re.search(r"/(contact|contact-us)(/|$|\?|#)", joined)),
        "privacy": bool(re.search(r"/(privacy|privacy-policy)(/|$|\?|#)", joined)),
        "terms": bool(re.search(r"/(terms|terms-of-use|terms-and-conditions)(/|$|\?|#)", joined)),
        "advertise": bool(re.search(r"/(advertise|advertising)(/|$|\?|#)", joined)),
        "masthead": bool(re.search(r"/(masthead|staff|editorial-team|team)(/|$|\?|#)", joined)),
        "corrections": bool(re.search(r"/(corrections|ethics|editorial-policy|standards)(/|$|\?|#)", joined)),
        "subscribe": bool(re.search(r"/(subscribe|subscription|membership)(/|$|\?|#)", joined)),
        "donate": bool(re.search(r"/(donate|donation|support-us|contribute)(/|$|\?|#)", joined)),
        "login": bool(re.search(r"/(login|sign-in|signin|account)(/|$|\?|#)", joined)),
    }


def summarize_html(html, final_url):
    parser = PageParser()
    try:
        parser.feed(html)
    except Exception:
        pass

    host = clean_host(final_url)
    root = base_domain(host)
    link_hosts = [host_of(link, final_url) for link in parser.links]
    link_hosts = [h for h in link_hosts if h]
    internal_links = sum(1 for h in link_hosts if is_internal(h, root))
    external_links = sum(1 for h in link_hosts if not is_internal(h, root))
    script_hosts = sorted({host_of(src, final_url) for src in parser.scripts if host_of(src, final_url)})
    stylesheet_hosts = sorted({host_of(src, final_url) for src in parser.stylesheets if host_of(src, final_url)})
    image_hosts = sorted({host_of(src, final_url) for src in parser.images if host_of(src, final_url)})
    title = re.sub(r"\s+", " ", " ".join(parser.title)).strip()
    meta_description = parser.meta.get("description", "")
    headings = [h["text"] for h in parser.headings]
    uppercase = [h for h in headings if len(h) >= 8 and sum(c.isupper() for c in h) / max(1, sum(c.isalpha() for c in h)) > 0.55]
    schema_types = set(parser.schema_types)

    for block in parser.jsonld_blocks:
        try:
            data = json.loads(block)
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict):
                value = item.get("@type")
                if isinstance(value, list):
                    schema_types.update(str(v) for v in value)
                elif value:
                    schema_types.add(str(value))

    return {
        "title": title,
        "title_length": len(title),
        "meta_description_length": len(meta_description),
        "meta_names": sorted(parser.meta.keys()),
        "og_tag_count": sum(1 for m in parser.meta_properties if m.startswith("og:")),
        "twitter_tag_count": sum(1 for m in parser.meta_properties if m.startswith("twitter:")),
        "has_canonical": any("canonical" in rel for rel in parser.link_rels),
        "has_rss_or_atom": bool(parser.feed_links),
        "has_schema_org": bool(schema_types),
        "has_news_schema": any("newsarticle" in s.lower() or "newspaper" in s.lower() for s in schema_types),
        "schema_types": sorted(schema_types),
        "links_total": len(parser.links),
        "link_hosts": sorted(set(link_hosts)),
        "internal_links": internal_links,
        "external_links": external_links,
        "script_count": len(parser.scripts),
        "script_hosts": script_hosts,
        "stylesheet_count": len(parser.stylesheets),
        "stylesheet_hosts": stylesheet_hosts,
        "image_count": len(parser.images),
        "image_hosts": image_hosts,
        "special_links": linked_page_flags(parser.links),
        "word_count": parser.word_count,
        "heading_count": len(headings),
        "h1_count": parser.tag_counts.get("h1", 0),
        "h2_count": parser.tag_counts.get("h2", 0),
        "h3_count": parser.tag_counts.get("h3", 0),
        "avg_heading_length": round(sum(len(h) for h in headings) / len(headings), 2) if headings else 0,
        "uppercase_heading_rate": round(len(uppercase) / len(headings), 4) if headings else 0,
        "article_tag_count": parser.tag_counts.get("article", 0),
        "nav_tag_count": parser.tag_counts.get("nav", 0),
        "form_count": parser.form_count,
        "has_search": parser.search_seen,
        "amp_detected": parser.amp_seen,
    }


def domain_metadata(url):
    parsed = urllib.parse.urlparse(url)
    host = clean_host(url)
    labels = [p for p in host.split(".") if p]
    return {
        "host": host,
        "base_domain": base_domain(host),
        "scheme": parsed.scheme,
        "tld": labels[-1] if labels else "",
        "subdomain_count": max(0, len(labels) - 2),
        "domain_length": len(host),
        "hyphen_count": host.count("-"),
        "digit_count": sum(c.isdigit() for c in host),
        "path_length": len(parsed.path or ""),
        "query_length": len(parsed.query or ""),
    }


def load_sources(metadata_dir):
    rows = []
    for path in sorted(metadata_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        url = normalize_url(data.get("media link"))
        if not url:
            continue
        rows.append({
            "media_name": str(data.get("media name") or path.stem).strip(),
            "source_file": path.name,
            "url": url,
            "factuality": str(data.get("factuality", "")).upper().strip(),
            "bias": str(data.get("bias", "")).upper().strip(),
        })
    return rows


def collect_one(source, timeout):
    fetched = fetch_url(source["url"], timeout)
    headers = fetched["headers"]
    final_url = fetched["final_url"] or source["url"]
    body = fetched.pop("body")
    text = decode_body(body, headers) if body else ""
    html = summarize_html(text, final_url) if text else {}
    return {
        "media_name": source["media_name"],
        "source_file": source["source_file"],
        "requested_url": source["url"],
        "final_url": final_url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth": source["factuality"],
        "bias": source["bias"],
        "ok": fetched["ok"],
        "status_code": fetched["status_code"],
        "content_type": headers.get("Content-Type") or headers.get("content-type") or "",
        "server": headers.get("Server") or headers.get("server") or "",
        "powered_by": headers.get("X-Powered-By") or headers.get("x-powered-by") or "",
        "elapsed_ms": fetched["elapsed_ms"],
        "bytes_read": len(body),
        "truncated": fetched["truncated"],
        "redirect_count": fetched["redirect_count"],
        "fetch_error": fetched["error"],
        "requested_domain": domain_metadata(source["url"]),
        "final_domain": domain_metadata(final_url),
        "ssl": ssl_metadata(final_url, timeout),
        "html": html,
    }


def existing_keys(path):
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            keys.add(row.get("media_name") or row.get("requested_url"))
    return keys


def main():
    parser = argparse.ArgumentParser(description="Extract lightweight website metadata for media outlets.")
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    sources = load_sources(args.metadata_dir)
    if args.resume:
        seen = existing_keys(args.output)
        sources = [s for s in sources if s["media_name"] not in seen and s["url"] not in seen]
    if args.limit:
        sources = sources[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    print(f"Extracting {len(sources)} outlets -> {args.output}")

    with args.output.open(mode, encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(collect_one, source, args.timeout): source for source in sources}
            for i, future in enumerate(as_completed(futures), start=1):
                source = futures[future]
                try:
                    row = future.result()
                except Exception as e:
                    row = {
                        "media_name": source["media_name"],
                        "source_file": source["source_file"],
                        "requested_url": source["url"],
                        "ground_truth": source["factuality"],
                        "bias": source["bias"],
                        "ok": False,
                        "fetch_error": str(e),
                    }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{i}/{len(sources)} {source['media_name']} ok={row.get('ok')}")


if __name__ == "__main__":
    main()
