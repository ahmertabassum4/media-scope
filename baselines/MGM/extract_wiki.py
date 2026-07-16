"""Group C — Wikipedia features (Baly ACL'20 §3.3, Tables 2/3 row 24).

Fetch phase (network, resumable): for every outlet, find its English Wikipedia
page and cache the plaintext in data/wiki/wiki_pages.jsonl. Matching is
precision-first: search candidates by media name, then accept a candidate only
if its Wikidata official-website (P856) registered domain equals the outlet's
domain; a normalized-exact-title + media-keyword fallback catches verified name
matches without a P856 claim (disabled by --strict). Non-notable outlets stay
found=false -> zero vector, the paper's missing-media handling.

Encode phase: first 510 tokens of each found page through *pre-trained*
bert-base-uncased (paper footnote 6 - no fine-tuning), second-to-last hidden
layer, mean over tokens -> 768-d. Saves data/articles/feats_wiki.npz keyed like
the article features (task-agnostic, shared by factuality and bias).
"""

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import tldextract
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import ARTICLES_DIR, load_outlets  # noqa: E402
from extract_features import mean_wordpiece_hidden, pick_device  # noqa: E402

WIKI_DIR = ARTICLES_DIR.parent / "wiki"
CACHE = WIKI_DIR / "wiki_pages.jsonl"
FEATS_OUT = ARTICLES_DIR / "feats_wiki.npz"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "MediascopeResearch/0.1 (academic media-profiling baseline; contact: delyanhristov06@gmail.com)"
CACHE_SCHEMA = "mgm_wiki_pages_v2"
FEATURE_SCHEMA = "mgm_wiki_features_v3"
# Four workers share this throttle.  Four requests/second avoids Wikimedia's
# sustained-rate throttling while still keeping the resumable fetch practical.
REQUEST_INTERVAL_SECONDS = 0.25
MEDIA_KEYWORDS = ("news", "newspaper", "magazine", "website", "media", "publication",
                  "broadcaster", "journal", "radio", "television", "tabloid", "blog")

_local = threading.local()
_request_lock = threading.Lock()
_next_request_at = 0.0


class WikiApiError(RuntimeError):
    """A transient or API-level failure that must not be persisted as no match."""


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = USER_AGENT
    return _local.s


def throttle_requests():
    """Keep the aggregate request rate modest even with several fetch workers."""
    global _next_request_at
    with _request_lock:
        now = time.monotonic()
        due = max(now, _next_request_at)
        _next_request_at = due + REQUEST_INTERVAL_SECONDS
    if due > now:
        time.sleep(due - now)


def retry_after(response, default):
    try:
        return float(response.headers.get("Retry-After", default))
    except (TypeError, ValueError):
        return default


def api_get(url, params, tries=6, maxlag=5):
    params = dict(params, format="json")
    if maxlag is not None:
        params["maxlag"] = maxlag
    last_error = None
    for attempt in range(tries):
        try:
            throttle_requests()
            r = session().get(url, params=params, timeout=20)
            if r.status_code in (429, 503):
                last_error = WikiApiError(f"HTTP {r.status_code}")
                time.sleep(retry_after(r, 2))
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                code = data["error"].get("code", "unknown")
                last_error = WikiApiError(f"API error: {code}")
                if code == "maxlag":
                    time.sleep(retry_after(r, 5))
                    continue
                break
            return data
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(1 + attempt)
    raise WikiApiError(f"request failed after {tries} attempts: {last_error}")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def search_candidates(name, limit=5):
    queries = [name]
    if name.lower().startswith("the "):
        queries.append(name[4:])
    titles, seen = [], set()
    for q in queries:
        data = api_get(WIKI_API, {"action": "query", "list": "search", "srsearch": q, "srlimit": limit})
        for hit in data.get("query", {}).get("search", []):
            if hit["title"] not in seen:
                seen.add(hit["title"])
                titles.append(hit["title"])
    return titles


def titles_to_qids(titles):
    """Resolve titles (following redirects) to Wikidata QIDs in one call."""
    if not titles:
        return {}
    data = api_get(WIKI_API, {"action": "query", "prop": "pageprops", "ppprop": "wikibase_item",
                              "titles": "|".join(titles), "redirects": 1})
    query = data.get("query", {})
    redirect = {r["from"]: r["to"] for r in query.get("redirects", [])}
    by_title = {}
    for page in query.get("pages", {}).values():
        qid = page.get("pageprops", {}).get("wikibase_item")
        if qid and "title" in page:
            by_title[page["title"]] = qid
    return {t: by_title.get(redirect.get(t, t)) for t in titles}


def official_domains(qids):
    """QID -> set of registered domains from Wikidata P856 (official website)."""
    qids = sorted({q for q in qids if q})
    if not qids:
        return {}
    # Wikidata currently reports replica lag above the standard five-second
    # threshold. P856 is static profile metadata, so a 100-second read
    # tolerance is accurate and avoids falsely treating a temporary lag as a
    # failed outlet lookup. Wikipedia page reads retain the normal maxlag=5.
    data = api_get(
        WIKIDATA_API,
        {"action": "wbgetentities", "ids": "|".join(qids), "props": "claims"},
        maxlag=100,
    )
    out = {}
    for qid, ent in data.get("entities", {}).items():
        doms = set()
        for claim in ent.get("claims", {}).get("P856", []):
            try:
                d = tldextract.extract(claim["mainsnak"]["datavalue"]["value"]).registered_domain
                if d:
                    doms.add(d)
            except (KeyError, TypeError):
                continue
        out[qid] = doms
    return out


def page_text(title, max_words=1500):
    data = api_get(WIKI_API, {"action": "query", "prop": "extracts", "explaintext": 1,
                              "titles": title, "redirects": 1})
    for page in data.get("query", {}).get("pages", {}).values():
        words = (page.get("extract") or "").split()
        return " ".join(words[:max_words])
    return ""


def match_outlet(outlet, strict):
    """Find the outlet's Wikipedia page. Returns a cache record."""
    rec = {
        "schema": CACHE_SCHEMA,
        "strict": bool(strict),
        "key": outlet["key"],
        "media_name": outlet.get("media_name", ""),
        "url": outlet.get("url", ""),
        "split": outlet.get("split", ""),
        "found": False,
        "matched_title": None,
        "qid": None,
        "match_reason": None,
        "n_words": 0,
        "text": "",
    }
    name = str(outlet.get("media_name", "")).strip()
    if not name:
        return rec
    domain = tldextract.extract(str(outlet.get("url", ""))).registered_domain
    candidates = search_candidates(name)
    if not candidates:
        return rec
    qids = titles_to_qids(candidates)
    doms = official_domains(set(qids.values()))

    accepted, reason = None, None
    for title in candidates:  # search order = relevance order
        if domain and domain in doms.get(qids.get(title), set()):
            accepted, reason = title, "wikidata_domain"
            break
    if accepted is None and not strict:
        for title in candidates:
            if norm(title) == norm(name):
                text = page_text(title)
                if any(k in text[:1500].lower() for k in MEDIA_KEYWORDS):
                    accepted, reason = title, "name_verified"
                    rec["text"] = text
                break  # only the exact-name candidate is eligible
    if accepted is None:
        return rec

    if not rec["text"]:
        rec["text"] = page_text(accepted)
    rec.update(found=bool(rec["text"]), matched_title=accepted, qid=qids.get(accepted),
               match_reason=reason, n_words=len(rec["text"].split()),
               wiki_url="https://en.wikipedia.org/wiki/" + accepted.replace(" ", "_"))
    return rec


def load_cache():
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    cache[rec["key"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return cache


def cache_compatible(record, strict):
    """A strict and non-strict lookup must never silently share a cache row."""
    return bool(record) and record.get("schema") == CACHE_SCHEMA and bool(record.get("strict")) == bool(strict)


def selected_keys(outlets, args):
    """Apply the same deterministic --keys/--limit selection to fetch and encode."""
    keys = sorted(outlets)
    if args.keys:
        wanted = {key.strip() for key in args.keys.split(",") if key.strip()}
        unknown = sorted(wanted - set(outlets))
        if unknown:
            raise SystemExit(f"unknown outlet keys: {unknown[:5]}")
        keys = [key for key in keys if key in wanted]
    if args.limit:
        keys = keys[:args.limit]
    return keys


def cache_fingerprint(keys, cache):
    """Detect a feature archive produced from a different cached page set."""
    digest = hashlib.sha256(CACHE_SCHEMA.encode("ascii"))
    for key in keys:
        rec = cache.get(key, {})
        payload = {
            "key": key,
            "found": bool(rec.get("found")),
            "title": rec.get("matched_title"),
            "qid": rec.get("qid"),
            "reason": rec.get("match_reason"),
            "text": rec.get("text", ""),
        }
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8"))
    return digest.hexdigest()


def wiki_text_hash(text):
    normalized = " ".join(str(text or "").split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def select_wiki_pages(keys, cache):
    """Keep one test-preferred copy when multiple outlets map to one wiki page."""
    records = []
    for key in keys:
        rec = cache.get(key, {})
        if not rec.get("found"):
            continue
        records.append((
            key,
            rec.get("split", ""),
            rec.get("qid"),
            norm(rec.get("matched_title", "")),
            wiki_text_hash(rec.get("text", "")),
        ))

    parent = list(range(len(records)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left, right):
        left, right = root(left), root(right)
        if left != right:
            parent[right] = left

    first_by_identifier = {}
    for index, (_, _, qid, title, text_hash) in enumerate(records):
        for identifier in (("qid", qid), ("title", title), ("text", text_hash)):
            if not identifier[1]:
                continue
            previous = first_by_identifier.setdefault(identifier, index)
            join(index, previous)

    components = {}
    for index in range(len(records)):
        components.setdefault(root(index), []).append(index)

    selected = set()
    duplicate_components = 0
    duplicate_pages_removed = 0
    for component in components.values():
        if len(component) > 1:
            duplicate_components += 1
        winner = min(
            component,
            key=lambda index: (0 if records[index][1] == "test" else 1, records[index][0]),
        )
        selected.add(records[winner][0])
        duplicate_pages_removed += len(component) - 1
    return sorted(selected), {
        "found_pages": len(records),
        "duplicate_components": duplicate_components,
        "duplicate_pages_removed": duplicate_pages_removed,
        "retained_pages": len(selected),
    }


def fetch(outlets, args):
    cache = load_cache()
    keys = selected_keys(outlets, args)
    work = [
        outlets[key] for key in keys
        if args.refresh or not cache_compatible(cache.get(key), args.strict)
    ]
    cached = len(keys) - len(work)
    print(f"wiki fetch: {cached}/{len(keys)} compatible cached, {len(work)} to fetch "
          f"({args.workers} workers, strict={args.strict})")
    if not work:
        return
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    failures = 0
    with CACHE.open("a") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(match_outlet, o, args.strict) for o in work]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="outlet", desc="Wikipedia"):
            try:
                rec = fut.result()
            except Exception:  # noqa: BLE001 - one outlet must not kill the fetch
                failures += 1
                continue
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
    if failures:
        print(f"wiki fetch: {failures} API failures were not cached; rerun to retry them")


@torch.no_grad()
def encode(outlets, args):
    cache = load_cache()
    keys = selected_keys(outlets, args)
    full_dataset = keys == sorted(outlets)
    missing = [key for key in keys if not cache_compatible(cache.get(key), args.strict)]
    if missing and not args.force_partial:
        raise SystemExit(f"{len(missing)} selected outlets are not fetched with strict={args.strict} "
                         f"(e.g. {missing[:5]}) — finish the fetch or pass --force-partial")
    found, dedup_summary = select_wiki_pages(
        [key for key in keys if cache_compatible(cache.get(key), args.strict)], cache)
    print(f"wiki encode: {len(keys)} outlets, {len(found)} retained pages, "
          f"{len(keys) - len(found)} zero vectors")
    print(f"wiki duplicate filter: {dedup_summary}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    device = pick_device()
    model.to(device).eval()

    X = np.zeros((len(keys), model.config.hidden_size), dtype=np.float32)
    pos = {k: i for i, k in enumerate(keys)}
    for start in tqdm(range(0, len(found), args.batch_size), unit="batch", desc="BERT encode"):
        chunk = found[start:start + args.batch_size]
        batch = tokenizer([cache[k]["text"] for k in chunk], truncation=True, max_length=512,
                          padding=True, return_special_tokens_mask=True, return_tensors="pt")
        special_tokens_mask = batch.pop("special_tokens_mask").to(device)
        batch = batch.to(device)
        out = model(**batch, output_hidden_states=True)
        hs = out.hidden_states[-2]                       # second-to-last layer
        vecs = mean_wordpiece_hidden(
            hs, batch["attention_mask"], special_tokens_mask).float().cpu().numpy()
        for k, v in zip(chunk, vecs):
            X[pos[k]] = v
    out_path = Path(args.out) if args.out else FEATS_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        keys=np.array(keys),
        wiki=X,
        wiki_feature_schema=np.array(FEATURE_SCHEMA),
        wiki_cache_schema=np.array(CACHE_SCHEMA),
        wiki_cache_fingerprint=np.array(cache_fingerprint(keys, cache)),
        wiki_strict=np.array(bool(args.strict)),
        # A smoke/subset archive must never be accepted by train_svm.py as the
        # full Group-C feature matrix.
        wiki_complete=np.array(full_dataset and not missing),
        wiki_dedup_summary=np.array(json.dumps(dedup_summary, sort_keys=True)),
    )
    print(f"saved {out_path}  wiki{X.shape}  zero rows={int((np.abs(X).sum(axis=1) == 0).sum())}")


def report(outlets, strict):
    cache = load_cache()
    total = len(outlets)
    recs = [cache[k] for k in outlets if cache_compatible(cache.get(k), strict)]
    found = [r for r in recs if r.get("found")]
    print(f"wiki cache: {len(recs)}/{total} outlets cached for strict={strict}, {len(found)} pages found "
          f"({len(found) / max(1, len(recs)):.1%} of fetched)")
    for split in ("train", "test"):
        sub = [r for r in recs if r.get("split") == split]
        hit = sum(1 for r in sub if r.get("found"))
        print(f"  {split:5s}: {hit}/{len(sub)} found")
    reasons = {}
    for r in found:
        reasons[r.get("match_reason")] = reasons.get(r.get("match_reason"), 0) + 1
    print(f"  match_reason: {reasons}")
    if found:
        w = [r["n_words"] for r in found]
        print(f"  page words: median={sorted(w)[len(w)//2]}  min={min(w)}  max={max(w)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--encode-only", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap outlets to fetch or encode (smoke tests)")
    ap.add_argument("--keys", default="", help="comma-separated outlet keys to fetch or encode (smoke tests)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--strict", action="store_true", help="accept Wikidata P856 domain matches only")
    ap.add_argument("--refresh", action="store_true", help="refetch selected outlets even when cached")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="", help="override feats_wiki.npz path (smoke tests)")
    ap.add_argument("--force-partial", action="store_true",
                    help="encode even if some outlets are not fetched yet (missing -> zero vectors)")
    args = ap.parse_args()
    if args.fetch_only and args.encode_only:
        raise SystemExit("--fetch-only and --encode-only cannot be used together")

    outlets = load_outlets()
    if args.report:
        report(outlets, args.strict)
        return
    if not args.encode_only:
        fetch(outlets, args)
        report(outlets, args.strict)
    if not args.fetch_only:
        encode(outlets, args)


if __name__ == "__main__":
    main()
