"""Shared plumbing for the Multi-View Media Profiling Suite reimplementation.

Adapts "A Multi-View Media Profiling Suite" (arXiv:2605.01336v1) to our
Mediascope-2800 train/test split. Five views per outlet:

- F(a) Alexa audience-overlap graph  (frozen 2022 crawl, shipped in MGM_code)
- F(h) Hyperlink (on-site) graph     (shipped in multi-graph-perspective)
- F(l) LLM similarity graph          (regenerated with the paper's prompt)
- F(t) media articles                (our scraped bags, DistilBERT embeddings)
- F(w) Wikipedia descriptions        (our wiki pages, DistilBERT embeddings)

Outlet universe, labels, article dedup, and the metrics row are reused from
baselines/MGM/pipeline.py so every benchmark row shares one definition.
"""

import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

MGM_DIR = Path(__file__).resolve().parents[1] / "MGM"
sys.path.insert(0, str(MGM_DIR))
import pipeline as mgm_pipeline  # noqa: E402
from dataset import (  # noqa: E402  (src/ is on sys.path via pipeline import)
    ROOT,
    TEST_CSV,
    TRAIN_CSV,
    normalize_key,
)

# Re-export the shared Group A/C contracts used by the command-line modules.
ARTICLES_DIR = mgm_pipeline.ARTICLES_DIR
ARTICLES_JSONL = mgm_pipeline.ARTICLES_JSONL
BIAS_MAP = mgm_pipeline.BIAS_MAP
RESULTS_DIR = mgm_pipeline.RESULTS_DIR
TASKS = mgm_pipeline.TASKS
build_content_filter = mgm_pipeline.build_content_filter
filtered_articles = mgm_pipeline.filtered_articles
outlet_label = mgm_pipeline.outlet_label
scores = mgm_pipeline.scores

GRAPHS_DIR = ROOT / "data" / "graphs"
METADATA_DIR = ROOT / "data" / "metadata"
MBFC_SOURCES_JSON = METADATA_DIR / "mbfc_sources_v5.json"
MGM_CODE_REV = "8831f3fb15cf69459549b637ca5f4863b02ec621"
MULTI_GRAPH_REV = "9ad577e413bd9b8318982295b143d24558c6fb92"
MBFC_EXT_REV = "d13aa536bef9db1c1676a49b8439ce66630fea83"
MBFC_SOURCES_URL = (
    "https://raw.githubusercontent.com/drmikecrowe/mbfcext/"
    f"{MBFC_EXT_REV}/docs/v5/data/sources.json"
)

# Upstream data files (verified July 2026). The Alexa graph is the frozen
# pre-shutdown Panayotov'22 crawl republished in the MGM (NAACL'25) repo; the
# hyperlink graph ships in the Multi-View suite repo itself.
MGM_CODE_RAW = (
    "https://raw.githubusercontent.com/marslanm/MGM_code/"
    f"{MGM_CODE_REV}/data/NMP"
)
MULTI_GRAPH_RAW = (
    "https://raw.githubusercontent.com/marslanm/multi-graph-perspective/"
    f"{MULTI_GRAPH_REV}/onsite_data"
)
ALEXA_EDGES_URL = f"{MGM_CODE_RAW}/Edge_level_3.csv"
ALEXA_NODES_URL = f"{MGM_CODE_RAW}/Fact/ACL_level_3_fact.csv"
HYPERLINK_GRAPH_ZIP_URL = f"{MULTI_GRAPH_RAW}/MBFC-2023/graph.zip"
HYPERLINK_NAME2ID_URL = f"{MULTI_GRAPH_RAW}/MBFC-2023/name2id.json"

SOURCE_SHA256 = {
    "alexa_edges": "084c01478895f7bb482053a7fbe947248e220e77acfd9e4f0e4fb9eec405fa0c",
    "alexa_nodes": "8e2dae891d45bfa66d44d51216abdf8711c75cc7515379003e3c11969cd38c7a",
    "hyperlink_graph": "3beaa3882870238b1b9dd6e59e52889472d2586e976849f64e004e003ea17e72",
    "hyperlink_name2id": "8704241c58c5db8489598f49f18dd05516681c12d49cb20b3b78d9f7385081bf",
    "hyperlink_edges": "1469a77905ab24c2245cb3abe7d1a7427561392d3afd02a1a85052d66d80e363",
    "mbfc_sources": "b561c21bc74690b17b41f4448a5cbaebf1bedf204f0755bfa61e1afd47a44349",
}

# The five Alexa node attributes their dataset.py feeds the GNN, in order.
ALEXA_FEATURE_COLUMNS = [
    "daily_pageviews_per_visitor",
    "daily_time_on_site",
    "bounce_rate",
    "normalized_alexa_rank",
    "normalized_total_sites_linked_in",
]

GRAPH_VIEWS = ("alexa", "hyperlink", "llm")
TEXT_VIEWS = ("articles", "wiki")
EMB_DIM_GNN = 64  # paper: 64-dimensional graph embeddings
URL_MATCH_PREFIX = "@url:"
HOST_MATCH_PREFIX = "@host:"
DOMAIN_FALLBACK_PREFIX = "@registered:"


def _tldextract():
    import tldextract
    # Offline suffix list: deterministic and no network call at import time.
    return tldextract.TLDExtract(suffix_list_urls=())


_EXTRACT = None


def normalized_host(value):
    """Lowercase hostname without a leading ``www.`` or port."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    if "://" not in text:
        text = "//" + text
    try:
        host = urlsplit(text).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return None
    return host if re.fullmatch(r"[a-z0-9.-]+", host) else None


def normalized_url(value):
    """Canonical HTTP(S) URL for exact graph-node matching, or ``None``."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = urlsplit(text)
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.lower().rstrip(".")
    try:
        host = host.encode("idna").decode("ascii")
        port = parsed.port
    except (UnicodeError, ValueError):
        return None
    if not re.fullmatch(r"[a-z0-9.-]+", host):
        return None
    authority = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        authority += f":{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "").rstrip("/")
    return f"{scheme}://{authority}{path}"


def registered_domain(value):
    """Registered domain ('nytimes.com') for a URL or bare host, or None."""
    global _EXTRACT
    host = normalized_host(value)
    if not host:
        return None
    if _EXTRACT is None:
        _EXTRACT = _tldextract()
    parts = _EXTRACT(host)
    if parts.domain and parts.suffix:
        return f"{parts.domain}.{parts.suffix}"
    return host or None


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path, payload):
    """Replace ``path`` only after all bytes are durable on the same volume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def atomic_write_text(path, text):
    atomic_write_bytes(path, str(text).encode("utf-8"))


@contextmanager
def exclusive_file_lock(path):
    """Fail fast when another process already owns the same long-running job."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            owner = handle.read().strip() or "unknown process"
            raise RuntimeError(f"another run holds {path} ({owner})") from None
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def download(url, dest, expect_min_bytes=100, expected_sha256=None):
    """Fetch a pinned URL atomically and verify its expected SHA-256."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= expect_min_bytes:
        if expected_sha256 is None or sha256_file(dest) == expected_sha256:
            return dest
    req = urllib.request.Request(url, headers={"User-Agent": "ugrip-multiview/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()
    if payload.startswith(b"version https://git-lfs.github.com"):
        meta = dict(
            line.split(" ", 1) for line in payload.decode().strip().splitlines()
        )
        oid = meta["oid"].split(":", 1)[1].strip()
        size = meta["size"].strip()
        lfs_url = url.replace(
            "https://raw.githubusercontent.com/", "https://media.githubusercontent.com/media/"
        )
        req = urllib.request.Request(lfs_url, headers={"User-Agent": "ugrip-multiview/1.0"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = resp.read()
        if str(len(payload)) != size:
            raise IOError(f"LFS download size mismatch for {url}: {len(payload)} != {size} (oid {oid[:12]})")
    if len(payload) < expect_min_bytes:
        raise IOError(f"suspiciously small download from {url}: {len(payload)} bytes")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise IOError(
            f"SHA-256 mismatch for {url}: {actual_sha256} != {expected_sha256}"
        )
    atomic_write_bytes(dest, payload)
    return dest


def load_mbfc_sources(path=MBFC_SOURCES_JSON):
    """domain -> source record from the free mbfcext v5 dump (downloads once)."""
    download(
        MBFC_SOURCES_URL, path, expect_min_bytes=100_000,
        expected_sha256=SOURCE_SHA256["mbfc_sources"],
    )
    raw = json.loads(Path(path).read_text())
    records = raw.get("sources", raw) if isinstance(raw, dict) else raw
    if isinstance(records, dict):
        records = list(records.values())
    by_domain = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        dom = normalized_host(rec.get("domain") or rec.get("url") or "")
        if dom:
            by_domain.setdefault(dom, rec)
    return by_domain


def _normalize_name(name):
    return " ".join(re.findall(r"[a-z0-9]+", str(name or "").lower()))


def candidate_exact_hosts(outlet, mbfc_by_name=None):
    """Ordered exact hostnames from the outlet record and MBFC alias."""
    values = [outlet.get("url", "")]
    if mbfc_by_name:
        rec = mbfc_by_name.get(_normalize_name(outlet.get("media_name")))
        if rec:
            values.append(rec.get("domain") or rec.get("url") or "")
    hosts = []
    for value in values:
        host = normalized_host(value)
        if host and host not in hosts:
            hosts.append(host)
    return tuple(hosts)


def candidate_node_keys(outlet, mbfc_by_name=None):
    """Namespaced URL, host, and registered-domain graph lookup keys."""
    values = [outlet.get("url", "")]
    if mbfc_by_name:
        rec = mbfc_by_name.get(_normalize_name(outlet.get("media_name")))
        if rec:
            values.append(rec.get("domain") or rec.get("url") or "")
    keys = []
    for value in values:
        url = normalized_url(value)
        host = normalized_host(value)
        domain = registered_domain(value)
        for key in (
            f"{URL_MATCH_PREFIX}{url}" if url else None,
            f"{HOST_MATCH_PREFIX}{host}" if host else None,
            f"{DOMAIN_FALLBACK_PREFIX}{domain}" if domain else None,
        ):
            if key and key not in keys:
                keys.append(key)
    return tuple(keys)


def mbfc_name_index(mbfc_by_domain):
    index = {}
    for rec in mbfc_by_domain.values():
        name = _normalize_name(rec.get("name"))
        if name:
            index.setdefault(name, rec)
    return index


def unique_node_host_index(name2id):
    """Host -> node id, excluding ambiguous hostname/domain fallbacks.

    Hosted sites such as ``foo.wordpress.com`` and ``bar.wordpress.com`` must
    not collapse to an arbitrary ``wordpress.com`` node. Exact host matches
    are retained; registered-domain fallbacks are added only when they resolve
    to exactly one node in the graph.
    """
    url_ids = defaultdict(set)
    host_ids = defaultdict(set)
    domain_ids = defaultdict(set)
    for name, node_id in name2id.items():
        node_id = int(node_id)
        url = normalized_url(name)
        host = normalized_host(name)
        domain = registered_domain(name)
        if url:
            url_ids[url].add(node_id)
        if host:
            host_ids[host].add(node_id)
        if domain:
            domain_ids[domain].add(node_id)
    index = {}
    for prefix, groups in (
        (URL_MATCH_PREFIX, url_ids),
        (HOST_MATCH_PREFIX, host_ids),
        (DOMAIN_FALLBACK_PREFIX, domain_ids),
    ):
        index.update({
            f"{prefix}{value}": next(iter(node_ids))
            for value, node_ids in groups.items() if len(node_ids) == 1
        })
    return index


def match_outlets_to_nodes(outlets, node_domain_to_id, mbfc_by_name=None):
    """Outlet key -> graph node id via ordered, unambiguous host matching."""
    matches = {}
    for key in sorted(outlets):
        for dom in candidate_node_keys(outlets[key], mbfc_by_name):
            node_id = node_domain_to_id.get(dom)
            if node_id is not None:
                matches[key] = int(node_id)
                break
    return matches


def deduplicate_node_matches(matches, outlets):
    """Keep one outlet per graph node, preferring held-out test outlets."""
    by_node = defaultdict(list)
    for key, node_id in matches.items():
        by_node[int(node_id)].append(key)

    def rank(key):
        outlet = outlets[key]
        split_rank = 0 if outlet.get("split") == "test" else 1
        n_labels = sum(outlet_label(outlet, task) is not None for task in TASKS)
        return split_rank, -n_labels, key

    clean = {}
    dropped = []
    for node_id, keys in by_node.items():
        winner = min(keys, key=rank)
        clean[winner] = node_id
        dropped.extend({"key": key, "kept": winner, "node_id": node_id}
                       for key in keys if key != winner)
    return clean, sorted(dropped, key=lambda row: (row["node_id"], row["key"]))


def match_path(view):
    return GRAPHS_DIR / f"graph_match_{view}.json"


def emb_path(view, encoder=None, task=None):
    """Canonical embedding archive path for one view."""
    if view in GRAPH_VIEWS:
        if not encoder:
            raise ValueError("graph views need an encoder name")
        return GRAPHS_DIR / f"emb_{view}_{encoder}.npz"
    if view in TEXT_VIEWS:
        if not task:
            raise ValueError("text views need a task name")
        return GRAPHS_DIR / f"emb_{view}_{task}.npz"
    raise ValueError(f"unknown view {view!r}")


def save_embeddings(path, keys, X, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(keys)
    X = np.asarray(X, dtype=np.float32)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate embedding keys")
    if X.ndim != 2 or X.shape[0] != len(keys):
        raise ValueError(f"{path}: embedding shape does not match keys")
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: non-finite embeddings")
    arrays = {"keys": np.array(keys), "X": X}
    for name, value in metadata.items():
        if name in arrays:
            raise ValueError(f"{path}: metadata cannot replace reserved field {name!r}")
        array = np.array(value)
        if array.dtype.kind == "O":
            raise ValueError(f"{path}: metadata {name!r} cannot use object arrays")
        arrays[name] = array
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def load_embeddings(path, expected_keys=None, expected_metadata=None):
    with np.load(path, allow_pickle=False) as data:
        if "keys" not in data.files or "X" not in data.files:
            raise ValueError(f"{path}: missing keys or X array")
        if data["keys"].dtype.kind not in "SU":
            raise ValueError(f"{path}: embedding keys must be strings")
        keys = [str(k) for k in data["keys"].tolist()]
        X = np.asarray(data["X"], dtype=np.float32)
        metadata = {
            name: data[name].item() if data[name].ndim == 0 else data[name].tolist()
            for name in data.files if name not in {"keys", "X"}
        }
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate embedding keys")
    if X.ndim != 2 or X.shape[0] != len(keys):
        raise ValueError(f"{path}: malformed embedding archive")
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: non-finite embeddings")
    if expected_keys is not None and keys != list(expected_keys):
        raise ValueError(f"{path}: outlet keys do not match the current universe")
    for name, expected in (expected_metadata or {}).items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"{path}: metadata {name!r} is {metadata.get(name)!r}, expected {expected!r}"
            )
    return keys, X


def deduplicate_embedding_rows(keys, X, outlets, task=None):
    """Zero repeated non-missing vectors, keeping one deterministic outlet."""
    matrix = np.asarray(X, dtype=np.float32).copy()
    groups = defaultdict(list)
    for index, (key, row) in enumerate(zip(keys, matrix)):
        if np.any(row):
            groups[np.ascontiguousarray(row).tobytes()].append((index, key))

    def rank(item):
        _, key = item
        outlet = outlets[key]
        split_rank = 0 if outlet.get("split") == "test" else 1
        labeled = outlet_label(outlet, task) is not None if task else any(
            outlet_label(outlet, name) is not None for name in TASKS
        )
        return split_rank, 0 if labeled else 1, key

    dropped = []
    for items in groups.values():
        if len(items) < 2:
            continue
        winner_index, winner_key = min(items, key=rank)
        for index, key in items:
            if index != winner_index:
                matrix[index] = 0.0
                dropped.append({"key": key, "kept": winner_key})
    return matrix, sorted(dropped, key=lambda row: row["key"])


def outlet_universe_sha256(outlets):
    rows = [
        (key, outlet.get("split"), outlet_label(outlet, "factuality"),
         outlet_label(outlet, "bias"))
        for key, outlet in sorted(outlets.items())
    ]
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_outlets():
    """Article records restricted to the canonical cleaned OCR outlet split."""
    article_outlets = mgm_pipeline.load_outlets()
    canonical = {}
    missing_articles = []
    for split, path in (("train", TRAIN_CSV), ("test", TEST_CSV)):
        with Path(path).open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = normalize_key(Path(str(row.get("image_path", ""))).name)
                if key in canonical:
                    raise ValueError(f"duplicate canonical outlet key {key!r}")
                if key not in article_outlets:
                    missing_articles.append(key)
                    continue
                record = dict(article_outlets[key])
                record.update(row)
                record["key"] = key
                record["split"] = split
                canonical[key] = record
    if missing_articles:
        raise ValueError(
            f"articles.jsonl lacks {len(missing_articles)} canonical outlets, "
            f"e.g. {missing_articles[:10]}"
        )
    return canonical


def task_split(outlets, task):
    """(train_keys, test_keys) of labeled outlets, sorted."""
    train = sorted(k for k, o in outlets.items()
                   if o.get("split") == "train" and outlet_label(o, task))
    test = sorted(k for k, o in outlets.items()
                  if o.get("split") == "test" and outlet_label(o, task))
    return train, test
