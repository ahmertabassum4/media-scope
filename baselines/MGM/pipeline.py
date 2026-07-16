"""Shared pieces of the Baly ACL'20 Group-A pipeline (articles features).

- loads the scraped article bags (deduplicated by outlet key),
- maps outlet labels for the two tasks (factuality 3-way, bias 5-level),
- builds the leakage-safe role split: train outlets are divided, stratified by
  the task label, into a `bert_tune` slice (fine-tunes BERT) and a disjoint
  `svm_train` slice (trains the SVM); the given test split stays untouched,
- provides the metrics row used by all result CSVs (same columns as
  src/metrics.py plus per-class F1 for the task's own classes).
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dataset import ROOT  # noqa: E402

ARTICLES_DIR = ROOT / "data" / "articles"
ARTICLES_JSONL = ARTICLES_DIR / "articles.jsonl"
MODELS_DIR = Path(__file__).resolve().parent / "models"
RESULTS_DIR = ROOT / "results" / "current"
DEFAULT_BERT_FRAC = 1 / 3
DEFAULT_ROLE_SEED = 42
ROLE_SCHEMA = "mgm_group_a_roles_v2"
CONTENT_FILTER_SCHEMA = "mgm_global_cross_outlet_dedup_v4"
MIN_SHARED_BODY_PREFIX_TOKENS = 80

# Query parameters that identify a visit rather than a distinct article.
TRACKING_QUERY_PARAMETERS = frozenset({
    "_ga", "_gl", "dclid", "fbclid", "gclid", "igshid", "mc_cid",
    "mc_eid", "msclkid", "yclid",
})

# extreme left/right merged into left/right (user's choice); UNRATED dropped
BIAS_MAP = {
    "LEFT": "left", "EXTREME LEFT": "left",
    "LEFT-CENTER": "left-center",
    "LEAST BIASED": "center",
    "RIGHT-CENTER": "right-center",
    "RIGHT": "right", "EXTREME RIGHT": "right",
}

TASKS = {
    "factuality": {
        "classes": ["LOW", "MIXED", "HIGH"],
        "ordinal": {"LOW": -1.0, "MIXED": 0.0, "HIGH": 1.0},
        # the paper fine-tunes BERT binary low-vs-high (footnote 8); MIXED
        # outlets still get features and are classified by the 3-way SVM
        "bert_classes": ["LOW", "HIGH"],
    },
    "bias": {
        "classes": ["left", "left-center", "center", "right-center", "right"],
        "ordinal": {"left": -2.0, "left-center": -1.0, "center": 0.0,
                    "right-center": 1.0, "right": 2.0},
        "bert_classes": ["left", "left-center", "center", "right-center", "right"],
    },
}


def outlet_label(outlet, task):
    """Task label for an outlet record, or None if the outlet has no label."""
    if task == "factuality":
        label = str(outlet.get("label_3class", "")).upper()
        return label if label in TASKS["factuality"]["classes"] else None
    return BIAS_MAP.get(str(outlet.get("bias_rating", "")).strip().upper())


def load_outlets(path=ARTICLES_JSONL):
    """articles.jsonl deduplicated by key: keep the record with most articles."""
    outlets = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec["key"]
            if key not in outlets or rec["n_articles"] > outlets[key]["n_articles"]:
                outlets[key] = rec
    return outlets


def roles_path(task):
    return ARTICLES_DIR / f"roles_{task}.json"


def eligible_outlet_keys(task, outlets, split):
    """Sorted outlet keys with a usable label for one source split."""
    return sorted(k for k, outlet in outlets.items()
                  if outlet.get("split") == split and outlet_label(outlet, task))


def outlet_fingerprint(task, outlets):
    """Fingerprint the labeled outlet universe used to construct role splits."""
    rows = [(key, str(outlet.get("split", "")), outlet_label(outlet, task))
            for key, outlet in sorted(outlets.items())
            if outlet_label(outlet, task)]
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_roles(task, outlets, bert_frac=DEFAULT_BERT_FRAC, seed=DEFAULT_ROLE_SEED):
    """Stratified, disjoint bert_tune/svm_train split of the train outlets."""
    if not 0.0 < bert_frac < 1.0:
        raise ValueError(f"bert_frac must be between 0 and 1, got {bert_frac}")
    train_keys = eligible_outlet_keys(task, outlets, "train")
    test_keys = eligible_outlet_keys(task, outlets, "test")
    labels = [outlet_label(outlets[k], task) for k in train_keys]
    try:
        bert_keys, svm_keys = train_test_split(
            train_keys, train_size=bert_frac, stratify=labels, random_state=seed)
    except ValueError as exc:
        raise ValueError(f"cannot build stratified roles for {task}: {exc}") from exc
    return {"schema": ROLE_SCHEMA, "task": task, "seed": seed, "bert_frac": bert_frac,
             "outlet_fingerprint": outlet_fingerprint(task, outlets),
             "bert_tune": sorted(bert_keys), "svm_train": sorted(svm_keys),
             "test": test_keys}


def make_roles(task, outlets, bert_frac=DEFAULT_BERT_FRAC, seed=DEFAULT_ROLE_SEED):
    roles = build_roles(task, outlets, bert_frac=bert_frac, seed=seed)
    roles_path(task).write_text(json.dumps(roles, indent=1))
    return roles


def validate_roles(task, roles, outlets):
    """Fail closed unless role membership exactly matches the current dataset."""
    names = ("bert_tune", "svm_train", "test")
    if any(name not in roles for name in names):
        raise ValueError(f"roles for {task} are missing a required partition")
    if roles.get("task") != task:
        raise ValueError(f"roles task is {roles.get('task')!r}, expected {task!r}")

    partitions = {name: list(roles[name]) for name in names}
    for name, keys in partitions.items():
        if len(keys) != len(set(keys)):
            raise ValueError(f"roles for {task} contain duplicate keys in {name}")
    bert, svm, test = map(set, (partitions["bert_tune"], partitions["svm_train"], partitions["test"]))
    if bert & svm or bert & test or svm & test:
        raise ValueError(f"roles for {task} overlap between partitions")

    all_keys = bert | svm | test
    missing = sorted(all_keys - set(outlets))
    if missing:
        raise ValueError(f"roles for {task} reference missing outlets, e.g. {missing[:5]}")
    wrong_train = sorted(k for k in bert | svm if outlets[k].get("split") != "train")
    wrong_test = sorted(k for k in test if outlets[k].get("split") != "test")
    if wrong_train or wrong_test:
        raise ValueError(f"roles for {task} contain keys from the wrong source split")

    expected_train = set(eligible_outlet_keys(task, outlets, "train"))
    expected_test = set(eligible_outlet_keys(task, outlets, "test"))
    if bert | svm != expected_train or test != expected_test:
        raise ValueError(f"roles for {task} do not exactly cover the current labeled outlets")
    return True


def _roles_need_refresh(task, roles, outlets, bert_frac, seed):
    if roles.get("schema") != ROLE_SCHEMA:
        return "role schema changed"
    if roles.get("task") != task:
        return "task changed"
    if roles.get("seed") != seed:
        return "seed changed"
    try:
        if not math.isclose(float(roles.get("bert_frac")), bert_frac, rel_tol=0.0, abs_tol=1e-12):
            return "BERT fraction changed"
    except (TypeError, ValueError):
        return "BERT fraction is invalid"
    if roles.get("outlet_fingerprint") != outlet_fingerprint(task, outlets):
        return "labeled outlet set changed"
    try:
        validate_roles(task, roles, outlets)
    except ValueError as exc:
        return str(exc)
    return None


def load_roles(task, outlets=None, bert_frac=DEFAULT_BERT_FRAC, seed=DEFAULT_ROLE_SEED):
    if outlets is None:
        outlets = load_outlets()
    path = roles_path(task)
    roles = None
    if path.exists():
        try:
            roles = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"refreshing {path}: invalid JSON")
    reason = _roles_need_refresh(task, roles, outlets, bert_frac, seed) if roles else "roles missing"
    if reason:
        print(f"refreshing {path}: {reason}")
        roles = make_roles(task, outlets, bert_frac=bert_frac, seed=seed)
    validate_roles(task, roles, outlets)
    return roles


def fine_tune_outlet_keys(task, roles, outlets):
    """BERT-tune outlets that have articles and a BERT-supervision label."""
    validate_roles(task, roles, outlets)
    allowed_labels = set(TASKS[task]["bert_classes"])
    return sorted(key for key in roles["bert_tune"]
                  if outlet_label(outlets[key], task) in allowed_labels
                  and outlets[key].get("articles"))


def split_fine_tune_outlets(task, roles, outlets, val_frac=0.1, seed=42):
    """Split BERT supervision at outlet granularity, never article granularity."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be between 0 and 1, got {val_frac}")
    keys = fine_tune_outlet_keys(task, roles, outlets)
    labels = [outlet_label(outlets[key], task) for key in keys]
    try:
        train_keys, val_keys = train_test_split(
            keys, test_size=val_frac, stratify=labels, random_state=seed)
    except ValueError as exc:
        raise ValueError(f"cannot make an outlet-level validation split for {task}: {exc}") from exc
    train_keys, val_keys = sorted(train_keys), sorted(val_keys)
    if set(train_keys) & set(val_keys):
        raise AssertionError("BERT train/validation outlet leakage")
    forbidden = set(roles["svm_train"]) | set(roles["test"])
    if (set(train_keys) | set(val_keys)) & forbidden:
        raise AssertionError("BERT supervision overlaps SVM or test outlets")
    return train_keys, val_keys


def normalize_article_text(text):
    """Normalize insignificant formatting differences before deduplication."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def article_text_hash(text):
    """Stable content hash after case, whitespace, Unicode, and punctuation cleanup."""
    normalized = normalize_article_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def article_body_prefix_hash(text, min_tokens=MIN_SHARED_BODY_PREFIX_TOKENS):
    """Hash a long article-body prefix to catch copies with changed headlines."""
    lines = str(text or "").splitlines()
    body = "\n".join(lines[1:]) if len(lines) > 1 else str(text or "")
    tokens = normalize_article_text(body).split()
    if len(tokens) < min_tokens:
        return None
    prefix = " ".join(tokens[:min_tokens])
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def article_url_key(url):
    """Canonical URL key used to detect a page despite tracking parameters."""
    value = str(url or "").strip()
    if not value:
        return None
    parsed = urlsplit(value)
    if not parsed.netloc and parsed.path:
        parsed = urlsplit(f"//{value}")
    netloc = parsed.netloc.casefold()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = re.sub(r"/{2,}", "/", parsed.path or "/").rstrip("/") or "/"
    query_pairs = [
        (key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_")
        and key.casefold() not in TRACKING_QUERY_PARAMETERS
    ]
    return urlunsplit(("", netloc, path,
                       urlencode(sorted(query_pairs)), ""))


def build_content_filter(outlets):
    """Keep one canonical copy of each high-confidence duplicate article.

    Records are connected when their normalized text, canonical URL, or first
    80 normalized body words match. One record is retained per connected
    duplicate group. A test-split record is preferred over a train-split record
    so no test content can remain in training. Ties are resolved by outlet key
    and original article order. The raw scrape is never modified; callers must
    obtain texts through ``filtered_articles``.
    """
    records = []
    text_outlets = defaultdict(set)
    url_outlets = defaultdict(set)
    body_prefix_outlets = defaultdict(set)

    for outlet_key in sorted(outlets):
        for index, article in enumerate(outlets[outlet_key].get("articles", [])):
            text = str(article.get("text", "") or "").strip()
            if not text:
                continue
            text_hash = article_text_hash(text)
            url_key = article_url_key(article.get("url", ""))
            body_prefix_hash = article_body_prefix_hash(text)
            records.append((outlet_key, index, text_hash, url_key, body_prefix_hash))
            text_outlets[text_hash].add(outlet_key)
            if url_key:
                url_outlets[url_key].add(outlet_key)
            if body_prefix_hash:
                body_prefix_outlets[body_prefix_hash].add(outlet_key)

    shared_texts = {value for value, keys in text_outlets.items() if len(keys) > 1}
    shared_urls = {value for value, keys in url_outlets.items() if len(keys) > 1}
    shared_body_prefixes = {
        value for value, keys in body_prefix_outlets.items() if len(keys) > 1
    }

    # Join records that share any high-confidence article identifier. This
    # avoids retaining two copies through a transitive text/URL/prefix match.
    parent = list(range(len(records)))

    def root(record_index):
        while parent[record_index] != record_index:
            parent[record_index] = parent[parent[record_index]]
            record_index = parent[record_index]
        return record_index

    def join(left, right):
        left, right = root(left), root(right)
        if left != right:
            parent[right] = left

    first_by_identifier = {}
    for record_index, (_, _, text_hash, url_key, body_prefix_hash) in enumerate(records):
        for identifier in (("text", text_hash), ("url", url_key), ("body_prefix", body_prefix_hash)):
            if not identifier[1]:
                continue
            previous = first_by_identifier.setdefault(identifier, record_index)
            join(record_index, previous)

    components = defaultdict(list)
    for record_index in range(len(records)):
        components[root(record_index)].append(record_index)

    excluded_indices = defaultdict(set)
    cross_outlet_removed = 0
    within_outlet_removed = 0
    cross_outlet_components = 0
    for component in components.values():
        if len(component) == 1:
            continue
        component_outlets = {records[record_index][0] for record_index in component}
        if len(component_outlets) > 1:
            cross_outlet_components += 1

        def retention_rank(record_index):
            outlet_key, article_index, _, _, _ = records[record_index]
            # Evaluation data wins any tie with training data. This is what
            # prevents a kept test article from also influencing the model.
            split_rank = 0 if outlets[outlet_key].get("split") == "test" else 1
            return split_rank, outlet_key, article_index

        winner = min(component, key=retention_rank)
        for record_index in component:
            if record_index == winner:
                continue
            outlet_key, article_index, _, _, _ = records[record_index]
            excluded_indices[outlet_key].add(article_index)
            if len(component_outlets) > 1:
                cross_outlet_removed += 1
            else:
                within_outlet_removed += 1

    digest = hashlib.sha256(CONTENT_FILTER_SCHEMA.encode("ascii"))
    for outlet_key, index, text_hash, url_key, body_prefix_hash in records:
        status = "excluded" if index in excluded_indices[outlet_key] else "kept"
        digest.update(
            f"\0{outlet_key}\0{index}\0{text_hash}\0{url_key or ''}\0"
            f"{body_prefix_hash or ''}\0{status}".encode("utf-8"))

    usable_by_outlet = defaultdict(int)
    kept_by_outlet = defaultdict(int)
    for outlet_key, index, _, _, _ in records:
        usable_by_outlet[outlet_key] += 1
        if index not in excluded_indices[outlet_key]:
            kept_by_outlet[outlet_key] += 1
    emptied_outlets = sum(
        usable_by_outlet[key] > 0 and kept_by_outlet[key] == 0
        for key in usable_by_outlet
    )
    return {
        "schema": CONTENT_FILTER_SCHEMA,
        "fingerprint": digest.hexdigest(),
        "excluded_indices": {key: frozenset(indices)
                             for key, indices in excluded_indices.items() if indices},
        "summary": {
            "usable_articles": len(records),
            "cross_outlet_text_groups": len(shared_texts),
            "cross_outlet_url_groups": len(shared_urls),
            "cross_outlet_body_prefix_groups": len(shared_body_prefixes),
            "cross_outlet_duplicate_components": cross_outlet_components,
            "cross_outlet_articles_removed": cross_outlet_removed,
            "within_outlet_duplicates_removed": within_outlet_removed,
            "retained_articles": len(records) - cross_outlet_removed - within_outlet_removed,
            "outlets_emptied": emptied_outlets,
        },
    }


def filtered_articles(outlets, outlet_key, content_filter):
    """Return the usable, de-duplicated article records for one outlet."""
    excluded = content_filter["excluded_indices"].get(outlet_key, frozenset())
    return [
        article for index, article in enumerate(outlets[outlet_key].get("articles", []))
        if index not in excluded and str(article.get("text", "") or "").strip()
    ]


def scores(model, experiment, components, y_true, y_pred, task):
    """One metrics row, same columns as src/metrics.py plus this task's classes."""
    classes = TASKS[task]["classes"]
    ordinal = TASKS[task]["ordinal"]
    rep = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    t = np.array([ordinal[c] for c in y_true], dtype=float)
    p = np.array([ordinal[c] for c in y_pred], dtype=float)
    row = {
        "model": model,
        "experiment": experiment,
        "components": components,
        "n_test": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred) * 100.0,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) * 100.0,
        "macro_precision": precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0) * 100.0,
        "macro_recall": recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0) * 100.0,
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0) * 100.0,
    }
    for c in classes:
        row[f"{c}_f1"] = rep[c]["f1-score"] * 100.0
    row["mae"] = float(np.mean(np.abs(t - p)))
    row["mse"] = float(np.mean((t - p) ** 2))
    return row


def report(task, outlets):
    roles = load_roles(task, outlets)
    print(f"\n=== roles for {task} (seed={roles['seed']}, bert_frac={roles['bert_frac']}) ===")
    slices = {"bert_tune": roles["bert_tune"], "svm_train": roles["svm_train"], "test": roles["test"]}
    classes = TASKS[task]["classes"]
    header = f"{'slice':10s} {'outlets':>7s} {'articles':>8s}  " + "  ".join(f"{c:>12s}" for c in classes)
    print(header)
    for name, keys in slices.items():
        counts = {c: 0 for c in classes}
        n_articles = 0
        for k in keys:
            counts[outlet_label(outlets[k], task)] += 1
            n_articles += outlets[k]["n_articles"]
        dist = "  ".join(f"{counts[c]:5d} ({counts[c] / max(1, len(keys)):5.1%})" for c in classes)
        print(f"{name:10s} {len(keys):7d} {n_articles:8d}  {dist}")
    bert, svm, test = map(set, slices.values())
    print(f"overlap bert∩svm={len(bert & svm)}  bert∩test={len(bert & test)}  svm∩test={len(svm & test)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="build/inspect the role splits")
    ap.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    args = ap.parse_args()
    outlets = load_outlets()
    print(f"outlets loaded (deduplicated): {len(outlets)}")
    for task in (list(TASKS) if args.task == "all" else [args.task]):
        report(task, outlets)
