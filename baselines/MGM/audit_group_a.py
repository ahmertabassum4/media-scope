"""Audit Group-A outlet partitions and the strict article de-duplication policy.

The hard checks ensure outlet keys and labels are isolated between BERT
fine-tuning, SVM training, and test. Before any model sees articles, one
canonical copy of matching text, canonical URLs, or long article-body prefixes
is retained globally, preferring test-split content. This audit verifies that
no matching article remains in any active partition.
"""

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import sys

import tldextract

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (  # noqa: E402
    TASKS,
    article_body_prefix_hash,
    article_text_hash,
    article_url_key,
    build_content_filter,
    filtered_articles,
    load_outlets,
    load_roles,
    outlet_label,
    split_fine_tune_outlets,
)


def article_records(outlets, keys, content_filter=None):
    records = []
    for key in keys:
        articles = (filtered_articles(outlets, key, content_filter)
                    if content_filter else outlets[key].get("articles", []))
        for article in articles:
            text = str(article.get("text", "") or "").strip()
            if not text:
                continue
            records.append((
                key,
                article_text_hash(text),
                article_url_key(article.get("url", "")),
                article_body_prefix_hash(text),
            ))
    return records


def summarize_records(records):
    return (
        len(records),
        {text_hash for _, text_hash, _, _ in records},
        {url for _, _, url, _ in records if url},
        {prefix for _, _, _, prefix in records if prefix},
    )


def outlet_domains(outlets, keys):
    domains = set()
    for key in keys:
        domain = tldextract.extract(str(outlets[key].get("url", ""))).registered_domain
        if domain:
            domains.add(domain)
    return domains


def audit_task(task, outlets, content_filter, val_frac, seed):
    roles = load_roles(task, outlets)
    bert_train, bert_val = split_fine_tune_outlets(task, roles, outlets, val_frac, seed)
    partitions = {
        "bert_train": bert_train,
        "bert_validation": bert_val,
        "svm_train": roles["svm_train"],
        "test": roles["test"],
    }

    all_keys = list(partitions.values())
    if any(set(left) & set(right) for left, right in combinations(all_keys, 2)):
        raise SystemExit(f"{task}: outlet-key overlap detected")
    if any(outlets[key].get("split") != "train"
           for key in partitions["bert_train"] + partitions["bert_validation"] + partitions["svm_train"]):
        raise SystemExit(f"{task}: a training partition contains a test-split outlet")
    if any(outlets[key].get("split") != "test" for key in partitions["test"]):
        raise SystemExit(f"{task}: test partition contains a train-split outlet")

    expected_bert_labels = set(TASKS[task]["bert_classes"])
    if any(outlet_label(outlets[key], task) not in expected_bert_labels
           for key in partitions["bert_train"] + partitions["bert_validation"]):
        raise SystemExit(f"{task}: invalid BERT supervision label")

    raw_records = {name: article_records(outlets, keys) for name, keys in partitions.items()}
    records = {
        name: article_records(outlets, keys, content_filter)
        for name, keys in partitions.items()
    }
    sets = {name: summarize_records(partition_records)
            for name, partition_records in records.items()}

    print(f"\n=== Group-A leakage audit: {task} ===")
    for name, keys in partitions.items():
        labels = Counter(outlet_label(outlets[key], task) for key in keys)
        n_articles, hashes, urls, prefixes = sets[name]
        n_removed = len(raw_records[name]) - n_articles
        print(f"{name:16s} outlets={len(keys):4d} articles={n_articles:5d} removed={n_removed:4d} "
              f"unique_text={len(hashes):5d} unique_url={len(urls):5d} "
              f"body_prefix={len(prefixes):5d} labels={dict(sorted(labels.items()))}")
    print("outlet-key and source-split checks: PASS")
    print(f"strict content filter: {content_filter['summary']}")

    content_overlap = 0
    names = list(partitions)
    for left, right in combinations(names, 2):
        shared_text = len(sets[left][1] & sets[right][1])
        shared_url = len(sets[left][2] & sets[right][2])
        shared_prefix = len(sets[left][3] & sets[right][3])
        content_overlap += shared_text + shared_url + shared_prefix
        print(f"{left} x {right}: exact_text={shared_text} exact_url={shared_url} "
              f"body_prefix={shared_prefix}")

    shared_domains = outlet_domains(outlets, partitions["svm_train"]) & outlet_domains(outlets, partitions["test"])
    print(f"svm_train x test shared outlet domains={len(shared_domains)}")
    if shared_domains:
        print(f"shared domain examples={sorted(shared_domains)[:10]}")
    if content_overlap:
        print("strict content audit: FAIL (residual duplicate articles remain after filtering)")
        return False
    print("strict content audit: PASS")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASKS, "all"], default="all")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-content", action="store_true",
                        help="deprecated compatibility flag; strict de-duplication is always checked")
    args = parser.parse_args()

    outlets = load_outlets()
    content_filter = build_content_filter(outlets)
    tasks = list(TASKS) if args.task == "all" else [args.task]
    outcomes = [audit_task(task, outlets, content_filter, args.val_frac, args.seed)
                for task in tasks]
    if not all(outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
