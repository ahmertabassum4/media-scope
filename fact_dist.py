#!/usr/bin/env python3
# Report the 5-point factuality distribution over the full snapshots.csv and
# over the subset whose provenance enrichment actually succeeded
# (provenance.fetch_ok == True, matching enrich_provenance.py's own success flag).
#
#   python fact_dist.py
#   python fact_dist.py --json-dir json_data --csv snapshots.csv

import os
import csv
import glob
import json
import argparse
from collections import Counter

FACT_ORDER = ["VERY LOW", "LOW", "MIXED", "HIGH", "VERY HIGH"]


def norm(v):
    return (v or "").strip().upper()


def load_snapshots(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def enriched_outlets(json_dir):
    """Media names (upper-cased) whose provenance fetch succeeded."""
    names = set()
    for path in glob.glob(os.path.join(json_dir, "*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        prov = obj.get("provenance") or {}
        if not prov.get("fetch_ok"):
            continue
        name = obj.get("media name") or obj.get("media_name") or ""
        names.add(name.strip().upper())
    return names


def tally(rows):
    return Counter(norm(r.get("factuality")) for r in rows)


def show(title, counter):
    total = sum(counter.values())
    print(f"\n{title}  (n={total})")
    print("-" * (len(title) + 12))
    for label in FACT_ORDER:
        if counter.get(label):
            print(f"  {label:<10} {counter[label]:>5}")
    for label in sorted(set(counter) - set(FACT_ORDER)):
        print(f"  {label or '(blank)':<10} {counter[label]:>5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="snapshots.csv")
    ap.add_argument("--json-dir", default="json_data")
    args = ap.parse_args()

    rows = load_snapshots(args.csv)
    show("Full snapshots.csv", tally(rows))

    if os.path.isdir(args.json_dir):
        keep = enriched_outlets(args.json_dir)
        subset = [r for r in rows if norm(r.get("media_name")) in keep]
        show(f"Enriched subset (matched {len(subset)} / {len(rows)})", tally(subset))
        unmatched = keep - {norm(r.get("media_name")) for r in subset}
        if unmatched:
            print(f"\nNote: {len(unmatched)} enriched outlets had no name match in the CSV.")
    else:
        print(f"\n(json dir '{args.json_dir}' not found — skipped subset breakdown.)")


if __name__ == "__main__":
    main()