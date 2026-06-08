#!/usr/bin/env python3
"""
postprocessing.py — Post-processing pipeline for Mixed_output snapshots.

Steps:
  1. index        — Build mixed-snapshots.csv from Mixed_output/batch_log.jsonl
                      • Same columns as snapshot_index.csv
                      • Excludes error entries (listed in mixed-errors.csv)
                      • trustworthiness = empty for all MIXED entries
  2. merge-errors — Merge errors.csv + mixed-errors.csv → error.csv
  3. merge-snaps  — Merge snapshot_index.csv + mixed-snapshots.csv → snapshots.csv

Usage:
    python postprocessing.py                        # run all steps
    python postprocessing.py --step index           # only build mixed-snapshots.csv
    python postprocessing.py --step merge-errors    # only merge error CSVs
    python postprocessing.py --step merge-snaps     # only merge snapshot CSVs
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

DATA_DIR        = Path("data/2291eng_dedup")
BATCH_LOG       = Path("tmp/Mixed_output/batch_log.jsonl")
MIXED_ERRORS    = Path("mixed-errors.csv")
ERRORS_CSV      = Path("errors.csv")
SNAPSHOT_INDEX  = Path("snapshot_index.csv")
MIXED_SNAPSHOTS = Path("mixed-snapshots.csv")
ERROR_OUT       = Path("error.csv")
SNAPSHOTS_OUT   = Path("snapshots.csv")

SNAPSHOT_COLUMNS = ["media_name", "url", "image_path", "timestamp", "country", "factuality", "trustworthiness"]
ERROR_COLUMNS    = ["filename", "name", "url", "issue", "timestamp", "factuality"]


def parse_timestamp(image_path: str) -> str:
    """Extract timestamp from filename like Name_20260608_163823.png → 2026-06-08 16:38:23."""
    m = re.search(r"_(\d{8})_(\d{6})(?:\.\w+)?$", Path(image_path).stem)
    if not m:
        return ""
    d, t = m.group(1), m.group(2)
    return f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}"


def build_url_to_country(data_dir: Path) -> dict:
    """Read all source JSON files and return a url → country mapping."""
    mapping = {}
    for json_file in data_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        url = (data.get("media link") or "").strip()
        country = (data.get("country") or "").strip()
        if url:
            mapping[url] = country
    return mapping


def load_error_filenames(errors_csv: Path) -> set:
    """Return the set of filenames that had errors (to exclude from the snapshot index)."""
    filenames = set()
    if not errors_csv.exists():
        return filenames
    with open(errors_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row.get("filename", "").strip()
            if fname:
                filenames.add(fname)
    return filenames


def build_mixed_snapshots():
    print("Step 1: Building mixed-snapshots.csv ...")

    url_to_country = build_url_to_country(DATA_DIR)
    print(f"  Loaded country data for {len(url_to_country)} sources")

    error_filenames = load_error_filenames(MIXED_ERRORS)
    print(f"  Excluding {len(error_filenames)} error filenames")

    rows = []
    skipped_captures = 0
    skipped_errors = 0

    with open(BATCH_LOG, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)

            if entry["status"] != "ok":
                skipped_captures += 1
                continue

            filename = Path(entry["path"]).name
            if filename in error_filenames:
                skipped_errors += 1
                continue

            url = entry["url"]
            rows.append({
                "media_name":      entry.get("name", ""),
                "url":             url,
                "image_path":      entry["path"],
                "timestamp":       parse_timestamp(entry["path"]),
                "country":         url_to_country.get(url, ""),
                "factuality":      "MIXED",
                "trustworthiness": "",
            })

    with open(MIXED_SNAPSHOTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Written {len(rows)} rows → {MIXED_SNAPSHOTS}")
    print(f"  Skipped: {skipped_captures} capture errors, {skipped_errors} visual errors")


def merge_errors():
    """Merge errors.csv + mixed-errors.csv → error.csv (union of all columns)."""
    print("Step 2: Merging error CSVs → error.csv ...")

    # errors.csv columns:       filename, url, issue, timestamp
    # mixed-errors.csv columns: filename, name, url, issue, factuality
    # merged columns:           filename, name, url, issue, timestamp, factuality

    def load(path):
        if not path.exists():
            print(f"  WARN: {path} not found, skipping", file=sys.stderr)
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    errors_rows = load(ERRORS_CSV)
    mixed_rows  = load(MIXED_ERRORS)

    print(f"  {ERRORS_CSV}: {len(errors_rows)} rows")
    print(f"  {MIXED_ERRORS}: {len(mixed_rows)} rows")

    all_rows = []
    for r in errors_rows:
        all_rows.append({
            "filename":    r.get("filename", ""),
            "name":        r.get("name", ""),
            "url":         r.get("url", ""),
            "issue":       r.get("issue", ""),
            "timestamp":   r.get("timestamp", ""),
            "factuality":  r.get("factuality", ""),
        })
    for r in mixed_rows:
        all_rows.append({
            "filename":    r.get("filename", ""),
            "name":        r.get("name", ""),
            "url":         r.get("url", ""),
            "issue":       r.get("issue", ""),
            "timestamp":   r.get("timestamp", ""),
            "factuality":  r.get("factuality", ""),
        })

    with open(ERROR_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  Written {len(all_rows)} total rows → {ERROR_OUT}")


def merge_snapshots():
    """Merge snapshot_index.csv + mixed-snapshots.csv → snapshots.csv."""
    print("Step 3: Merging snapshot CSVs → snapshots.csv ...")

    if not MIXED_SNAPSHOTS.exists():
        print(f"  ERROR: {MIXED_SNAPSHOTS} not found — run --step index first", file=sys.stderr)
        return

    rows = []
    with open(SNAPSHOT_INDEX, newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    print(f"  {SNAPSHOT_INDEX}: {len(rows)} rows")

    mixed_rows = []
    with open(MIXED_SNAPSHOTS, newline="", encoding="utf-8") as f:
        mixed_rows.extend(csv.DictReader(f))
    print(f"  {MIXED_SNAPSHOTS}: {len(mixed_rows)} rows")

    rows.extend(mixed_rows)

    with open(SNAPSHOTS_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Written {len(rows)} total rows → {SNAPSHOTS_OUT}")


def main():
    parser = argparse.ArgumentParser(description="Post-process Mixed_output snapshots.")
    parser.add_argument(
        "--step",
        choices=["index", "merge-errors", "merge-snaps", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    if args.step in ("index", "all"):
        build_mixed_snapshots()

    if args.step in ("merge-errors", "all"):
        merge_errors()

    if args.step in ("merge-snaps", "all"):
        merge_snapshots()

    print("\nDone.")


if __name__ == "__main__":
    main()
