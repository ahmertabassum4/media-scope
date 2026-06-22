#!/usr/bin/env python3
"""
postprocessing.py — Post-processing pipeline for Mixed_output snapshots.

Steps:
  1. index        — Build mixed-snapshots.csv from Mixed_output/batch_log.jsonl
                      • Same columns as snapshot_index.csv
                      • Excludes error entries (listed in mixed-errors.csv)
                      • trustworthiness = empty for all MIXED entries
  2. merge-errors — Merge errors.csv + mixed-errors.csv → error.csv
                      • Deduplicates by URL (first occurrence wins)
                      • Removes URLs already in snapshots.csv (successful capture takes priority)
                      • Adds sources with no media link as "No URL available"
                      • Enriches factuality from JSON source files
                      • Final row count: exactly (total_jsons - rows_in_snapshots.csv)
  3. merge-snaps  — Merge snapshot_index.csv + mixed-snapshots.csv → snapshots.csv
                      • Deduplicates by URL (first occurrence wins)

Invariant enforced: rows(snapshots.csv) + rows(error.csv) == total JSON files in DATA_DIR

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
MIXED_ERRORS    = Path("tmp/mixed-errors.csv")
ERRORS_CSV      = Path("tmp/errors.csv")
SNAPSHOT_INDEX  = Path("tmp/snapshot_index.csv")
MIXED_SNAPSHOTS = Path("tmp/mixed-snapshots.csv")
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


def load_source_metadata(data_dir: Path) -> tuple[dict, list]:
    """
    Read all source JSON files.
    Returns:
      url_meta   — dict of url → {name, country, factuality} (for lookups; deduped by URL)
      all_sources — list of ALL sources including duplicate-URL entries (len == total JSON files)
    """
    url_meta = {}
    all_sources = []
    for json_file in sorted(data_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        url = (data.get("media link") or "").strip()
        name = (data.get("media name") or json_file.stem).strip()
        country = (data.get("country") or "").strip()
        factuality = (data.get("factuality") or "").strip().upper()
        entry = {"name": name, "url": url, "country": country, "factuality": factuality}
        all_sources.append(entry)
        if url and url not in url_meta:
            url_meta[url] = entry
    return url_meta, all_sources


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

    url_meta, _ = load_source_metadata(DATA_DIR)
    print(f"  Loaded metadata for {len(url_meta)} sources")

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
            meta = url_meta.get(url, {})
            rows.append({
                "media_name":      entry.get("name", ""),
                "url":             url,
                "image_path":      entry["path"],
                "timestamp":       parse_timestamp(entry["path"]),
                "country":         meta.get("country", ""),
                "factuality":      "MIXED",
                "trustworthiness": "",
            })

    with open(MIXED_SNAPSHOTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Written {len(rows)} rows → {MIXED_SNAPSHOTS}")
    print(f"  Skipped: {skipped_captures} capture errors, {skipped_errors} visual errors")


def merge_snapshots():
    """
    Merge snapshot_index.csv + mixed-snapshots.csv → snapshots.csv.

    One row per JSON source file (not per URL). The 11 pairs of JSON files that share
    a URL each get their own row — both point to the same captured image.
    """
    print("Step 2: Merging snapshot CSVs → snapshots.csv ...")

    if not MIXED_SNAPSHOTS.exists():
        print(f"  ERROR: {MIXED_SNAPSHOTS} not found — run --step index first", file=sys.stderr)
        return

    # Build url → snapshot row lookup from existing index CSVs
    url_to_snap = {}
    for path in (SNAPSHOT_INDEX, MIXED_SNAPSHOTS):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                url = r.get("url", "").strip()
                if url and url not in url_to_snap:
                    url_to_snap[url] = r
    print(f"  Unique captured URLs: {len(url_to_snap)}")

    # Emit one row per JSON source whose URL was captured (all_sources preserves duplicates)
    _, all_sources = load_source_metadata(DATA_DIR)
    rows = []
    for src in all_sources:
        url = src["url"]
        if url in url_to_snap:
            snap = dict(url_to_snap[url])
            snap["url"] = url
            rows.append(snap)

    with open(SNAPSHOTS_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    captured_urls = set(url_to_snap.keys())
    print(f"  Written {len(rows)} rows → {SNAPSHOTS_OUT}")
    return captured_urls


def merge_errors(captured_urls: set = None):
    """
    Merge errors.csv + mixed-errors.csv → error.csv.

    One row per JSON source file whose URL was NOT successfully captured.
    Uses the JSON source files as ground truth so the invariant holds:
      rows(snapshots.csv) + rows(error.csv) == total JSON files in DATA_DIR
    """
    print("Step 3: Merging error CSVs → error.csv ...")

    _, all_sources = load_source_metadata(DATA_DIR)

    if captured_urls is None:
        if SNAPSHOTS_OUT.exists():
            with open(SNAPSHOTS_OUT, newline="", encoding="utf-8") as f:
                captured_urls = {r["url"].strip() for r in csv.DictReader(f)}
        else:
            captured_urls = set()
    print(f"  Captured URLs (excluded from errors): {len(captured_urls)}")

    # Build url → best error row lookup from both error CSVs
    def load(path):
        if not path.exists():
            print(f"  WARN: {path} not found, skipping", file=sys.stderr)
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    url_to_err = {}
    for r in load(ERRORS_CSV) + load(MIXED_ERRORS):
        url = r.get("url", "").strip()
        if url and url not in url_to_err:
            url_to_err[url] = r

    # Emit one row per JSON source whose URL was NOT captured
    all_rows = []
    not_in_either = 0

    for src in all_sources:
        url = src["url"]
        if url in captured_urls:
            continue

        err = url_to_err.get(url, {})
        factuality = err.get("factuality", "").strip() or src["factuality"]
        name = (err.get("name") or err.get("filename") or src["name"]).strip()
        issue = err.get("issue", "").strip() or "Not captured"

        all_rows.append({
            "filename":   err.get("filename", ""),
            "name":       name,
            "url":        url,
            "issue":      issue,
            "timestamp":  err.get("timestamp", ""),
            "factuality": factuality,
        })

        if not err:
            not_in_either += 1

    with open(ERROR_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  Sources not in any error CSV (labelled 'Not captured'): {not_in_either}")
    print(f"  Written {len(all_rows)} rows → {ERROR_OUT}")


def verify():
    """Verify that snapshots.csv + error.csv == total JSON files."""
    print("\nVerification:")
    total_jsons = sum(1 for _ in DATA_DIR.glob("*.json"))

    with open(SNAPSHOTS_OUT, newline="", encoding="utf-8") as f:
        snap_count = sum(1 for _ in csv.DictReader(f))
    with open(ERROR_OUT, newline="", encoding="utf-8") as f:
        err_count = sum(1 for _ in csv.DictReader(f))

    combined = snap_count + err_count
    status = "OK" if combined == total_jsons else "MISMATCH"

    print(f"  snapshots.csv rows : {snap_count}")
    print(f"  error.csv rows     : {err_count}")
    print(f"  Combined           : {combined}")
    print(f"  JSON files         : {total_jsons}")
    print(f"  Status             : {status} {'✓' if status == 'OK' else f'(diff: {combined - total_jsons:+d})'}")


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

    snap_urls = None
    if args.step in ("merge-snaps", "all"):
        snap_urls = merge_snapshots()

    if args.step in ("merge-errors", "all"):
        merge_errors(snap_urls)

    if args.step == "all":
        verify()

    print("\nDone.")


if __name__ == "__main__":
    main()
