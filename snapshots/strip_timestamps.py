#!/usr/bin/env python3
"""
strip_timestamps.py — Remove _YYYYMMDD_HHMMSS suffix from filenames in
tmp/questionable_sources/ and update image_path in tmp/qs_snapshots.csv.
"""

import csv
import re
import sys
from pathlib import Path

QS_DIR     = Path("tmp/questionable_sources")
CSV_PATH   = Path("tmp/qs_snapshots.csv")
TS_PATTERN = re.compile(r"(_\d{8}_\d{6})(\.\w+)$")


def strip_ts(filename: str) -> str:
    return TS_PATTERN.sub(r"\2", filename)


def main():
    # --- 1. Rename files in tmp/questionable_sources/ ---
    renamed = 0
    conflicts = []

    for src in sorted(QS_DIR.glob("*.png")):
        dst_name = strip_ts(src.name)
        if dst_name == src.name:
            continue  # already stripped
        dst = src.parent / dst_name
        if dst.exists():
            conflicts.append((src.name, dst_name))
            continue
        src.rename(dst)
        renamed += 1

    print(f"Renamed {renamed} files in {QS_DIR}")
    if conflicts:
        print(f"Skipped {len(conflicts)} conflicts (destination already exists):")
        for s, d in conflicts[:10]:
            print(f"  {s} → {d}")

    # --- 2. Update image_path in qs_snapshots.csv ---
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            old_path = row["image_path"]
            new_name = strip_ts(Path(old_path).name)
            row["image_path"] = str(Path(old_path).parent / new_name)
            rows.append(row)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {len(rows)} image_path entries in {CSV_PATH}")


if __name__ == "__main__":
    sys.exit(main())
