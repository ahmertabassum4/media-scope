# sample_mixed_subset.py
# Build a ~200-image subset, as balanced as possible across the FIVE factuality
# classes (VERY LOW / LOW / MIXED / HIGH / VERY HIGH), using the updated index
# that now carries a "Mixed Factuality" column for the Mixed_output captures.
#
#   python sample_mixed_subset.py
#   python sample_mixed_subset.py --target 200 --index snapshots.csv
#
# Output: snapshot_sample_mixed_index.csv  (+ copies images into a sample dir)

import os
import csv
import math
import random
import shutil
import argparse
from datetime import datetime

from mixed_config import (
    FACT_LABELS, SAMPLE_INDEX, SAMPLE_IMAGE_DIR, effective_tier,
)

random.seed(14)

SOURCE_INDEX_DEFAULT = "snapshots.csv"


def load_rows(index_path):
    with open(index_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bucket_rows(rows):
    """Group rows by their effective 5-class tier; skip rows we can't place
    or whose screenshot is missing on disk."""
    buckets = {lab: [] for lab in FACT_LABELS}
    skipped_nolabel = skipped_noimg = 0
    for row in rows:
        tier = effective_tier(row)
        if tier not in buckets:
            skipped_nolabel += 1
            continue
        img = row.get("image_path", "").strip()
        if not img or not os.path.exists(img):
            skipped_noimg += 1
            continue
        buckets[tier].append(row)
    return buckets, skipped_nolabel, skipped_noimg


def allocate(buckets, target):
    """Decide how many to take per class: an even split, but never more than a
    class can supply. Redistribute the shortfall to classes that still have room."""
    classes = list(buckets)
    take = {c: 0 for c in classes}
    remaining = target
    # Iteratively give each non-exhausted class an equal share of what's left.
    open_classes = [c for c in classes if len(buckets[c]) > 0]
    while remaining > 0 and open_classes:
        share = max(1, remaining // len(open_classes))
        progressed = False
        for c in open_classes:
            room = len(buckets[c]) - take[c]
            if room <= 0:
                continue
            add = min(share, room, remaining)
            take[c] += add
            remaining -= add
            if add > 0:
                progressed = True
            if remaining <= 0:
                break
        open_classes = [c for c in classes if len(buckets[c]) - take[c] > 0]
        if not progressed:
            break
    return take


def run(index_path, target):
    rows = load_rows(index_path)
    buckets, no_label, no_img = bucket_rows(rows)

    print("Available per class (with image on disk):")
    for lab in FACT_LABELS:
        print(f"  {lab:<10} {len(buckets[lab])}")
    if no_label:
        print(f"  ({no_label} rows skipped: no usable factuality tier)")
    if no_img:
        print(f"  ({no_img} rows skipped: image missing on disk)")

    take = allocate(buckets, target)
    print("\nSampling:")
    for lab in FACT_LABELS:
        print(f"  {lab:<10} {take[lab]}")
    print(f"  total      {sum(take.values())}")

    os.makedirs(SAMPLE_IMAGE_DIR, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fieldnames = ["media_name", "url", "image_path", "timestamp",
                  "country", "factuality", "trustworthiness"]
    sampled = []
    for lab in FACT_LABELS:
        chosen = random.sample(buckets[lab], take[lab])
        for row in chosen:
            src = row["image_path"].strip()
            fname = os.path.basename(src)
            dst = os.path.join(SAMPLE_IMAGE_DIR, fname)
            shutil.copy(src, dst)
            sampled.append({
                "media_name":      row["media_name"],
                "url":             row["url"],
                "image_path":      os.path.join(SAMPLE_IMAGE_DIR, fname),
                "timestamp":       now,
                "country":         row.get("country", ""),
                # Write the EFFECTIVE 5-class tier into `factuality` so the rest
                # of the pipeline reads one column, unchanged.
                "factuality":      lab,
                "trustworthiness": row.get("trustworthiness", ""),
            })

    with open(SAMPLE_INDEX, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(sampled, key=lambda r: r["media_name"]):
            writer.writerow(row)

    print(f"\nWrote {len(sampled)} rows -> {SAMPLE_INDEX}")
    print(f"Images copied -> {SAMPLE_IMAGE_DIR}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", default=SOURCE_INDEX_DEFAULT,
                   help="Source index CSV with a 'Mixed Factuality' column.")
    p.add_argument("--target", type=int, default=200)
    args = p.parse_args()
    run(args.index, args.target)