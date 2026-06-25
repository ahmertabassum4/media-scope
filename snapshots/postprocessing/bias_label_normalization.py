"""
bias_label_normalization.py — normalize the bias_rating column in snapshots.csv

Run from project root:
    python snapshots/postprocessing/bias_label_normalization.py
"""

import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_CSV = ROOT / "snapshots" / "data" / "snapshots.csv"

REPLACEMENTS = {
    "FAR RIGHT":           "RIGHT",
    "FAR LEFT":            "LEFT",
    "FAR RIGHT-BIAS":      "RIGHT",
    "EXTREME-RIGHT":       "EXTREME RIGHT",
    "RIGHT-CENTER (4.3":   "RIGHT-CENTER",
    "NOT RATED":           "UNRATED",
    "":                    "UNRATED",
}

REMOVE = {"JUNK NEWS", "PRO-SCIENCE", "RIGHT CONSPIRACY-PSEUDOSCIENCE"}

with open(SNAPSHOTS_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

kept = []
removed = 0
for row in rows:
    bias = row["bias_rating"].strip()
    if bias in REMOVE:
        removed += 1
        continue
    row["bias_rating"] = REPLACEMENTS.get(bias, bias)
    kept.append(row)

with open(SNAPSHOTS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept)

counts = Counter(r["bias_rating"] for r in kept)
print(f"Removed rows : {removed}")
print(f"Remaining    : {len(kept)}\n")
print("Final distribution:")
for val, cnt in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {cnt:4d}  {val}")
