"""
add_metadata.py — enrich snapshots/data/snapshots.csv with:
  bias_rating

Sources (no labels generated from model knowledge):
  1. tmp/questionable_sources_with_dates.csv  (bias_rating)
  2. raw_data/2291eng_dedup/*.json            (bias → bias_rating)

Run from project root:
    python snapshots/postprocessing/add_metadata.py
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT            = Path(__file__).resolve().parents[2]
SNAPSHOTS_CSV   = ROOT / "snapshots" / "data" / "snapshots.csv"
QS_CSV          = ROOT / "tmp" / "questionable_sources_with_dates.csv"
JSON_DIR        = ROOT / "raw_data" / "2291eng_dedup"


def normalize_url(url: str) -> str:
    """Lowercase, strip trailing slash, drop www. prefix for robust matching."""
    url = url.strip().lower().rstrip("/")
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url


# ── 1. Build lookup from questionable sources CSV ─────────────────────────────

qs_lookup: dict[str, dict] = {}

with open(QS_CSV, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        raw_url = row.get("media_source_url", "").strip()
        if not raw_url:
            continue
        key = normalize_url(raw_url)
        qs_lookup[key] = {
            "bias_rating": row.get("bias_rating", "").strip(),
        }

print(f"Loaded {len(qs_lookup):,} entries from {QS_CSV}")


# ── 2. Build lookup from JSON files ───────────────────────────────────────────

json_lookup: dict[str, dict] = {}

for json_path in JSON_DIR.glob("*.json"):
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        continue
    raw_url = data.get("media link", "").strip()
    if not raw_url:
        continue
    key = normalize_url(raw_url)
    json_lookup[key] = {
        "bias_rating": data.get("bias", "").strip(),
    }

print(f"Loaded {len(json_lookup):,} entries from {JSON_DIR}")


# ── 3. Enrich snapshots.csv ───────────────────────────────────────────────────

rows: list[dict] = []
with open(SNAPSHOTS_CSV, newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    original_fields = reader.fieldnames or []
    rows = list(reader)

new_fields = list(original_fields)
if "bias_rating" not in new_fields:
    new_fields.append("bias_rating")

matched_qs   = 0
matched_json = 0
unmatched    = 0

for row in rows:
    key = normalize_url(row.get("url", ""))

    meta = qs_lookup.get(key) or json_lookup.get(key)

    if meta:
        if key in qs_lookup:
            matched_qs += 1
        else:
            matched_json += 1

        if not row.get("bias_rating"):
            row["bias_rating"] = meta["bias_rating"]
    else:
        unmatched += 1
        row.setdefault("bias_rating", "")


with open(SNAPSHOTS_CSV, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=new_fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nResults:")
print(f"  Matched via questionable CSV : {matched_qs:,}")
print(f"  Matched via JSON files       : {matched_json:,}")
print(f"  No match found               : {unmatched:,}")
print(f"  Total rows                   : {len(rows):,}")
print(f"\nUpdated: {SNAPSHOTS_CSV}")
