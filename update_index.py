import csv
import json
import os
import sys

SAMPLE_INDEX = "snapshot_sample_index.csv"
JSON_DIR = "json_data"
OUTPUT_INDEX = "snapshot_sample_index.csv"  # overwrite in place

NEW_FIELDS = ["bias", "genre", "credibility", "media_type", "freedom_rating", "traffic"]


def load_json_lookup(json_dir):
    lookup = {}

    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(json_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: skipping {fname} — {e}")
            continue

        name = data.get("media name", "").strip()
        if not name:
            continue

        lookup[name.lower()] = {
            "bias":           data.get("bias", "").strip(),
            "genre":          data.get("genre", "").strip(),
            "credibility":    data.get("credibility", "").strip(),
            "media_type":     data.get("media_type", "").strip(),
            "freedom_rating": data.get("freedom_rating", "").strip(),
            "traffic":        data.get("traffic", "").strip(),
        }

    print(f"Loaded metadata for {len(lookup)} sources from {json_dir}/")
    return lookup


def update_csv(sample_index, json_dir, output_path):
    lookup = load_json_lookup(json_dir)

    with open(sample_index, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        rows = list(reader)

    # Don't duplicate columns if u rerun the script
    added = [c for c in NEW_FIELDS if c not in existing_fields]
    out_fields = existing_fields + added

    matched = 0
    for row in rows:
        name_key = row.get("media_name", "").strip().lower()
        meta = lookup.get(name_key, {})
        if meta:
            matched += 1
        for col in NEW_FIELDS:
            # Preserve any existing value; only fill if blank or column is new
            if not row.get(col):
                row[col] = meta.get(col, "")

    print(f"Matched {matched}/{len(rows)} rows to JSON metadata")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written to {output_path}")


if __name__ == "__main__":
    update_csv(SAMPLE_INDEX, JSON_DIR, OUTPUT_INDEX)