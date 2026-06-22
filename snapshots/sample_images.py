import os
import csv
import random
import shutil
from datetime import datetime

random.seed(14)

SNAPSHOT_INDEX = "snapshot_index.csv"
OUTPUT_DIR     = "output"
DEST_DIR       = "snapshot-samples"

SAMPLE_SIZES = {
    "VERY LOW":  50,
    "LOW":       50,
    "VERY HIGH": 11,
    "HIGH":      89,
}

buckets = {factuality_level: [] for factuality_level in SAMPLE_SIZES}

with open(SNAPSHOT_INDEX, newline="", encoding="utf-8") as f:
    all_rows = {
        os.path.basename(row["image_path"].strip()): row
        for row in csv.DictReader(f)
    }

for filename, row in all_rows.items():
    factuality_level = row["factuality"].strip().upper()
    if factuality_level not in buckets:
        continue
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        buckets[factuality_level].append(filename)

os.makedirs(DEST_DIR, exist_ok=True)

sampled_filenames = set()
for factuality_level, n in SAMPLE_SIZES.items():
    pool = buckets[factuality_level]
    if len(pool) < n:
        print(f"Only {len(pool)} available for {factuality_level}, requested {n}")
        n = len(pool)
    for fname in random.sample(pool, n):
        shutil.copy(os.path.join(OUTPUT_DIR, fname), os.path.join(DEST_DIR, fname))
        sampled_filenames.add(fname)
    print(f"{factuality_level}: copied {n} images")

# CSV for the samples
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
out_csv = "snapshot_sample_index.csv"

fieldnames = ["media_name", "url", "image_path", "timestamp", "country", "factuality", "trustworthiness"]

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for fname in sorted(sampled_filenames):
        row = all_rows[fname]
        writer.writerow({
            "media_name":      row["media_name"],
            "url":             row["url"],
            "image_path":      f"sub-samples/{fname}",
            "timestamp":       now,
            "country":         row["country"],
            "factuality":      row["factuality"],
            "trustworthiness": row["trustworthiness"],
        })

print(f"\nDone")
