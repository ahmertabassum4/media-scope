"""
copy_to_mediascope.py — deduplicate snapshots.csv by keeping the latest row
per image filename, then copy all images into mediascopeAll/.

Source directories:
    output/
    tmp/Mixed_output/
    tmp/questionable_sources/

Run from project root:
    python snapshots/postprocessing/copy_to_mediascope.py
"""

import csv
import shutil
from collections import defaultdict
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[2]
SNAPSHOTS_CSV = ROOT / "snapshots" / "data" / "snapshots.csv"
DEST_DIR      = ROOT / "mediascopeAll"

SOURCE_DIRS = [
    ROOT / "output",
    ROOT / "tmp" / "Mixed_output",
    ROOT / "tmp" / "questionable_sources",
]

# ── 1. Read all rows ──────────────────────────────────────────────────────────

with open(SNAPSHOTS_CSV, newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    fieldnames = reader.fieldnames
    rows = list(reader)

# ── 2. Deduplicate — keep latest timestamp per image filename ─────────────────

by_fname: dict[str, list] = defaultdict(list)
for row in rows:
    fname = Path(row["image_path"]).name
    by_fname[fname].append(row)

deduped = []
dropped = 0

for fname, group in by_fname.items():
    if len(group) == 1:
        deduped.append(group[0])
    else:
        group.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        deduped.append(group[0])
        dropped += len(group) - 1

print(f"Original rows  : {len(rows):,}")
print(f"Dropped (older): {dropped:,}")
print(f"Remaining      : {len(deduped):,}")

# ── 3. Write deduplicated snapshots.csv ──────────────────────────────────────

with open(SNAPSHOTS_CSV, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(deduped)

print(f"Updated        : {SNAPSHOTS_CSV}")

# ── 4. Build filename → actual path index from source dirs ───────────────────

file_index: dict[str, Path] = {}
for src_dir in SOURCE_DIRS:
    for f in src_dir.glob("*.png"):
        file_index[f.name] = f

print(f"\nIndexed {len(file_index):,} images across source directories")

# ── 5. Copy images to mediascopeAll/ ─────────────────────────────────────────

DEST_DIR.mkdir(exist_ok=True)

copied = missing = 0

for row in deduped:
    fname = Path(row["image_path"]).name
    src = file_index.get(fname)
    if src is None:
        print(f"MISSING  {fname}")
        missing += 1
        continue
    shutil.copy2(src, DEST_DIR / fname)
    copied += 1

print(f"Copied to {DEST_DIR}/: {copied:,} images")
if missing:
    print(f"Missing (not copied): {missing:,}")
print(f"Files in {DEST_DIR}/: {len(list(DEST_DIR.glob('*.png'))):,}")
