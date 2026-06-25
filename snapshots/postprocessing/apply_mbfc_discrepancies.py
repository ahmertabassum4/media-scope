#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_CSV = ROOT / "snapshots" / "data" / "snapshots.csv"
DEFAULT_DISCREPANCIES_CSV = ROOT / "snapshots" / "data" / "mbfc_discrepancies.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--discrepancies",
        default=str(DEFAULT_DISCREPANCIES_CSV),
        help="Path to the discrepancy CSV to apply.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    discrepancies_csv = Path(args.discrepancies)

    with discrepancies_csv.open(newline="", encoding="utf-8") as fh:
        discrepancies = {
            row["media_name"]: row
            for row in csv.DictReader(fh)
        }

    with SNAPSHOTS_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = 0
    skipped = []
    for row in rows:
        media_name = row["media_name"]
        discrepancy = discrepancies.get(media_name)
        if not discrepancy:
            continue

        current_bias = (discrepancy.get("current_bias_rating") or "").strip()
        current_factuality = (discrepancy.get("current_factuality") or "").strip()

        # Avoid erasing existing labels when the live MBFC parse was incomplete.
        if not current_bias or not current_factuality:
            skipped.append(media_name)
            continue

        changed = False
        if row.get("bias_rating", "").strip() != current_bias:
            row["bias_rating"] = current_bias
            changed = True
        if row.get("factuality", "").strip() != current_factuality:
            row["factuality"] = current_factuality
            changed = True

        if changed:
            updated += 1

    with SNAPSHOTS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated rows: {updated}")
    print(f"Skipped rows with incomplete current labels: {len(skipped)}")
    for media_name in skipped:
        print(f"SKIPPED\t{media_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
