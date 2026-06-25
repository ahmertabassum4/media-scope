#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_CSV = ROOT / "snapshots" / "data" / "snapshots.csv"
RAW_DATA_DIR = ROOT / "raw_data" / "2291eng_dedup"
DEFAULT_OUTPUT_CSV = ROOT / "snapshots" / "data" / "mbfc_discrepancies.csv"
QUESTIONABLE_URL_SOURCE_CSV = ROOT / "tmp" / "questionable_sources_with_dates.csv"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)

REVIEW_JSON_RE = re.compile(
    r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)


def normalize_label(value: str) -> str:
    value = (value or "").strip().upper()
    value = value.replace("_", "-")
    value = re.sub(r"\s+", " ", value)
    return value


def is_nullish(value: str) -> bool:
    value = (value or "").strip().lower()
    return value in {"", "none", "null", "nan", "n/a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        choices=["nullish", "questionable"],
        default="nullish",
        help="Which snapshots.csv MBFC_source subset to review.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Path to the discrepancy CSV to write.",
    )
    parser.add_argument(
        "--questionable-url-source",
        default=str(QUESTIONABLE_URL_SOURCE_CSV),
        help="CSV containing questionable-source MBFC URLs.",
    )
    return parser.parse_args()


def fetch(url: str, timeout: int = 30) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_mbfc_page(html: str) -> tuple[str, str]:
    for match in REVIEW_JSON_RE.finditer(html):
        raw = unescape(match.group(1).strip())
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if payload.get("@type") != "Review":
            continue
        props = (
            payload.get("itemReviewed", {})
            .get("additionalProperty", [])
        )
        bias = ""
        factuality = ""
        for prop in props:
            name = prop.get("name", "").strip().lower()
            value = prop.get("value", "").strip()
            if name == "bias rating":
                bias = value
            elif name == "factual reporting":
                factuality = value
        if bias or factuality:
            return normalize_label(bias), normalize_label(factuality)
    raise ValueError("Could not find MBFC review schema with bias/factuality")


def load_snapshot_rows(subset: str) -> dict[str, dict]:
    rows = {}
    with SNAPSHOTS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            source = (row["MBFC_source"] or "").strip()
            if subset == "nullish" and not is_nullish(source):
                continue
            if subset == "questionable" and source != "Questionable Source":
                continue
            rows[row["media_name"]] = row
    return rows


def load_json_records(snapshot_rows: dict[str, dict]) -> list[dict]:
    records = []
    for path in glob.glob(str(RAW_DATA_DIR / "*.json")):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        media_name = data.get("media name") or Path(path).stem
        if media_name not in snapshot_rows:
            continue
        mbfc_url = (data.get("mbfc link") or "").strip()
        if not mbfc_url:
            continue
        snapshot = snapshot_rows[media_name]
        records.append(
            {
                "media_name": media_name,
                "existing_bias_rating": normalize_label(snapshot.get("bias_rating", "")),
                "existing_factuality": normalize_label(snapshot.get("factuality", "")),
                "source": snapshot.get("MBFC_source", "").strip(),
                "mbfc_url": mbfc_url,
            }
        )
    return records


def load_questionable_csv_records(snapshot_rows: dict[str, dict], source_csv: Path) -> list[dict]:
    records = []
    with source_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            media_name = (row.get("media_source_name") or "").strip()
            if media_name not in snapshot_rows:
                continue
            mbfc_url = (row.get("mbfc_report_url") or "").strip()
            if not mbfc_url:
                continue
            snapshot = snapshot_rows[media_name]
            records.append(
                {
                    "media_name": media_name,
                    "existing_bias_rating": normalize_label(snapshot.get("bias_rating", "")),
                    "existing_factuality": normalize_label(snapshot.get("factuality", "")),
                    "source": snapshot.get("MBFC_source", "").strip(),
                    "mbfc_url": mbfc_url,
                }
            )
    return records


def verify_record(record: dict) -> dict:
    url = record["mbfc_url"]
    for attempt in range(3):
        try:
            html = fetch(url)
            current_bias, current_factuality = parse_mbfc_page(html)
            result = dict(record)
            result["current_bias_rating"] = current_bias
            result["current_factuality"] = current_factuality
            result["changed"] = (
                current_bias != record["existing_bias_rating"]
                or current_factuality != record["existing_factuality"]
            )
            result["error"] = ""
            return result
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            if attempt == 2:
                result = dict(record)
                result["current_bias_rating"] = ""
                result["current_factuality"] = ""
                result["changed"] = False
                result["error"] = str(exc)
                return result
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    output_csv = Path(args.output)
    snapshot_rows = load_snapshot_rows(args.subset)
    if args.subset == "questionable":
        records = load_questionable_csv_records(
            snapshot_rows,
            Path(args.questionable_url_source),
        )
    else:
        records = load_json_records(snapshot_rows)
    if not records:
        print("No matching records found.", file=sys.stderr)
        return 1

    results = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(verify_record, r): r for r in records}
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if idx % 100 == 0 or idx == len(records):
                print(f"Verified {idx}/{len(records)}", file=sys.stderr)

    results.sort(key=lambda row: row["media_name"].lower())
    discrepancies = [row for row in results if row["changed"]]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "media_name",
                "existing_bias_rating",
                "existing_factuality",
                "current_bias_rating",
                "current_factuality",
                "source",
                "mbfc_url",
                "error",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(discrepancies)

    error_count = sum(1 for row in results if row["error"])
    print(f"Subset: {args.subset}")
    print(f"Matched sources: {len(records)}")
    print(f"Discrepancies: {len(discrepancies)}")
    print(f"Errors: {error_count}")
    print(f"Wrote: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
