#!/usr/bin/env python3
"""
batch_snapshot.py — Snapshot every media source whose factuality is MIXED.

Usage:
    python batch_snapshot.py                                                    # skip already-captured (default)
    python batch_snapshot.py --force                                            # re-capture even if screenshot exists
    python batch_snapshot.py --retry-timeouts                                   # only retry previous timeout errors
    python batch_snapshot.py --retry-timeouts --timeout 60000                   # retry with higher timeout
    python batch_snapshot.py --from-csv output/image_issues.csv                 # rerun URLs listed in a CSV
    python batch_snapshot.py --from-csv output/image_issues.csv --output rerun_output --timeout 30000
    python batch_snapshot.py --from-logs output/batch_log.jsonl rerun_output/batch_log.jsonl --output error_output --timeout 60000
    python batch_snapshot.py --workers 4                                        # parallel workers (default 4)
    python batch_snapshot.py --output shots                                     # custom output dir (default: output)
    python batch_snapshot.py --dry-run                                          # print URLs without capturing
"""

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from snapshot import take_snapshot

TARGET_FACTUALITY = {"MIXED"}
DATA_DIR = Path("data/2291eng_dedup")


def load_qualifying_sources(data_dir: Path) -> list[dict]:
    sources = []
    for json_file in sorted(data_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Could not parse {json_file.name}: {e}", file=sys.stderr)
            continue

        factuality = (data.get("factuality") or "").strip().upper()
        if factuality not in TARGET_FACTUALITY:
            continue

        media_link = (data.get("media link") or "").strip()
        if not media_link:
            continue

        sources.append(
            {
                "name": data.get("media name", json_file.stem),
                "url": media_link,
                "factuality": factuality,
                "source_file": json_file.name,
            }
        )
    return sources


def build_captured_slugs(output_dir: Path) -> set:
    """Build a set of slugs that already have a screenshot, scanned once upfront."""
    slugs = set()
    # Strip the trailing _YYYYMMDD_HHMMSS timestamp — works for any slug including
    # ones with underscores (e.g. BBC_News_20260604_120000 → BBC_News).
    ts_pattern = re.compile(r"_\d{8}_\d{6}$")
    for ext in ("*.png", "*.jpeg"):
        for f in output_dir.glob(ext):
            slug = ts_pattern.sub("", f.stem)
            if slug:
                slugs.add(slug)
    return slugs


def load_timeout_urls(log_path: Path) -> set:
    """Return deduplicated URLs that had a page load timeout in a previous run."""
    urls = set()
    if not log_path.exists():
        return urls
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "error" and "Timeout" in r.get("error", ""):
                urls.add(r["url"])
    return urls


def load_error_urls_from_logs(log_paths: list[Path]) -> set:
    """Return deduplicated URLs whose latest status across all given logs is 'error'."""
    latest = {}
    for log_path in log_paths:
        if not log_path.exists():
            print(f"[WARN] Log not found, skipping: {log_path}", file=sys.stderr)
            continue
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                latest[r["url"]] = r["status"]
    return {url for url, status in latest.items() if status == "error"}


def load_csv_sources(csv_path: Path) -> list[dict]:
    """Load sources from a CSV that has at least a 'url' column."""
    sources = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            url = row.get("url", "").strip()
            if url:
                sources.append({"name": url, "url": url, "factuality": ""})
    return sources


def capture_one(source: dict, output_dir: Path, timeout_ms: int) -> dict:
    url = source["url"]
    label = source.get("name") or None
    try:
        path = take_snapshot(
            url=url,
            output_dir=output_dir,
            label=label,
            full_page=True,
            timeout_ms=timeout_ms,
        )
        return {"status": "ok", "url": url, "path": str(path), "name": source.get("name", "")}
    except Exception as e:
        return {"status": "error", "url": url, "error": str(e), "name": source.get("name", "")}


def main():
    parser = argparse.ArgumentParser(description="Batch snapshot media sources by factuality filter.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Directory containing source JSON files")
    parser.add_argument("-o", "--output", default="output", help="Output directory for screenshots (default: output)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel browser workers (default: 4)")
    parser.add_argument("--timeout", type=int, default=30000, help="Per-page timeout in ms (default: 30000)")
    parser.add_argument("--force", action="store_true", help="Re-capture even if a screenshot already exists")
    parser.add_argument("--retry-timeouts", action="store_true", help="Only retry URLs that previously timed out")
    parser.add_argument("--from-csv", metavar="CSV", help="Run only the URLs listed in this CSV file (must have a 'url' column)")
    parser.add_argument("--from-logs", nargs="+", metavar="LOG", help="Retry all error-status URLs found across one or more log files")
    parser.add_argument("--dry-run", action="store_true", help="Print qualifying URLs without capturing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    log_path = output_dir / "batch_log.jsonl"

    if args.from_logs:
        error_urls = load_error_urls_from_logs([Path(p) for p in args.from_logs])
        if not error_urls:
            print("No error-status URLs found in the provided logs. Nothing to run.")
            return 0
        # Enrich with media names from the JSON source files
        url_to_name = {s["url"]: s["name"] for s in load_qualifying_sources(data_dir)}
        sources = [
            {"name": url_to_name.get(url, url), "url": url, "factuality": ""}
            for url in sorted(error_urls)
        ]
        print(f"Found {len(sources)} error-status URLs across {len(args.from_logs)} log(s) (timeout={args.timeout}ms)")
    elif args.from_csv:
        sources = load_csv_sources(Path(args.from_csv))
        print(f"Loaded {len(sources)} URLs from {args.from_csv}")
    else:
        sources = load_qualifying_sources(data_dir)
        print(f"Found {len(sources)} qualifying sources with factuality in {TARGET_FACTUALITY}")

    if args.retry_timeouts:
        timeout_urls = load_timeout_urls(log_path)
        if not timeout_urls:
            print(f"No timeout errors found in {log_path}. Nothing to retry.")
            return 0
        sources = [s for s in sources if s["url"] in timeout_urls]
        print(f"Retrying {len(sources)} URLs that previously timed out (timeout={args.timeout}ms)")
    elif not args.force and output_dir.exists():
        from snapshot import slugify_url
        captured = build_captured_slugs(output_dir)
        before = len(sources)
        sources = [s for s in sources if slugify_url(s["url"]) not in captured]
        print(f"Skipping {before - len(sources)} already-captured, {len(sources)} remaining")

    if args.dry_run:
        for s in sources:
            print(f"[{s['factuality']:12s}] {s['url']}  ({s['name']})")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    err_count = 0
    start = time.time()

    with open(log_path, "a", encoding="utf-8") as log_fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(capture_one, s, output_dir, args.timeout): s
                for s in sources
            }
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                log_fh.write(json.dumps(result) + "\n")
                log_fh.flush()

                elapsed = time.time() - start
                if result["status"] == "ok":
                    ok_count += 1
                    print(f"[{i}/{len(sources)}] OK    {result['url']}  -> {result['path']}")
                else:
                    err_count += 1
                    print(f"[{i}/{len(sources)}] ERROR {result['url']}  -- {result['error']}")

                # ETA
                rate = i / elapsed if elapsed > 0 else 0
                remaining = len(sources) - i
                eta = remaining / rate if rate > 0 else 0
                print(f"         elapsed={elapsed:.0f}s  eta≈{eta:.0f}s  ok={ok_count}  err={err_count}")

    print(f"\nDone. {ok_count} captured, {err_count} errors. Log: {log_path}")
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
