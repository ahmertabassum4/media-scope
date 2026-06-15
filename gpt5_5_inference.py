# Joint inference over a stratified sample drawn from snapshots.csv (all outlets).
# Reads image_path values directly from the CSV — paths like output/Foo.png and
# Mixed_output/Foo_ts.png are resolved relative to the project root.
# Writes to bias_results/gpt-5-5_joint_engineered_large.jsonl (resume-safe).
#
#   python inference_joint_large.py
#   python inference_joint_large.py --workers 6

import argparse
import csv
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

from inference_multiclass import encode_image, query_model
from inference_joint import DIMENSIONS, PROMPTS, parse_joint

load_dotenv()
API_KEY = os.getenv("OPENROUTER_KEY")

SNAPSHOTS   = "snapshots.csv"
RESULTS_DIR = Path("bias_results")
OUT_PATH    = RESULTS_DIR / "gpt-5-5_joint_engineered_large.jsonl"

MODEL_SHORT = "gpt-5-5"
MODEL_ID    = "openai/gpt-5.5"

SEED = 14

CAPS = {
    "VERY LOW":  None,
    "LOW":       None,
    "MIXED":     200,
    "HIGH":      200,
    "VERY HIGH": None,
}


def load_snapshots(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stratified_sample(rows):
    random.seed(SEED)
    buckets = {k: [] for k in CAPS}
    for row in rows:
        fact = (row.get("factuality") or "").strip().upper()
        if fact in buckets:
            buckets[fact].append(row)

    sampled = []
    for fact, pool in buckets.items():
        cap = CAPS[fact]
        n   = min(len(pool), cap) if cap else len(pool)
        sampled.extend(random.sample(pool, n))
        print(f"  {fact:<10} pool={len(pool):>4}  sampled={n}")

    random.shuffle(sampled)
    return sampled


def load_done(path):
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["media_name"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def process_row(row, system, user_text):
    """Run inference for a single row. Returns (record, tag, summary) or (None, ...) on missing image."""
    media_name = row["media_name"]
    img_path   = row["image_path"].strip()

    if not os.path.exists(img_path):
        return None, "MISSING", media_name

    truth = {}
    for dim, (col, labels) in DIMENSIONS.items():
        val = (row.get(col) or "").strip().upper()
        truth[dim] = val if val in labels else ""

    # Retry up to 3 times on 429 with exponential backoff
    for attempt in range(3):
        try:
            image_b64 = encode_image(img_path)
            response  = query_model(MODEL_ID, system, user_text, image_b64)
            verdicts  = parse_joint(response)

            record = {
                "media_name":    media_name,
                "model":         MODEL_SHORT,
                "task":          "joint",
                "prompt":        "engineered",
                "verdicts":      verdicts,
                "ground_truth":  truth,
                "full_response": response,
                "factuality":    row["factuality"],
            }

            miss    = [d for d, v in verdicts.items() if v == "UNKNOWN"]
            tag     = "OK" if not miss else f"missing:{','.join(miss)}"
            summary = " ".join(f"{d[0].upper()}={verdicts[d]}" for d in DIMENSIONS)
            return record, tag, summary

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                wait = 10 * (2 ** attempt)
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"429 after 3 retries — {media_name}")


def run(workers):
    system, user_text = PROMPTS["engineered"]

    all_rows = load_snapshots(SNAPSHOTS)
    print("Sampling:")
    rows = stratified_sample(all_rows)
    print(f"Total: {len(rows)}\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    done = load_done(OUT_PATH)
    if done:
        print(f"Resuming — {len(done)} already done\n")

    pending = [r for r in rows
               if r["media_name"] not in done
               and os.path.exists(r["image_path"].strip())]

    write_lock = threading.Lock()
    completed  = 0
    total      = len(pending)

    with open(OUT_PATH, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_row, row, system, user_text): row
                       for row in pending}

            for future in as_completed(futures):
                row        = futures[future]
                media_name = row["media_name"]
                completed += 1

                try:
                    record, tag, summary = future.result()
                    if record is None:
                        print(f"  [{completed}/{total}] MISSING IMAGE — {media_name}")
                        continue

                    with write_lock:
                        out_f.write(json.dumps(record) + "\n")
                        out_f.flush()

                    print(f"  [{completed}/{total}] {tag:<22} {summary:<40} {media_name}")

                except Exception as e:
                    print(f"  [{completed}/{total}] ERROR — {media_name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel API workers (default: 5)")
    args = parser.parse_args()
    run(args.workers)