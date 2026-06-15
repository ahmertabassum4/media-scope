# Ordinal error metrics (MSE / RMSE / MAE) for the 5-class mixed factuality task.
# Maps VERY LOW..VERY HIGH -> 0..4 and reports error per (source, model, prompt).
#
#   python eval_mse_mixed.py                         # scans both mixed folders
#   python eval_mse_mixed.py --results mixed_results_factuality
#   python eval_mse_mixed.py --results A B           # any number of folders
#
# Reads both record shapes used in this project:
#   - flat  (mixed_results_factuality): {"verdict": "...", "ground_truth": "..."}
#   - joint (mixed_results_joint):      {"verdicts": {"factuality": "..."},
#                                        "ground_truth": {"factuality": "..."}}
# No assumption is made about the "task" string.

import os
import json
import argparse
import glob

import numpy as np
import pandas as pd

DEFAULT_DIRS = ["mixed_results_factuality", "mixed_results_joint"]

# Ordinal scale. Index position is the score used for the squared error.
SCALE = ["VERY LOW", "LOW", "MIXED", "HIGH", "VERY HIGH"]
SCORE = {label: i for i, label in enumerate(SCALE)}


def norm(value):
    """Upper-case, strip, collapse internal whitespace; return None if off-scale."""
    if not isinstance(value, str):
        return None
    v = " ".join(value.strip().upper().split())
    return v if v in SCORE else None


def extract(rec):
    """Pull (model, prompt, pred, truth) from either record shape, or None."""
    model  = rec.get("model", "?")
    prompt = rec.get("prompt", "?")

    if "verdict" in rec:  # flat
        return model, prompt, norm(rec.get("verdict")), norm(rec.get("ground_truth"))

    verdicts = rec.get("verdicts")  # joint
    gt = rec.get("ground_truth")
    if isinstance(verdicts, dict):
        truth = gt.get("factuality") if isinstance(gt, dict) else None
        return model, prompt, norm(verdicts.get("factuality")), norm(truth)

    return None


def load_dir(results_dir):
    raw = []
    for path in glob.glob(os.path.join(results_dir, "*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return raw


def compute_ordinal(y_true, y_pred):
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    mse = float(np.mean(err ** 2))
    return {
        "mse":  round(mse, 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "mae":  round(float(np.mean(np.abs(err))), 4),
        "n":    len(y_true),
    }


def print_summary(summary):
    header = (f"{'Source':<24} {'Model':<22} {'Prompt':<12} {'N':>5} "
              f"{'MSE':>8} {'RMSE':>8} {'MAE':>8}")
    print("\n" + header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['source']:<24} {row['model']:<22} {row['prompt']:<12} "
            f"{row['n']:>5} {row['mse']:>8.4f} {row['rmse']:>8.4f} {row['mae']:>8.4f}"
        )


def run(results_dirs):
    rows, dropped = [], 0
    found_any = False

    for results_dir in results_dirs:
        raw = load_dir(results_dir)
        if not raw:
            print(f"(no .jsonl records under {results_dir!r})")
            continue
        found_any = True
        source = os.path.basename(results_dir.rstrip("/")) or results_dir
        for rec in raw:
            parsed = extract(rec)
            if parsed is None:
                continue
            model, prompt, pred, truth = parsed
            if pred is None or truth is None:
                dropped += 1
                continue
            rows.append({"source": source, "model": model, "prompt": prompt,
                         "y_true": SCORE[truth], "y_pred": SCORE[pred]})

    if not found_any:
        print("No result folders found. Pass --results with the right path(s).")
        return
    if not rows:
        print("Records loaded but none carried an on-scale factuality "
              "prediction + ground truth.")
        return

    if dropped:
        print(f"{dropped} records skipped (off-scale or UNKNOWN prediction/truth).")

    df = pd.DataFrame(rows)
    summary = []
    for (source, model, prompt), group in df.groupby(["source", "model", "prompt"]):
        metrics = compute_ordinal(group["y_true"].tolist(),
                                  group["y_pred"].tolist())
        summary.append({"source": source, "model": model, "prompt": prompt,
                        **metrics})

    print_summary(summary)

    out = "summary_mse_mixed.csv"
    pd.DataFrame(summary).to_csv(out, index=False)
    print(f"\n  summary saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", default=DEFAULT_DIRS,
                        help="One or more result folders (default: both mixed folders)")
    args = parser.parse_args()
    run(args.results)