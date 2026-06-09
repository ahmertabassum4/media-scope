# Compare model performance on imposter vs non-imposter sources, across the
# factuality, trustworthiness, and bias axes. Reads predictions from the joint
# results and uses the index CSV's genre_class column to split the rows.
#
#   python imposter_breakdown.py
#
# Prints one table per axis: accuracy and macro-F1 for imposter rows vs the
# rest, per model and prompt. No plots.

import os
import csv
import json
import glob
import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

RESULTS_DIR  = "results"
SAMPLE_INDEX = "snapshot_sample_index.csv"

AXIS_LABELS = {
    "factuality":      ["VERY LOW", "LOW", "HIGH", "VERY HIGH"],
    "trustworthiness": ["NOT FACTUAL", "FACTUAL"],
    "bias":            ["LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT"],
}


def _norm(s):
    return (s or "").strip().upper()


def factuality_to_trust(tier):
    t = _norm(tier)
    if t in ("HIGH", "VERY HIGH"):
        return "FACTUAL"
    if t in ("LOW", "VERY LOW"):
        return "NOT FACTUAL"
    return None


def load_genre_split(index_path):
    """media_name -> 'imposter' or 'non-imposter' from the index genre_class."""
    split = {}
    with open(index_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gc = (row.get("genre_class") or "").strip().upper()
            split[row["media_name"]] = "imposter" if gc == "IMPOSTER" else "non-imposter"
    return split


def load_joint(results_dir):
    recs = []
    for path in glob.glob(os.path.join(results_dir, "*_joint_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("task") == "joint":
                    recs.append(r)
    return recs


def axis_truth_pred(rec, axis):
    """Return (ground_truth, predicted) strings for an axis, or (None, None)."""
    if axis == "trustworthiness":
        gt = factuality_to_trust(rec["ground_truth"].get("factuality"))
        pred = factuality_to_trust(rec["verdicts"].get("factuality"))
        return gt, (pred if pred is not None else "__UNKNOWN__")
    gt = _norm(rec["ground_truth"].get(axis))
    pred = _norm(rec["verdicts"].get(axis))
    if gt not in AXIS_LABELS[axis]:
        return None, None
    return gt, (pred if pred in AXIS_LABELS[axis] else "__UNKNOWN__")


def run(results_dir, index_path):
    split = load_genre_split(index_path)
    recs = load_joint(results_dir)
    if not recs:
        print("No joint results found.")
        return

    for axis, labels in AXIS_LABELS.items():
        rows = []
        # group by model, prompt, and imposter/non-imposter
        buckets = {}
        for rec in recs:
            grp = split.get(rec["media_name"], "non-imposter")
            gt, pred = axis_truth_pred(rec, axis)
            if gt is None:
                continue
            buckets.setdefault((rec["model"], rec["prompt"], grp), []).append((gt, pred))

        for (model, prompt, grp), pairs in buckets.items():
            y_true = [g for g, _ in pairs]
            y_pred = [p for _, p in pairs]
            rows.append({
                "model": model,
                "prompt": prompt,
                "group": grp,
                "n": len(pairs),
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "macro_f1": round(f1_score(y_true, y_pred, labels=labels,
                                           average="macro", zero_division=0), 4),
            })

        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values(["model", "prompt", "group"])
        print(f"\n=== {axis}: imposter vs non-imposter ===")
        print(df.to_string(index=False))

    # Note: for the bias axis, the imposter group is uniformly RIGHT in the
    # dataset, so its macro-F1 may be degenerate on a one-class slice.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--index", default=SAMPLE_INDEX)
    args = parser.parse_args()
    run(args.results, args.index)