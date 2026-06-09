# Reads two result sets from results/:
#   isolated  — {model}_{task}_{prompt}.jsonl   (binary / factuality / genre / bias)
#   joint     — {model}_joint_{prompt}.jsonl    (factuality + genre + bias in one call)

import os
import json
import glob
import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

RESULTS_DIR = "results"

# Per-axis label spaces. The "binary" axis is the standalone trustworthiness
# task; in the joint run it is derived from factuality.
AXIS_LABELS = {
    "binary":      ["NOT FACTUAL", "FACTUAL"],
    "factuality":  ["VERY LOW", "LOW", "HIGH", "VERY HIGH"],
    "genre":       ["CONSPIRACY", "PSEUDOSCIENCE", "IMPOSTER", "OTHER"],
    "bias":        ["LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT"],
}

# How each axis is named inside a joint record's dicts. binary is derived.
JOINT_DIM = {"factuality": "factuality", "genre": "genre", "bias": "bias"}


def _norm(s):
    return (s or "").strip().upper()


def factuality_to_trust(tier):
    t = _norm(tier)
    if t in ("HIGH", "VERY HIGH"):
        return "FACTUAL"
    if t in ("LOW", "VERY LOW"):
        return "NOT FACTUAL"
    return None


def _binary_gt_to_label(gt):
    # Isolated binary task stores ground_truth as 1/0 (trustworthiness column).
    s = str(gt).strip()
    if s in ("1", "1.0"):
        return "FACTUAL"
    if s in ("0", "0.0"):
        return "NOT FACTUAL"
    return _norm(gt)


def metrics(y_true, y_pred, labels):
    if not y_true:
        return None
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0), 4),
    }


def score_isolated(results_dir):
    """One row per (axis, model, prompt) from isolated single-task jsonl files."""
    rows = []
    pattern = os.path.join(results_dir, "*.jsonl")
    for path in glob.glob(pattern):
        recs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Only flat single-task records (joint records have "verdicts").
                if "verdicts" in r or r.get("task") == "joint":
                    continue
                recs.append(r)
        if not recs:
            continue

        axis = recs[0].get("task")
        if axis not in AXIS_LABELS:
            continue
        labels = AXIS_LABELS[axis]

        by_key = {}
        for r in recs:
            by_key.setdefault((r["model"], r["prompt"]), []).append(r)

        for (model, prompt), group in by_key.items():
            y_true, y_pred = [], []
            for r in group:
                if axis == "binary":
                    gt = _binary_gt_to_label(r["ground_truth"])
                else:
                    gt = _norm(r["ground_truth"])
                if gt not in labels:
                    continue
                pred = _norm(r["verdict"])
                y_true.append(gt)
                y_pred.append(pred if pred in labels else "__UNKNOWN__")
            m = metrics(y_true, y_pred, labels)
            if m:
                rows.append({"axis": axis, "model": model, "prompt": prompt,
                             "condition": "isolated", **m})
    return rows


def score_joint(results_dir):
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

    rows = []
    by_key = {}
    for r in recs:
        by_key.setdefault((r["model"], r["prompt"]), []).append(r)

    for (model, prompt), group in by_key.items():
        # Three predicted axes.
        for axis, dim in JOINT_DIM.items():
            labels = AXIS_LABELS[axis]
            y_true, y_pred = [], []
            for r in group:
                gt = _norm(r["ground_truth"].get(dim))
                if gt not in labels:
                    continue
                pred = _norm(r["verdicts"].get(dim))
                y_true.append(gt)
                y_pred.append(pred if pred in labels else "__UNKNOWN__")
            m = metrics(y_true, y_pred, labels)
            if m:
                rows.append({"axis": axis, "model": model, "prompt": prompt,
                             "condition": "joint", **m})

        # Derived binary from factuality.
        labels = AXIS_LABELS["binary"]
        y_true, y_pred = [], []
        for r in group:
            gt = factuality_to_trust(r["ground_truth"].get("factuality"))
            if gt is None:
                continue
            pred = factuality_to_trust(r["verdicts"].get("factuality"))
            y_true.append(gt)
            y_pred.append(pred if pred is not None else "__UNKNOWN__")
        m = metrics(y_true, y_pred, labels)
        if m:
            rows.append({"axis": "binary", "model": model, "prompt": prompt,
                         "condition": "joint", **m})
    return rows


def run(results_dir):
    iso = score_isolated(results_dir)
    joint = score_joint(results_dir)
    if not iso and not joint:
        print("No results found.")
        return

    df = pd.DataFrame(iso + joint)

    # Pivot so isolated and joint sit side by side per (axis, model, prompt).
    wide = df.pivot_table(
        index=["axis", "model", "prompt"],
        columns="condition",
        values=["accuracy", "macro_f1", "n"],
        aggfunc="first",
    )
    # Flatten the column MultiIndex.
    wide.columns = [f"{metric}_{cond}" for metric, cond in wide.columns]
    wide = wide.reset_index()

    # Deltas (joint - isolated): positive means joint helped.
    for metric in ("accuracy", "macro_f1"):
        i, j = f"{metric}_isolated", f"{metric}_joint"
        if i in wide and j in wide:
            wide[f"{metric}_delta"] = (wide[j] - wide[i]).round(4)

    axis_order = {"binary": 0, "factuality": 1, "genre": 2, "bias": 3}
    wide["__o"] = wide["axis"].map(axis_order).fillna(9)
    wide = wide.sort_values(["__o", "model", "prompt"]).drop(columns="__o")

    # Console view, per axis.
    show_cols = [c for c in [
        "model", "prompt",
        "accuracy_isolated", "accuracy_joint", "accuracy_delta",
        "macro_f1_isolated", "macro_f1_joint", "macro_f1_delta",
    ] if c in wide.columns]

    for axis in sorted(wide["axis"].unique(), key=lambda a: axis_order.get(a, 9)):
        sub = wide[wide["axis"] == axis]
        print(f"\n=== {axis}: joint vs isolated ===")
        print(sub[show_cols].to_string(index=False))

    out = os.path.join(results_dir, "summary_joint_vs_isolated.csv")
    wide.to_csv(out, index=False)
    print(f"\nComparison saved to {out}")

    # Compact aggregate: mean delta per axis (averaged over models/prompts).
    if "macro_f1_delta" in wide.columns:
        agg = (wide.groupby("axis")[["accuracy_delta", "macro_f1_delta"]]
               .mean().round(4).reset_index())
        agg["__o"] = agg["axis"].map(axis_order).fillna(9)
        agg = agg.sort_values("__o").drop(columns="__o")
        print("\n=== mean delta (joint - isolated), averaged over model & prompt ===")
        print(agg.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS_DIR)
    args = parser.parse_args()
    run(args.results)