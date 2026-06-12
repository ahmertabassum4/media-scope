# eval_joint_mixed.py
# Evaluate joint-prediction results on the MIXED subset (factuality 5-class + genre
# + bias in one call). Mirrors eval_joint.py but with the 5-class scale and the
# eval_plots_mixed/eval_plots_joint output folder.
#   python eval_joint_mixed.py

import os
import json
import glob
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix,
)

from mixed_config import (
    FACT_LABELS, BIAS_LABELS, GENRE_LABELS, RESULTS_DIRS, PLOTS_DIRS,
)

RESULTS_DIR = str(RESULTS_DIRS["joint"])
PLOTS_DIR   = PLOTS_DIRS["joint"]

DIM_LABELS = {
    "factuality": FACT_LABELS,
    "genre":      GENRE_LABELS,
    "bias":       BIAS_LABELS,
}

# Derived trustworthiness, now with MIXED treated as the unreliable half (so it
# stays binary and comparable to the standalone task). Adjust if you'd rather drop
# MIXED from the trust view.
TRUST_LABELS = ["NOT FACTUAL", "FACTUAL"]


def factuality_to_trust(tier):
    t = (tier or "").strip().upper()
    if t in ("HIGH", "VERY HIGH"):
        return "FACTUAL"
    if t in ("LOW", "VERY LOW", "MIXED"):
        return "NOT FACTUAL"
    return None


def load_joint(results_dir):
    records = []
    for path in glob.glob(os.path.join(results_dir, "*_joint_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("task") == "joint":
                    records.append(rec)
    return records


def to_long(records):
    rows = []
    for rec in records:
        for dim, labels in DIM_LABELS.items():
            gt = (rec["ground_truth"].get(dim) or "").strip().upper()
            pred = (rec["verdicts"].get(dim) or "UNKNOWN").strip().upper()
            rows.append({"media_name": rec["media_name"], "model": rec["model"],
                         "prompt": rec["prompt"], "dimension": dim,
                         "ground_truth": gt,
                         "predicted": pred if pred in labels else "__UNKNOWN__"})
        gt_trust   = factuality_to_trust(rec["ground_truth"].get("factuality"))
        pred_trust = factuality_to_trust(rec["verdicts"].get("factuality"))
        rows.append({"media_name": rec["media_name"], "model": rec["model"],
                     "prompt": rec["prompt"], "dimension": "trustworthiness",
                     "ground_truth": gt_trust or "",
                     "predicted": pred_trust or "__UNKNOWN__"})
    return pd.DataFrame(rows)


def compute_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    row = {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4), "n": len(y_true)}
    for i, lab in enumerate(labels):
        row[f"{lab}_precision"] = round(prec[i], 4)
        row[f"{lab}_recall"]    = round(rec[i], 4)
        row[f"{lab}_f1"]        = round(f1[i], 4)
        row[f"{lab}_support"]   = int(support[i])
    return row


def plot_confusion_matrix(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(4, n * 1.1), max(3.5, n)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    ax.set_title(title, fontsize=10)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"  saved {out_path}")


def consistency(records):
    """Fraction of rows where the model's own genre/factuality verdicts agree
    with the heuristic that problematic genres imply unreliable factuality."""
    PROBLEM_GENRE = {"CONSPIRACY", "PSEUDOSCIENCE", "IMPOSTER"}
    UNRELIABLE = {"VERY LOW", "LOW", "MIXED"}
    rows = []
    for (model, prompt), recs in _group(records):
        n = agree = 0
        for r in recs:
            g = (r["verdicts"].get("genre") or "").upper()
            fct = (r["verdicts"].get("factuality") or "").upper()
            if g in PROBLEM_GENRE:
                n += 1
                if fct in UNRELIABLE:
                    agree += 1
        rate = round(agree / n, 4) if n else float("nan")
        rows.append({"model": model, "prompt": prompt,
                     "problem_genre_n": n, "genre_factuality_consistency": rate})
    return pd.DataFrame(rows)


def _group(records):
    buckets = {}
    for r in records:
        buckets.setdefault((r["model"], r["prompt"]), []).append(r)
    return buckets.items()


def run():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    records = load_joint(RESULTS_DIR)
    if not records:
        print(f"No joint results in {RESULTS_DIR}")
        return

    long = to_long(records)

    eval_dims = dict(DIM_LABELS)
    eval_dims["trustworthiness"] = TRUST_LABELS

    for dim, labels in eval_dims.items():
        ddf = long[long["dimension"] == dim]
        ddf = ddf[ddf["ground_truth"].isin(labels)]
        if ddf.empty:
            continue
        unknown = (ddf["predicted"] == "__UNKNOWN__").sum()
        if unknown:
            print(f"[joint/{dim}] {unknown} unparseable predictions; counted as wrong.")

        summary = []
        for (model, prompt), group in ddf.groupby(["model", "prompt"]):
            y_true = group["ground_truth"].tolist()
            y_pred = group["predicted"].tolist()
            row = {"task": "joint", "dimension": dim, "model": model, "prompt": prompt}
            row.update(compute_metrics(y_true, y_pred, labels))
            summary.append(row)
            safe = f"{model}_joint_{dim}_{prompt}".replace(" ", "_")
            plot_confusion_matrix(y_true, y_pred, labels,
                                  f"{model} / joint:{dim} / {prompt}",
                                  os.path.join(PLOTS_DIR, f"cm_{safe}.png"))

        sdf = pd.DataFrame(summary)
        print(f"\n=== joint / {dim} ===")
        print(sdf[["model", "prompt", "n", "accuracy", "macro_f1"]].to_string(index=False))
        out = os.path.join(RESULTS_DIR, f"summary_joint_{dim}.csv")
        sdf.to_csv(out, index=False)
        print(f"  summary saved to {out}")

    cons = consistency(records)
    print("\n=== joint / internal consistency ===")
    print(cons.to_string(index=False))
    cons.to_csv(os.path.join(RESULTS_DIR, "summary_joint_consistency.csv"), index=False)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()