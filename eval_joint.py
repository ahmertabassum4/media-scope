# Evaluate joint-prediction results (factuality + genre + bias in one call). Just enter python eval_joint.py


import os
import json
import argparse
import glob

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
)

RESULTS_DIR = "results"
PLOTS_DIR   = "eval_plots"

DIM_LABELS = {
    "factuality": ["VERY LOW", "LOW", "HIGH", "VERY HIGH"],
    "genre":      ["CONSPIRACY", "PSEUDOSCIENCE", "IMPOSTER", "OTHER"],
    "bias":       ["LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT"],
}

# Trustworthiness is not predicted directly in the joint prompt; it is derived
# from the factuality verdict so it lines up with the standalone binary task.
# Label space matches the binary task exactly (FACTUAL / NOT FACTUAL).
TRUST_LABELS = ["NOT FACTUAL", "FACTUAL"]


def factuality_to_trust(tier):
    """HIGH / VERY HIGH -> FACTUAL ; LOW / VERY LOW -> NOT FACTUAL ; else None."""
    t = (tier or "").strip().upper()
    if t in ("HIGH", "VERY HIGH"):
        return "FACTUAL"
    if t in ("LOW", "VERY LOW"):
        return "NOT FACTUAL"
    return None

# Genres whose presence implies the model should also see low reliability.
PROBLEMATIC_GENRE = {"CONSPIRACY", "PSEUDOSCIENCE", "IMPOSTER"}
# Reliable factuality tiers, for the genre<->factuality consistency check.
UNRELIABLE_FACT = {"VERY LOW", "LOW"}


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
    """Explode each joint record into one row per dimension.

    Adds a derived 'trustworthiness' dimension from the factuality verdict so
    it can be compared against the standalone binary task. A factuality verdict
    that does not map (UNKNOWN) yields a __UNKNOWN__ trust prediction, counted
    as wrong; a row with no factuality ground truth is dropped downstream.
    """
    rows = []
    for rec in records:
        for dim, labels in DIM_LABELS.items():
            gt = (rec["ground_truth"].get(dim) or "").strip().upper()
            pred = (rec["verdicts"].get(dim) or "UNKNOWN").strip().upper()
            rows.append({
                "media_name": rec["media_name"],
                "model": rec["model"],
                "prompt": rec["prompt"],
                "dimension": dim,
                "ground_truth": gt,
                "predicted": pred if pred in labels else "__UNKNOWN__",
            })

        # Derived trustworthiness from factuality.
        gt_trust   = factuality_to_trust(rec["ground_truth"].get("factuality"))
        pred_trust = factuality_to_trust(rec["verdicts"].get("factuality"))
        rows.append({
            "media_name": rec["media_name"],
            "model": rec["model"],
            "prompt": rec["prompt"],
            "dimension": "trustworthiness",
            "ground_truth": gt_trust or "",
            "predicted": pred_trust or "__UNKNOWN__",
        })
    return pd.DataFrame(rows)


def compute_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
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
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title, fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path}")


def consistency(records):
    """Internal agreement between the model's own joint labels.

    Two checks, computed per (model, prompt) over rows where both relevant
    predictions parsed:
      - genre_vs_fact: when the model assigns a problematic genre (CONSPIRACY,
        PSEUDOSCIENCE, or IMPOSTER), does it also place factuality in an
        unreliable tier (VERY LOW / LOW)? Reported as the fraction that agree.
      - other_vs_fact: when the model says OTHER, does it place factuality in a
        reliable tier (HIGH / VERY HIGH)?
    These measure the model's self-consistency, independent of ground truth.
    """
    rows = []
    by_key = {}
    for rec in records:
        by_key.setdefault((rec["model"], rec["prompt"]), []).append(rec)

    for (model, prompt), recs in sorted(by_key.items()):
        g_prob = g_prob_ok = 0   # genre is problematic (conspiracy/pseudo/imposter)
        g_other = g_other_ok = 0 # genre is OTHER
        for rec in recs:
            genre = rec["verdicts"].get("genre", "UNKNOWN")
            fact  = rec["verdicts"].get("factuality", "UNKNOWN")
            if fact not in DIM_LABELS["factuality"]:
                continue
            if genre in PROBLEMATIC_GENRE:
                g_prob += 1
                if fact in UNRELIABLE_FACT:
                    g_prob_ok += 1
            elif genre == "OTHER":
                g_other += 1
                if fact not in UNRELIABLE_FACT:
                    g_other_ok += 1
        rows.append({
            "model": model,
            "prompt": prompt,
            "problematic_genre_n": g_prob,
            "genre_vs_fact_agree": round(g_prob_ok / g_prob, 4) if g_prob else None,
            "other_genre_n": g_other,
            "other_vs_fact_agree": round(g_other_ok / g_other, 4) if g_other else None,
        })
    return pd.DataFrame(rows)


def run(results_dir):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    records = load_joint(results_dir)
    if not records:
        print("No joint results found")
        return

    long = to_long(records)

    # Evaluate the three predicted axes plus the derived binary trustworthiness.
    eval_dims = dict(DIM_LABELS)
    eval_dims["trustworthiness"] = TRUST_LABELS

    for dim, labels in eval_dims.items():
        ddf = long[long["dimension"] == dim]
        # Drop rows with no ground-truth label for this dimension.
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
            plot_confusion_matrix(
                y_true, y_pred, labels,
                title=f"{model} / joint:{dim} / {prompt}",
                out_path=os.path.join(PLOTS_DIR, f"cm_{safe}.png"),
            )

        sdf = pd.DataFrame(summary)
        print(f"\n=== joint / {dim} ===")
        cols = ["model", "prompt", "n", "accuracy", "macro_f1"]
        print(sdf[cols].to_string(index=False))
        out = os.path.join(RESULTS_DIR, f"summary_joint_{dim}.csv")
        sdf.to_csv(out, index=False)
        print(f"  summary saved to {out}")

    cons = consistency(records)
    print("\n=== joint / internal consistency ===")
    print(cons.to_string(index=False))
    out = os.path.join(RESULTS_DIR, "summary_joint_consistency.csv")
    cons.to_csv(out, index=False)
    print(f"  consistency saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS_DIR)
    args = parser.parse_args()
    run(args.results)