# eval_multiclass_mixed.py
# Evaluate multiclass results on the MIXED subset (factuality 5-class / genre / bias).
#   python eval_multiclass_mixed.py                  # all tasks found
#   python eval_multiclass_mixed.py --task factuality
 
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
 
TASK_LABELS = {
    "factuality": FACT_LABELS,
    "genre":      GENRE_LABELS,
    "bias":       BIAS_LABELS,
}
 
 
def load_results(results_dir):
    records = []
    for path in glob.glob(os.path.join(results_dir, "*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "task" in rec and "verdict" in rec:
                    records.append(rec)
    return pd.DataFrame(records)
 
 
def compute_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {}
    for i, lab in enumerate(labels):
        per_class[lab] = {"precision": round(prec[i], 4), "recall": round(rec[i], 4),
                          "f1": round(f1[i], 4), "support": int(support[i])}
    return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "per_class": per_class, "n": len(y_true)}
 
 
def print_summary(task, summary):
    print(f"\n=== {task} ===")
    header = f"{'Model':<22} {'Prompt':<12} {'N':>5} {'Acc':>7} {'MacroF1':>8}"
    print(header); print("-" * len(header))
    for row in summary:
        print(f"{row['model']:<22} {row['prompt']:<12} {row['n']:>5} "
              f"{row['accuracy']:>7.4f} {row['macro_f1']:>8.4f}")
 
 
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
 
 
def plot_macro_f1(task, summary, out_path):
    df = pd.DataFrame(summary)
    df["label"] = df["model"] + "\n(" + df["prompt"] + ")"
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 1.2), 4))
    bars = ax.bar(df["label"], df["macro_f1"], color="steelblue",
                  edgecolor="white", width=0.5)
    ax.set_ylim(0, 1); ax.set_ylabel("Macro-F1")
    ax.set_title(f"Macro-F1 by Model & Prompt — {task}")
    for bar, val in zip(bars, df["macro_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"  saved {out_path}")
 
 
def plot_per_class_accuracy(task, df, labels, out_path):
    df = df.copy()
    df["correct"] = (df["predicted"] == df["ground_truth"]).astype(int)
    for prompt in df["prompt"].unique():
        sub = df[df["prompt"] == prompt]
        grouped = (sub.groupby(["model", "ground_truth"])["correct"]
                   .mean().reset_index())
        pivot = (grouped.pivot(index="ground_truth", columns="model",
                               values="correct").reindex(labels))
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
        pivot.plot(kind="bar", ax=ax, edgecolor="white")
        ax.set_ylim(0, 1); ax.set_ylabel("Per-class accuracy (recall)")
        ax.set_title(f"Per-class accuracy — {task} / {prompt} prompt")
        ax.set_xlabel(""); ax.legend(title="Model", bbox_to_anchor=(1, 1))
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        save_path = out_path.replace(".png", f"_{prompt}.png")
        plt.savefig(save_path, dpi=150); plt.close()
        print(f"  saved {save_path}")
 
 
def run(only_task):
    all_summaries = []
    for task, labels in TASK_LABELS.items():
        if only_task and task != only_task:
            continue
        results_dir = str(RESULTS_DIRS[task])
        plots_dir   = PLOTS_DIRS[task]
        os.makedirs(plots_dir, exist_ok=True)
 
        df = load_results(results_dir)
        if df.empty:
            print(f"No results in {results_dir}, skipping {task}")
            continue
 
        df = df[df["task"] == task].copy()
        df["predicted"]    = df["verdict"].str.upper().str.strip()
        df["ground_truth"] = df["ground_truth"].str.upper().str.strip()
 
        unknown = (~df["predicted"].isin(labels)).sum()
        if unknown:
            print(f"[{task}] {unknown} unparseable verdicts; counted as wrong.")
        df.loc[~df["predicted"].isin(labels), "predicted"] = "__UNKNOWN__"
 
        summary = []
        for (model, prompt), group in df.groupby(["model", "prompt"]):
            y_true = group["ground_truth"].tolist()
            y_pred = group["predicted"].tolist()
            m = compute_metrics(y_true, y_pred, labels)
            row = {"task": task, "model": model, "prompt": prompt,
                   "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "n": m["n"]}
            for lab, vals in m["per_class"].items():
                row[f"{lab}_precision"] = vals["precision"]
                row[f"{lab}_recall"]    = vals["recall"]
                row[f"{lab}_f1"]        = vals["f1"]
                row[f"{lab}_support"]   = vals["support"]
            summary.append(row)
 
            safe = f"{model}_{task}_{prompt}".replace(" ", "_")
            plot_confusion_matrix(y_true, y_pred, labels,
                                  f"{model} / {task} / {prompt}",
                                  os.path.join(plots_dir, f"cm_{safe}.png"))
 
        print_summary(task, summary)
        plot_macro_f1(task, summary, os.path.join(plots_dir, f"macro_f1_{task}.png"))
        plot_per_class_accuracy(task, df, labels,
                                os.path.join(plots_dir, f"per_class_acc_{task}.png"))
 
        out = os.path.join(results_dir, f"summary_{task}.csv")
        pd.DataFrame(summary).to_csv(out, index=False)
        print(f"  summary saved to {out}")
        all_summaries.extend(summary)
 
    if all_summaries:
        os.makedirs(PLOTS_DIRS["factuality"].rsplit("/", 1)[0], exist_ok=True)
        overview = pd.DataFrame(all_summaries)[
            ["task", "model", "prompt", "n", "accuracy", "macro_f1"]]
        out = os.path.join(PLOTS_DIRS["factuality"].rsplit("/", 1)[0],
                           "summary_multiclass_mixed.csv")
        overview.to_csv(out, index=False)
        print(f"\nOverview saved to {out}")
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASK_LABELS), default=None)
    args = p.parse_args()
    run(args.task)