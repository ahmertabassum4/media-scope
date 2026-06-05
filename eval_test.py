import os
import json
import argparse
import glob
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

RESULTS_DIR = "results"
PLOTS_DIR   = "eval_plots"


def load_results(results_dir):
    records = []
    for path in glob.glob(os.path.join(results_dir, "*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(records)


def verdict_to_label(verdict):
    # FACTUAL -> 1, NOT FACTUAL -> 0, anything else -> NaN
    if verdict == "FACTUAL":
        return 1
    if verdict == "NOT FACTUAL":
        return 0
    return None


def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "n":         len(y_true),
    }


def print_summary(summary):
    header = f"{'Model':<22} {'Prompt':<12} {'N':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}"
    print("\n" + header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['model']:<22} {row['prompt']:<12} {row['n']:>5} "
            f"{row['accuracy']:>7.4f} {row['precision']:>7.4f} "
            f"{row['recall']:>7.4f} {row['f1']:>7.4f}"
        )


def plot_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NOT FACTUAL", "FACTUAL"],
        yticklabels=["NOT FACTUAL", "FACTUAL"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved to {out_path}")


def plot_f1_comparison(summary, out_path):
    df = pd.DataFrame(summary)
    df["label"] = df["model"] + "\n(" + df["prompt"] + ")"

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 1.2), 4))
    bars = ax.bar(df["label"], df["f1"], color="steelblue", edgecolor="white", width=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score by Model & Prompt")
    for bar, val in zip(bars, df["f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved to {out_path}")


def plot_factuality_breakdown(df, out_path):
    # For each (model, prompt, factuality) triplet, calculate fraction gotten right
    df = df.copy()
    df["correct"] = (df["predicted"] == df["ground_truth"]).astype(int)

    grouped = (
        df.groupby(["model", "prompt", "factuality"])["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )

    models = grouped["model"].unique()
    prompts = grouped["prompt"].unique()

    for prompt in prompts:
        subset = grouped[grouped["prompt"] == prompt]
        pivot = subset.pivot(index="factuality", columns="model", values="accuracy")

        fig, ax = plt.subplots(figsize=(max(5, len(models) * 1.5), 4))
        pivot.plot(kind="bar", ax=ax, edgecolor="white")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Per-factuality accuracy — {prompt} prompt")
        ax.set_xlabel("")
        ax.legend(title="Model", bbox_to_anchor=(1, 1))
        plt.xticks(rotation=0)
        plt.tight_layout()
        save_path = out_path.replace(".png", f"_{prompt}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  saved to {save_path}")


def run(results_dir):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_results(results_dir)
    if df.empty:
        print("No results found")
        return

    df["predicted"] = df["verdict"].apply(verdict_to_label)

    unknown = df["predicted"].isna().sum()
    if unknown > 0:
        print(f"\nOOF {unknown} rows have unparseable verdicts and will be skipped.")
    df = df.dropna(subset=["predicted"])
    df["predicted"]    = df["predicted"].astype(int)
    df["ground_truth"] = df["ground_truth"].astype(int)

    summary = []

    for (model, prompt), group in df.groupby(["model", "prompt"]):
        y_true = group["ground_truth"].tolist()
        y_pred = group["predicted"].tolist()

        metrics = compute_metrics(y_true, y_pred)
        metrics.update({"model": model, "prompt": prompt})
        summary.append(metrics)

        title     = f"{model} / {prompt}"
        safe_name = f"{model}_{prompt}".replace(" ", "_")
        plot_confusion_matrix(
            y_true, y_pred,
            title=title,
            out_path=os.path.join(PLOTS_DIR, f"cm_{safe_name}.png"),
        )

    print_summary(summary)

    plot_f1_comparison(
        summary,
        out_path=os.path.join(PLOTS_DIR, "f1_comparison.png"),
    )

    plot_factuality_breakdown(
        df,
        out_path=os.path.join(PLOTS_DIR, "factuality_breakdown.png"),
    )

    # Save summary table to CSV as well
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=RESULTS_DIR,
        help="Folder containing .jsonl inference outputs (default: results/)",
    )
    args = parser.parse_args()
    run(args.results)