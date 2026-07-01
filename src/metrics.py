import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dataset import CLASS_ORDER, ORDINAL


def scores(name, label, components, y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    t = np.array([ORDINAL[c] for c in y_true], dtype=float)
    p = np.array([ORDINAL[c] for c in y_pred], dtype=float)
    return {
        "model": name,
        "experiment": label,
        "components": components,
        "n_test": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred) * 100.0,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) * 100.0,
        "macro_precision": precision_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0) * 100.0,
        "macro_recall": recall_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0) * 100.0,
        "macro_f1": f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0) * 100.0,
        "LOW_f1": rep["LOW"]["f1-score"] * 100.0,
        "MIXED_f1": rep["MIXED"]["f1-score"] * 100.0,
        "HIGH_f1": rep["HIGH"]["f1-score"] * 100.0,
        "mae": float(np.mean(np.abs(t - p))),
        "mse": float(np.mean((t - p) ** 2)),
    }


def report_rows(name, label, y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    return [{
        "model": name, "experiment": label, "label": c,
        "precision": rep[c]["precision"], "recall": rep[c]["recall"],
        "f1-score": rep[c]["f1-score"], "support": rep[c]["support"],
    } for c in CLASS_ORDER]


def confusion_rows(name, label, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    return [{
        "model": name, "experiment": label,
        "truth": CLASS_ORDER[i], "pred": CLASS_ORDER[j], "count": int(cm[i, j]),
    } for i in range(len(CLASS_ORDER)) for j in range(len(CLASS_ORDER))]


def plot(metrics, path):
    x = np.arange(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - width / 2, metrics["accuracy"], width, label="Accuracy", color="#4C72B0")
    b2 = ax.bar(x + width / 2, metrics["macro_f1"], width, label="Macro-F1", color="#DD8452")
    ax.set_ylabel("Score (%)")
    ax.set_title("Factuality models (Mediascope 2800)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["experiment"], rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.1f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write(out_dir, metrics, predictions, reports, confusions):
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(reports).to_csv(out_dir / "classification_report.csv", index=False)
    pd.DataFrame(confusions).to_csv(out_dir / "confusion_matrix.csv", index=False)
    plot(pd.DataFrame(metrics), out_dir / "accuracy_f1.png")
