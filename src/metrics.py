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


def scores(name, label, components, y_true, y_pred, classes, ordinal):
    rep = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0,
    )
    truth = np.array([ordinal[value] for value in y_true], dtype=float)
    predictions = np.array([ordinal[value] for value in y_pred], dtype=float)
    row = {
        "model": name,
        "experiment": label,
        "components": components,
        "n_test": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred) * 100.0,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) * 100.0,
        "macro_precision": precision_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0,
        ) * 100.0,
        "macro_recall": recall_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0,
        ) * 100.0,
        "macro_f1": f1_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0,
        ) * 100.0,
        "mae": float(np.mean(np.abs(truth - predictions))),
        "mse": float(np.mean((truth - predictions) ** 2)),
    }
    for class_name in classes:
        row[f"{class_name}_f1"] = rep[class_name]["f1-score"] * 100.0
    return row


def report_rows(name, label, y_true, y_pred, classes):
    rep = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0,
    )
    return [{
        "model": name, "experiment": label, "label": c,
        "precision": rep[c]["precision"], "recall": rep[c]["recall"],
        "f1-score": rep[c]["f1-score"], "support": rep[c]["support"],
    } for c in classes]


def confusion_rows(name, label, y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    return [{
        "model": name, "experiment": label,
        "truth": classes[i], "pred": classes[j], "count": int(cm[i, j]),
    } for i in range(len(classes)) for j in range(len(classes))]


def plot(metrics, path, task):
    x = np.arange(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - width / 2, metrics["accuracy"], width, label="Accuracy", color="#4C72B0")
    b2 = ax.bar(x + width / 2, metrics["macro_f1"], width, label="Macro-F1", color="#DD8452")
    ax.set_ylabel("Score (%)")
    ax.set_title(f"{task.title()} models (Mediascope 2800)")
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


def write(out_dir, metrics, predictions, reports, confusions, filenames, task):
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(out_dir / filenames["metrics"], index=False)
    pd.DataFrame(predictions).to_csv(out_dir / filenames["predictions"], index=False)
    pd.DataFrame(reports).to_csv(out_dir / filenames["report"], index=False)
    pd.DataFrame(confusions).to_csv(out_dir / filenames["confusion"], index=False)
    plot(pd.DataFrame(metrics), out_dir / filenames["plot"], task)
