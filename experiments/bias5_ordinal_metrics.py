from experiment_bootstrap import activate_project_root

activate_project_root()

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


LABEL_ORDER = ["left", "left-center", "least biased", "right-center", "right"]
LABEL_TO_VALUE = {label: index for index, label in enumerate(LABEL_ORDER)}
MAX_DISTANCE = len(LABEL_ORDER) - 1
MAX_SQUARED_DISTANCE = MAX_DISTANCE ** 2

OUTPUT_PATH = Path("bias5_ordinal_metrics.csv")
SELECTED_OUTPUT_PATH = Path("bias5_ordinal_selected_metrics.csv")
PER_CLASS_OUTPUT_PATH = Path("bias5_ordinal_per_class_metrics.csv")
REPORT_PATH = Path("bias5_ordinal_metrics_report.md")

PREDICTION_SOURCES = [
    (
        "nested_ablation",
        Path("bias5_top_nested_source_ablation_predictions.csv"),
        "true_bias_5",
    ),
    (
        "semantic_visual_ablation",
        Path("bias5_top_semantic_visual_group_ablation_predictions.csv"),
        "true_bias_5",
    ),
    (
        "semantic_full",
        Path("bias5_semantic_full_ml_predictions.csv"),
        "true_bias_5",
    ),
    (
        "ml_full",
        Path("bias5_ml_predictions.csv"),
        "true_bias_5",
    ),
    (
        "llm_direct",
        Path("bias_predictions.csv"),
        "true_bias_5",
    ),
]

SELECTED_METHODS = [
    ("nested_ablation", "nested_stack::core_gpt55_ocr_text_siglip"),
    ("nested_ablation", "nested_stack::pruned_without_visual_prompt_dino_ocrnum"),
    ("nested_ablation", "nested_stack::full"),
    ("semantic_visual_ablation", "semantic_visual_extra_trees::without_topics_boolean_and_baseline"),
    ("semantic_visual_ablation", "semantic_visual_extra_trees::full"),
    ("semantic_full", "semantic_only_random_forest_cv"),
    ("ml_full", "all_model_stack_plus_visual_extra_trees_cv"),
    ("ml_full", "gpt55_label_plus_visual_extra_trees_cv"),
    ("llm_direct", "bias_5:openai/gpt-5.5"),
    ("llm_direct", "bias_5:anthropic/claude-sonnet-4.6"),
    ("llm_direct", "bias_5:moonshotai/kimi-k2.6"),
    ("llm_direct", "bias_5:qwen/qwen3.7-plus"),
]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def valid_prediction_mask(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    return y_true.isin(LABEL_TO_VALUE) & y_pred.isin(LABEL_TO_VALUE)


def ordinal_arrays(y_true: pd.Series, y_pred: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    mask = valid_prediction_mask(y_true, y_pred)
    true_values = y_true[mask].map(LABEL_TO_VALUE).to_numpy(dtype=int)
    predicted_values = y_pred[mask].map(LABEL_TO_VALUE).to_numpy(dtype=int)
    return true_values, predicted_values


def macro_and_max_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    class_maes = []
    for class_value in range(len(LABEL_ORDER)):
        class_mask = y_true == class_value
        if class_mask.any():
            class_maes.append(float(np.abs(y_pred[class_mask] - y_true[class_mask]).mean()))
    return float(np.mean(class_maes)), float(np.max(class_maes))


def evaluate(source: str, method: str, y_true_labels: pd.Series, y_pred_labels: pd.Series) -> dict[str, str]:
    y_true, y_pred = ordinal_arrays(y_true_labels, y_pred_labels)
    errors = y_pred - y_true
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2
    macro_mae, max_mae = macro_and_max_mae(y_true, y_pred)
    mse = float(squared_errors.mean())

    return {
        "source": source,
        "method": method,
        "sample_size": str(len(y_true_labels)),
        "evaluated": str(len(y_true)),
        "accuracy": f"{accuracy_score(y_true, y_pred):.6f}",
        "macro_f1": f"{f1_score(y_true, y_pred, labels=list(range(len(LABEL_ORDER))), average='macro', zero_division=0):.6f}",
        "mae_steps": f"{absolute_errors.mean():.6f}",
        "macro_mae_steps": f"{macro_mae:.6f}",
        "max_class_mae_steps": f"{max_mae:.6f}",
        "mse_steps2": f"{mse:.6f}",
        "rmse_steps": f"{math.sqrt(mse):.6f}",
        "normalized_mse": f"{mse / MAX_SQUARED_DISTANCE:.6f}",
        "mse_closeness_score": f"{1.0 - mse / MAX_SQUARED_DISTANCE:.6f}",
        "within_1_accuracy": f"{(absolute_errors <= 1).mean():.6f}",
        "within_2_accuracy": f"{(absolute_errors <= 2).mean():.6f}",
        "severe_error_rate_ge_2": f"{(absolute_errors >= 2).mean():.6f}",
        "extreme_error_rate_ge_3": f"{(absolute_errors >= 3).mean():.6f}",
        "quadratic_weighted_kappa": f"{cohen_kappa_score(y_true, y_pred, weights='quadratic'):.6f}",
        "mean_signed_error": f"{errors.mean():.6f}",
    }


def per_class_rows(source: str, method: str, y_true_labels: pd.Series, y_pred_labels: pd.Series) -> list[dict[str, str]]:
    y_true, y_pred = ordinal_arrays(y_true_labels, y_pred_labels)
    rows = []
    for class_value, class_label in enumerate(LABEL_ORDER):
        class_mask = y_true == class_value
        errors = y_pred[class_mask] - y_true[class_mask]
        absolute_errors = np.abs(errors)
        squared_errors = errors ** 2
        rows.append(
            {
                "source": source,
                "method": method,
                "true_label": class_label,
                "count": str(int(class_mask.sum())),
                "accuracy": f"{(absolute_errors == 0).mean():.6f}",
                "mae_steps": f"{absolute_errors.mean():.6f}",
                "mse_steps2": f"{squared_errors.mean():.6f}",
                "within_1_accuracy": f"{(absolute_errors <= 1).mean():.6f}",
                "mean_signed_error": f"{errors.mean():.6f}",
            }
        )
    return rows


def load_prediction_sources() -> dict[str, tuple[pd.DataFrame, str]]:
    sources = {}
    for source, path, truth_column in PREDICTION_SOURCES:
        if path.exists():
            sources[source] = (pd.read_csv(path), truth_column)
    return sources


def center_baseline(truth: pd.Series) -> pd.Series:
    return pd.Series(["least biased"] * len(truth), index=truth.index)


def method_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sources = load_prediction_sources()
    metric_rows = []
    class_rows = []

    for source, (frame, truth_column) in sources.items():
        truth = frame[truth_column].astype(str)
        for column in frame.columns:
            if column in {"image", truth_column, "true_bias_3", "bias_3_included"}:
                continue
            predictions = frame[column].astype(str)
            if not valid_prediction_mask(truth, predictions).any():
                continue
            metric_rows.append(evaluate(source, column, truth, predictions))
            class_rows.extend(per_class_rows(source, column, truth, predictions))

    reference_frame, reference_truth_column = next(iter(sources.values()))
    reference_truth = reference_frame[reference_truth_column].astype(str)
    center_predictions = center_baseline(reference_truth)
    metric_rows.append(evaluate("reference", "always_least_biased", reference_truth, center_predictions))
    class_rows.extend(per_class_rows("reference", "always_least_biased", reference_truth, center_predictions))

    metric_rows.sort(
        key=lambda row: (
            float(row["mse_steps2"]),
            float(row["mae_steps"]),
            -float(row["quadratic_weighted_kappa"]),
        )
    )
    return metric_rows, class_rows


def selected_rows(all_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    lookup = {(row["source"], row["method"]): row for row in all_rows}
    rows = [
        lookup[key]
        for key in SELECTED_METHODS
        if key in lookup
    ]
    reference = lookup.get(("reference", "always_least_biased"))
    if reference:
        rows.append(reference)
    rows.sort(key=lambda row: float(row["mse_steps2"]))
    return rows


def write_report(selected: list[dict[str, str]], all_rows: list[dict[str, str]]) -> None:
    best_mse = min(selected, key=lambda row: float(row["mse_steps2"]))
    best_mae = min(selected, key=lambda row: float(row["mae_steps"]))
    best_qwk = max(selected, key=lambda row: float(row["quadratic_weighted_kappa"]))

    lines = [
        "# Bias 5 Ordinal Metrics Report",
        "",
        "Dataset: full balanced 250-site bias dataset.",
        "",
        "Ordinal encoding: `left=0`, `left-center=1`, `least biased=2`, `right-center=3`, `right=4`.",
        "",
        "Lower is better for MAE, MSE, RMSE, normalized MSE, and severe-error rates. "
        "Higher is better for accuracy, within-N accuracy, MSE closeness, and quadratic weighted kappa.",
        "",
        "## Best Results",
        "",
        f"- Best MSE: `{best_mse['method']}` = `{best_mse['mse_steps2']}`.",
        f"- Best MAE: `{best_mae['method']}` = `{best_mae['mae_steps']}` label steps.",
        f"- Best quadratic weighted kappa: `{best_qwk['method']}` = `{best_qwk['quadratic_weighted_kappa']}`.",
        "",
        "## Selected Methods",
        "",
        "| Method | Accuracy | Macro-F1 | MAE | MSE | RMSE | Within 1 | Severe >=2 | QWK |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| `{row['method']}` | {row['accuracy']} | {row['macro_f1']} | "
            f"{row['mae_steps']} | {row['mse_steps2']} | {row['rmse_steps']} | "
            f"{row['within_1_accuracy']} | {row['severe_error_rate_ge_2']} | "
            f"{row['quadratic_weighted_kappa']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Exact accuracy and Macro-F1 treat every wrong class as fully wrong.",
            "- MAE reports the average number of label positions missed.",
            "- MSE penalizes distant errors quadratically, so an error of two positions costs four times an adjacent error.",
            "- Within-1 accuracy gives credit when the prediction is exact or one neighboring bias class away.",
            "- Quadratic weighted kappa measures chance-corrected ordinal agreement and penalizes distant disagreements more strongly.",
            "- `mse_closeness_score = 1 - MSE/16` is a convenience normalization for this five-class scale, not a standard research metric.",
            "",
            "Ranked Probability Score was not calculated because the saved pipelines expose hard labels rather than calibrated per-class probabilities.",
            "",
            "## Research Context",
            "",
            "- Gaudette and Japkowicz compare ordinal evaluation methods and emphasize that plain accuracy ignores error severity: "
            "https://doi.org/10.1007/978-3-642-01818-3_25",
            "- Cohen's weighted kappa gives partial credit for disagreements of different severity and corrects for chance agreement: "
            "https://doi.org/10.1037/h0026256",
            "- Baccianella, Esuli, and Sebastiani discuss MAE and macro-averaged ordinal error measures: "
            "https://iris.cnr.it/retrieve/5bcf86c7-cd68-4884-93b5-ff86095082ec/prod_91979-doc_199135.pdf",
            "- Galdran recommends Ranked Probability Score for probabilistic ordinal predictions: "
            "https://arxiv.org/abs/2309.08701",
            "",
            "## Files",
            "",
            f"- `{OUTPUT_PATH}`: all {len(all_rows)} evaluated prediction columns.",
            f"- `{SELECTED_OUTPUT_PATH}`: selected top methods and baselines.",
            f"- `{PER_CLASS_OUTPUT_PATH}`: per-class distance errors.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    metric_rows, class_rows = method_rows()
    selected = selected_rows(metric_rows)

    metric_fieldnames = [
        "source",
        "method",
        "sample_size",
        "evaluated",
        "accuracy",
        "macro_f1",
        "mae_steps",
        "macro_mae_steps",
        "max_class_mae_steps",
        "mse_steps2",
        "rmse_steps",
        "normalized_mse",
        "mse_closeness_score",
        "within_1_accuracy",
        "within_2_accuracy",
        "severe_error_rate_ge_2",
        "extreme_error_rate_ge_3",
        "quadratic_weighted_kappa",
        "mean_signed_error",
    ]
    write_csv(OUTPUT_PATH, metric_rows, metric_fieldnames)
    write_csv(SELECTED_OUTPUT_PATH, selected, metric_fieldnames)
    write_csv(
        PER_CLASS_OUTPUT_PATH,
        class_rows,
        [
            "source",
            "method",
            "true_label",
            "count",
            "accuracy",
            "mae_steps",
            "mse_steps2",
            "within_1_accuracy",
            "mean_signed_error",
        ],
    )
    write_report(selected, metric_rows)

    print(f"wrote {OUTPUT_PATH}", flush=True)
    print(f"wrote {SELECTED_OUTPUT_PATH}", flush=True)
    print(f"wrote {PER_CLASS_OUTPUT_PATH}", flush=True)
    print(f"wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
