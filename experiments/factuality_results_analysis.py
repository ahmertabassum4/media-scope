from experiment_bootstrap import activate_project_root

activate_project_root()

from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from factuality_common import BINARY_LABELS, MULTICLASS_LABELS, write_csv


METRICS_PATH = Path("factuality_all_methods_metrics.csv")
PER_CLASS_PATH = Path("factuality_selected_per_class_metrics.csv")
SUMMARY_PATH = Path("factuality_experiment_summary.md")
USAGE_PATH = Path("factuality_openrouter_usage.csv")

SELECTED_METHODS = {
    "binary": [
        "baseline_gpt5.5",
        "baseline_plus_semantic_visual_extra_trees_cv",
        "full_local_stack_pca_ridge_cv",
        "clip_vit_b32_cosine_centroid_cv",
        "ocr_text_tfidf_svc_cv",
        "scratch_small_cnn_cv",
    ],
    "multiclass": [
        "baseline_gpt5.5",
        "baseline_plus_semantic_visual_extra_trees_cv",
        "pca_fusion::compact::pruned_components::linear_svc_c0.2",
        "pca_fusion::large::core_without_clip::linear_svc_c0.2",
        "pca_fusion::medium::core_without_clip::extra_trees",
        "nested_core_gpt55_ocr_text_siglip_cv",
        "semantic_only_mlp_cv",
        "siglip_b16_cosine_centroid_cv",
        "scratch_small_cnn_cv",
    ],
}


def selected_per_class_rows() -> list[dict[str, str]]:
    output_rows = []
    for task, methods in SELECTED_METHODS.items():
        predictions = pd.read_csv(f"factuality_{task}_all_methods_predictions.csv")
        labels = BINARY_LABELS if task == "binary" else MULTICLASS_LABELS
        for method in methods:
            precision, recall, f1, support = precision_recall_fscore_support(
                predictions["true_label"],
                predictions[method],
                labels=labels,
                zero_division=0,
            )
            for index, label in enumerate(labels):
                output_rows.append(
                    {
                        "task": task,
                        "method": method,
                        "label": label,
                        "support": str(int(support[index])),
                        "precision": f"{precision[index]:.6f}",
                        "recall": f"{recall[index]:.6f}",
                        "f1": f"{f1[index]:.6f}",
                    }
                )
    return output_rows


def metric_lookup(metrics: pd.DataFrame, task: str, method: str) -> pd.Series:
    matches = metrics[(metrics["task"] == task) & (metrics["method"] == method)]
    if len(matches) != 1:
        raise ValueError(f"Expected one metric row for {task}/{method}, found {len(matches)}")
    return matches.iloc[0]


def usage_lines() -> list[str]:
    if not USAGE_PATH.exists():
        return []
    usage = pd.read_csv(USAGE_PATH)
    grouped = usage.groupby("request_type")[
        ["prompt_tokens", "completion_tokens", "total_tokens"]
    ].sum()
    lines = ["## OpenRouter Usage", ""]
    for request_type, row in grouped.iterrows():
        lines.append(
            f"- `{request_type}`: prompt `{int(row['prompt_tokens'])}`, "
            f"completion `{int(row['completion_tokens'])}`, total `{int(row['total_tokens'])}` tokens."
        )
    lines.append(
        f"- Total measured for the new 400 requests: `{int(grouped['total_tokens'].sum())}` tokens."
    )
    lines.extend(
        [
            "- The earlier cached binary GPT-5.5 run is not included because its usage was not recorded.",
            "",
        ]
    )
    return lines


def write_summary(metrics: pd.DataFrame, per_class: pd.DataFrame) -> None:
    binary_baseline = metric_lookup(metrics, "binary", "baseline_gpt5.5")
    binary_semantic = metric_lookup(
        metrics, "binary", "baseline_plus_semantic_visual_extra_trees_cv"
    )
    multiclass_baseline = metric_lookup(metrics, "multiclass", "baseline_gpt5.5")
    multiclass_semantic = metric_lookup(
        metrics, "multiclass", "baseline_plus_semantic_visual_extra_trees_cv"
    )
    multiclass_pca = metric_lookup(
        metrics,
        "multiclass",
        "pca_fusion::compact::pruned_components::linear_svc_c0.2",
    )
    multiclass_accuracy = metric_lookup(
        metrics,
        "multiclass",
        "pca_fusion::large::core_without_clip::linear_svc_c0.2",
    )
    multiclass_ordinal = metric_lookup(
        metrics,
        "multiclass",
        "pca_fusion::medium::core_without_clip::extra_trees",
    )
    multiclass_nested = metric_lookup(
        metrics, "multiclass", "nested_core_gpt55_ocr_text_siglip_cv"
    )

    def class_recall(method: str, label: str) -> float:
        row = per_class[
            (per_class["task"] == "multiclass")
            & (per_class["method"] == method)
            & (per_class["label"] == label)
        ].iloc[0]
        return float(row["recall"])

    lines = [
        "# Factuality Experiment Summary",
        "",
        "## Dataset",
        "",
        "- 200 landing-page screenshots matched exactly to `snapshot_index.csv`.",
        "- Binary target: 100 low (`VERY LOW` + `LOW`) and 100 high (`HIGH` + `VERY HIGH`).",
        "- Four-class target: 50 very low, 50 low, 89 high, and 11 very high.",
        "- The requested multi-label setup is implemented as four-class single-label classification, because each dataset row has exactly one factuality label.",
        "",
        "## Evaluation",
        "",
        "- 147 binary and 149 four-class methods/configurations were evaluated.",
        "- Learned methods use stratified 5-fold CV.",
        "- PCA, OCR TF-IDF/SVD, nested base models, and final classifiers are fitted inside training folds.",
        "- Primary metric: Macro-F1. Accuracy, balanced accuracy, per-class metrics, MSE, and quadratic weighted kappa are also reported.",
        "",
        "## Binary Result",
        "",
        f"- GPT-5.5 baseline: accuracy `{binary_baseline['accuracy']:.3f}`, Macro-F1 `{binary_baseline['macro_f1']:.3f}`.",
        f"- Best semantic hybrid: accuracy `{binary_semantic['accuracy']:.3f}`, Macro-F1 `{binary_semantic['macro_f1']:.3f}`.",
        "- The semantic hybrid changes two predictions, correcting one and breaking one. It does not improve binary accuracy.",
        "- Best fully local method: `full_local_stack_pca_ridge_cv`, accuracy `0.845`, Macro-F1 `0.845`.",
        "",
        "## Four-Class Result",
        "",
        f"- Baseline GPT-5.5: accuracy `{multiclass_baseline['accuracy']:.3f}`, Macro-F1 `{multiclass_baseline['macro_f1']:.3f}`.",
        f"- Best Macro-F1: `baseline_plus_semantic_visual_extra_trees_cv`, accuracy `{multiclass_semantic['accuracy']:.3f}`, Macro-F1 `{multiclass_semantic['macro_f1']:.3f}`, MSE `{multiclass_semantic['mse_steps2']:.3f}`, QWK `{multiclass_semantic['quadratic_weighted_kappa']:.3f}`.",
        f"- Nearly tied direct fusion: `pca_fusion::compact::pruned_components::linear_svc_c0.2`, accuracy `{multiclass_pca['accuracy']:.3f}`, Macro-F1 `{multiclass_pca['macro_f1']:.3f}`, MSE `{multiclass_pca['mse_steps2']:.3f}`, QWK `{multiclass_pca['quadratic_weighted_kappa']:.3f}`.",
        f"- Best accuracy: `pca_fusion::large::core_without_clip::linear_svc_c0.2`, accuracy `{multiclass_accuracy['accuracy']:.3f}`, Macro-F1 `{multiclass_accuracy['macro_f1']:.3f}`.",
        f"- Best ordinal errors: `pca_fusion::medium::core_without_clip::extra_trees`, MSE `{multiclass_ordinal['mse_steps2']:.3f}`, QWK `{multiclass_ordinal['quadratic_weighted_kappa']:.3f}`.",
        f"- Best balanced accuracy and very-high recall: `nested_core_gpt55_ocr_text_siglip_cv`, balanced accuracy `{multiclass_nested['balanced_accuracy']:.3f}`, very-high recall `{class_recall('nested_core_gpt55_ocr_text_siglip_cv', 'very high'):.3f}`.",
        "",
        "## Interpretation",
        "",
        "- The four-class gain comes from calibrating the GPT label with structured visible-page features and local OCR/SigLIP signals.",
        "- The best semantic hybrid raises Macro-F1 by about 0.100 and accuracy by 0.065 over the GPT baseline.",
        "- Maximum accuracy is not the best selection criterion: the 0.820-accuracy model recalls only 0.273 of the rare `very high` class.",
        "- The nested core stack recalls 0.818 of `very high`, but over-predicts that class and loses overall accuracy.",
        "- With only 11 `very high` examples, differences involving that class have high variance and should be confirmed on a larger external test set.",
        "",
        "## Main Files",
        "",
        "- `factuality_dataset.csv`",
        "- `factuality_all_methods_metrics.csv`",
        "- `factuality_selected_per_class_metrics.csv`",
        "- `factuality_binary_all_methods_predictions.csv`",
        "- `factuality_multiclass_all_methods_predictions.csv`",
        "- `factuality_all_methods_confusions.csv`",
        "- `factuality_all_methods_report.md`",
        "",
        *usage_lines(),
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    metrics = pd.read_csv(METRICS_PATH)
    per_class_rows = selected_per_class_rows()
    write_csv(
        PER_CLASS_PATH,
        per_class_rows,
        ["task", "method", "label", "support", "precision", "recall", "f1"],
    )
    write_summary(metrics, pd.DataFrame(per_class_rows))
    print(f"wrote {PER_CLASS_PATH}")
    print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
