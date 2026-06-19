from experiment_bootstrap import activate_project_root

activate_project_root()

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import bias5_image_feature_advanced_experiments as advanced
import bias5_image_feature_experiments as image_features
import bias5_semantic_feature_experiments as semantic
from bias5_image_feature_full_advanced_experiments import full_prompt_score_path
from bias5_image_feature_full_experiments import FULL_PREFIX, full_embedding_cache_path, load_full_rows
from image_process import BIAS_5_LABELS, write_csv


RANDOM_STATE = advanced.RANDOM_STATE

NESTED_METRICS_PATH = Path("bias5_top_nested_source_ablation_metrics.csv")
NESTED_PREDICTIONS_PATH = Path("bias5_top_nested_source_ablation_predictions.csv")
NESTED_CONFUSIONS_PATH = Path("bias5_top_nested_source_ablation_confusions.csv")

SEMANTIC_GROUP_METRICS_PATH = Path("bias5_top_semantic_visual_group_ablation_metrics.csv")
SEMANTIC_SINGLE_METRICS_PATH = Path("bias5_top_semantic_visual_single_feature_ablation.csv")
SEMANTIC_PREDICTIONS_PATH = Path("bias5_top_semantic_visual_group_ablation_predictions.csv")
REPORT_PATH = Path("bias5_top_ablation_report.md")


def configure_full_caches() -> None:
    image_features.OCR_RAW_PATH = Path(f"{FULL_PREFIX}_ocr_raw.jsonl")
    image_features.OCR_FEATURES_PATH = Path(f"{FULL_PREFIX}_ocr_features.csv")
    image_features.cache_path_for_model = full_embedding_cache_path

    advanced.cache_path_for_model = full_embedding_cache_path
    advanced.prompt_score_path = full_prompt_score_path
    advanced.ensure_ocr_features = image_features.ensure_ocr_features
    advanced.handcrafted_visual_matrix = image_features.handcrafted_visual_matrix


def metric_row(method: str, y_true: np.ndarray, y_pred: np.ndarray, ablation: str = "") -> dict[str, str]:
    return {
        "method": method,
        "ablation": ablation,
        "sample_size": str(len(y_true)),
        "accuracy": f"{accuracy_score(y_true, y_pred):.6f}",
        "macro_f1": f"{f1_score(y_true, y_pred, labels=BIAS_5_LABELS, average='macro', zero_division=0):.6f}",
        "weighted_f1": f"{f1_score(y_true, y_pred, labels=BIAS_5_LABELS, average='weighted', zero_division=0):.6f}",
    }


def confusion_rows(method: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, str]]:
    matrix = confusion_matrix(y_true, y_pred, labels=BIAS_5_LABELS)
    rows = []
    for true_index, true_label in enumerate(BIAS_5_LABELS):
        for pred_index, pred_label in enumerate(BIAS_5_LABELS):
            rows.append(
                {
                    "method": method,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": str(int(matrix[true_index, pred_index])),
                }
            )
    return rows


def nested_stack_ablation() -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    rows = load_full_rows()
    y = advanced.labels_for_rows(rows)
    baseline_rows = advanced.load_baseline_predictions()
    gpt55 = np.asarray([baseline_rows[row["image"]][advanced.GPT55_COLUMN] for row in rows], dtype=object)

    ocr_df = image_features.ensure_ocr_features(rows)
    ocr_x = image_features.ocr_numeric_matrix(ocr_df)
    ocr_texts = ocr_df["ocr_text"].fillna("").astype(str).tolist()
    visual_x = image_features.handcrafted_visual_matrix(rows)
    if visual_x is None:
        raise RuntimeError("Missing handcrafted visual features")

    clip_x = advanced.embedding_matrix("clip_vit_b32")
    siglip_x = advanced.embedding_matrix("siglip_b16")
    dino_x = advanced.embedding_matrix("dinov2_small")
    clip_scores = advanced.prompt_score_matrix(advanced.ensure_prompt_scores(rows, image_features.HF_IMAGE_MODELS[0]))
    siglip_scores = advanced.prompt_score_matrix(advanced.ensure_prompt_scores(rows, image_features.HF_IMAGE_MODELS[1]))
    prompt_scores = np.hstack([clip_scores, siglip_scores])

    dense_specs: list[tuple[str, np.ndarray, Any]] = [
        ("visual_extra_trees", visual_x, advanced.extra_trees_estimator()),
        ("ocr_numeric_visual_extra_trees", np.hstack([ocr_x, visual_x]), advanced.extra_trees_estimator()),
        ("clip_pca_ridge", clip_x, advanced.pca_ridge_estimator(clip_x)),
        ("siglip_pca_ridge", siglip_x, advanced.pca_ridge_estimator(siglip_x)),
        ("dino_pca_ridge", dino_x, advanced.pca_ridge_estimator(dino_x)),
        (
            "prompt_scores_logreg",
            prompt_scores,
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", C=0.7, random_state=RANDOM_STATE),
            ),
        ),
    ]
    text_specs: list[tuple[str, list[str], Any]] = [
        (
            "ocr_text_tfidf",
            ocr_texts,
            make_pipeline(
                TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), max_features=1500),
                LinearSVC(C=0.7, class_weight="balanced", random_state=RANDOM_STATE),
            ),
        )
    ]

    source_names = [name for name, _, _ in dense_specs] + [name for name, _, _ in text_specs] + ["gpt55_cached"]
    configs: dict[str, list[str]] = {
        "full": source_names,
        "local_only_without_gpt55_cached": [name for name in source_names if name != "gpt55_cached"],
        "gpt55_cached_only": ["gpt55_cached"],
        "vision_embeddings_only": ["clip_pca_ridge", "siglip_pca_ridge", "dino_pca_ridge"],
        "ocr_only": ["ocr_numeric_visual_extra_trees", "ocr_text_tfidf"],
        "visual_style_only": ["visual_extra_trees"],
        "pruned_without_visual_and_prompt": [
            name for name in source_names
            if name not in {"visual_extra_trees", "prompt_scores_logreg"}
        ],
        "pruned_without_visual_prompt_dino": [
            name for name in source_names
            if name not in {"visual_extra_trees", "prompt_scores_logreg", "dino_pca_ridge"}
        ],
        "pruned_without_visual_prompt_dino_ocrnum": [
            name for name in source_names
            if name not in {
                "visual_extra_trees",
                "prompt_scores_logreg",
                "dino_pca_ridge",
                "ocr_numeric_visual_extra_trees",
            }
        ],
        "core_gpt55_ocr_text_siglip": ["gpt55_cached", "ocr_text_tfidf", "siglip_pca_ridge"],
        "core_gpt55_ocr_text_siglip_clip": ["gpt55_cached", "ocr_text_tfidf", "siglip_pca_ridge", "clip_pca_ridge"],
        "core_gpt55_ocr_text_siglip_ocrnum": [
            "gpt55_cached",
            "ocr_text_tfidf",
            "siglip_pca_ridge",
            "ocr_numeric_visual_extra_trees",
        ],
    }
    for source_name in source_names:
        configs[f"without_{source_name}"] = [name for name in source_names if name != source_name]
        configs[f"only_{source_name}"] = [source_name]

    final_predictions = {config_name: np.empty(len(y), dtype=object) for config_name in configs}
    raw_source_predictions = {source_name: np.empty(len(y), dtype=object) for source_name in source_names}
    cv = advanced.fixed_cv()

    for fold_index, (train_index, test_index) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        print(f"nested ablation fold {fold_index}/5", flush=True)
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + fold_index)
        train_source_predictions: dict[str, np.ndarray] = {}
        test_source_predictions: dict[str, np.ndarray] = {}

        for source_name, x, estimator in dense_specs:
            x = advanced.clean_matrix(x)
            train_pred = cross_val_predict(clone(estimator), x[train_index], y[train_index], cv=inner_cv)
            model = clone(estimator)
            model.fit(x[train_index], y[train_index])
            test_pred = model.predict(x[test_index])
            train_source_predictions[source_name] = train_pred
            test_source_predictions[source_name] = test_pred
            raw_source_predictions[source_name][test_index] = test_pred

        for source_name, texts, estimator in text_specs:
            train_texts = [texts[index] for index in train_index]
            test_texts = [texts[index] for index in test_index]
            train_pred = cross_val_predict(clone(estimator), train_texts, y[train_index], cv=inner_cv)
            model = clone(estimator)
            model.fit(train_texts, y[train_index])
            test_pred = model.predict(test_texts)
            train_source_predictions[source_name] = train_pred
            test_source_predictions[source_name] = test_pred
            raw_source_predictions[source_name][test_index] = test_pred

        train_source_predictions["gpt55_cached"] = gpt55[train_index]
        test_source_predictions["gpt55_cached"] = gpt55[test_index]
        raw_source_predictions["gpt55_cached"][test_index] = gpt55[test_index]

        for config_name, selected_sources in configs.items():
            train_meta_x = advanced.prediction_one_hot([train_source_predictions[name] for name in selected_sources])
            test_meta_x = advanced.prediction_one_hot([test_source_predictions[name] for name in selected_sources])
            meta_model = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.8, random_state=RANDOM_STATE)
            meta_model.fit(train_meta_x, y[train_index])
            final_predictions[config_name][test_index] = meta_model.predict(test_meta_x)

    metric_rows = []
    for config_name, predictions in final_predictions.items():
        metric_rows.append(metric_row(f"nested_stack::{config_name}", y, predictions, config_name))
    for source_name, predictions in raw_source_predictions.items():
        metric_rows.append(metric_row(f"raw_source::{source_name}", y, predictions, source_name))

    metric_rows = sorted(metric_rows, key=lambda row: float(row["macro_f1"]), reverse=True)
    write_csv(NESTED_METRICS_PATH, metric_rows, ["method", "ablation", "sample_size", "accuracy", "macro_f1", "weighted_f1"])

    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_row = {"image": row["image"], "true_bias_5": str(y[index])}
        for method, predictions in final_predictions.items():
            prediction_row[f"nested_stack::{method}"] = predictions[index]
        for source_name, predictions in raw_source_predictions.items():
            prediction_row[f"raw_source::{source_name}"] = predictions[index]
        prediction_rows.append(prediction_row)
    write_csv(NESTED_PREDICTIONS_PATH, prediction_rows, list(prediction_rows[0].keys()))

    all_confusions = []
    for config_name, predictions in final_predictions.items():
        all_confusions.extend(confusion_rows(f"nested_stack::{config_name}", y, predictions))
    for source_name, predictions in raw_source_predictions.items():
        all_confusions.extend(confusion_rows(f"raw_source::{source_name}", y, predictions))
    write_csv(NESTED_CONFUSIONS_PATH, all_confusions, ["method", "true_label", "predicted_label", "count"])

    return metric_rows, {f"nested_stack::{key}": value for key, value in final_predictions.items()}


def configure_semantic_full_paths() -> None:
    semantic.FEATURES_PATH = Path("bias5_semantic_full_features.csv")
    semantic.VISUAL_FEATURES_PATH = Path("bias5_ml_visual_features.csv")


def semantic_feature_groups(feature_names: list[str], semantic_name_count: int) -> dict[str, list[int]]:
    semantic_groups = semantic.grouped_feature_indices(feature_names[:semantic_name_count])
    groups = {name: indices[:] for name, indices in semantic_groups.items()}
    groups["all_semantic_excluding_baseline"] = [
        index for index, name in enumerate(feature_names[:semantic_name_count])
        if not name.startswith("baseline_gpt5.5=")
    ]
    groups["visual_features"] = list(range(semantic_name_count, len(feature_names)))
    return groups


def cv_extra_trees(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return semantic.cv_predict(x, y, semantic.classifier_extra_trees(), scale=False)


def semantic_visual_ablation() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    configure_semantic_full_paths()
    feature_df = pd.read_csv(semantic.FEATURES_PATH)
    images = feature_df["image"].tolist()
    visual_df = semantic.load_visual_features_for_sample(images)
    y = feature_df["true_bias_5"].to_numpy()

    semantic_x, semantic_names = semantic.semantic_matrix(feature_df)
    visual_x = semantic.visual_matrix(visual_df)
    combined_x = np.hstack([semantic_x, visual_x])
    visual_names = [f"visual::{column}" for column in visual_df.columns if column != "image"]
    feature_names = [*semantic_names, *visual_names]

    groups = semantic_feature_groups(feature_names, len(semantic_names))
    groups["topics_plus_boolean_cues"] = [*groups["topics"], *groups["boolean_cues"]]
    groups["topics_boolean_and_baseline"] = [
        *groups["topics"],
        *groups["boolean_cues"],
        *groups["baseline_label"],
    ]
    groups["topics_boolean_and_numeric"] = [
        *groups["topics"],
        *groups["boolean_cues"],
        *groups["numeric_levels_counts"],
    ]
    groups["topics_boolean_baseline_and_numeric"] = [
        *groups["topics"],
        *groups["boolean_cues"],
        *groups["baseline_label"],
        *groups["numeric_levels_counts"],
    ]
    all_indices = np.arange(combined_x.shape[1])

    predictions_by_method: dict[str, np.ndarray] = {}
    predictions_by_method["full"] = cv_extra_trees(combined_x, y)

    group_metric_rows = [metric_row("semantic_visual_extra_trees::full", y, predictions_by_method["full"], "none")]
    for group_name, remove_indices in groups.items():
        keep_indices = np.setdiff1d(all_indices, np.asarray(remove_indices))
        method_name = f"without_{group_name}"
        print(f"semantic group ablation {method_name}", flush=True)
        predictions = cv_extra_trees(combined_x[:, keep_indices], y)
        predictions_by_method[method_name] = predictions
        group_metric_rows.append(
            metric_row(f"semantic_visual_extra_trees::{method_name}", y, predictions, method_name)
        )

    # Component-only controls.
    baseline_indices = np.asarray(groups["baseline_label"])
    visual_indices = np.asarray(groups["visual_features"])
    semantic_without_baseline = np.asarray(groups["all_semantic_excluding_baseline"])
    component_specs = {
        "only_baseline_label": baseline_indices,
        "only_visual_features": visual_indices,
        "only_semantic_excluding_baseline": semantic_without_baseline,
        "baseline_plus_visual_only": np.concatenate([baseline_indices, visual_indices]),
        "baseline_plus_semantic_no_visual": np.arange(len(semantic_names)),
    }
    for method_name, keep_indices in component_specs.items():
        print(f"semantic component control {method_name}", flush=True)
        predictions = cv_extra_trees(combined_x[:, keep_indices], y)
        predictions_by_method[method_name] = predictions
        group_metric_rows.append(
            metric_row(f"semantic_visual_extra_trees::{method_name}", y, predictions, method_name)
        )

    group_metric_rows = sorted(group_metric_rows, key=lambda row: float(row["macro_f1"]), reverse=True)
    write_csv(
        SEMANTIC_GROUP_METRICS_PATH,
        group_metric_rows,
        ["method", "ablation", "sample_size", "accuracy", "macro_f1", "weighted_f1"],
    )

    prediction_rows = []
    for index, image_name in enumerate(images):
        prediction_row = {"image": image_name, "true_bias_5": str(y[index])}
        for method_name, predictions in predictions_by_method.items():
            prediction_row[f"semantic_visual_extra_trees::{method_name}"] = predictions[index]
        prediction_rows.append(prediction_row)
    write_csv(SEMANTIC_PREDICTIONS_PATH, prediction_rows, list(prediction_rows[0].keys()))

    model = semantic.classifier_extra_trees()
    model.fit(combined_x, y)
    importance_rows = sorted(
        [
            {"feature": feature_names[index], "importance": float(importance)}
            for index, importance in enumerate(model.feature_importances_)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )
    candidate_features = []
    for row in importance_rows[:12]:
        candidate_features.append(row)
    for row in reversed(importance_rows[-12:]):
        if row["feature"] not in {item["feature"] for item in candidate_features}:
            candidate_features.append(row)

    feature_to_index = {feature: index for index, feature in enumerate(feature_names)}
    full_macro_f1 = float(next(row for row in group_metric_rows if row["ablation"] == "none")["macro_f1"])
    single_rows = []
    for row in candidate_features:
        feature = row["feature"]
        feature_index = feature_to_index[feature]
        keep_indices = np.setdiff1d(all_indices, np.asarray([feature_index]))
        print(f"semantic single-feature ablation {feature}", flush=True)
        predictions = cv_extra_trees(combined_x[:, keep_indices], y)
        macro_f1 = f1_score(y, predictions, labels=BIAS_5_LABELS, average="macro", zero_division=0)
        single_rows.append(
            {
                "removed_feature": feature,
                "feature_importance": f"{row['importance']:.8f}",
                "accuracy": f"{accuracy_score(y, predictions):.6f}",
                "macro_f1": f"{macro_f1:.6f}",
                "macro_f1_delta_vs_full": f"{macro_f1 - full_macro_f1:.6f}",
            }
        )

    single_rows = sorted(single_rows, key=lambda row: float(row["macro_f1_delta_vs_full"]), reverse=True)
    write_csv(
        SEMANTIC_SINGLE_METRICS_PATH,
        single_rows,
        ["removed_feature", "feature_importance", "accuracy", "macro_f1", "macro_f1_delta_vs_full"],
    )

    return group_metric_rows, single_rows


def write_report(nested_rows: list[dict[str, str]], semantic_group_rows: list[dict[str, str]], semantic_single_rows: list[dict[str, str]]) -> None:
    nested_full = next(row for row in nested_rows if row["method"] == "nested_stack::full")
    semantic_full = next(row for row in semantic_group_rows if row["method"] == "semantic_visual_extra_trees::full")
    nested_without = [
        row for row in nested_rows
        if row["method"].startswith("nested_stack::without_")
    ]
    semantic_without = [
        row for row in semantic_group_rows
        if row["method"].startswith("semantic_visual_extra_trees::without_")
    ]

    lines = [
        "# Bias 5 Top Method Ablation Report",
        "",
        "Dataset: full balanced 250-site bias dataset.",
        "",
        "## Nested Stack Source Ablation",
        "",
        f"Full nested stack: accuracy `{nested_full['accuracy']}`, macro-F1 `{nested_full['macro_f1']}`.",
        "",
        "Largest drops when a source is removed:",
        "",
    ]
    for row in sorted(nested_without, key=lambda item: float(item["macro_f1"]))[:8]:
        delta = float(row["macro_f1"]) - float(nested_full["macro_f1"])
        lines.append(f"- `{row['ablation']}`: macro-F1 `{row['macro_f1']}` ({delta:+.6f})")

    lines.extend(
        [
            "",
            "Best nested/source rows:",
            "",
        ]
    )
    for row in nested_rows[:12]:
        lines.append(f"- `{row['method']}`: accuracy `{row['accuracy']}`, macro-F1 `{row['macro_f1']}`")

    lines.extend(
        [
            "",
            "## Semantic + Visual ExtraTrees Ablation",
            "",
            f"Full semantic+visual model: accuracy `{semantic_full['accuracy']}`, macro-F1 `{semantic_full['macro_f1']}`.",
            "",
            "Largest drops when a group is removed:",
            "",
        ]
    )
    for row in sorted(semantic_without, key=lambda item: float(item["macro_f1"]))[:8]:
        delta = float(row["macro_f1"]) - float(semantic_full["macro_f1"])
        lines.append(f"- `{row['ablation']}`: macro-F1 `{row['macro_f1']}` ({delta:+.6f})")

    lines.extend(
        [
            "",
            "Best single-feature cuts among tested high/low-importance candidates:",
            "",
        ]
    )
    for row in semantic_single_rows[:10]:
        lines.append(
            f"- remove `{row['removed_feature']}`: accuracy `{row['accuracy']}`, "
            f"macro-F1 `{row['macro_f1']}` ({float(row['macro_f1_delta_vs_full']):+.6f})"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{NESTED_METRICS_PATH}`",
            f"- `{NESTED_PREDICTIONS_PATH}`",
            f"- `{NESTED_CONFUSIONS_PATH}`",
            f"- `{SEMANTIC_GROUP_METRICS_PATH}`",
            f"- `{SEMANTIC_SINGLE_METRICS_PATH}`",
            f"- `{SEMANTIC_PREDICTIONS_PATH}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_full_caches()
    nested_rows, _ = nested_stack_ablation()
    semantic_group_rows, semantic_single_rows = semantic_visual_ablation()
    write_report(nested_rows, semantic_group_rows, semantic_single_rows)
    print(f"wrote {NESTED_METRICS_PATH}", flush=True)
    print(f"wrote {SEMANTIC_GROUP_METRICS_PATH}", flush=True)
    print(f"wrote {SEMANTIC_SINGLE_METRICS_PATH}", flush=True)
    print(f"wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
