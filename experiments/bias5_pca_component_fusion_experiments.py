from experiment_bootstrap import activate_project_root

activate_project_root()

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from image_process import BIAS_5_LABELS, write_csv


RANDOM_STATE = 83
CV_SPLITS = 5

LABEL_TO_VALUE = {
    "left": 0,
    "left-center": 1,
    "least biased": 2,
    "right-center": 3,
    "right": 4,
}

PREDICTIONS_INPUT_PATH = Path("bias_predictions.csv")
VISUAL_FEATURES_PATH = Path("bias5_ml_visual_features.csv")
OCR_FEATURES_PATH = Path("bias5_image_feature_full_ocr_features.csv")
CLIP_EMBEDDINGS_PATH = Path("bias5_image_feature_full_clip_vit_b32_embeddings.npz")
SIGLIP_EMBEDDINGS_PATH = Path("bias5_image_feature_full_siglip_b16_embeddings.npz")
DINO_EMBEDDINGS_PATH = Path("bias5_image_feature_full_dinov2_small_embeddings.npz")
CLIP_PROMPT_SCORES_PATH = Path("bias5_image_feature_full_clip_vit_b32_prompt_scores.csv")
SIGLIP_PROMPT_SCORES_PATH = Path("bias5_image_feature_full_siglip_b16_prompt_scores.csv")

METRICS_PATH = Path("bias5_pca_component_fusion_metrics.csv")
PREDICTIONS_PATH = Path("bias5_pca_component_fusion_predictions.csv")
FOLD_METRICS_PATH = Path("bias5_pca_component_fusion_fold_metrics.csv")
REPORT_PATH = Path("bias5_pca_component_fusion_report.md")

GPT55_COLUMN = "bias_5:openai/gpt-5.5"

COMPONENT_CONFIGS = {
    "compact": {
        "clip": 8,
        "siglip": 8,
        "dino": 8,
        "ocr_text": 16,
        "visual": 8,
    },
    "medium": {
        "clip": 16,
        "siglip": 16,
        "dino": 16,
        "ocr_text": 32,
        "visual": 16,
    },
    "large": {
        "clip": 32,
        "siglip": 32,
        "dino": 32,
        "ocr_text": 48,
        "visual": 24,
    },
}


def load_rows() -> list[dict[str, str]]:
    with PREDICTIONS_INPUT_PATH.open("r", encoding="utf-8", newline="") as input_file:
        return sorted(list(csv.DictReader(input_file)), key=lambda row: row["image"])


def aligned_frame(path: Path, image_names: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path).set_index("image")
    return frame.loc[image_names].reset_index()


def load_embedding(path: Path, image_names: list[str]) -> np.ndarray:
    cache = np.load(path, allow_pickle=True)
    cached_names = cache["image_names"].tolist()
    index_by_name = {name: index for index, name in enumerate(cached_names)}
    return np.vstack([cache["embeddings"][index_by_name[name]] for name in image_names]).astype(np.float32)


def one_hot_gpt(labels: list[str]) -> np.ndarray:
    return np.asarray(
        [
            [1.0 if label == candidate else 0.0 for candidate in BIAS_5_LABELS]
            for label in labels
        ],
        dtype=np.float32,
    )


def safe_component_count(requested: int, train_rows: int, feature_count: int) -> int:
    return max(1, min(requested, train_rows - 1, feature_count))


def pca_transform(
    train_x: np.ndarray,
    test_x: np.ndarray,
    components: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    test_scaled = scaler.transform(test_x)
    count = safe_component_count(components, train_scaled.shape[0], train_scaled.shape[1])
    pca = PCA(n_components=count, whiten=True, random_state=RANDOM_STATE)
    return pca.fit_transform(train_scaled), pca.transform(test_scaled)


def text_svd_transform(
    train_texts: list[str],
    test_texts: list[str],
    components: int,
) -> tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
        sublinear_tf=True,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)
    count = max(1, min(components, train_tfidf.shape[0] - 1, train_tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=count, random_state=RANDOM_STATE)
    train_svd = svd.fit_transform(train_tfidf)
    test_svd = svd.transform(test_tfidf)
    scaler = StandardScaler()
    return scaler.fit_transform(train_svd), scaler.transform(test_svd)


def scaled_transform(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(train_x), scaler.transform(test_x)


def build_fold_components(
    train_index: np.ndarray,
    test_index: np.ndarray,
    config: dict[str, int],
    data: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    components = {}
    for name in ("clip", "siglip", "dino"):
        components[name] = pca_transform(
            data[name][train_index],
            data[name][test_index],
            config[name],
        )

    train_texts = [data["ocr_texts"][index] for index in train_index]
    test_texts = [data["ocr_texts"][index] for index in test_index]
    components["ocr_text"] = text_svd_transform(
        train_texts,
        test_texts,
        config["ocr_text"],
    )
    components["visual"] = pca_transform(
        data["visual"][train_index],
        data["visual"][test_index],
        config["visual"],
    )
    components["ocr_numeric"] = scaled_transform(
        data["ocr_numeric"][train_index],
        data["ocr_numeric"][test_index],
    )
    components["prompt_scores"] = scaled_transform(
        data["prompt_scores"][train_index],
        data["prompt_scores"][test_index],
    )
    components["gpt55"] = (
        data["gpt55"][train_index],
        data["gpt55"][test_index],
    )
    return components


def concatenate_components(
    components: dict[str, tuple[np.ndarray, np.ndarray]],
    selected: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_x = np.hstack([components[name][0] for name in selected])
    test_x = np.hstack([components[name][1] for name in selected])
    return train_x.astype(np.float32), test_x.astype(np.float32)


def classifiers() -> dict[str, Any]:
    return {
        "logreg_c0.2": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                C=0.2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "logreg_c1": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "ridge_a3": make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=3.0, class_weight="balanced"),
        ),
        "linear_svc_c0.2": make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.2, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            random_state=RANDOM_STATE,
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 24),
                alpha=0.02,
                learning_rate_init=0.001,
                max_iter=600,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def feature_sets() -> dict[str, list[str]]:
    return {
        "all_components": [
            "clip",
            "siglip",
            "dino",
            "ocr_text",
            "visual",
            "ocr_numeric",
            "prompt_scores",
            "gpt55",
        ],
        "pruned_components": [
            "clip",
            "siglip",
            "ocr_text",
            "gpt55",
        ],
        "core_without_clip": [
            "siglip",
            "ocr_text",
            "gpt55",
        ],
        "all_without_gpt55": [
            "clip",
            "siglip",
            "dino",
            "ocr_text",
            "visual",
            "ocr_numeric",
            "prompt_scores",
        ],
        "vision_components_only": [
            "clip",
            "siglip",
            "dino",
        ],
    }


def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_values = np.asarray([LABEL_TO_VALUE[label] for label in y_true], dtype=int)
    predicted_values = np.asarray([LABEL_TO_VALUE[label] for label in y_pred], dtype=int)
    errors = predicted_values - true_values
    absolute_errors = np.abs(errors)
    mse = float(np.mean(errors ** 2))
    return {
        "mae_steps": float(absolute_errors.mean()),
        "mse_steps2": mse,
        "rmse_steps": math.sqrt(mse),
        "within_1_accuracy": float((absolute_errors <= 1).mean()),
        "severe_error_rate_ge_2": float((absolute_errors >= 2).mean()),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(true_values, predicted_values, weights="quadratic")
        ),
    }


def metric_row(method: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, str]:
    ordinal = ordinal_metrics(y_true, y_pred)
    return {
        "method": method,
        "sample_size": str(len(y_true)),
        "accuracy": f"{accuracy_score(y_true, y_pred):.6f}",
        "macro_f1": f"{f1_score(y_true, y_pred, labels=BIAS_5_LABELS, average='macro', zero_division=0):.6f}",
        "mae_steps": f"{ordinal['mae_steps']:.6f}",
        "mse_steps2": f"{ordinal['mse_steps2']:.6f}",
        "rmse_steps": f"{ordinal['rmse_steps']:.6f}",
        "within_1_accuracy": f"{ordinal['within_1_accuracy']:.6f}",
        "severe_error_rate_ge_2": f"{ordinal['severe_error_rate_ge_2']:.6f}",
        "quadratic_weighted_kappa": f"{ordinal['quadratic_weighted_kappa']:.6f}",
    }


def load_data(rows: list[dict[str, str]]) -> dict[str, Any]:
    image_names = [row["image"] for row in rows]
    ocr_df = aligned_frame(OCR_FEATURES_PATH, image_names)
    visual_df = aligned_frame(VISUAL_FEATURES_PATH, image_names)
    clip_scores = aligned_frame(CLIP_PROMPT_SCORES_PATH, image_names)
    siglip_scores = aligned_frame(SIGLIP_PROMPT_SCORES_PATH, image_names)

    return {
        "clip": load_embedding(CLIP_EMBEDDINGS_PATH, image_names),
        "siglip": load_embedding(SIGLIP_EMBEDDINGS_PATH, image_names),
        "dino": load_embedding(DINO_EMBEDDINGS_PATH, image_names),
        "ocr_texts": ocr_df["ocr_text"].fillna("").astype(str).tolist(),
        "ocr_numeric": ocr_df.drop(columns=["image", "ocr_text"]).to_numpy(dtype=np.float32),
        "visual": visual_df.drop(columns=["image"]).to_numpy(dtype=np.float32),
        "prompt_scores": np.hstack(
            [
                clip_scores.drop(columns=["image"]).to_numpy(dtype=np.float32),
                siglip_scores.drop(columns=["image"]).to_numpy(dtype=np.float32),
            ]
        ),
        "gpt55": one_hot_gpt([row[GPT55_COLUMN] for row in rows]),
    }


def run() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, np.ndarray]]:
    rows = load_rows()
    y = np.asarray([row["true_bias_5"] for row in rows], dtype=object)
    data = load_data(rows)
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    classifier_templates = classifiers()
    selected_feature_sets = feature_sets()
    predictions = {
        f"{config_name}::{feature_name}::{classifier_name}": np.empty(len(y), dtype=object)
        for config_name in COMPONENT_CONFIGS
        for feature_name in selected_feature_sets
        for classifier_name in classifier_templates
    }
    fold_rows = []

    for fold, (train_index, test_index) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        print(f"component fusion fold {fold}/{CV_SPLITS}", flush=True)
        for config_name, config in COMPONENT_CONFIGS.items():
            components = build_fold_components(train_index, test_index, config, data)
            for feature_name, selected_components in selected_feature_sets.items():
                train_x, test_x = concatenate_components(components, selected_components)
                for classifier_name, classifier_template in classifier_templates.items():
                    method = f"{config_name}::{feature_name}::{classifier_name}"
                    model = clone(classifier_template)
                    model.fit(train_x, y[train_index])
                    fold_predictions = model.predict(test_x)
                    predictions[method][test_index] = fold_predictions
                    fold_metric = metric_row(method, y[test_index], fold_predictions)
                    fold_metric["fold"] = str(fold)
                    fold_metric["feature_count"] = str(train_x.shape[1])
                    fold_rows.append(fold_metric)

    metric_rows = [metric_row(method, y, method_predictions) for method, method_predictions in predictions.items()]
    metric_rows.sort(key=lambda row: float(row["macro_f1"]), reverse=True)
    return metric_rows, fold_rows, predictions


def write_outputs(
    rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    fold_rows: list[dict[str, str]],
    predictions: dict[str, np.ndarray],
) -> None:
    metric_fields = [
        "method",
        "sample_size",
        "accuracy",
        "macro_f1",
        "mae_steps",
        "mse_steps2",
        "rmse_steps",
        "within_1_accuracy",
        "severe_error_rate_ge_2",
        "quadratic_weighted_kappa",
    ]
    write_csv(METRICS_PATH, metric_rows, metric_fields)

    fold_fields = ["fold", "feature_count", *metric_fields]
    write_csv(FOLD_METRICS_PATH, fold_rows, fold_fields)

    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_row = {
            "image": row["image"],
            "true_bias_5": row["true_bias_5"],
        }
        for method, method_predictions in predictions.items():
            prediction_row[method] = method_predictions[index]
        prediction_rows.append(prediction_row)
    write_csv(PREDICTIONS_PATH, prediction_rows, list(prediction_rows[0].keys()))


def write_report(metric_rows: list[dict[str, str]]) -> None:
    best_f1 = max(metric_rows, key=lambda row: float(row["macro_f1"]))
    best_mse = min(metric_rows, key=lambda row: float(row["mse_steps2"]))
    best_qwk = max(metric_rows, key=lambda row: float(row["quadratic_weighted_kappa"]))
    lines = [
        "# Bias 5 PCA Component Fusion Experiment",
        "",
        "Dataset: full balanced 250-site bias dataset.",
        "",
        "Each fold independently fits PCA for CLIP, SigLIP, DINOv2 and visual features, "
        "plus TF-IDF/TruncatedSVD for OCR text. The resulting continuous components are concatenated "
        "with OCR numeric features, prompt scores, and one-hot GPT-5.5 labels and passed directly to one final classifier.",
        "",
        "## Best Results",
        "",
        f"- Best Macro-F1: `{best_f1['method']}` = `{best_f1['macro_f1']}`; accuracy `{best_f1['accuracy']}`.",
        f"- Best MSE: `{best_mse['method']}` = `{best_mse['mse_steps2']}`; Macro-F1 `{best_mse['macro_f1']}`.",
        f"- Best QWK: `{best_qwk['method']}` = `{best_qwk['quadratic_weighted_kappa']}`.",
        "",
        "## Top 15 By Macro-F1",
        "",
        "| Method | Accuracy | Macro-F1 | MAE | MSE | Within-1 | QWK |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows[:15]:
        lines.append(
            f"| `{row['method']}` | {row['accuracy']} | {row['macro_f1']} | "
            f"{row['mae_steps']} | {row['mse_steps2']} | {row['within_1_accuracy']} | "
            f"{row['quadratic_weighted_kappa']} |"
        )
    lines.extend(
        [
            "",
            "## Reference",
            "",
            "- Best nested label-level stack before this experiment: accuracy `0.772000`, Macro-F1 `0.775729`.",
            "- Best ordinal nested configuration: MSE `0.408000`, QWK `0.893305`.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = load_rows()
    metric_rows, fold_rows, predictions = run()
    write_outputs(rows, metric_rows, fold_rows, predictions)
    write_report(metric_rows)
    print(f"wrote {METRICS_PATH}", flush=True)
    print(f"wrote {PREDICTIONS_PATH}", flush=True)
    print(f"wrote {FOLD_METRICS_PATH}", flush=True)
    print(f"wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
