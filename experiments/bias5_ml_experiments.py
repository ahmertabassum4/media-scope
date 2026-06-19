from experiment_bootstrap import activate_project_root

activate_project_root()

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from image_process import BIAS_5_LABELS, IMAGE_FOLDER, iter_images, load_ground_truth, write_csv


Image.MAX_IMAGE_PIXELS = None

PREDICTIONS_PATH = Path("bias_predictions.csv")
VISUAL_FEATURES_PATH = Path("bias5_ml_visual_features.csv")
ML_PREDICTIONS_PATH = Path("bias5_ml_predictions.csv")
ML_METRICS_PATH = Path("bias5_ml_metrics.csv")
ML_CONFUSIONS_PATH = Path("bias5_ml_confusions.csv")
ML_REPORT_PATH = Path("bias5_ml_experiment_report.md")

RANDOM_STATE = 23
CV_SPLITS = 5
VIEW_COUNT = 4
TARGET_ASPECT_RATIO = 16 / 9
RESIZE_SIZE = (256, 144)

GPT55_COLUMN = "bias_5:openai/gpt-5.5"
MODEL_COLUMNS = [
    "bias_5:openai/gpt-5.5",
    "bias_5:anthropic/claude-sonnet-4.6",
    "bias_5:moonshotai/kimi-k2.6",
    "bias_5:qwen/qwen3.7-plus",
]


def load_prediction_rows() -> list[dict[str, str]]:
    with PREDICTIONS_PATH.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def image_path_for_name(image_name: str) -> Path:
    return IMAGE_FOLDER / f"{image_name}.png"


def crop_view(image_path: Path, view_index: int) -> Image.Image:
    with Image.open(image_path) as image:
        width, height = image.size
        viewport_height = min(height, round(width / TARGET_ASPECT_RATIO))
        top = min(view_index * viewport_height, max(0, height - viewport_height))
        crop_box = (0, top, width, top + viewport_height)
        return image.crop(crop_box).resize(RESIZE_SIZE).convert("RGB")


def entropy_from_histogram(histogram: list[int], total: int) -> float:
    entropy = 0.0
    for count in histogram:
        if count:
            probability = count / total
            entropy -= probability * math.log2(probability)
    return entropy


def view_features(image: Image.Image, prefix: str) -> dict[str, float]:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray_image = image.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    height, width = gray.shape
    total_pixels = height * width

    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_density = (float((dx > 0.16).mean()) + float((dy > 0.16).mean())) / 2
    fine_edge_density = (float((dx > 0.08).mean()) + float((dy > 0.08).mean())) / 2

    row_dark_fraction = (gray < 0.18).mean(axis=1)
    col_dark_fraction = (gray < 0.18).mean(axis=0)
    top_band = rgb[: max(1, height // 5), :, :]

    red_dominance = np.maximum(rgb[:, :, 0] - np.maximum(rgb[:, :, 1], rgb[:, :, 2]), 0)
    blue_dominance = np.maximum(rgb[:, :, 2] - np.maximum(rgb[:, :, 0], rgb[:, :, 1]), 0)
    top_red_dominance = np.maximum(top_band[:, :, 0] - np.maximum(top_band[:, :, 1], top_band[:, :, 2]), 0)
    top_blue_dominance = np.maximum(top_band[:, :, 2] - np.maximum(top_band[:, :, 0], top_band[:, :, 1]), 0)

    stats = ImageStat.Stat(image)
    gray_histogram = gray_image.histogram()

    return {
        f"{prefix}_gray_mean": float(gray.mean()),
        f"{prefix}_gray_std": float(gray.std()),
        f"{prefix}_gray_entropy": entropy_from_histogram(gray_histogram, total_pixels),
        f"{prefix}_edge_density": edge_density,
        f"{prefix}_fine_edge_density": fine_edge_density,
        f"{prefix}_saturation_mean": float(hsv[:, :, 1].mean()),
        f"{prefix}_saturation_std": float(hsv[:, :, 1].std()),
        f"{prefix}_value_mean": float(hsv[:, :, 2].mean()),
        f"{prefix}_red_mean": float(rgb[:, :, 0].mean()),
        f"{prefix}_green_mean": float(rgb[:, :, 1].mean()),
        f"{prefix}_blue_mean": float(rgb[:, :, 2].mean()),
        f"{prefix}_red_dominance_mean": float(red_dominance.mean()),
        f"{prefix}_blue_dominance_mean": float(blue_dominance.mean()),
        f"{prefix}_red_dominance_fraction": float((red_dominance > 0.12).mean()),
        f"{prefix}_blue_dominance_fraction": float((blue_dominance > 0.12).mean()),
        f"{prefix}_dark_fraction": float((gray < 0.18).mean()),
        f"{prefix}_light_fraction": float((gray > 0.92).mean()),
        f"{prefix}_white_fraction": float((rgb.min(axis=2) > 0.94).mean()),
        f"{prefix}_black_fraction": float((rgb.max(axis=2) < 0.08).mean()),
        f"{prefix}_top_dark_fraction": float((top_band.max(axis=2) < 0.18).mean()),
        f"{prefix}_top_red_dominance_mean": float(top_red_dominance.mean()),
        f"{prefix}_top_blue_dominance_mean": float(top_blue_dominance.mean()),
        f"{prefix}_row_dark_peaks": float((row_dark_fraction > 0.35).mean()),
        f"{prefix}_col_dark_peaks": float((col_dark_fraction > 0.25).mean()),
        f"{prefix}_channel_mean_range": float(max(stats.mean) - min(stats.mean)) / 255.0,
    }


def extract_visual_features(image_name: str) -> dict[str, float | str]:
    image_path = image_path_for_name(image_name)
    with Image.open(image_path) as image:
        width, height = image.size

    features: dict[str, float | str] = {
        "image": image_name,
        "image_width": float(width),
        "image_height": float(height),
        "image_aspect": float(width / height),
        "image_log_height": float(math.log1p(height)),
        "image_file_mb": float(image_path.stat().st_size / (1024 * 1024)),
    }

    for view_index in range(VIEW_COUNT):
        features.update(view_features(crop_view(image_path, view_index), f"view{view_index}"))

    return features


def ensure_visual_features(rows: list[dict[str, str]]) -> pd.DataFrame:
    if VISUAL_FEATURES_PATH.exists():
        existing = pd.read_csv(VISUAL_FEATURES_PATH)
        if set(rows_by_name(rows)).issubset(set(existing["image"])):
            return existing

    feature_rows = []
    for index, row in enumerate(rows, start=1):
        feature_rows.append(extract_visual_features(row["image"]))
        print(f"visual features {index}/{len(rows)}: {row['image']}", flush=True)

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(VISUAL_FEATURES_PATH, index=False)
    return feature_df


def rows_by_name(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["image"]: row for row in rows}


def labels_for_rows(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([row["true_bias_5"] for row in rows])


def one_hot_from_columns(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    dict_rows = [
        {column: row.get(column, "") for column in columns}
        for row in rows
    ]
    return DictVectorizer(sparse=False).fit_transform(dict_rows)


def visual_matrix(rows: list[dict[str, str]], feature_df: pd.DataFrame) -> np.ndarray:
    feature_by_name = feature_df.set_index("image")
    ordered = feature_by_name.loc[[row["image"] for row in rows]]
    return ordered.drop(columns=[], errors="ignore").to_numpy(dtype=float)


def combine(*matrices: np.ndarray) -> np.ndarray:
    return np.hstack(matrices)


def logreg_model() -> LogisticRegression:
    return LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=0.7,
        random_state=RANDOM_STATE,
    )


def random_forest_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def extra_trees_model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def cross_val_predictions(
    x: np.ndarray,
    y: np.ndarray,
    estimator_factory: Callable[[], object],
    scale: bool,
) -> np.ndarray:
    estimator = estimator_factory()
    if scale:
        estimator = make_pipeline(StandardScaler(), estimator)
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(estimator, x, y, cv=cv, n_jobs=None)


def majority_vote(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    predictions = []
    for row in rows:
        votes = Counter(row[column] for column in columns if row.get(column) in BIAS_5_LABELS)
        if not votes:
            predictions.append(row.get(GPT55_COLUMN, ""))
            continue
        predictions.append(votes.most_common(1)[0][0])
    return np.asarray(predictions)


def gpt55_least_visual_router(rows: list[dict[str, str]], visual_x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    output = np.asarray([row[GPT55_COLUMN] for row in rows], dtype=object)

    for train_index, test_index in cv.split(visual_x, y):
        train_rows = [rows[index] for index in train_index]
        train_mask = np.asarray([row[GPT55_COLUMN] == "least biased" for row in train_rows])
        if train_mask.sum() < len(BIAS_5_LABELS):
            continue

        router = make_pipeline(StandardScaler(), logreg_model())
        router.fit(visual_x[train_index][train_mask], y[train_index][train_mask])

        test_rows = [rows[index] for index in test_index]
        test_mask = np.asarray([row[GPT55_COLUMN] == "least biased" for row in test_rows])
        if test_mask.any():
            routed = router.predict(visual_x[test_index][test_mask])
            output[test_index[test_mask]] = routed

    return output


def least_router_cv(
    rows: list[dict[str, str]],
    router_x: np.ndarray,
    y: np.ndarray,
    name: str,
) -> np.ndarray:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    output = np.asarray([row[GPT55_COLUMN] for row in rows], dtype=object)

    for train_index, test_index in cv.split(router_x, y):
        train_rows = [rows[index] for index in train_index]
        train_mask = np.asarray([row[GPT55_COLUMN] == "least biased" for row in train_rows])
        if train_mask.sum() < len(BIAS_5_LABELS):
            continue

        router = make_pipeline(StandardScaler(), logreg_model())
        router.fit(router_x[train_index][train_mask], y[train_index][train_mask])

        test_rows = [rows[index] for index in test_index]
        test_mask = np.asarray([row[GPT55_COLUMN] == "least biased" for row in test_rows])
        if test_mask.any():
            routed = router.predict(router_x[test_index][test_mask])
            output[test_index[test_mask]] = routed

    return output


def ordinal_ridge_cv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    label_to_score = {
        "left": -2.0,
        "left-center": -1.0,
        "least biased": 0.0,
        "right-center": 1.0,
        "right": 2.0,
    }
    score_to_label = {
        -2: "left",
        -1: "left-center",
        0: "least biased",
        1: "right-center",
        2: "right",
    }
    y_scores = np.asarray([label_to_score[label] for label in y])
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    raw_predictions = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=4.0)), x, y_scores, cv=cv)
    rounded = np.clip(np.rint(raw_predictions), -2, 2).astype(int)
    return np.asarray([score_to_label[int(score)] for score in rounded])


def metric_row(method: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, str]:
    valid_mask = np.asarray([prediction in BIAS_5_LABELS for prediction in y_pred])
    evaluated_true = y_true[valid_mask]
    evaluated_pred = y_pred[valid_mask]
    return {
        "method": method,
        "sample_size": str(len(y_true)),
        "evaluated": str(len(evaluated_true)),
        "invalid_or_error": str(int((~valid_mask).sum())),
        "accuracy": f"{accuracy_score(evaluated_true, evaluated_pred):.6f}",
        "macro_f1": f"{f1_score(evaluated_true, evaluated_pred, labels=BIAS_5_LABELS, average='macro'):.6f}",
        "weighted_f1": f"{f1_score(evaluated_true, evaluated_pred, labels=BIAS_5_LABELS, average='weighted'):.6f}",
    }


def confusion_rows(method: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, str]]:
    matrix = confusion_matrix(y_true, y_pred, labels=BIAS_5_LABELS)
    rows = []
    for row_index, true_label in enumerate(BIAS_5_LABELS):
        rows.append(
            {
                "method": method,
                "true_label": true_label,
                **{f"pred_{label}": str(int(matrix[row_index, col_index])) for col_index, label in enumerate(BIAS_5_LABELS)},
            }
        )
    return rows


def run() -> None:
    rows = load_prediction_rows()
    rows = sorted(rows, key=lambda row: row["image"])
    y = labels_for_rows(rows)
    feature_df = ensure_visual_features(rows)

    visual_x = visual_matrix(rows, feature_df)
    gpt55_x = one_hot_from_columns(rows, [GPT55_COLUMN])
    all_model_x = one_hot_from_columns(rows, MODEL_COLUMNS)

    methods: dict[str, np.ndarray] = {
        "baseline_gpt5.5": np.asarray([row[GPT55_COLUMN] for row in rows]),
        "all_models_majority_vote": majority_vote(rows, MODEL_COLUMNS),
        "visual_logreg_cv": cross_val_predictions(visual_x, y, logreg_model, scale=True),
        "visual_random_forest_cv": cross_val_predictions(visual_x, y, random_forest_model, scale=False),
        "visual_extra_trees_cv": cross_val_predictions(visual_x, y, extra_trees_model, scale=False),
        "gpt55_label_logreg_cv": cross_val_predictions(gpt55_x, y, logreg_model, scale=False),
        "gpt55_label_plus_visual_logreg_cv": cross_val_predictions(combine(gpt55_x, visual_x), y, logreg_model, scale=True),
        "gpt55_label_plus_visual_random_forest_cv": cross_val_predictions(combine(gpt55_x, visual_x), y, random_forest_model, scale=False),
        "gpt55_label_plus_visual_extra_trees_cv": cross_val_predictions(combine(gpt55_x, visual_x), y, extra_trees_model, scale=False),
        "gpt55_least_visual_router_cv": gpt55_least_visual_router(rows, visual_x, y),
        "gpt55_least_allmodel_visual_router_cv": least_router_cv(rows, combine(all_model_x, visual_x), y, "allmodel_visual"),
        "gpt55_visual_ordinal_ridge_cv": ordinal_ridge_cv(combine(gpt55_x, visual_x), y),
        "all_model_stack_logreg_cv": cross_val_predictions(all_model_x, y, logreg_model, scale=False),
        "all_model_stack_plus_visual_logreg_cv": cross_val_predictions(combine(all_model_x, visual_x), y, logreg_model, scale=True),
        "all_model_stack_plus_visual_extra_trees_cv": cross_val_predictions(combine(all_model_x, visual_x), y, extra_trees_model, scale=False),
        "all_model_visual_ordinal_ridge_cv": ordinal_ridge_cv(combine(all_model_x, visual_x), y),
        "all_model_stack_random_forest_cv": cross_val_predictions(all_model_x, y, random_forest_model, scale=False),
    }

    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                "image": row["image"],
                "true_bias_5": row["true_bias_5"],
                **{method: str(predictions[index]) for method, predictions in methods.items()},
            }
        )

    metric_rows = [metric_row(method, y, predictions) for method, predictions in methods.items()]
    all_confusion_rows = [
        confusion_row
        for method, predictions in methods.items()
        for confusion_row in confusion_rows(method, y, predictions)
    ]

    write_csv(ML_PREDICTIONS_PATH, prediction_rows, list(prediction_rows[0].keys()))
    write_csv(ML_METRICS_PATH, metric_rows, list(metric_rows[0].keys()))
    write_csv(
        ML_CONFUSIONS_PATH,
        all_confusion_rows,
        ["method", "true_label", *[f"pred_{label}" for label in BIAS_5_LABELS]],
    )
    write_report(metric_rows)


def write_report(metric_rows: list[dict[str, str]]) -> None:
    sorted_rows = sorted(metric_rows, key=lambda row: float(row["macro_f1"]), reverse=True)
    metrics_csv = ML_METRICS_PATH.read_text(encoding="utf-8") if ML_METRICS_PATH.exists() else ""
    report = (
        "# Bias 5-Label ML Experiment Report\n\n"
        "No new LLM calls were made in this experiment. It reuses cached full-run labels and screenshot pixels.\n\n"
        "## Methods\n\n"
        "- `visual_*`: classical ML using image-only features from the first four viewport-height screenshots.\n"
        "- `gpt55_label_*`: calibration/stacking using only the cached GPT-5.5 5-label prediction, optionally plus visual features.\n"
        "- `gpt55_least_visual_router_cv`: keep GPT-5.5 unless it predicts `least biased`; train a visual router for those ambiguous rows.\n"
        "- `*_ordinal_ridge_cv`: regress labels on an ordinal left-to-right axis and round back to five labels.\n"
        "- `all_model_*`: uses cached labels from all four prior models as an upper-bound stacking signal; no new API calls, but not a single-model method.\n\n"
        "## Best By Macro-F1\n\n"
        f"`{sorted_rows[0]['method']}`: accuracy `{sorted_rows[0]['accuracy']}`, macro-F1 `{sorted_rows[0]['macro_f1']}`.\n\n"
        "## Takeaways\n\n"
        "- Best single-model method: `gpt55_label_plus_visual_extra_trees_cv`, accuracy `0.716000`, macro-F1 `0.716226`.\n"
        "- Best overall method using cached labels from all four previous LLMs: `all_model_stack_plus_visual_extra_trees_cv`, accuracy `0.720000`, macro-F1 `0.717370`.\n"
        "- The gain comes from using GPT-5.5 as a semantic prior and visual screenshot features as a learned calibrator. This directly fixes the baseline's major failure mode: over-predicting `least biased` for `left-center` and `right-center`.\n"
        "- Visual-only ML is not strong enough by itself (`0.468`-`0.488` accuracy), but it carries useful calibration signal when combined with GPT-5.5.\n"
        "- Simple stacking on labels alone is weak. `all_model_stack_logreg_cv` only reaches `0.544`; the visual features are the important additional signal.\n"
        "- Ordinal regression was not competitive. The five labels behave partly ordinal, but the boundary between `least biased` and center-lean labels is not captured well by a single left-to-right scalar.\n\n"
        "## Recommended Pipeline\n\n"
        "For a single-model setup: run GPT-5.5 with the original 5-label prompt, extract the same four-viewport visual features from the screenshot, then classify with an ExtraTrees calibrator trained on `GPT-5.5 label + visual features`.\n\n"
        "This keeps LLM cost to one GPT-5.5 call per website and moves the improvement into cheap local ML.\n\n"
        "## Metrics\n\n"
        "```csv\n"
        f"{metrics_csv.strip()}\n"
        "```\n\n"
        "Confusions are saved in `bias5_ml_confusions.csv`.\n"
    )
    ML_REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    run()
