from experiment_bootstrap import activate_project_root

activate_project_root()

import base64
import csv
import io
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold


Image.MAX_IMAGE_PIXELS = None

SNAPSHOT_INDEX_PATH = Path("snapshot_index.csv")
IMAGE_FOLDER = Path("snapshot-samples")
DATASET_PATH = Path("factuality_dataset.csv")
GPT_PREDICTIONS_PATH = Path("factuality_gpt55_predictions.csv")

TARGET_ASPECT_RATIO = 16 / 9
VIEW_COUNT = 4
CV_SPLITS = 5
RANDOM_STATE = 101

BINARY_LABELS = ["low", "high"]
MULTICLASS_LABELS = ["very low", "low", "high", "very high"]
MULTICLASS_TO_BINARY = {
    "very low": "low",
    "low": "low",
    "high": "high",
    "very high": "high",
}
LABEL_TO_ORDINAL = {
    "very low": 0,
    "low": 1,
    "high": 2,
    "very high": 3,
}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_dataset() -> list[dict[str, str]]:
    with SNAPSHOT_INDEX_PATH.open("r", encoding="utf-8-sig", newline="") as input_file:
        index_rows = list(csv.DictReader(input_file))

    rows_by_filename: dict[str, dict[str, str]] = {}
    for row in index_rows:
        filename = Path(row["image_path"]).name
        if filename in rows_by_filename:
            raise ValueError(f"Duplicate image filename in snapshot index: {filename}")
        rows_by_filename[filename] = row

    output_rows = []
    for image_path in sorted(IMAGE_FOLDER.glob("*.png"), key=lambda path: path.stem.lower()):
        source = rows_by_filename.get(image_path.name)
        if source is None:
            raise ValueError(f"Image is missing from snapshot index: {image_path.name}")

        multiclass = source["factuality"].strip().lower()
        if multiclass not in MULTICLASS_LABELS:
            raise ValueError(f"Unexpected factuality label for {image_path.name}: {multiclass}")
        binary = MULTICLASS_TO_BINARY[multiclass]
        expected_trustworthiness = "1" if binary == "high" else "0"
        if source["trustworthiness"].strip() != expected_trustworthiness:
            raise ValueError(f"Inconsistent trustworthiness for {image_path.name}")

        output_rows.append(
            {
                "image": image_path.stem,
                "image_file": image_path.name,
                "image_path": str(image_path.as_posix()),
                "media_name": source["media_name"],
                "url": source["url"],
                "country": source["country"],
                "factuality_4": multiclass,
                "factuality_2": binary,
                "trustworthiness": source["trustworthiness"],
            }
        )

    if len(output_rows) != 200:
        raise ValueError(f"Expected 200 factuality screenshots, found {len(output_rows)}")
    if Counter(row["factuality_2"] for row in output_rows) != Counter({"low": 100, "high": 100}):
        raise ValueError("Binary factuality dataset is not balanced 100/100")

    write_csv(DATASET_PATH, output_rows, list(output_rows[0].keys()))
    return output_rows


def load_dataset() -> list[dict[str, str]]:
    if not DATASET_PATH.exists():
        return build_dataset()
    with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if len(rows) != 200:
        return build_dataset()
    return rows


def image_path(image_name: str) -> Path:
    return IMAGE_FOLDER / f"{image_name}.png"


def crop_viewport(path: Path, viewport_index: int, max_width: int | None = None) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        viewport_height = min(height, round(width / TARGET_ASPECT_RATIO))
        top = min(viewport_index * viewport_height, max(0, height - viewport_height))
        crop = image.crop((0, top, width, top + viewport_height))

    if max_width and crop.width > max_width:
        new_height = round(crop.height * max_width / crop.width)
        crop = crop.resize((max_width, new_height), Image.Resampling.LANCZOS)
    return crop


def viewport_data_url(path: Path, viewport_index: int, max_width: int = 1600) -> str:
    crop = crop_viewport(path, viewport_index, max_width=max_width)
    output = io.BytesIO()
    crop.save(output, format="PNG", optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def fixed_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def labels_for_task(rows: list[dict[str, str]], task: str) -> tuple[np.ndarray, list[str]]:
    if task == "binary":
        return np.asarray([row["factuality_2"] for row in rows], dtype=object), BINARY_LABELS
    if task == "multiclass":
        return np.asarray([row["factuality_4"] for row in rows], dtype=object), MULTICLASS_LABELS
    raise ValueError(f"Unknown task: {task}")


def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_values = np.asarray([LABEL_TO_ORDINAL[label] for label in y_true], dtype=int)
    predicted_values = np.asarray([LABEL_TO_ORDINAL[label] for label in y_pred], dtype=int)
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


def metric_row(
    task: str,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, str]:
    row = {
        "task": task,
        "method": method,
        "sample_size": str(len(y_true)),
        "accuracy": f"{accuracy_score(y_true, y_pred):.6f}",
        "balanced_accuracy": f"{balanced_accuracy_score(y_true, y_pred):.6f}",
        "macro_precision": f"{precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.6f}",
        "macro_recall": f"{recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.6f}",
        "macro_f1": f"{f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.6f}",
        "weighted_f1": f"{f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0):.6f}",
        "mae_steps": "",
        "mse_steps2": "",
        "rmse_steps": "",
        "within_1_accuracy": "",
        "severe_error_rate_ge_2": "",
        "quadratic_weighted_kappa": "",
    }
    if task == "multiclass":
        row.update({key: f"{value:.6f}" for key, value in ordinal_metrics(y_true, y_pred).items()})
    return row


def confusion_rows(
    task: str,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> list[dict[str, str]]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return [
        {
            "task": task,
            "method": method,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "count": str(int(matrix[true_index, predicted_index])),
        }
        for true_index, true_label in enumerate(labels)
        for predicted_index, predicted_label in enumerate(labels)
    ]
