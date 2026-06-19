from experiment_bootstrap import activate_project_root

activate_project_root()

import csv
import io
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from image_process import BIAS_5_LABELS, IMAGE_FOLDER, write_csv


Image.MAX_IMAGE_PIXELS = None

RANDOM_STATE = 41
CV_SPLITS = 5
VIEW_COUNT = 4
TARGET_ASPECT_RATIO = 16 / 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREDICTIONS_INPUT_PATH = Path("bias_predictions.csv")
HANDCRAFTED_VISUAL_PATH = Path("bias5_ml_visual_features.csv")
EMBEDDINGS_PATH = Path("bias5_resnet18_embeddings.npz")
METRICS_PATH = Path("bias5_cnn_metrics.csv")
PREDICTIONS_PATH = Path("bias5_cnn_predictions.csv")
CONFUSIONS_PATH = Path("bias5_cnn_confusions.csv")
SCRATCH_FOLD_METRICS_PATH = Path("bias5_scratch_cnn_fold_metrics.csv")
REPORT_PATH = Path("bias5_cnn_experiment_report.md")

GPT55_COLUMN = "bias_5:openai/gpt-5.5"
LABEL_TO_INDEX = {label: index for index, label in enumerate(BIAS_5_LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows() -> list[dict[str, str]]:
    with PREDICTIONS_INPUT_PATH.open("r", encoding="utf-8", newline="") as input_file:
        return sorted(list(csv.DictReader(input_file)), key=lambda row: row["image"])


def image_path(image_name: str) -> Path:
    return IMAGE_FOLDER / f"{image_name}.png"


def crop_viewport(path: Path, viewport_index: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        viewport_height = min(height, round(width / TARGET_ASPECT_RATIO))
        top = min(viewport_index * viewport_height, max(0, height - viewport_height))
        return image.crop((0, top, width, top + viewport_height))


def build_view_grid(path: Path, resize_size: tuple[int, int] = (160, 90)) -> Image.Image:
    crops = [crop_viewport(path, index).resize(resize_size) for index in range(VIEW_COUNT)]
    width, height = resize_size
    grid = Image.new("RGB", (width * 2, height * 2), "white")
    for index, crop in enumerate(crops):
        left = (index % 2) * width
        top = (index // 2) * height
        grid.paste(crop, (left, top))
    return grid


def load_resnet18() -> tuple[nn.Module, transforms.Compose]:
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        mean = weights.transforms().mean
        std = weights.transforms().std
    except Exception:
        model = models.resnet18(weights=None)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    model.fc = nn.Identity()
    model.eval()
    model.to(DEVICE)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return model, transform


def ensure_resnet_embeddings(rows: list[dict[str, str]]) -> np.ndarray:
    image_names = [row["image"] for row in rows]
    if EMBEDDINGS_PATH.exists():
        cached = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        cached_names = cached["image_names"].tolist()
        if cached_names == image_names:
            return cached["embeddings"]

    model, transform = load_resnet18()
    embeddings = []

    with torch.no_grad():
        for row_index, row in enumerate(rows, start=1):
            tensors = []
            path = image_path(row["image"])
            for viewport_index in range(VIEW_COUNT):
                tensors.append(transform(crop_viewport(path, viewport_index)))
            batch = torch.stack(tensors).to(DEVICE)
            features = model(batch).detach().cpu().numpy()
            combined = np.concatenate(
                [
                    features.reshape(-1),
                    features.mean(axis=0),
                    features.std(axis=0),
                ]
            )
            embeddings.append(combined.astype(np.float32))
            print(f"resnet embeddings {row_index}/{len(rows)}: {row['image']}", flush=True)

    embedding_matrix = np.vstack(embeddings)
    np.savez_compressed(EMBEDDINGS_PATH, image_names=np.asarray(image_names), embeddings=embedding_matrix)
    return embedding_matrix


def one_hot_labels(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    dict_rows = [{column: row.get(column, "") for column in columns} for row in rows]
    return DictVectorizer(sparse=False).fit_transform(dict_rows)


def handcrafted_visual_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    visual_df = pd.read_csv(HANDCRAFTED_VISUAL_PATH).set_index("image")
    ordered = visual_df.loc[[row["image"] for row in rows]]
    return ordered.to_numpy(dtype=float)


def combine(*matrices: np.ndarray) -> np.ndarray:
    return np.hstack(matrices)


def ridge() -> RidgeClassifier:
    return RidgeClassifier(alpha=3.0, class_weight="balanced")


def extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=60,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def cv_predict(x: np.ndarray, y: np.ndarray, estimator: Any, scale: bool) -> np.ndarray:
    model = make_pipeline(StandardScaler(), estimator) if scale else estimator
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, x, y, cv=cv)


def cv_predict_pca_ridge(x: np.ndarray, y: np.ndarray, n_components: int = 32) -> np.ndarray:
    components = min(n_components, x.shape[0] - x.shape[0] // CV_SPLITS - 1, x.shape[1])
    model = make_pipeline(StandardScaler(), PCA(n_components=components, random_state=RANDOM_STATE), ridge())
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, x, y, cv=cv)


def cosine_centroid_cv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    output = np.empty(len(y), dtype=object)

    normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    for train_index, test_index in cv.split(normalized, y):
        centroids = []
        for label in BIAS_5_LABELS:
            class_rows = normalized[train_index][y[train_index] == label]
            centroid = class_rows.mean(axis=0)
            centroid = centroid / max(np.linalg.norm(centroid), 1e-8)
            centroids.append(centroid)
        centroid_matrix = np.vstack(centroids)
        similarities = normalized[test_index] @ centroid_matrix.T
        output[test_index] = [BIAS_5_LABELS[index] for index in similarities.argmax(axis=1)]

    return output


def cosine_knn_cv(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    output = np.empty(len(y), dtype=object)
    normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    for train_index, test_index in cv.split(normalized, y):
        similarities = normalized[test_index] @ normalized[train_index].T
        neighbor_indices = np.argsort(-similarities, axis=1)[:, :k]
        train_labels = y[train_index]
        for row_offset, neighbors in enumerate(neighbor_indices):
            votes = {}
            for rank, neighbor in enumerate(neighbors):
                label = train_labels[neighbor]
                votes[label] = votes.get(label, 0.0) + 1.0 / (rank + 1)
            output[test_index[row_offset]] = max(votes.items(), key=lambda item: item[1])[0]

    return output


def build_grid_cache(rows: list[dict[str, str]]) -> dict[str, Image.Image]:
    cache = {}
    for index, row in enumerate(rows, start=1):
        cache[row["image"]] = build_view_grid(image_path(row["image"]))
        print(f"scratch grid cache {index}/{len(rows)}: {row['image']}", flush=True)
    return cache


class ViewGridDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], grid_cache: dict[str, Image.Image], train: bool) -> None:
        self.rows = rows
        self.grid_cache = grid_cache
        jitter = []
        if train:
            jitter = [
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
            ]
        self.transform = transforms.Compose(
            [
                *jitter,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        grid = self.grid_cache[row["image"]]
        return self.transform(grid), LABEL_TO_INDEX[row["true_bias_5"]]


class SmallScreenshotCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, len(BIAS_5_LABELS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train_one_scratch_fold(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    grid_cache: dict[str, Image.Image],
    fold: int,
    epochs: int = 18,
) -> tuple[list[str], dict[str, str]]:
    set_seed(RANDOM_STATE + fold)
    train_loader = DataLoader(ViewGridDataset(train_rows, grid_cache, train=True), batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(ViewGridDataset(test_rows, grid_cache, train=False), batch_size=32, shuffle=False, num_workers=0)
    model = SmallScreenshotCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    predictions = []
    truth = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x.to(DEVICE))
            predicted = logits.argmax(dim=1).detach().cpu().numpy()
            predictions.extend(INDEX_TO_LABEL[int(index)] for index in predicted)
            truth.extend(INDEX_TO_LABEL[int(index)] for index in batch_y.numpy())

    fold_metrics = {
        "fold": str(fold),
        "accuracy": f"{accuracy_score(truth, predictions):.6f}",
        "macro_f1": f"{f1_score(truth, predictions, labels=BIAS_5_LABELS, average='macro'):.6f}",
    }
    return predictions, fold_metrics


def scratch_cnn_cv(rows: list[dict[str, str]], y: np.ndarray) -> tuple[np.ndarray, list[dict[str, str]]]:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    output = np.empty(len(rows), dtype=object)
    fold_metrics = []
    grid_cache = build_grid_cache(rows)

    for fold, (train_index, test_index) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        train_rows = [rows[index] for index in train_index]
        test_rows = [rows[index] for index in test_index]
        predictions, metrics = train_one_scratch_fold(train_rows, test_rows, grid_cache, fold)
        output[test_index] = predictions
        fold_metrics.append(metrics)
        print(f"scratch cnn fold {fold}/{CV_SPLITS}: {metrics}", flush=True)

    return output, fold_metrics


def metric_row(method: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, str]:
    return {
        "method": method,
        "sample_size": str(len(y_true)),
        "accuracy": f"{accuracy_score(y_true, y_pred):.6f}",
        "macro_f1": f"{f1_score(y_true, y_pred, labels=BIAS_5_LABELS, average='macro'):.6f}",
        "weighted_f1": f"{f1_score(y_true, y_pred, labels=BIAS_5_LABELS, average='weighted'):.6f}",
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
    set_seed(RANDOM_STATE)
    rows = load_rows()
    y = np.asarray([row["true_bias_5"] for row in rows])
    resnet_x = ensure_resnet_embeddings(rows)
    resnet_mean_x = resnet_x[:, 2048:2560]
    resnet_meanstd_x = resnet_x[:, 2048:3072]

    methods: dict[str, np.ndarray] = {
        "baseline_gpt5.5": np.asarray([row[GPT55_COLUMN] for row in rows]),
        "resnet18_mean_cosine_centroid_cv": cosine_centroid_cv(resnet_mean_x, y),
        "resnet18_meanstd_cosine_centroid_cv": cosine_centroid_cv(resnet_meanstd_x, y),
        "resnet18_meanstd_cosine_knn5_cv": cosine_knn_cv(resnet_meanstd_x, y, k=5),
        "resnet18_meanstd_cosine_knn9_cv": cosine_knn_cv(resnet_meanstd_x, y, k=9),
    }

    scratch_predictions, scratch_fold_metrics = scratch_cnn_cv(rows, y)
    methods["scratch_small_cnn_cv"] = scratch_predictions

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
        row
        for method, predictions in methods.items()
        for row in confusion_rows(method, y, predictions)
    ]

    write_csv(PREDICTIONS_PATH, prediction_rows, list(prediction_rows[0].keys()))
    write_csv(METRICS_PATH, metric_rows, list(metric_rows[0].keys()))
    write_csv(CONFUSIONS_PATH, all_confusion_rows, ["method", "true_label", *[f"pred_{label}" for label in BIAS_5_LABELS]])
    write_csv(SCRATCH_FOLD_METRICS_PATH, scratch_fold_metrics, ["fold", "accuracy", "macro_f1"])
    write_report(metric_rows, scratch_fold_metrics)


def write_report(metric_rows: list[dict[str, str]], scratch_fold_metrics: list[dict[str, str]]) -> None:
    metrics_text = METRICS_PATH.read_text(encoding="utf-8") if METRICS_PATH.exists() else ""
    fold_text = SCRATCH_FOLD_METRICS_PATH.read_text(encoding="utf-8") if SCRATCH_FOLD_METRICS_PATH.exists() else ""
    best = max(metric_rows, key=lambda row: float(row["macro_f1"]))
    report = (
        "# Bias 5 CNN Experiment Report\n\n"
        f"Device: `{DEVICE}`.\n\n"
        "Dataset: full 250-site bias dataset, evaluated with 5-fold stratified CV.\n\n"
        "## Methods\n\n"
        "- `resnet18_embedding_*`: first four viewport crops are passed through frozen ImageNet ResNet18; per-viewport embeddings are concatenated with mean/std pooled embeddings, then classical ML predicts the label.\n"
        "- `*_plus_gpt55_*`: adds one-hot GPT-5.5 baseline label to the ResNet embedding features.\n"
        "- `*_plus_visual_*`: adds the handcrafted visual features from the earlier experiment.\n"
        "- `scratch_small_cnn_cv`: a small 4-layer CNN trained from scratch on a 2x2 grid of the first four viewport crops.\n\n"
        "## Best Method\n\n"
        f"`{best['method']}`: accuracy `{best['accuracy']}`, macro-F1 `{best['macro_f1']}`.\n\n"
        "## Metrics\n\n"
        "```csv\n"
        f"{metrics_text.strip()}\n"
        "```\n\n"
        "## Scratch CNN Fold Metrics\n\n"
        "```csv\n"
        f"{fold_text.strip()}\n"
        "```\n"
    )
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    run()
