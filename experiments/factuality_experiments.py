from experiment_bootstrap import activate_project_root

activate_project_root()

import csv
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.base import clone
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from factuality_common import (
    CV_SPLITS,
    GPT_PREDICTIONS_PATH,
    LABEL_TO_ORDINAL,
    MULTICLASS_LABELS,
    RANDOM_STATE,
    VIEW_COUNT,
    confusion_rows,
    crop_viewport,
    fixed_cv,
    image_path,
    labels_for_task,
    load_dataset,
    metric_row,
    write_csv,
)
from factuality_prepare_features import (
    BOOLEAN_FIELDS,
    CATEGORICAL_FIELDS,
    HF_IMAGE_MODELS,
    NUMERIC_FIELDS,
    OCR_FEATURES_PATH,
    RESNET_EMBEDDINGS_PATH,
    SEMANTIC_FEATURES_PATH,
    TOPICS,
    VISUAL_FEATURES_PATH,
    embedding_path,
    prompt_score_path,
)


Image.MAX_IMAGE_PIXELS = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METRICS_PATH = Path("factuality_all_methods_metrics.csv")
CONFUSIONS_PATH = Path("factuality_all_methods_confusions.csv")
REPORT_PATH = Path("factuality_all_methods_report.md")
CNN_FOLD_METRICS_PATH = Path("factuality_scratch_cnn_fold_metrics.csv")

COMPONENT_CONFIGS = {
    "compact": {"clip": 8, "siglip": 8, "dino": 8, "ocr_text": 16, "visual": 8},
    "medium": {"clip": 16, "siglip": 16, "dino": 16, "ocr_text": 32, "visual": 16},
    "large": {"clip": 32, "siglip": 32, "dino": 32, "ocr_text": 48, "visual": 24},
}


def aligned_frame(path: Path, image_names: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path).set_index("image")
    missing = sorted(set(image_names) - set(frame.index))
    if missing:
        raise ValueError(f"{path} is missing rows: {missing[:5]}")
    return frame.loc[image_names].reset_index()


def load_embedding(path: Path, image_names: list[str]) -> np.ndarray:
    cache = np.load(path, allow_pickle=True)
    names = cache["image_names"].tolist()
    index_by_name = {name: index for index, name in enumerate(names)}
    missing = sorted(set(image_names) - set(index_by_name))
    if missing:
        raise ValueError(f"{path} is missing embeddings: {missing[:5]}")
    return np.vstack([cache["embeddings"][index_by_name[name]] for name in image_names]).astype(np.float32)


def clean_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def load_features(rows: list[dict[str, str]], task: str, labels: list[str]) -> dict[str, Any]:
    image_names = [row["image"] for row in rows]
    visual = aligned_frame(VISUAL_FEATURES_PATH, image_names)
    ocr = aligned_frame(OCR_FEATURES_PATH, image_names)
    semantic = aligned_frame(SEMANTIC_FEATURES_PATH, image_names)
    gpt = aligned_frame(GPT_PREDICTIONS_PATH, image_names)
    clip_scores = aligned_frame(prompt_score_path("clip_vit_b32"), image_names)
    siglip_scores = aligned_frame(prompt_score_path("siglip_b16"), image_names)
    baseline_column = f"{task}:openai/gpt-5.5"
    baseline = gpt[baseline_column].astype(str).to_numpy(dtype=object)
    invalid = sorted(set(baseline) - set(labels))
    if invalid:
        raise ValueError(f"Invalid GPT labels for {task}: {invalid}")

    return {
        "image_names": image_names,
        "visual": clean_matrix(visual.drop(columns=["image"]).to_numpy(dtype=np.float32)),
        "ocr_texts": ocr["ocr_text"].fillna("").astype(str).tolist(),
        "ocr_numeric": clean_matrix(
            ocr.drop(columns=["image", "ocr_text"], errors="ignore").to_numpy(dtype=np.float32)
        ),
        "semantic": semantic,
        "clip": load_embedding(embedding_path("clip_vit_b32"), image_names),
        "siglip": load_embedding(embedding_path("siglip_b16"), image_names),
        "dino": load_embedding(embedding_path("dinov2_small"), image_names),
        "resnet": load_embedding(RESNET_EMBEDDINGS_PATH, image_names),
        "prompt_scores": clean_matrix(
            np.hstack(
                [
                    clip_scores.drop(columns=["image"]).to_numpy(dtype=np.float32),
                    siglip_scores.drop(columns=["image"]).to_numpy(dtype=np.float32),
                ]
            )
        ),
        "gpt_labels": baseline,
        "gpt_one_hot": np.asarray(
            [[1.0 if prediction == label else 0.0 for label in labels] for prediction in baseline],
            dtype=np.float32,
        ),
    }


def extra_trees(max_depth: int = 7, estimators: int = 500) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def logreg(c_value: float = 0.7) -> LogisticRegression:
    return LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=c_value,
        random_state=RANDOM_STATE,
    )


def pca_components(matrix: np.ndarray, requested: int = 24) -> int:
    train_size = math.floor(matrix.shape[0] * (CV_SPLITS - 1) / CV_SPLITS)
    return max(2, min(requested, train_size - 1, matrix.shape[1]))


def pca_ridge_estimator(matrix: np.ndarray, requested: int = 24) -> Any:
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=pca_components(matrix, requested), random_state=RANDOM_STATE),
        RidgeClassifier(alpha=3.0, class_weight="balanced"),
    )


def dense_cv(matrix: np.ndarray, y: np.ndarray, estimator: Any, scale: bool = False) -> np.ndarray:
    model = make_pipeline(StandardScaler(), estimator) if scale else estimator
    return cross_val_predict(model, clean_matrix(matrix), y, cv=fixed_cv())


def text_cv(texts: list[str], y: np.ndarray) -> np.ndarray:
    estimator = make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_features=3000,
            sublinear_tf=True,
        ),
        LinearSVC(C=0.7, class_weight="balanced", random_state=RANDOM_STATE),
    )
    return cross_val_predict(estimator, texts, y, cv=fixed_cv())


def cosine_centroid_cv(matrix: np.ndarray, y: np.ndarray, labels: list[str]) -> np.ndarray:
    matrix = clean_matrix(matrix)
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    output = np.empty(len(y), dtype=object)
    for train_index, test_index in fixed_cv().split(normalized, y):
        centroids = []
        for label in labels:
            centroid = normalized[train_index][y[train_index] == label].mean(axis=0)
            centroids.append(centroid / max(np.linalg.norm(centroid), 1e-8))
        similarities = normalized[test_index] @ np.vstack(centroids).T
        output[test_index] = [labels[index] for index in similarities.argmax(axis=1)]
    return output


def cosine_knn_cv(matrix: np.ndarray, y: np.ndarray, k: int = 5) -> np.ndarray:
    estimator = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    return cross_val_predict(estimator, clean_matrix(matrix), y, cv=fixed_cv())


def majority_vote(predictions: list[np.ndarray], labels: list[str]) -> np.ndarray:
    output = []
    tie_order = list(reversed(labels))
    for values in zip(*predictions):
        counts = Counter(values)
        top_count = max(counts.values())
        tied = {label for label, count in counts.items() if count == top_count}
        output.append(next(label for label in tie_order if label in tied))
    return np.asarray(output, dtype=object)


def prediction_one_hot(predictions: list[np.ndarray], labels: list[str]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(prediction_set[row_index] == label)
                for prediction_set in predictions
                for label in labels
            ]
            for row_index in range(len(predictions[0]))
        ],
        dtype=np.float32,
    )


def nested_stack_cv(
    y: np.ndarray,
    labels: list[str],
    dense_specs: list[tuple[str, np.ndarray, Any]],
    text_specs: list[tuple[str, list[str], Any]],
    static_predictions: list[np.ndarray],
) -> np.ndarray:
    output = np.empty(len(y), dtype=object)
    for fold, (train_index, test_index) in enumerate(fixed_cv().split(np.zeros(len(y)), y), start=1):
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + fold)
        train_predictions = []
        test_predictions = []
        for _, matrix, estimator in dense_specs:
            matrix = clean_matrix(matrix)
            train_predictions.append(
                cross_val_predict(clone(estimator), matrix[train_index], y[train_index], cv=inner_cv)
            )
            model = clone(estimator)
            model.fit(matrix[train_index], y[train_index])
            test_predictions.append(model.predict(matrix[test_index]))
        for _, texts, estimator in text_specs:
            train_texts = [texts[index] for index in train_index]
            test_texts = [texts[index] for index in test_index]
            train_predictions.append(
                cross_val_predict(clone(estimator), train_texts, y[train_index], cv=inner_cv)
            )
            model = clone(estimator)
            model.fit(train_texts, y[train_index])
            test_predictions.append(model.predict(test_texts))
        for predictions in static_predictions:
            train_predictions.append(predictions[train_index])
            test_predictions.append(predictions[test_index])
        meta_model = logreg(0.8)
        meta_model.fit(prediction_one_hot(train_predictions, labels), y[train_index])
        output[test_index] = meta_model.predict(prediction_one_hot(test_predictions, labels))
    return output


def semantic_matrices(
    semantic_frame: pd.DataFrame,
    baseline: np.ndarray,
    visual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    categorical_columns = list(CATEGORICAL_FIELDS)
    vectorizer = DictVectorizer(sparse=False)
    category_matrix = vectorizer.fit_transform(
        semantic_frame[categorical_columns].to_dict(orient="records")
    )
    numeric_columns = [
        column
        for column in semantic_frame.columns
        if column.startswith("topic_") or column in BOOLEAN_FIELDS or column in NUMERIC_FIELDS
    ]
    numeric_matrix = semantic_frame[numeric_columns].to_numpy(dtype=np.float32)
    semantic_only = clean_matrix(np.hstack([category_matrix, numeric_matrix]))
    baseline_vectorizer = DictVectorizer(sparse=False)
    baseline_matrix = baseline_vectorizer.fit_transform(
        [{"baseline": prediction} for prediction in baseline]
    )
    semantic_plus_baseline = clean_matrix(np.hstack([semantic_only, baseline_matrix]))
    return semantic_only, semantic_plus_baseline, clean_matrix(
        np.hstack([semantic_plus_baseline, visual])
    )


def ordinal_ridge_cv(matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    y_numeric = np.asarray([LABEL_TO_ORDINAL[label] for label in y], dtype=np.float32)
    output = np.empty(len(y), dtype=object)
    for train_index, test_index in fixed_cv().split(matrix, y):
        components = max(2, min(24, len(train_index) - 1, matrix.shape[1]))
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=components, random_state=RANDOM_STATE),
            Ridge(alpha=4.0),
        )
        model.fit(matrix[train_index], y_numeric[train_index])
        predicted = np.clip(np.rint(model.predict(matrix[test_index])), 0, 3).astype(int)
        output[test_index] = [MULTICLASS_LABELS[index] for index in predicted]
    return output


def classical_methods(
    task: str,
    y: np.ndarray,
    labels: list[str],
    data: dict[str, Any],
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {
        "majority_class": np.repeat(Counter(y).most_common(1)[0][0], len(y)),
        "baseline_gpt5.5": data["gpt_labels"],
        "visual_logreg_cv": dense_cv(data["visual"], y, logreg(), scale=True),
        "visual_random_forest_cv": dense_cv(data["visual"], y, random_forest()),
        "visual_extra_trees_cv": dense_cv(data["visual"], y, extra_trees()),
        "ocr_text_tfidf_svc_cv": text_cv(data["ocr_texts"], y),
        "ocr_numeric_extra_trees_cv": dense_cv(data["ocr_numeric"], y, extra_trees()),
        "ocr_numeric_plus_visual_extra_trees_cv": dense_cv(
            np.hstack([data["ocr_numeric"], data["visual"]]), y, extra_trees()
        ),
        "clip_vit_b32_cosine_centroid_cv": cosine_centroid_cv(data["clip"], y, labels),
        "clip_vit_b32_knn5_cv": cosine_knn_cv(data["clip"], y),
        "clip_vit_b32_pca_ridge_cv": dense_cv(data["clip"], y, pca_ridge_estimator(data["clip"])),
        "siglip_b16_cosine_centroid_cv": cosine_centroid_cv(data["siglip"], y, labels),
        "siglip_b16_knn5_cv": cosine_knn_cv(data["siglip"], y),
        "siglip_b16_pca_ridge_cv": dense_cv(data["siglip"], y, pca_ridge_estimator(data["siglip"])),
        "dinov2_small_cosine_centroid_cv": cosine_centroid_cv(data["dino"], y, labels),
        "dinov2_small_knn5_cv": cosine_knn_cv(data["dino"], y),
        "dinov2_small_pca_ridge_cv": dense_cv(data["dino"], y, pca_ridge_estimator(data["dino"])),
        "resnet18_cosine_centroid_cv": cosine_centroid_cv(data["resnet"], y, labels),
        "resnet18_knn5_cv": cosine_knn_cv(data["resnet"], y),
        "resnet18_pca_ridge_cv": dense_cv(data["resnet"], y, pca_ridge_estimator(data["resnet"], 32)),
        "clip_siglip_prompt_scores_logreg_cv": dense_cv(
            data["prompt_scores"], y, logreg(), scale=True
        ),
        "gpt55_label_logreg_cv": dense_cv(data["gpt_one_hot"], y, logreg()),
        "gpt55_label_plus_visual_logreg_cv": dense_cv(
            np.hstack([data["gpt_one_hot"], data["visual"]]), y, logreg(), scale=True
        ),
        "gpt55_label_plus_visual_extra_trees_cv": dense_cv(
            np.hstack([data["gpt_one_hot"], data["visual"]]), y, extra_trees()
        ),
    }

    local_stack = np.hstack(
        [
            data["clip"],
            data["siglip"],
            data["dino"],
            data["visual"],
            data["ocr_numeric"],
            data["prompt_scores"],
        ]
    )
    predictions["full_local_stack_pca_ridge_cv"] = dense_cv(
        local_stack, y, pca_ridge_estimator(local_stack, 32)
    )
    predictions["full_local_stack_extra_trees_cv"] = dense_cv(local_stack, y, extra_trees())
    predictions["strong_local_majority_vote"] = majority_vote(
        [
            predictions["ocr_text_tfidf_svc_cv"],
            predictions["visual_extra_trees_cv"],
            predictions["clip_vit_b32_pca_ridge_cv"],
            predictions["siglip_b16_pca_ridge_cv"],
            predictions["dinov2_small_pca_ridge_cv"],
        ],
        labels,
    )
    predictions["gpt55_plus_strong_local_majority_vote"] = majority_vote(
        [
            data["gpt_labels"],
            predictions["ocr_text_tfidf_svc_cv"],
            predictions["visual_extra_trees_cv"],
            predictions["clip_vit_b32_pca_ridge_cv"],
            predictions["siglip_b16_pca_ridge_cv"],
            predictions["dinov2_small_pca_ridge_cv"],
        ],
        labels,
    )

    print(f"{task}: starting nested stacks", flush=True)
    dense_specs = [
        ("visual_extra_trees", data["visual"], extra_trees(6, 300)),
        (
            "ocr_numeric_visual_extra_trees",
            np.hstack([data["ocr_numeric"], data["visual"]]),
            extra_trees(6, 300),
        ),
        ("clip_pca_ridge", data["clip"], pca_ridge_estimator(data["clip"])),
        ("siglip_pca_ridge", data["siglip"], pca_ridge_estimator(data["siglip"])),
        ("dino_pca_ridge", data["dino"], pca_ridge_estimator(data["dino"])),
        (
            "prompt_scores_logreg",
            data["prompt_scores"],
            make_pipeline(StandardScaler(), logreg()),
        ),
    ]
    text_specs = [
        (
            "ocr_text_tfidf",
            data["ocr_texts"],
            make_pipeline(
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=3000,
                    sublinear_tf=True,
                ),
                LinearSVC(C=0.7, class_weight="balanced", random_state=RANDOM_STATE),
            ),
        )
    ]
    predictions["nested_local_prediction_stack_logreg_cv"] = nested_stack_cv(
        y, labels, dense_specs, text_specs, []
    )
    predictions["nested_gpt55_plus_local_prediction_stack_logreg_cv"] = nested_stack_cv(
        y, labels, dense_specs, text_specs, [data["gpt_labels"]]
    )
    predictions["nested_core_gpt55_ocr_text_siglip_cv"] = nested_stack_cv(
        y,
        labels,
        [("siglip_pca_ridge", data["siglip"], pca_ridge_estimator(data["siglip"]))],
        text_specs,
        [data["gpt_labels"]],
    )
    predictions["nested_pruned_gpt55_ocr_text_clip_siglip_cv"] = nested_stack_cv(
        y,
        labels,
        [
            ("clip_pca_ridge", data["clip"], pca_ridge_estimator(data["clip"])),
            ("siglip_pca_ridge", data["siglip"], pca_ridge_estimator(data["siglip"])),
        ],
        text_specs,
        [data["gpt_labels"]],
    )

    print(f"{task}: starting semantic classifiers", flush=True)
    semantic_only, semantic_baseline, semantic_baseline_visual = semantic_matrices(
        data["semantic"], data["gpt_labels"], data["visual"]
    )
    predictions.update(
        {
            "semantic_only_logreg_cv": dense_cv(semantic_only, y, logreg(), scale=True),
            "semantic_only_random_forest_cv": dense_cv(semantic_only, y, random_forest()),
            "semantic_only_extra_trees_cv": dense_cv(semantic_only, y, extra_trees()),
            "semantic_only_mlp_cv": dense_cv(
                semantic_only,
                y,
                MLPClassifier(
                    hidden_layer_sizes=(32,),
                    alpha=0.05,
                    max_iter=1500,
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                ),
                scale=True,
            ),
            "baseline_plus_semantic_logreg_cv": dense_cv(
                semantic_baseline, y, logreg(), scale=True
            ),
            "baseline_plus_semantic_random_forest_cv": dense_cv(
                semantic_baseline, y, random_forest()
            ),
            "baseline_plus_semantic_extra_trees_cv": dense_cv(
                semantic_baseline, y, extra_trees()
            ),
            "baseline_plus_semantic_mlp_cv": dense_cv(
                semantic_baseline,
                y,
                MLPClassifier(
                    hidden_layer_sizes=(32,),
                    alpha=0.05,
                    max_iter=1500,
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                ),
                scale=True,
            ),
            "baseline_plus_semantic_visual_extra_trees_cv": dense_cv(
                semantic_baseline_visual, y, extra_trees()
            ),
        }
    )
    if task == "multiclass":
        predictions["siglip_ordinal_ridge_cv"] = ordinal_ridge_cv(data["siglip"], y)
        predictions["full_local_stack_ordinal_ridge_cv"] = ordinal_ridge_cv(local_stack, y)
    return predictions


def safe_component_count(requested: int, rows: int, columns: int) -> int:
    return max(1, min(requested, rows - 1, columns))


def fold_pca(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    requested: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    test_scaled = scaler.transform(test_matrix)
    count = safe_component_count(requested, train_scaled.shape[0], train_scaled.shape[1])
    pca = PCA(n_components=count, whiten=True, random_state=RANDOM_STATE)
    return pca.fit_transform(train_scaled), pca.transform(test_scaled)


def fold_text_svd(
    train_texts: list[str],
    test_texts: list[str],
    requested: int,
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
    count = max(1, min(requested, train_tfidf.shape[0] - 1, train_tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=count, random_state=RANDOM_STATE)
    train_svd = svd.fit_transform(train_tfidf)
    test_svd = svd.transform(test_tfidf)
    scaler = StandardScaler()
    return scaler.fit_transform(train_svd), scaler.transform(test_svd)


def fold_scaled(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(train_matrix), scaler.transform(test_matrix)


def fusion_classifiers() -> dict[str, Any]:
    return {
        "logreg_c0.2": make_pipeline(StandardScaler(), logreg(0.2)),
        "logreg_c1": make_pipeline(StandardScaler(), logreg(1.0)),
        "ridge_a3": make_pipeline(
            StandardScaler(), RidgeClassifier(alpha=3.0, class_weight="balanced")
        ),
        "linear_svc_c0.2": make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.2, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "extra_trees": extra_trees(),
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


def fusion_feature_sets() -> dict[str, list[str]]:
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
        "pruned_components": ["clip", "siglip", "ocr_text", "gpt55"],
        "core_without_clip": ["siglip", "ocr_text", "gpt55"],
        "all_without_gpt55": [
            "clip",
            "siglip",
            "dino",
            "ocr_text",
            "visual",
            "ocr_numeric",
            "prompt_scores",
        ],
        "vision_components_only": ["clip", "siglip", "dino"],
    }


def pca_fusion_methods(y: np.ndarray, data: dict[str, Any]) -> dict[str, np.ndarray]:
    classifier_templates = fusion_classifiers()
    feature_sets = fusion_feature_sets()
    predictions = {
        f"pca_fusion::{config_name}::{feature_name}::{classifier_name}": np.empty(
            len(y), dtype=object
        )
        for config_name in COMPONENT_CONFIGS
        for feature_name in feature_sets
        for classifier_name in classifier_templates
    }

    for fold, (train_index, test_index) in enumerate(fixed_cv().split(np.zeros(len(y)), y), start=1):
        print(f"PCA fusion fold {fold}/{CV_SPLITS}", flush=True)
        for config_name, config in COMPONENT_CONFIGS.items():
            components: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for name in ("clip", "siglip", "dino"):
                components[name] = fold_pca(
                    data[name][train_index], data[name][test_index], config[name]
                )
            components["visual"] = fold_pca(
                data["visual"][train_index], data["visual"][test_index], config["visual"]
            )
            components["ocr_text"] = fold_text_svd(
                [data["ocr_texts"][index] for index in train_index],
                [data["ocr_texts"][index] for index in test_index],
                config["ocr_text"],
            )
            components["ocr_numeric"] = fold_scaled(
                data["ocr_numeric"][train_index], data["ocr_numeric"][test_index]
            )
            components["prompt_scores"] = fold_scaled(
                data["prompt_scores"][train_index], data["prompt_scores"][test_index]
            )
            components["gpt55"] = (
                data["gpt_one_hot"][train_index],
                data["gpt_one_hot"][test_index],
            )

            for feature_name, selected in feature_sets.items():
                train_matrix = np.hstack([components[name][0] for name in selected])
                test_matrix = np.hstack([components[name][1] for name in selected])
                for classifier_name, classifier in classifier_templates.items():
                    method = (
                        f"pca_fusion::{config_name}::{feature_name}::{classifier_name}"
                    )
                    model = clone(classifier)
                    model.fit(train_matrix, y[train_index])
                    predictions[method][test_index] = model.predict(test_matrix)
    return predictions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_view_grid(path: Path, resize_size: tuple[int, int] = (160, 90)) -> Image.Image:
    crops = [
        crop_viewport(path, index).resize(resize_size, Image.Resampling.BILINEAR)
        for index in range(VIEW_COUNT)
    ]
    width, height = resize_size
    grid = Image.new("RGB", (width * 2, height * 2), "white")
    for index, crop in enumerate(crops):
        grid.paste(crop, ((index % 2) * width, (index // 2) * height))
    return grid


class ScreenshotDataset(Dataset):
    def __init__(
        self,
        image_names: list[str],
        y_indices: np.ndarray,
        grid_cache: dict[str, Image.Image],
        train: bool,
    ) -> None:
        self.image_names = image_names
        self.y_indices = y_indices
        self.grid_cache = grid_cache
        augmentation: list[Any] = []
        if train:
            augmentation = [
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
            ]
        self.transform = transforms.Compose(
            [
                *augmentation,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return (
            self.transform(self.grid_cache[self.image_names[index]]),
            int(self.y_indices[index]),
        )


class SmallScreenshotCNN(nn.Module):
    def __init__(self, classes: int) -> None:
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
            nn.Linear(256, classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def scratch_cnn_cv(
    task: str,
    image_names: list[str],
    y: np.ndarray,
    labels: list[str],
    epochs: int = 15,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    cache_path = Path(f"factuality_{task}_scratch_cnn_predictions.csv")
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if cached["image"].tolist() == image_names:
            return cached["prediction"].to_numpy(dtype=object), []

    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    y_indices = np.asarray([label_to_index[label] for label in y], dtype=np.int64)
    grid_cache = {
        image_name: build_view_grid(image_path(image_name))
        for image_name in image_names
    }
    output = np.empty(len(y), dtype=object)
    fold_rows = []

    for fold, (train_index, test_index) in enumerate(fixed_cv().split(np.zeros(len(y)), y), start=1):
        set_seed(RANDOM_STATE + fold)
        train_loader = DataLoader(
            ScreenshotDataset(
                [image_names[index] for index in train_index],
                y_indices[train_index],
                grid_cache,
                train=True,
            ),
            batch_size=16,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            ScreenshotDataset(
                [image_names[index] for index in test_index],
                y_indices[test_index],
                grid_cache,
                train=False,
            ),
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )
        model = SmallScreenshotCNN(len(labels)).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        class_counts = np.bincount(y_indices[train_index], minlength=len(labels))
        class_weights = len(train_index) / (len(labels) * np.maximum(class_counts, 1))
        loss_function = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
        )
        for _ in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        fold_predictions = []
        model.eval()
        with torch.no_grad():
            for batch_x, _ in test_loader:
                indices = model(batch_x.to(DEVICE)).argmax(dim=1).cpu().numpy()
                fold_predictions.extend(index_to_label[int(index)] for index in indices)
        output[test_index] = fold_predictions
        fold_rows.append(
            {
                "task": task,
                "fold": str(fold),
                "accuracy": f"{accuracy_score(y[test_index], fold_predictions):.6f}",
                "macro_f1": f"{f1_score(y[test_index], fold_predictions, labels=labels, average='macro', zero_division=0):.6f}",
            }
        )
        print(f"scratch CNN {task} fold {fold}/{CV_SPLITS}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(
        cache_path,
        [
            {"image": image_name, "prediction": output[index]}
            for index, image_name in enumerate(image_names)
        ],
        ["image", "prediction"],
    )
    return output, fold_rows


def family_for_method(method: str) -> str:
    if method.startswith("pca_fusion::"):
        return "pca_component_fusion"
    if method.startswith("nested_"):
        return "nested_stacking"
    if "semantic" in method:
        return "semantic_features"
    if method.startswith("scratch_"):
        return "cnn_from_scratch"
    if method.startswith("baseline") or method.startswith("gpt55"):
        return "gpt_baseline_or_calibration"
    if method.startswith(("clip_", "siglip_", "dinov2_", "resnet18_")):
        return "vision_embeddings"
    if method.startswith("ocr_"):
        return "ocr"
    if method.startswith("visual_"):
        return "handcrafted_visual"
    if method.startswith(("full_local_", "strong_local_")):
        return "local_feature_fusion"
    return "control_or_ensemble"


def write_report(metric_rows: list[dict[str, str]]) -> None:
    lines = [
        "# Factuality Classification: All Methods",
        "",
        "Dataset: 200 landing-page screenshots.",
        "",
        "- Binary task: 100 low (`VERY LOW` + `LOW`) and 100 high (`HIGH` + `VERY HIGH`).",
        "- Multiclass task: 50 very low, 50 low, 89 high, and 11 very high.",
        "- All learned methods use stratified 5-fold CV.",
        "- PCA, SVD, base-model stacking, and final classifiers are fit inside their training folds.",
        "",
    ]
    for task in ("binary", "multiclass"):
        task_rows = sorted(
            [row for row in metric_rows if row["task"] == task],
            key=lambda row: float(row["macro_f1"]),
            reverse=True,
        )
        lines.extend(
            [
                f"## {task.title()} Results",
                "",
                "| Rank | Method | Family | Accuracy | Balanced accuracy | Macro-F1 | MSE | QWK |",
                "|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(task_rows[:25], start=1):
            lines.append(
                f"| {rank} | `{row['method']}` | {row['family']} | {row['accuracy']} | "
                f"{row['balanced_accuracy']} | {row['macro_f1']} | "
                f"{row['mse_steps2'] or '-'} | {row['quadratic_weighted_kappa'] or '-'} |"
            )
        lines.append("")

        best_by_family = {}
        for row in task_rows:
            best_by_family.setdefault(row["family"], row)
        lines.extend(["### Best By Family", ""])
        for family, row in sorted(best_by_family.items()):
            lines.append(
                f"- `{family}`: `{row['method']}`, accuracy `{row['accuracy']}`, "
                f"Macro-F1 `{row['macro_f1']}`."
            )
        lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(RANDOM_STATE)
    rows = load_dataset()
    all_metric_rows = []
    all_confusion_rows = []
    cnn_fold_rows = []

    for task in ("binary", "multiclass"):
        print(f"starting {task} experiments", flush=True)
        y, labels = labels_for_task(rows, task)
        data = load_features(rows, task, labels)
        predictions = classical_methods(task, y, labels, data)
        predictions.update(pca_fusion_methods(y, data))
        cnn_predictions, task_fold_rows = scratch_cnn_cv(
            task, data["image_names"], y, labels
        )
        predictions["scratch_small_cnn_cv"] = cnn_predictions
        cnn_fold_rows.extend(task_fold_rows)

        prediction_rows = []
        for index, image_name in enumerate(data["image_names"]):
            prediction_rows.append(
                {
                    "image": image_name,
                    "true_label": y[index],
                    **{
                        method: method_predictions[index]
                        for method, method_predictions in predictions.items()
                    },
                }
            )
        write_csv(
            Path(f"factuality_{task}_all_methods_predictions.csv"),
            prediction_rows,
            list(prediction_rows[0].keys()),
        )

        for method, method_predictions in predictions.items():
            metrics = metric_row(task, method, y, method_predictions, labels)
            metrics["family"] = family_for_method(method)
            all_metric_rows.append(metrics)
            all_confusion_rows.extend(
                confusion_rows(task, method, y, method_predictions, labels)
            )
        print(f"completed {task} experiments: {len(predictions)} methods", flush=True)

    metric_fields = [
        "task",
        "method",
        "family",
        "sample_size",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "mae_steps",
        "mse_steps2",
        "rmse_steps",
        "within_1_accuracy",
        "severe_error_rate_ge_2",
        "quadratic_weighted_kappa",
    ]
    all_metric_rows.sort(
        key=lambda row: (row["task"], -float(row["macro_f1"]))
    )
    write_csv(METRICS_PATH, all_metric_rows, metric_fields)
    write_csv(
        CONFUSIONS_PATH,
        all_confusion_rows,
        ["task", "method", "true_label", "predicted_label", "count"],
    )
    if cnn_fold_rows:
        write_csv(
            CNN_FOLD_METRICS_PATH,
            cnn_fold_rows,
            ["task", "fold", "accuracy", "macro_f1"],
        )
    write_report(all_metric_rows)
    print(f"wrote {METRICS_PATH}", flush=True)
    print(f"wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
