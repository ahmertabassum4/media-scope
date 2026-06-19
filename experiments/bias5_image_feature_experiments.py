from experiment_bootstrap import activate_project_root

activate_project_root()

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from image_process import BIAS_5_LABELS, IMAGE_FOLDER, write_csv


Image.MAX_IMAGE_PIXELS = None

RANDOM_STATE = 67
CV_SPLITS = 5
VIEW_COUNT = 4
OCR_VIEW_COUNT = 2
TARGET_ASPECT_RATIO = 16 / 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_PATH = Path("bias5_experiment_sample.csv")
BASELINE_PREDICTIONS_PATH = Path("bias_predictions.csv")
HANDCRAFTED_VISUAL_PATH = Path("bias5_ml_visual_features.csv")

METRICS_PATH = Path("bias5_image_feature_metrics.csv")
PREDICTIONS_PATH = Path("bias5_image_feature_predictions.csv")
CONFUSIONS_PATH = Path("bias5_image_feature_confusions.csv")
REPORT_PATH = Path("bias5_image_feature_report.md")
OCR_RAW_PATH = Path("bias5_image_feature_ocr_raw.jsonl")
OCR_FEATURES_PATH = Path("bias5_image_feature_ocr_features.csv")

GPT55_COLUMN = "bias_5:openai/gpt-5.5"

HF_IMAGE_MODELS = [
    {
        "name": "clip_vit_b32",
        "model_id": "openai/clip-vit-base-patch32",
        "kind": "clip_like",
    },
    {
        "name": "siglip_b16",
        "model_id": "google/siglip-base-patch16-224",
        "kind": "clip_like",
    },
    {
        "name": "dinov2_small",
        "model_id": "facebook/dinov2-small",
        "kind": "vision_encoder",
    },
]

ZERO_SHOT_PROMPTS = {
    "left": [
        "a screenshot of a strongly left biased political news website",
        "a liberal advocacy news homepage with left partisan framing",
    ],
    "left-center": [
        "a screenshot of a mildly center-left news website",
        "a news homepage with mild liberal framing but not extreme bias",
    ],
    "least biased": [
        "a screenshot of a neutral least biased news website",
        "a factual nonpartisan news or research homepage",
    ],
    "right-center": [
        "a screenshot of a mildly center-right news website",
        "a news homepage with mild conservative framing but not extreme bias",
    ],
    "right": [
        "a screenshot of a strongly right biased political news website",
        "a conservative advocacy news homepage with right partisan framing",
    ],
}

LEFT_TERMS = [
    "progressive",
    "liberal",
    "democrat",
    "democrats",
    "equity",
    "social justice",
    "climate justice",
    "workers",
    "union",
    "labor",
    "racism",
    "inequality",
    "lgbtq",
    "reproductive",
    "abortion rights",
    "medicare for all",
    "green new deal",
]

RIGHT_TERMS = [
    "conservative",
    "republican",
    "republicans",
    "patriot",
    "liberty",
    "freedom",
    "border",
    "woke",
    "pro-life",
    "second amendment",
    "faith",
    "values",
    "taxpayer",
    "america first",
    "illegal immigration",
    "radical left",
]

NEUTRAL_TERMS = [
    "fact check",
    "nonpartisan",
    "research",
    "data",
    "analysis",
    "explainer",
    "corrections",
    "methodology",
    "local",
    "weather",
    "sports",
    "business",
]

SENSATIONAL_TERMS = [
    "breaking",
    "shocking",
    "exposed",
    "secret",
    "crisis",
    "scandal",
    "destroy",
    "betrayal",
    "outrage",
    "threat",
]

CTA_TERMS = [
    "donate",
    "join",
    "subscribe",
    "petition",
    "sign up",
    "support us",
    "take action",
]


def load_sample_rows() -> list[dict[str, str]]:
    with SAMPLE_PATH.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def load_baseline_predictions() -> dict[str, dict[str, str]]:
    with BASELINE_PREDICTIONS_PATH.open("r", encoding="utf-8", newline="") as input_file:
        return {row["image"]: row for row in csv.DictReader(input_file)}


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


def image_views(path: Path, max_width: int | None = None) -> list[Image.Image]:
    return [crop_viewport(path, index, max_width=max_width) for index in range(VIEW_COUNT)]


def labels_for_rows(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([row["true_bias_5"] for row in rows], dtype=object)


def clean_matrix(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def pca_components(x: np.ndarray) -> int:
    train_size = math.floor(x.shape[0] * (CV_SPLITS - 1) / CV_SPLITS)
    return max(2, min(24, train_size - 1, x.shape[1]))


def ridge_classifier() -> RidgeClassifier:
    return RidgeClassifier(alpha=3.0, class_weight="balanced")


def extra_trees_classifier() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def fixed_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def cv_predict_dense(x: np.ndarray, y: np.ndarray, estimator: Any, scale: bool = True) -> np.ndarray:
    model = make_pipeline(StandardScaler(), estimator) if scale else estimator
    return cross_val_predict(model, clean_matrix(x), y, cv=fixed_cv())


def cv_predict_pca_ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = clean_matrix(x)
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=pca_components(x), random_state=RANDOM_STATE),
        ridge_classifier(),
    )
    return cross_val_predict(model, x, y, cv=fixed_cv())


def cosine_centroid_cv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = normalize_rows(clean_matrix(x))
    output = np.empty(len(y), dtype=object)

    for train_index, test_index in fixed_cv().split(x, y):
        centroids = []
        for label in BIAS_5_LABELS:
            class_rows = x[train_index][y[train_index] == label]
            centroid = class_rows.mean(axis=0)
            centroids.append(centroid / max(np.linalg.norm(centroid), 1e-8))
        similarities = x[test_index] @ np.vstack(centroids).T
        output[test_index] = [BIAS_5_LABELS[index] for index in similarities.argmax(axis=1)]

    return output


def cosine_knn_cv(x: np.ndarray, y: np.ndarray, k: int = 5) -> np.ndarray:
    model = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    return cross_val_predict(model, clean_matrix(x), y, cv=fixed_cv())


def evaluate(method: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, str]:
    return {
        "method": method,
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


def encode_prediction_columns(prediction_sets: dict[str, np.ndarray], method_names: list[str]) -> np.ndarray:
    dict_rows = []
    for row_index in range(len(next(iter(prediction_sets.values())))):
        dict_rows.append({method: prediction_sets[method][row_index] for method in method_names})
    return DictVectorizer(sparse=False).fit_transform(dict_rows)


def majority_vote(predictions: list[np.ndarray]) -> np.ndarray:
    output = []
    center_first_order = ["least biased", "left-center", "right-center", "left", "right"]
    for values in zip(*predictions):
        counts = Counter(values)
        best_score = max(counts.values())
        tied = {label for label, score in counts.items() if score == best_score}
        for label in center_first_order:
            if label in tied:
                output.append(label)
                break
    return np.asarray(output, dtype=object)


def cache_path_for_model(name: str) -> Path:
    return Path(f"bias5_image_feature_{name}_embeddings.npz")


def pool_view_features(view_features: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            view_features[0],
            view_features.mean(axis=0),
            view_features.std(axis=0),
        ]
    )


def tensor_from_model_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    for attribute in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attribute, None)
        if torch.is_tensor(value):
            return value
    value = getattr(output, "last_hidden_state", None)
    if torch.is_tensor(value):
        return value[:, 0]
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Cannot extract tensor features from {type(output).__name__}")


def ensure_hf_image_embeddings(rows: list[dict[str, str]], spec: dict[str, str]) -> np.ndarray:
    cache_path = cache_path_for_model(spec["name"])
    image_names = [row["image"] for row in rows]
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if cached["image_names"].tolist() == image_names:
            return cached["embeddings"]

    from transformers import AutoImageProcessor, AutoModel, AutoProcessor

    if spec["kind"] == "vision_encoder":
        processor = AutoImageProcessor.from_pretrained(spec["model_id"])
    else:
        processor = AutoProcessor.from_pretrained(spec["model_id"])
    model = AutoModel.from_pretrained(spec["model_id"]).to(DEVICE)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            views = image_views(image_path(row["image"]))
            inputs = processor(images=views, return_tensors="pt")
            tensor_inputs = {key: value.to(DEVICE) for key, value in inputs.items() if torch.is_tensor(value)}

            if hasattr(model, "get_image_features"):
                features = tensor_from_model_output(model.get_image_features(**tensor_inputs))
            else:
                output = model(**tensor_inputs)
                features = tensor_from_model_output(output)

            view_features = features.detach().float().cpu().numpy()
            embeddings.append(pool_view_features(view_features).astype(np.float32))
            print(f"{spec['name']} embeddings {index}/{len(rows)}: {row['image']}", flush=True)

    embedding_matrix = np.vstack(embeddings)
    np.savez_compressed(cache_path, image_names=np.asarray(image_names), embeddings=embedding_matrix)
    return embedding_matrix


def ensure_clip_zero_shot(rows: list[dict[str, str]], spec: dict[str, str]) -> np.ndarray:
    cache_path = Path(f"bias5_image_feature_{spec['name']}_zeroshot.csv")
    image_names = [row["image"] for row in rows]
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if cached["image"].tolist() == image_names:
            return cached["prediction"].to_numpy(dtype=object)

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(spec["model_id"])
    model = AutoModel.from_pretrained(spec["model_id"]).to(DEVICE)
    model.eval()
    if not hasattr(model, "get_image_features") or not hasattr(model, "get_text_features"):
        raise RuntimeError(f"{spec['model_id']} does not expose CLIP-style text/image features")

    prompt_texts = []
    prompt_labels = []
    for label, prompts in ZERO_SHOT_PROMPTS.items():
        prompt_texts.extend(prompts)
        prompt_labels.extend([label] * len(prompts))

    text_inputs = processor(text=prompt_texts, padding=True, return_tensors="pt")
    text_inputs = {key: value.to(DEVICE) for key, value in text_inputs.items() if torch.is_tensor(value)}

    with torch.no_grad():
        text_features = tensor_from_model_output(model.get_text_features(**text_inputs))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    predictions = []
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            views = image_views(image_path(row["image"]))
            image_inputs = processor(images=views, return_tensors="pt")
            image_inputs = {key: value.to(DEVICE) for key, value in image_inputs.items() if torch.is_tensor(value)}
            image_features = tensor_from_model_output(model.get_image_features(**image_inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            similarities = image_features @ text_features.T
            label_scores = {}
            for label in BIAS_5_LABELS:
                prompt_indices = [i for i, prompt_label in enumerate(prompt_labels) if prompt_label == label]
                label_scores[label] = similarities[:, prompt_indices].mean().item()
            predictions.append(max(label_scores, key=label_scores.get))
            print(f"{spec['name']} zero-shot {index}/{len(rows)}: {row['image']}", flush=True)

    df = pd.DataFrame({"image": image_names, "prediction": predictions})
    df.to_csv(cache_path, index=False)
    return np.asarray(predictions, dtype=object)


def load_ocr_raw() -> dict[str, dict[str, Any]]:
    if not OCR_RAW_PATH.exists():
        return {}
    rows = {}
    with OCR_RAW_PATH.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                row = json.loads(line)
                rows[row["image"]] = row
    return rows


def count_terms(text: str, terms: list[str]) -> int:
    lower = text.lower()
    return sum(lower.count(term) for term in terms)


def summarize_ocr(image_name: str, results_by_view: list[list[Any]]) -> dict[str, Any]:
    pieces = []
    confidences = []
    boxes = 0
    for results in results_by_view:
        boxes += len(results)
        for item in results:
            text = item[1] if len(item) > 1 else ""
            confidence = float(item[2]) if len(item) > 2 else 0.0
            if text:
                pieces.append(text)
                confidences.append(confidence)

    text = " ".join(pieces)
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", text)
    upper_words = [word for word in words if len(word) > 2 and word.isupper()]
    word_count = len(words)

    return {
        "image": image_name,
        "ocr_text": text,
        "ocr_box_count": boxes,
        "ocr_word_count": word_count,
        "ocr_unique_word_count": len({word.lower() for word in words}),
        "ocr_avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "ocr_upper_word_fraction": len(upper_words) / max(1, word_count),
        "ocr_exclamation_count": text.count("!"),
        "ocr_question_count": text.count("?"),
        "ocr_digit_count": len(re.findall(r"\d", text)),
        "ocr_left_term_count": count_terms(text, LEFT_TERMS),
        "ocr_right_term_count": count_terms(text, RIGHT_TERMS),
        "ocr_neutral_term_count": count_terms(text, NEUTRAL_TERMS),
        "ocr_sensational_term_count": count_terms(text, SENSATIONAL_TERMS),
        "ocr_cta_term_count": count_terms(text, CTA_TERMS),
        "ocr_left_minus_right": count_terms(text, LEFT_TERMS) - count_terms(text, RIGHT_TERMS),
    }


def ensure_ocr_features(rows: list[dict[str, str]]) -> pd.DataFrame:
    image_names = [row["image"] for row in rows]
    if OCR_FEATURES_PATH.exists():
        cached = pd.read_csv(OCR_FEATURES_PATH)
        if cached["image"].tolist() == image_names:
            return cached

    import easyocr

    raw_rows = load_ocr_raw()
    missing = [row for row in rows if row["image"] not in raw_rows]
    if missing:
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        with OCR_RAW_PATH.open("a", encoding="utf-8") as output_file:
            for index, row in enumerate(missing, start=1):
                path = image_path(row["image"])
                results_by_view = []
                for view_index in range(OCR_VIEW_COUNT):
                    crop = crop_viewport(path, view_index, max_width=1280)
                    results = reader.readtext(
                        np.asarray(crop),
                        detail=1,
                        paragraph=False,
                        text_threshold=0.6,
                        low_text=0.35,
                    )
                    results_by_view.append(results)
                summary = summarize_ocr(row["image"], results_by_view)
                output_file.write(json.dumps(summary, ensure_ascii=False) + "\n")
                output_file.flush()
                raw_rows[row["image"]] = summary
                print(f"easyocr {index}/{len(missing)}: {row['image']}", flush=True)

    feature_df = pd.DataFrame([raw_rows[name] for name in image_names])
    feature_df.to_csv(OCR_FEATURES_PATH, index=False)
    return feature_df


def ocr_numeric_matrix(ocr_df: pd.DataFrame) -> np.ndarray:
    return ocr_df.drop(columns=["image", "ocr_text"], errors="ignore").to_numpy(dtype=np.float32)


def cv_predict_ocr_tfidf(texts: list[str], y: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_features=1500,
        ),
        LinearSVC(C=0.7, class_weight="balanced", random_state=RANDOM_STATE),
    )
    return cross_val_predict(model, texts, y, cv=fixed_cv())


def handcrafted_visual_matrix(rows: list[dict[str, str]]) -> np.ndarray | None:
    if not HANDCRAFTED_VISUAL_PATH.exists():
        return None
    df = pd.read_csv(HANDCRAFTED_VISUAL_PATH).set_index("image")
    ordered = df.loc[[row["image"] for row in rows]]
    return ordered.to_numpy(dtype=np.float32)


def add_dense_methods(
    prediction_sets: dict[str, np.ndarray],
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    include_tree: bool = True,
) -> None:
    prediction_sets[f"{name}_centroid_cv"] = cosine_centroid_cv(x, y)
    prediction_sets[f"{name}_knn5_cv"] = cosine_knn_cv(x, y, k=5)
    prediction_sets[f"{name}_pca_ridge_cv"] = cv_predict_pca_ridge(x, y)
    if include_tree:
        prediction_sets[f"{name}_extra_trees_cv"] = cv_predict_dense(x, y, extra_trees_classifier(), scale=False)


def write_predictions(rows: list[dict[str, str]], y_true: np.ndarray, prediction_sets: dict[str, np.ndarray]) -> None:
    output_rows = []
    for index, row in enumerate(rows):
        output = {"image": row["image"], "true_bias_5": y_true[index]}
        for method, predictions in prediction_sets.items():
            output[method] = predictions[index]
        output_rows.append(output)
    fieldnames = ["image", "true_bias_5", *prediction_sets.keys()]
    write_csv(PREDICTIONS_PATH, output_rows, fieldnames)


def write_report(metric_rows: list[dict[str, str]], skipped: list[str]) -> None:
    sorted_metrics = sorted(metric_rows, key=lambda row: float(row["macro_f1"]), reverse=True)
    best = sorted_metrics[0]
    lines = [
        "# Bias 5 Image Feature Experiment Report",
        "",
        f"Device: `{DEVICE}`.",
        "",
        "Dataset: balanced exploratory 50-site subset from `bias5_experiment_sample.csv`.",
        "",
        "This run avoids new paid LLM calls. It tests image-derived OCR, local vision embeddings, and ensembles.",
        "",
        "## Best Method",
        "",
        f"`{best['method']}`: accuracy `{best['accuracy']}`, macro-F1 `{best['macro_f1']}`.",
        "",
        "## Methods",
        "",
        "- OCR: EasyOCR on the first two 16:9 viewport crops, then TF-IDF and numeric lexicon/count features.",
        "- CLIP/SigLIP/DINOv2: first four 16:9 viewport crops are embedded locally, then pooled as first-view + mean + std vectors.",
        "- Ensembles: majority voting and small CV classifiers over local extracted-feature predictions.",
        "- Cached GPT-5.5 labels are included only as a reference and one hybrid vote; no new LLM requests are made.",
        "",
    ]
    if skipped:
        lines.extend(["## Skipped Blocks", ""])
        lines.extend([f"- {item}" for item in skipped])
        lines.append("")
    lines.extend(["## Metrics", "", "```csv"])
    lines.append(",".join(["method", "sample_size", "accuracy", "macro_f1", "weighted_f1"]))
    for row in sorted_metrics:
        lines.append(",".join([row["method"], row["sample_size"], row["accuracy"], row["macro_f1"], row["weighted_f1"]]))
    lines.extend(["```", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = load_sample_rows()
    y = labels_for_rows(rows)
    baseline_rows = load_baseline_predictions()
    prediction_sets: dict[str, np.ndarray] = {}
    skipped: list[str] = []

    prediction_sets["baseline_gpt5.5_cached"] = np.asarray(
        [baseline_rows[row["image"]][GPT55_COLUMN] for row in rows],
        dtype=object,
    )

    visual_x = handcrafted_visual_matrix(rows)
    if visual_x is not None:
        add_dense_methods(prediction_sets, "handcrafted_visual", visual_x, y, include_tree=True)

    try:
        ocr_df = ensure_ocr_features(rows)
        ocr_texts = ocr_df["ocr_text"].fillna("").astype(str).tolist()
        ocr_x = ocr_numeric_matrix(ocr_df)
        prediction_sets["ocr_text_tfidf_svc_cv"] = cv_predict_ocr_tfidf(ocr_texts, y)
        add_dense_methods(prediction_sets, "ocr_numeric", ocr_x, y, include_tree=True)
        if visual_x is not None:
            add_dense_methods(prediction_sets, "ocr_numeric_plus_visual", np.hstack([ocr_x, visual_x]), y, include_tree=True)
    except Exception as exc:
        skipped.append(f"OCR block: {type(exc).__name__}: {exc}")

    available_embedding_names = []
    for spec in HF_IMAGE_MODELS:
        try:
            embeddings = ensure_hf_image_embeddings(rows, spec)
            add_dense_methods(prediction_sets, spec["name"], embeddings, y, include_tree=False)
            available_embedding_names.append(spec["name"])

            if spec["kind"] == "clip_like":
                try:
                    prediction_sets[f"{spec['name']}_zeroshot"] = ensure_clip_zero_shot(rows, spec)
                except Exception as exc:
                    skipped.append(f"{spec['name']} zero-shot: {type(exc).__name__}: {exc}")
        except Exception as exc:
            skipped.append(f"{spec['name']} embeddings: {type(exc).__name__}: {exc}")

    dense_stack_parts = []
    for name in available_embedding_names:
        cache = np.load(cache_path_for_model(name), allow_pickle=True)
        dense_stack_parts.append(cache["embeddings"])
    if visual_x is not None:
        dense_stack_parts.append(visual_x)

    if len(dense_stack_parts) >= 2:
        stacked_x = np.hstack([clean_matrix(part) for part in dense_stack_parts])
        prediction_sets["local_embedding_visual_stack_pca_ridge_cv"] = cv_predict_pca_ridge(stacked_x, y)
        prediction_sets["local_embedding_visual_stack_extra_trees_cv"] = cv_predict_dense(
            stacked_x,
            y,
            extra_trees_classifier(),
            scale=False,
        )

    local_vote_candidates = [
        method
        for method in prediction_sets
        if method.endswith("_pca_ridge_cv") and not method.startswith("baseline")
    ]
    if len(local_vote_candidates) >= 3:
        prediction_sets["local_pca_ridge_majority_vote"] = majority_vote(
            [prediction_sets[method] for method in local_vote_candidates]
        )

    strong_local_candidates = [
        method
        for method in [
            "ocr_text_tfidf_svc_cv",
            "handcrafted_visual_extra_trees_cv",
            "clip_vit_b32_pca_ridge_cv",
            "siglip_b16_pca_ridge_cv",
            "dinov2_small_pca_ridge_cv",
        ]
        if method in prediction_sets
    ]
    if len(strong_local_candidates) >= 2:
        prediction_sets["strong_local_majority_vote"] = majority_vote(
            [prediction_sets[method] for method in strong_local_candidates]
        )
        hybrid_inputs = [prediction_sets["baseline_gpt5.5_cached"], *[prediction_sets[method] for method in strong_local_candidates]]
        prediction_sets["gpt55_plus_strong_local_majority_vote"] = majority_vote(hybrid_inputs)

        meta_x = encode_prediction_columns(prediction_sets, strong_local_candidates)
        prediction_sets["strong_local_prediction_stack_logreg_cv"] = cv_predict_dense(
            meta_x,
            y,
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.8, random_state=RANDOM_STATE),
            scale=False,
        )

    metric_rows = [evaluate(method, y, predictions) for method, predictions in prediction_sets.items()]
    write_csv(METRICS_PATH, metric_rows, ["method", "sample_size", "accuracy", "macro_f1", "weighted_f1"])

    all_confusions = []
    for method, predictions in prediction_sets.items():
        all_confusions.extend(confusion_rows(method, y, predictions))
    write_csv(CONFUSIONS_PATH, all_confusions, ["method", "true_label", "predicted_label", "count"])

    write_predictions(rows, y, prediction_sets)
    write_report(metric_rows, skipped)

    print(f"wrote {METRICS_PATH}", flush=True)
    print(f"wrote {PREDICTIONS_PATH}", flush=True)
    print(f"wrote {CONFUSIONS_PATH}", flush=True)
    print(f"wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
