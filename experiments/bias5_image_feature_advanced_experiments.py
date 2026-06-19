from experiment_bootstrap import activate_project_root

activate_project_root()

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from image_process import BIAS_5_LABELS, write_csv
from bias5_image_feature_experiments import (
    GPT55_COLUMN,
    HF_IMAGE_MODELS,
    PREDICTIONS_PATH as BASE_IMAGE_FEATURE_PREDICTIONS_PATH,
    RANDOM_STATE,
    ZERO_SHOT_PROMPTS,
    cache_path_for_model,
    clean_matrix,
    ensure_ocr_features,
    fixed_cv,
    handcrafted_visual_matrix,
    labels_for_rows,
    load_baseline_predictions,
    load_sample_rows,
    ocr_numeric_matrix,
    pca_components,
    tensor_from_model_output,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

METRICS_PATH = Path("bias5_image_feature_advanced_metrics.csv")
PREDICTIONS_PATH = Path("bias5_image_feature_advanced_predictions.csv")
CONFUSIONS_PATH = Path("bias5_image_feature_advanced_confusions.csv")
REPORT_PATH = Path("bias5_image_feature_advanced_report.md")

LABEL_TO_ORDINAL = {
    "left": -2.0,
    "left-center": -1.0,
    "least biased": 0.0,
    "right-center": 1.0,
    "right": 2.0,
}
ORDINAL_TO_LABEL = {
    -2: "left",
    -1: "left-center",
    0: "least biased",
    1: "right-center",
    2: "right",
}


def embedding_matrix(name: str) -> np.ndarray:
    cache = np.load(cache_path_for_model(name), allow_pickle=True)
    return clean_matrix(cache["embeddings"])


def prompt_score_path(name: str) -> Path:
    return Path(f"bias5_image_feature_{name}_prompt_scores.csv")


def mean_embedding_from_pooled(x: np.ndarray) -> np.ndarray:
    segment = x.shape[1] // 3
    return x[:, segment : segment * 2]


def ensure_prompt_scores(rows: list[dict[str, str]], spec: dict[str, str]) -> pd.DataFrame:
    output_path = prompt_score_path(spec["name"])
    image_names = [row["image"] for row in rows]
    if output_path.exists():
        cached = pd.read_csv(output_path)
        if cached["image"].tolist() == image_names:
            return cached

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(spec["model_id"])
    model = AutoModel.from_pretrained(spec["model_id"]).to(DEVICE)
    model.eval()
    if not hasattr(model, "get_text_features"):
        raise RuntimeError(f"{spec['model_id']} has no text feature head")

    prompt_texts = []
    prompt_labels = []
    for label in BIAS_5_LABELS:
        for prompt in ZERO_SHOT_PROMPTS[label]:
            prompt_labels.append(label)
            prompt_texts.append(prompt)

    text_inputs = processor(text=prompt_texts, padding=True, return_tensors="pt")
    text_inputs = {key: value.to(DEVICE) for key, value in text_inputs.items() if torch.is_tensor(value)}
    with torch.no_grad():
        text_features = tensor_from_model_output(model.get_text_features(**text_inputs)).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    image_features = torch.tensor(mean_embedding_from_pooled(embedding_matrix(spec["name"])), device=DEVICE)
    if image_features.shape[1] != text_features.shape[1]:
        raise RuntimeError(
            f"Dimension mismatch for {spec['name']}: image {image_features.shape[1]}, text {text_features.shape[1]}"
        )
    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scores = (image_features @ text_features.T).detach().cpu().numpy()

    output_rows = []
    for row_index, image_name in enumerate(image_names):
        output: dict[str, float | str] = {"image": image_name}
        label_means = {}
        for label in BIAS_5_LABELS:
            indices = [i for i, prompt_label in enumerate(prompt_labels) if prompt_label == label]
            label_scores = scores[row_index, indices]
            output[f"{label}_prompt_mean"] = float(label_scores.mean())
            output[f"{label}_prompt_max"] = float(label_scores.max())
            label_means[label] = float(label_scores.mean())
        sorted_scores = sorted(label_means.values(), reverse=True)
        output["prompt_top2_margin"] = sorted_scores[0] - sorted_scores[1]
        output["prompt_left_right_axis"] = (
            label_means["left"]
            + 0.5 * label_means["left-center"]
            - 0.5 * label_means["right-center"]
            - label_means["right"]
        )
        output["prompt_center_advantage"] = label_means["least biased"] - max(
            label_means["left-center"],
            label_means["right-center"],
        )
        output_rows.append(output)

    score_df = pd.DataFrame(output_rows)
    score_df.to_csv(output_path, index=False)
    return score_df


def prompt_score_matrix(score_df: pd.DataFrame) -> np.ndarray:
    return score_df.drop(columns=["image"], errors="ignore").to_numpy(dtype=np.float32)


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


def dense_logreg_cv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.7, random_state=RANDOM_STATE),
    )
    return cross_val_predict(model, clean_matrix(x), y, cv=fixed_cv())


def dense_extra_trees_cv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return cross_val_predict(model, clean_matrix(x), y, cv=fixed_cv())


def dense_pca_ridge_cv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = clean_matrix(x)
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=pca_components(x), random_state=RANDOM_STATE),
        RidgeClassifier(alpha=3.0, class_weight="balanced"),
    )
    return cross_val_predict(model, x, y, cv=fixed_cv())


def ordinal_ridge_cv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = clean_matrix(x)
    y_num = np.asarray([LABEL_TO_ORDINAL[label] for label in y], dtype=np.float32)
    output = np.empty(len(y), dtype=object)

    for train_index, test_index in fixed_cv().split(x, y):
        components = max(2, min(24, len(train_index) - 1, x.shape[1]))
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=components, random_state=RANDOM_STATE),
            Ridge(alpha=4.0),
        )
        model.fit(x[train_index], y_num[train_index])
        predicted = np.rint(model.predict(x[test_index])).astype(int)
        predicted = np.clip(predicted, -2, 2)
        output[test_index] = [ORDINAL_TO_LABEL[int(value)] for value in predicted]

    return output


def train_predict_classifier(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    unique = np.unique(y_train)
    if len(unique) == 1:
        return np.repeat(unique[0], len(x_test))
    steps: list[Any] = [StandardScaler()]
    if x_train.shape[1] > 20 and len(y_train) > 8:
        steps.append(PCA(n_components=max(2, min(12, len(y_train) - 1, x_train.shape[1])), random_state=RANDOM_STATE))
    steps.append(RidgeClassifier(alpha=2.0, class_weight="balanced"))
    model = make_pipeline(*steps)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def side_label(label: str) -> str:
    if label in {"left", "left-center"}:
        return "left_side"
    if label in {"right", "right-center"}:
        return "right_side"
    return "center"


def hierarchical_side_intensity_cv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = clean_matrix(x)
    output = np.empty(len(y), dtype=object)
    side_y = np.asarray([side_label(label) for label in y], dtype=object)

    for train_index, test_index in fixed_cv().split(x, y):
        predicted_side = train_predict_classifier(x[train_index], side_y[train_index], x[test_index])

        left_train = train_index[np.isin(y[train_index], ["left", "left-center"])]
        right_train = train_index[np.isin(y[train_index], ["right", "right-center"])]
        left_test_offsets = np.where(predicted_side == "left_side")[0]
        right_test_offsets = np.where(predicted_side == "right_side")[0]
        center_test_offsets = np.where(predicted_side == "center")[0]

        output[test_index[center_test_offsets]] = "least biased"
        if len(left_test_offsets):
            output[test_index[left_test_offsets]] = train_predict_classifier(
                x[left_train],
                y[left_train],
                x[test_index[left_test_offsets]],
            )
        if len(right_test_offsets):
            output[test_index[right_test_offsets]] = train_predict_classifier(
                x[right_train],
                y[right_train],
                x[test_index[right_test_offsets]],
            )

    return output


def prediction_one_hot(predictions: list[np.ndarray]) -> np.ndarray:
    rows = []
    for row_index in range(len(predictions[0])):
        row_features = []
        for pred in predictions:
            row_features.extend([1.0 if pred[row_index] == label else 0.0 for label in BIAS_5_LABELS])
        rows.append(row_features)
    return np.asarray(rows, dtype=np.float32)


def nested_prediction_stack_cv(
    y: np.ndarray,
    dense_specs: list[tuple[str, np.ndarray, Any]],
    text_specs: list[tuple[str, list[str], Any]],
    static_predictions: list[np.ndarray] | None = None,
) -> np.ndarray:
    output = np.empty(len(y), dtype=object)
    static_predictions = static_predictions or []

    for fold_index, (train_index, test_index) in enumerate(fixed_cv().split(np.zeros(len(y)), y), start=1):
        train_meta_predictions: list[np.ndarray] = []
        test_meta_predictions: list[np.ndarray] = []
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + fold_index)

        for _, x, estimator in dense_specs:
            x = clean_matrix(x)
            train_pred = cross_val_predict(clone(estimator), x[train_index], y[train_index], cv=inner_cv)
            model = clone(estimator)
            model.fit(x[train_index], y[train_index])
            test_pred = model.predict(x[test_index])
            train_meta_predictions.append(train_pred)
            test_meta_predictions.append(test_pred)

        for _, texts, estimator in text_specs:
            train_texts = [texts[index] for index in train_index]
            test_texts = [texts[index] for index in test_index]
            train_pred = cross_val_predict(clone(estimator), train_texts, y[train_index], cv=inner_cv)
            model = clone(estimator)
            model.fit(train_texts, y[train_index])
            test_pred = model.predict(test_texts)
            train_meta_predictions.append(train_pred)
            test_meta_predictions.append(test_pred)

        for static_pred in static_predictions:
            train_meta_predictions.append(static_pred[train_index])
            test_meta_predictions.append(static_pred[test_index])

        meta_train_x = prediction_one_hot(train_meta_predictions)
        meta_test_x = prediction_one_hot(test_meta_predictions)
        meta_model = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.8, random_state=RANDOM_STATE)
        meta_model.fit(meta_train_x, y[train_index])
        output[test_index] = meta_model.predict(meta_test_x)

    return output


def pca_ridge_estimator(x: np.ndarray) -> Any:
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=pca_components(x), random_state=RANDOM_STATE),
        RidgeClassifier(alpha=3.0, class_weight="balanced"),
    )


def extra_trees_estimator() -> Any:
    return ExtraTreesClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def ocr_text_estimator() -> Any:
    return make_pipeline(
        TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), max_features=1500),
        LinearSVC(C=0.7, class_weight="balanced", random_state=RANDOM_STATE),
    )


def write_predictions(rows: list[dict[str, str]], y_true: np.ndarray, prediction_sets: dict[str, np.ndarray]) -> None:
    output_rows = []
    for index, row in enumerate(rows):
        output = {"image": row["image"], "true_bias_5": y_true[index]}
        for method, predictions in prediction_sets.items():
            output[method] = predictions[index]
        output_rows.append(output)
    write_csv(PREDICTIONS_PATH, output_rows, ["image", "true_bias_5", *prediction_sets.keys()])


def write_report(metric_rows: list[dict[str, str]]) -> None:
    sorted_metrics = sorted(metric_rows, key=lambda row: float(row["macro_f1"]), reverse=True)
    best_overall = sorted_metrics[0]
    best_new = next(row for row in sorted_metrics if not row["method"].startswith("reference_non_nested_"))
    lines = [
        "# Bias 5 Advanced Image Feature Experiment Report",
        "",
        f"Device: `{DEVICE}`.",
        "",
        "Dataset: balanced exploratory 50-site subset.",
        "",
        "This run reuses cached OCR, CLIP, SigLIP, DINOv2, and handcrafted visual features.",
        "",
        "## Best New Method",
        "",
        f"`{best_new['method']}`: accuracy `{best_new['accuracy']}`, macro-F1 `{best_new['macro_f1']}`.",
        "",
        "## Best Overall Reference",
        "",
        f"`{best_overall['method']}`: accuracy `{best_overall['accuracy']}`, macro-F1 `{best_overall['macro_f1']}`.",
        "",
        "## Added Methods",
        "",
        "- Prompt-score classifiers: supervised models over CLIP/SigLIP similarities to label descriptions.",
        "- Ordinal ridge: treats labels as ordered positions left=-2 ... right=2 and rounds predictions.",
        "- Hierarchical side/intensity: first predicts left/center/right side, then center-vs-extreme strength.",
        "- Nested stacking: base image/OCR classifiers are stacked inside outer CV to reduce optimistic leakage.",
        "",
        "## Metrics",
        "",
        "```csv",
        "method,sample_size,accuracy,macro_f1,weighted_f1",
    ]
    for row in sorted_metrics:
        lines.append(",".join([row["method"], row["sample_size"], row["accuracy"], row["macro_f1"], row["weighted_f1"]]))
    lines.extend(["```", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = load_sample_rows()
    y = labels_for_rows(rows)
    baseline_rows = load_baseline_predictions()
    prediction_sets: dict[str, np.ndarray] = {}

    gpt55 = np.asarray([baseline_rows[row["image"]][GPT55_COLUMN] for row in rows], dtype=object)
    prediction_sets["baseline_gpt5.5_cached"] = gpt55

    ocr_df = ensure_ocr_features(rows)
    ocr_x = ocr_numeric_matrix(ocr_df)
    ocr_texts = ocr_df["ocr_text"].fillna("").astype(str).tolist()
    visual_x = handcrafted_visual_matrix(rows)
    if visual_x is None:
        raise RuntimeError("Handcrafted visual features are required for this script")

    clip_x = embedding_matrix("clip_vit_b32")
    siglip_x = embedding_matrix("siglip_b16")
    dino_x = embedding_matrix("dinov2_small")

    clip_scores = prompt_score_matrix(ensure_prompt_scores(rows, HF_IMAGE_MODELS[0]))
    siglip_scores = prompt_score_matrix(ensure_prompt_scores(rows, HF_IMAGE_MODELS[1]))
    prompt_scores = np.hstack([clip_scores, siglip_scores])

    stack_x = np.hstack([clip_x, siglip_x, dino_x, visual_x, ocr_x, prompt_scores])
    semantic_light_x = np.hstack([siglip_x, visual_x, ocr_x, prompt_scores])

    prediction_sets["clip_prompt_scores_logreg_cv"] = dense_logreg_cv(clip_scores, y)
    prediction_sets["siglip_prompt_scores_logreg_cv"] = dense_logreg_cv(siglip_scores, y)
    prediction_sets["clip_siglip_prompt_scores_logreg_cv"] = dense_logreg_cv(prompt_scores, y)
    prediction_sets["prompt_scores_plus_ocr_visual_extra_trees_cv"] = dense_extra_trees_cv(
        np.hstack([prompt_scores, ocr_x, visual_x]),
        y,
    )
    prediction_sets["siglip_ordinal_ridge_cv"] = ordinal_ridge_cv(siglip_x, y)
    prediction_sets["stack_ordinal_ridge_cv"] = ordinal_ridge_cv(stack_x, y)
    prediction_sets["siglip_hierarchical_side_intensity_cv"] = hierarchical_side_intensity_cv(siglip_x, y)
    prediction_sets["semantic_light_hierarchical_side_intensity_cv"] = hierarchical_side_intensity_cv(semantic_light_x, y)
    prediction_sets["full_local_stack_pca_ridge_cv"] = dense_pca_ridge_cv(stack_x, y)
    prediction_sets["full_local_stack_extra_trees_cv"] = dense_extra_trees_cv(stack_x, y)

    dense_specs = [
        ("visual_extra_trees", visual_x, extra_trees_estimator()),
        ("ocr_numeric_visual_extra_trees", np.hstack([ocr_x, visual_x]), extra_trees_estimator()),
        ("clip_pca_ridge", clip_x, pca_ridge_estimator(clip_x)),
        ("siglip_pca_ridge", siglip_x, pca_ridge_estimator(siglip_x)),
        ("dino_pca_ridge", dino_x, pca_ridge_estimator(dino_x)),
        ("prompt_scores_logreg", prompt_scores, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.7, random_state=RANDOM_STATE))),
    ]
    text_specs = [("ocr_text_tfidf", ocr_texts, ocr_text_estimator())]

    prediction_sets["nested_local_prediction_stack_logreg_cv"] = nested_prediction_stack_cv(
        y,
        dense_specs,
        text_specs,
    )
    prediction_sets["nested_gpt55_plus_local_prediction_stack_logreg_cv"] = nested_prediction_stack_cv(
        y,
        dense_specs,
        text_specs,
        static_predictions=[gpt55],
    )

    if BASE_IMAGE_FEATURE_PREDICTIONS_PATH.exists():
        base_predictions = pd.read_csv(BASE_IMAGE_FEATURE_PREDICTIONS_PATH)
        if base_predictions["image"].tolist() == [row["image"] for row in rows]:
            for method in [
                "strong_local_prediction_stack_logreg_cv",
                "local_embedding_visual_stack_extra_trees_cv",
                "siglip_b16_pca_ridge_cv",
            ]:
                if method in base_predictions:
                    prediction_sets[f"reference_non_nested_{method}"] = base_predictions[method].to_numpy(dtype=object)

    metric_rows = [evaluate(method, y, predictions) for method, predictions in prediction_sets.items()]
    write_csv(METRICS_PATH, metric_rows, ["method", "sample_size", "accuracy", "macro_f1", "weighted_f1"])

    all_confusions = []
    for method, predictions in prediction_sets.items():
        all_confusions.extend(confusion_rows(method, y, predictions))
    write_csv(CONFUSIONS_PATH, all_confusions, ["method", "true_label", "predicted_label", "count"])

    write_predictions(rows, y, prediction_sets)
    write_report(metric_rows)
    print(f"wrote {METRICS_PATH}", flush=True)
    print(f"wrote {PREDICTIONS_PATH}", flush=True)
    print(f"wrote {CONFUSIONS_PATH}", flush=True)
    print(f"wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
