from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


BIAS_LABELS = ["left", "left-center", "least biased", "right-center", "right"]
FACTUALITY_LABELS = ["very low", "low", "high", "very high"]
BIAS_RANDOM_STATE = 67
FACTUALITY_RANDOM_STATE = 101

BIAS_GPT_COLUMN = "bias_5:openai/gpt-5.5"
FACTUALITY_GPT_COLUMN = "multiclass:openai/gpt-5.5"

FACTUALITY_CATEGORICAL_FIELDS = [
    "source_type",
    "main_content_format",
    "evidence_style",
    "headline_style",
    "overall_tone",
]
FACTUALITY_NUMERIC_FIELDS = [
    "visible_source_attribution_level",
    "evidence_and_data_level",
    "editorial_transparency_level",
    "sensationalism_level",
    "emotional_language_level",
    "opinion_prominence_level",
    "headline_specificity_level",
    "visible_named_sources_count",
    "visible_links_or_citations_count",
    "visible_byline_or_date_count",
]
FACTUALITY_BOOLEAN_FIELDS = [
    "has_named_authors",
    "has_visible_dates",
    "has_links_to_primary_sources",
    "has_research_or_data_visuals",
    "has_fact_check_or_correction_cues",
    "has_about_contact_or_staff_cues",
    "has_clear_news_opinion_separation",
    "has_loaded_or_absolute_claims",
    "has_conspiracy_or_anti_establishment_cues",
    "has_clickbait_or_curiosity_gap_headlines",
    "has_urgent_or_alarmist_calls",
    "has_balanced_or_qualified_language",
]


def pca_components(matrix: np.ndarray, cv_splits: int = 5, requested: int = 24) -> int:
    train_size = math.floor(matrix.shape[0] * (cv_splits - 1) / cv_splits)
    return max(2, min(requested, train_size - 1, matrix.shape[1]))


def prediction_one_hot(predictions: list[np.ndarray], labels: list[str]) -> np.ndarray:
    return np.asarray(
        [
            [float(prediction_set[row_index] == label) for prediction_set in predictions for label in labels]
            for row_index in range(len(predictions[0]))
        ],
        dtype=np.float32,
    )


def train_bias(project_root: Path, output_dir: Path) -> None:
    rows = pd.read_csv(project_root / "bias_predictions.csv")
    rows = rows[rows["image"].notna() & rows["true_bias_5"].notna()].sort_values("image").reset_index(drop=True)
    image_names = rows["image"].tolist()
    y = rows["true_bias_5"].astype(str).to_numpy(dtype=object)
    gpt = rows[BIAS_GPT_COLUMN].astype(str).to_numpy(dtype=object)

    ocr = pd.read_csv(project_root / "bias5_image_feature_full_ocr_features.csv").set_index("image").loc[image_names]
    ocr_texts = ocr["ocr_text"].fillna("").astype(str).tolist()

    embedding_cache = np.load(project_root / "bias5_image_feature_full_siglip_b16_embeddings.npz", allow_pickle=True)
    embedding_names = embedding_cache["image_names"].tolist()
    embedding_index = {name: index for index, name in enumerate(embedding_names)}
    siglip = np.vstack([embedding_cache["embeddings"][embedding_index[name]] for name in image_names]).astype(np.float32)

    oof = pd.read_csv(project_root / "bias5_top_nested_source_ablation_predictions.csv").set_index("image").loc[image_names]
    oof_ocr = oof["raw_source::ocr_text_tfidf"].astype(str).to_numpy(dtype=object)
    oof_siglip = oof["raw_source::siglip_pca_ridge"].astype(str).to_numpy(dtype=object)
    oof_gpt = oof["raw_source::gpt55_cached"].astype(str).to_numpy(dtype=object)

    meta_model = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.8, random_state=BIAS_RANDOM_STATE)
    meta_model.fit(prediction_one_hot([oof_gpt, oof_ocr, oof_siglip], BIAS_LABELS), y)

    ocr_model = make_pipeline(
        TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), max_features=1500),
        LinearSVC(C=0.7, class_weight="balanced", random_state=BIAS_RANDOM_STATE),
    )
    ocr_model.fit(ocr_texts, y)

    siglip_model = make_pipeline(
        StandardScaler(),
        PCA(n_components=pca_components(siglip), random_state=BIAS_RANDOM_STATE),
        RidgeClassifier(alpha=3.0, class_weight="balanced"),
    )
    siglip_model.fit(np.nan_to_num(siglip, nan=0.0, posinf=0.0, neginf=0.0), y)

    artifact = {
        "name": "nested_stack::core_gpt55_ocr_text_siglip",
        "labels": BIAS_LABELS,
        "gpt_column": BIAS_GPT_COLUMN,
        "siglip_model_id": "google/siglip-base-patch16-224",
        "ocr_model": ocr_model,
        "siglip_model": siglip_model,
        "meta_model": meta_model,
        "source_order": ["gpt55_cached", "ocr_text_tfidf", "siglip_pca_ridge"],
    }
    joblib.dump(artifact, output_dir / "bias_core_gpt55_ocr_text_siglip.joblib")


def train_factuality(project_root: Path, output_dir: Path) -> None:
    dataset = pd.read_csv(project_root / "factuality_dataset.csv").sort_values("image").reset_index(drop=True)
    image_names = dataset["image"].tolist()
    y = dataset["factuality_4"].astype(str).to_numpy(dtype=object)
    semantic = pd.read_csv(project_root / "factuality_semantic_features.csv").set_index("image").loc[image_names]
    visual = pd.read_csv(project_root / "factuality_visual_features.csv").set_index("image").loc[image_names]
    gpt = pd.read_csv(project_root / "factuality_gpt55_predictions.csv").set_index("image").loc[image_names][FACTUALITY_GPT_COLUMN]

    category_vectorizer = DictVectorizer(sparse=False)
    category_matrix = category_vectorizer.fit_transform(semantic[FACTUALITY_CATEGORICAL_FIELDS].to_dict(orient="records"))
    numeric_columns = [
        column
        for column in semantic.columns
        if column.startswith("topic_") or column in FACTUALITY_BOOLEAN_FIELDS or column in FACTUALITY_NUMERIC_FIELDS
    ]
    numeric_matrix = semantic[numeric_columns].to_numpy(dtype=np.float32)

    baseline_vectorizer = DictVectorizer(sparse=False)
    baseline_matrix = baseline_vectorizer.fit_transform([{"baseline": value} for value in gpt.astype(str).tolist()])
    visual_columns = visual.columns.tolist()
    matrix = np.nan_to_num(
        np.hstack([category_matrix, numeric_matrix, baseline_matrix, visual[visual_columns].to_numpy(dtype=np.float32)]),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=FACTUALITY_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(matrix, y)

    artifact = {
        "name": "baseline_plus_semantic_visual_extra_trees_cv",
        "labels": FACTUALITY_LABELS,
        "gpt_column": FACTUALITY_GPT_COLUMN,
        "category_vectorizer": category_vectorizer,
        "baseline_vectorizer": baseline_vectorizer,
        "categorical_fields": FACTUALITY_CATEGORICAL_FIELDS,
        "numeric_columns": numeric_columns,
        "visual_columns": visual_columns,
        "model": model,
    }
    joblib.dump(artifact, output_dir / "factuality_semantic_visual_extra_trees.joblib")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "models")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_bias(args.project_root, args.output_dir)
    train_factuality(args.project_root, args.output_dir)
    print(f"wrote {args.output_dir / 'bias_core_gpt55_ocr_text_siglip.joblib'}")
    print(f"wrote {args.output_dir / 'factuality_semantic_visual_extra_trees.joblib'}")


if __name__ == "__main__":
    main()
