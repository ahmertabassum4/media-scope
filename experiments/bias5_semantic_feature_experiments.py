from experiment_bootstrap import activate_project_root

activate_project_root()

import base64
import concurrent.futures
import csv
import io
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from image_process import BIAS_5_LABELS, IMAGE_FOLDER, OpenRouterLLM, extract_content, safe_divide, write_csv


Image.MAX_IMAGE_PIXELS = None

MODEL = "openai/gpt-5.5"
TARGET_ASPECT_RATIO = 16 / 9
VIEW_COUNT = 4
REQUEST_TIMEOUT = 180
REQUEST_RETRIES = 2
MAX_WORKERS = 4
RANDOM_STATE = 29
CV_SPLITS = 5

SAMPLE_PATH = Path("bias5_experiment_sample.csv")
BASELINE_PREDICTIONS_PATH = Path("bias_predictions.csv")
VISUAL_FEATURES_PATH = Path("bias5_ml_visual_features.csv")
RAW_FEATURES_PATH = Path("bias5_semantic_features_raw.jsonl")
FEATURES_PATH = Path("bias5_semantic_features.csv")
PREDICTIONS_PATH = Path("bias5_semantic_ml_predictions.csv")
METRICS_PATH = Path("bias5_semantic_ml_metrics.csv")
CONFUSIONS_PATH = Path("bias5_semantic_ml_confusions.csv")
IMPORTANCE_PATH = Path("bias5_semantic_feature_importance.csv")
HYBRID_IMPORTANCE_PATH = Path("bias5_semantic_hybrid_feature_importance.csv")
GROUP_ABLATION_PATH = Path("bias5_semantic_group_ablation.csv")
SINGLE_FEATURE_ABLATION_PATH = Path("bias5_semantic_single_feature_ablation.csv")
PAIR_FEATURE_ABLATION_PATH = Path("bias5_semantic_pair_feature_ablation.csv")
REPORT_PATH = Path("bias5_semantic_feature_report.md")

GPT55_COLUMN = "bias_5:openai/gpt-5.5"

SOURCE_TYPES = [
    "straight_news",
    "local_news",
    "opinion_magazine",
    "think_tank_or_research",
    "advocacy_or_activist",
    "single_issue_site",
    "government_or_public_institution",
    "entertainment_lifestyle",
    "sports",
    "business_finance",
    "technology",
    "other_or_unclear",
]
PAGE_SCOPES = ["local", "national", "international", "topic_vertical", "mixed_or_unclear"]
DIRECTION_HINTS = ["none", "left", "right", "mixed_or_unclear"]
INTENSITY_HINTS = ["none", "mild", "strong", "unclear"]
HEADLINE_FRAMES = ["mostly_neutral", "left_framed", "right_framed", "mixed", "unclear"]
CONTENT_FORMATS = [
    "news_homepage",
    "blog_or_commentary",
    "advocacy_homepage",
    "magazine",
    "research_org",
    "aggregator",
    "institutional_homepage",
    "unclear",
]
TONE_VALUES = ["neutral", "analytical", "activist", "sensational", "promotional", "mixed_or_unclear"]
TOPICS = [
    "politics_government",
    "elections_campaigns",
    "policy_law",
    "economy_business",
    "immigration_border",
    "climate_environment",
    "social_justice_identity",
    "religion_values",
    "crime_policing",
    "foreign_policy_security",
    "health_science",
    "technology",
    "culture_education_gender",
    "local_community",
    "entertainment_lifestyle",
    "sports",
]
NUMERIC_FIELDS = [
    "political_content_level",
    "advocacy_language_level",
    "opinion_prominence_level",
    "sensationalism_level",
    "partisan_branding_level",
    "call_to_action_level",
    "emotional_language_level",
    "visible_named_political_entities",
    "political_headline_count",
    "policy_issue_count",
]
BOOLEAN_FIELDS = [
    "has_opinion_section",
    "has_fact_check_or_corrections_cues",
    "has_donate_or_join_cta",
    "has_petition_or_campaign_cta",
    "has_local_weather_or_traffic",
    "has_wire_service_style",
    "has_research_reports_or_data",
    "has_religious_or_values_language",
    "has_conspiracy_or_anti_establishment_language",
    "has_loaded_labels_for_opponents",
    "has_balanced_multi_side_framing",
    "has_prominent_author_opinion",
]
CATEGORICAL_FIELDS = {
    "source_type": SOURCE_TYPES,
    "page_scope": PAGE_SCOPES,
    "ideological_direction_hint": DIRECTION_HINTS,
    "ideological_intensity_hint": INTENSITY_HINTS,
    "headline_frame": HEADLINE_FRAMES,
    "main_content_format": CONTENT_FORMATS,
    "overall_tone": TONE_VALUES,
}

FEATURE_PROMPT = f"""
Extract structured visible-page features for political-bias classification.

Use only the supplied landing-page screenshots. Do not browse the web. Do not use prior
knowledge about the outlet. Do not output the final bias label. Your job is to describe
visible cues as structured features for a downstream classifier.

Return valid compact JSON with exactly these keys:
- source_type: one of {SOURCE_TYPES}
- page_scope: one of {PAGE_SCOPES}
- ideological_direction_hint: one of {DIRECTION_HINTS}
- ideological_intensity_hint: one of {INTENSITY_HINTS}
- headline_frame: one of {HEADLINE_FRAMES}
- main_content_format: one of {CONTENT_FORMATS}
- overall_tone: one of {TONE_VALUES}
- topics: object with exactly these boolean keys: {TOPICS}
- numeric fields, integers 0..3 unless count field says 0..10:
  political_content_level, advocacy_language_level, opinion_prominence_level,
  sensationalism_level, partisan_branding_level, call_to_action_level,
  emotional_language_level, visible_named_political_entities,
  political_headline_count, policy_issue_count
- boolean fields: {BOOLEAN_FIELDS}

Scale guidance:
0 = absent/not visible, 1 = weak/minor, 2 = clear/moderate, 3 = strong/prominent.
For count fields use an approximate visible count from 0 to 10.

JSON only. No markdown.
""".strip()


def crop_viewport(image_path: Path, viewport_index: int) -> bytes:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        viewport_height = min(height, round(width / TARGET_ASPECT_RATIO))
        top = min(viewport_index * viewport_height, max(0, height - viewport_height))
        crop_box = (0, top, width, top + viewport_height)
        cropped = image.crop(crop_box)
        output = io.BytesIO()
        cropped.save(output, format="PNG")
        return output.getvalue()


def viewport_to_data_url(image_path: Path, viewport_index: int) -> str:
    encoded = base64.b64encode(crop_viewport(image_path, viewport_index)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_messages(image_name: str) -> list[dict[str, Any]]:
    image_path = IMAGE_FOLDER / f"{image_name}.png"
    content: list[dict[str, Any]] = [{"type": "text", "text": FEATURE_PROMPT}]
    for index in range(VIEW_COUNT):
        content.append({"type": "image_url", "image_url": {"url": viewport_to_data_url(image_path, index)}})
    return [{"role": "user", "content": content}]


def load_sample_rows() -> list[dict[str, str]]:
    with SAMPLE_PATH.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def load_baseline() -> dict[str, str]:
    with BASELINE_PREDICTIONS_PATH.open("r", encoding="utf-8", newline="") as input_file:
        return {
            row["image"]: row[GPT55_COLUMN]
            for row in csv.DictReader(input_file)
            if row.get("image")
        }


def load_raw_features() -> dict[str, dict[str, Any]]:
    if not RAW_FEATURES_PATH.exists():
        return {}
    rows = {}
    with RAW_FEATURES_PATH.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["image"]] = row["features"]
    return rows


def append_raw_feature(image_name: str, features: dict[str, Any]) -> None:
    with RAW_FEATURES_PATH.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps({"image": image_name, "features": features}, ensure_ascii=False) + "\n")


def parse_json_response(response_text: str) -> dict[str, Any]:
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found: {response_text[:200]}")
    return json.loads(match.group(0))


def request_features(llm: OpenRouterLLM, image_name: str) -> dict[str, Any]:
    last_error: Exception | None = None
    messages = build_messages(image_name)
    for attempt in range(REQUEST_RETRIES + 1):
        try:
            response = llm.openr_llm(model=MODEL, messages=messages, timeout=REQUEST_TIMEOUT)
            return parse_json_response(extract_content(response))
        except Exception as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(2 ** attempt)
    raise last_error or RuntimeError("Feature extraction failed")


def coerce_category(value: Any, allowed: list[str], fallback: str) -> str:
    value_text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    normalized = {item.replace("-", "_"): item for item in allowed}
    return normalized.get(value_text, fallback)


def coerce_bool(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    return int(str(value).strip().lower() in {"true", "yes", "1", "present"})


def coerce_int(value: Any, upper: int) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = 0
    return max(0, min(upper, number))


def flatten_features(image_name: str, truth: str, baseline: str, raw: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image": image_name,
        "true_bias_5": truth,
        "baseline_gpt5.5": baseline,
    }

    for field, allowed in CATEGORICAL_FIELDS.items():
        fallback = allowed[-1]
        row[field] = coerce_category(raw.get(field, fallback), allowed, fallback)

    topics = raw.get("topics", {})
    if not isinstance(topics, dict):
        topics = {}
    for topic in TOPICS:
        row[f"topic_{topic}"] = coerce_bool(topics.get(topic, False))

    for field in BOOLEAN_FIELDS:
        row[field] = coerce_bool(raw.get(field, False))

    for field in NUMERIC_FIELDS:
        upper = 10 if field.endswith("_count") else 3
        row[field] = coerce_int(raw.get(field, 0), upper)

    return row


def ensure_feature_table() -> pd.DataFrame:
    sample_rows = load_sample_rows()
    baseline = load_baseline()
    raw_features = load_raw_features()
    missing = [row["image"] for row in sample_rows if row["image"] not in raw_features]

    if missing:
        llm = OpenRouterLLM()
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(request_features, llm, image_name): image_name for image_name in missing}
            for future in concurrent.futures.as_completed(futures):
                image_name = futures[future]
                features = future.result()
                raw_features[image_name] = features
                append_raw_feature(image_name, features)
                print(f"semantic features {len(raw_features)}/{len(sample_rows)}: {image_name}", flush=True)

    flat_rows = []
    truth_by_name = {row["image"]: row["true_bias_5"] for row in sample_rows}
    for image_name in sorted(truth_by_name):
        flat_rows.append(flatten_features(image_name, truth_by_name[image_name], baseline[image_name], raw_features[image_name]))

    feature_df = pd.DataFrame(flat_rows)
    feature_df.to_csv(FEATURES_PATH, index=False)
    return feature_df


def load_visual_features_for_sample(images: list[str]) -> pd.DataFrame:
    visual_df = pd.read_csv(VISUAL_FEATURES_PATH)
    return visual_df.set_index("image").loc[images].reset_index()


def one_hot(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    dict_rows = df[columns].to_dict(orient="records")
    return DictVectorizer(sparse=False).fit_transform(dict_rows)


def semantic_matrix(feature_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    categorical_columns = ["baseline_gpt5.5", *CATEGORICAL_FIELDS.keys()]
    one_hot_matrix = one_hot(feature_df, categorical_columns)
    one_hot_names = DictVectorizer(sparse=False).fit(feature_df[categorical_columns].to_dict(orient="records")).get_feature_names_out().tolist()
    numeric_columns = [
        column for column in feature_df.columns
        if column.startswith("topic_") or column in BOOLEAN_FIELDS or column in NUMERIC_FIELDS
    ]
    numeric_matrix = feature_df[numeric_columns].to_numpy(dtype=float)
    return np.hstack([one_hot_matrix, numeric_matrix]), [*one_hot_names, *numeric_columns]


def semantic_only_matrix(feature_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    categorical_columns = list(CATEGORICAL_FIELDS.keys())
    vectorizer = DictVectorizer(sparse=False)
    one_hot_matrix = vectorizer.fit_transform(feature_df[categorical_columns].to_dict(orient="records"))
    numeric_columns = [
        column for column in feature_df.columns
        if column.startswith("topic_") or column in BOOLEAN_FIELDS or column in NUMERIC_FIELDS
    ]
    numeric_matrix = feature_df[numeric_columns].to_numpy(dtype=float)
    return np.hstack([one_hot_matrix, numeric_matrix]), [*vectorizer.get_feature_names_out().tolist(), *numeric_columns]


def visual_matrix(visual_df: pd.DataFrame) -> np.ndarray:
    return visual_df.drop(columns=["image"], errors="ignore").to_numpy(dtype=float)


def classifier_logreg() -> LogisticRegression:
    return LogisticRegression(max_iter=5000, class_weight="balanced", C=0.7, random_state=RANDOM_STATE)


def classifier_extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def classifier_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def classifier_mlp() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(32,),
        alpha=0.05,
        max_iter=3000,
        random_state=RANDOM_STATE,
        early_stopping=False,
    )


def cv_predict(x: np.ndarray, y: np.ndarray, estimator: Any, scale: bool) -> np.ndarray:
    model = make_pipeline(StandardScaler(), estimator) if scale else estimator
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, x, y, cv=cv)


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


def feature_importances(x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> list[dict[str, str]]:
    model = classifier_extra_trees()
    model.fit(x, y)
    rows = []
    for name, importance in sorted(zip(feature_names, model.feature_importances_), key=lambda item: item[1], reverse=True):
        rows.append({"feature": name, "importance": f"{importance:.8f}"})
    return rows


def cv_macro_f1(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    predictions = cv_predict(x, y, classifier_extra_trees(), scale=False)
    return (
        accuracy_score(y, predictions),
        f1_score(y, predictions, labels=BIAS_5_LABELS, average="macro"),
    )


def single_feature_ablation(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    importance_rows: list[dict[str, str]],
    base_macro_f1: float,
    candidate_count: int = 30,
) -> list[dict[str, str]]:
    importance_by_feature = {row["feature"]: float(row["importance"]) for row in importance_rows}
    candidates = [
        row["feature"]
        for row in sorted(importance_rows, key=lambda item: float(item["importance"]))[:candidate_count]
    ]
    feature_to_index = {feature: index for index, feature in enumerate(feature_names)}
    rows = []
    all_indices = np.arange(x.shape[1])

    for feature in candidates:
        keep = np.setdiff1d(all_indices, np.asarray([feature_to_index[feature]]))
        accuracy, macro_f1 = cv_macro_f1(x[:, keep], y)
        rows.append(
            {
                "removed_feature": feature,
                "feature_importance": f"{importance_by_feature[feature]:.8f}",
                "accuracy": f"{accuracy:.6f}",
                "macro_f1": f"{macro_f1:.6f}",
                "macro_f1_delta_vs_full": f"{macro_f1 - base_macro_f1:.6f}",
            }
        )

    return sorted(rows, key=lambda row: float(row["macro_f1_delta_vs_full"]), reverse=True)


def pair_feature_ablation(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    importance_rows: list[dict[str, str]],
    base_macro_f1: float,
    candidate_count: int = 12,
) -> list[dict[str, str]]:
    candidates = [
        row["feature"]
        for row in sorted(importance_rows, key=lambda item: float(item["importance"]))[:candidate_count]
    ]
    feature_to_index = {feature: index for index, feature in enumerate(feature_names)}
    all_indices = np.arange(x.shape[1])
    rows = []

    for left_index, left_feature in enumerate(candidates):
        for right_feature in candidates[left_index + 1:]:
            removed = np.asarray([feature_to_index[left_feature], feature_to_index[right_feature]])
            keep = np.setdiff1d(all_indices, removed)
            accuracy, macro_f1 = cv_macro_f1(x[:, keep], y)
            rows.append(
                {
                    "removed_feature_1": left_feature,
                    "removed_feature_2": right_feature,
                    "accuracy": f"{accuracy:.6f}",
                    "macro_f1": f"{macro_f1:.6f}",
                    "macro_f1_delta_vs_full": f"{macro_f1 - base_macro_f1:.6f}",
                }
            )

    return sorted(rows, key=lambda row: float(row["macro_f1_delta_vs_full"]), reverse=True)


def grouped_feature_indices(feature_names: list[str]) -> dict[str, list[int]]:
    groups = {
        "baseline_label": [],
        "categorical_source_page": [],
        "ideology_hint": [],
        "topics": [],
        "boolean_cues": [],
        "numeric_levels_counts": [],
    }
    for index, name in enumerate(feature_names):
        if name.startswith("baseline_gpt5.5="):
            groups["baseline_label"].append(index)
        elif name.startswith("ideological_") or name.startswith("headline_frame="):
            groups["ideology_hint"].append(index)
        elif name.startswith("topic_"):
            groups["topics"].append(index)
        elif name in BOOLEAN_FIELDS:
            groups["boolean_cues"].append(index)
        elif name in NUMERIC_FIELDS:
            groups["numeric_levels_counts"].append(index)
        else:
            groups["categorical_source_page"].append(index)
    return groups


def group_ablation(x: np.ndarray, y: np.ndarray, feature_names: list[str], base_macro_f1: float) -> list[dict[str, str]]:
    rows = []
    groups = grouped_feature_indices(feature_names)
    all_indices = np.arange(x.shape[1])

    for group, indices in groups.items():
        if not indices:
            continue
        keep = np.setdiff1d(all_indices, np.asarray(indices))
        predictions = cv_predict(x[:, keep], y, classifier_extra_trees(), scale=False)
        macro_f1 = f1_score(y, predictions, labels=BIAS_5_LABELS, average="macro")
        rows.append(
            {
                "removed_group": group,
                "removed_features": str(len(indices)),
                "accuracy": f"{accuracy_score(y, predictions):.6f}",
                "macro_f1": f"{macro_f1:.6f}",
                "macro_f1_delta_vs_full": f"{macro_f1 - base_macro_f1:.6f}",
            }
        )

    return sorted(rows, key=lambda row: float(row["macro_f1_delta_vs_full"]))


def run() -> None:
    feature_df = ensure_feature_table()
    images = feature_df["image"].tolist()
    visual_df = load_visual_features_for_sample(images)
    y = feature_df["true_bias_5"].to_numpy()

    semantic_x, semantic_names = semantic_matrix(feature_df)
    semantic_only_x, semantic_only_names = semantic_only_matrix(feature_df)
    visual_x = visual_matrix(visual_df)
    combined_x = np.hstack([semantic_x, visual_x])
    combined_names = [*semantic_names, *[f"visual::{column}" for column in visual_df.columns if column != "image"]]

    methods = {
        "baseline_gpt5.5": feature_df["baseline_gpt5.5"].to_numpy(),
        "semantic_only_logreg_cv": cv_predict(semantic_only_x, y, classifier_logreg(), scale=True),
        "semantic_only_random_forest_cv": cv_predict(semantic_only_x, y, classifier_random_forest(), scale=False),
        "semantic_only_extra_trees_cv": cv_predict(semantic_only_x, y, classifier_extra_trees(), scale=False),
        "semantic_only_mlp_cv": cv_predict(semantic_only_x, y, classifier_mlp(), scale=True),
        "baseline_plus_semantic_logreg_cv": cv_predict(semantic_x, y, classifier_logreg(), scale=True),
        "baseline_plus_semantic_random_forest_cv": cv_predict(semantic_x, y, classifier_random_forest(), scale=False),
        "baseline_plus_semantic_extra_trees_cv": cv_predict(semantic_x, y, classifier_extra_trees(), scale=False),
        "baseline_plus_semantic_mlp_cv": cv_predict(semantic_x, y, classifier_mlp(), scale=True),
        "baseline_plus_semantic_visual_extra_trees_cv": cv_predict(combined_x, y, classifier_extra_trees(), scale=False),
    }

    prediction_rows = []
    for index, image_name in enumerate(images):
        prediction_rows.append(
            {
                "image": image_name,
                "true_bias_5": str(y[index]),
                **{method: str(predictions[index]) for method, predictions in methods.items()},
            }
        )

    metric_rows = [metric_row(method, y, predictions) for method, predictions in methods.items()]
    confusion_output_rows = [
        row
        for method, predictions in methods.items()
        for row in confusion_rows(method, y, predictions)
    ]

    write_csv(PREDICTIONS_PATH, prediction_rows, list(prediction_rows[0].keys()))
    write_csv(METRICS_PATH, metric_rows, list(metric_rows[0].keys()))
    write_csv(CONFUSIONS_PATH, confusion_output_rows, ["method", "true_label", *[f"pred_{label}" for label in BIAS_5_LABELS]])

    importance_rows = feature_importances(semantic_only_x, y, semantic_only_names)
    hybrid_importance_rows = feature_importances(semantic_x, y, semantic_names)
    write_csv(IMPORTANCE_PATH, importance_rows, ["feature", "importance"])
    write_csv(HYBRID_IMPORTANCE_PATH, hybrid_importance_rows, ["feature", "importance"])

    best_semantic = next(row for row in metric_rows if row["method"] == "semantic_only_extra_trees_cv")
    base_macro_f1 = float(best_semantic["macro_f1"])
    ablation_rows = group_ablation(semantic_only_x, y, semantic_only_names, base_macro_f1)
    write_csv(GROUP_ABLATION_PATH, ablation_rows, ["removed_group", "removed_features", "accuracy", "macro_f1", "macro_f1_delta_vs_full"])

    single_ablation_rows = single_feature_ablation(semantic_only_x, y, semantic_only_names, importance_rows, base_macro_f1)
    pair_ablation_rows = pair_feature_ablation(semantic_only_x, y, semantic_only_names, importance_rows, base_macro_f1)
    write_csv(
        SINGLE_FEATURE_ABLATION_PATH,
        single_ablation_rows,
        ["removed_feature", "feature_importance", "accuracy", "macro_f1", "macro_f1_delta_vs_full"],
    )
    write_csv(
        PAIR_FEATURE_ABLATION_PATH,
        pair_ablation_rows,
        ["removed_feature_1", "removed_feature_2", "accuracy", "macro_f1", "macro_f1_delta_vs_full"],
    )

    write_report(metric_rows, importance_rows, ablation_rows, single_ablation_rows, pair_ablation_rows)


def write_report(
    metric_rows: list[dict[str, str]],
    importance_rows: list[dict[str, str]],
    ablation_rows: list[dict[str, str]],
    single_ablation_rows: list[dict[str, str]],
    pair_ablation_rows: list[dict[str, str]],
) -> None:
    metrics_text = METRICS_PATH.read_text(encoding="utf-8")
    top_features = "\n".join(
        f"- `{row['feature']}`: {row['importance']}"
        for row in importance_rows[:25]
    )
    ablation_text = GROUP_ABLATION_PATH.read_text(encoding="utf-8")
    single_ablation_text = "\n".join(
        ",".join(row[field] for field in ["removed_feature", "feature_importance", "accuracy", "macro_f1", "macro_f1_delta_vs_full"])
        for row in single_ablation_rows[:15]
    )
    pair_ablation_text = "\n".join(
        ",".join(row[field] for field in ["removed_feature_1", "removed_feature_2", "accuracy", "macro_f1", "macro_f1_delta_vs_full"])
        for row in pair_ablation_rows[:15]
    )
    best = max(metric_rows, key=lambda row: float(row["macro_f1"]))

    report = (
        "# Bias 5 Semantic Feature Experiment Report\n\n"
        f"Model used for extraction: `{MODEL}`.\n\n"
        "Dataset: balanced dev subset of 50 sites, 10 per 5-label class.\n\n"
        "The LLM extracts structured visible-page features from the first four viewport screenshots. "
        "The downstream classifiers then predict the 5-label bias class from those features.\n\n"
        "## Best Method\n\n"
        f"`{best['method']}`: accuracy `{best['accuracy']}`, macro-F1 `{best['macro_f1']}`.\n\n"
        "## Metrics\n\n"
        "```csv\n"
        f"{metrics_text.strip()}\n"
        "```\n\n"
        "## Top ExtraTrees Feature Importances\n\n"
        f"{top_features}\n\n"
        "## Feature Group Ablation\n\n"
        "Negative delta means removing that group hurt the model. Positive delta means removing it improved the model on this small subset.\n\n"
        "```csv\n"
        f"{ablation_text.strip()}\n"
        "```\n\n"
        "## Best Single-Feature Cuts Among Low-Importance Features\n\n"
        "```csv\n"
        "removed_feature,feature_importance,accuracy,macro_f1,macro_f1_delta_vs_full\n"
        f"{single_ablation_text}\n"
        "```\n\n"
        "## Best Pair Cuts Among Low-Importance Features\n\n"
        "```csv\n"
        "removed_feature_1,removed_feature_2,accuracy,macro_f1,macro_f1_delta_vs_full\n"
        f"{pair_ablation_text}\n"
        "```\n"
    )
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    run()
