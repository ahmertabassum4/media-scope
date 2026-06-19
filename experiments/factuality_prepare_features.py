from experiment_bootstrap import activate_project_root

activate_project_root()

import concurrent.futures
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageStat

from factuality_common import (
    BINARY_LABELS,
    GPT_PREDICTIONS_PATH,
    MULTICLASS_LABELS,
    VIEW_COUNT,
    build_dataset,
    crop_viewport,
    image_path,
    load_dataset,
    viewport_data_url,
    write_csv,
)
from llm_client import OpenRouterLLM


Image.MAX_IMAGE_PIXELS = None

MODEL = "openai/gpt-5.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WORKERS = 4
REQUEST_TIMEOUT = 180
REQUEST_RETRIES = 2
OCR_VIEW_COUNT = 2

EXISTING_BINARY_PREDICTIONS_PATH = Path("factuality_labels.csv")
MULTICLASS_RAW_PATH = Path("factuality_gpt55_multiclass_raw.jsonl")
SEMANTIC_RAW_PATH = Path("factuality_gpt55_semantic_raw.jsonl")
USAGE_PATH = Path("factuality_openrouter_usage.csv")
VISUAL_FEATURES_PATH = Path("factuality_visual_features.csv")
OCR_RAW_PATH = Path("factuality_ocr_raw.jsonl")
OCR_FEATURES_PATH = Path("factuality_ocr_features.csv")
SEMANTIC_FEATURES_PATH = Path("factuality_semantic_features.csv")
RESNET_EMBEDDINGS_PATH = Path("factuality_resnet18_embeddings.npz")

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

FACTUALITY_PROMPTS = {
    "very low": [
        "a screenshot of a very low factuality website with unsupported or deceptive claims",
        "a website using conspiracy-like, fabricated, or highly unreliable reporting",
    ],
    "low": [
        "a screenshot of a low factuality website with weak sourcing and misleading framing",
        "a website whose reporting has limited evidence and notable reliability problems",
    ],
    "high": [
        "a screenshot of a high factuality website with professional reporting and clear sourcing",
        "a generally reliable news or research website using evidence and attribution",
    ],
    "very high": [
        "a screenshot of a very high factuality website with exceptional sourcing and transparency",
        "a highly rigorous evidence-based website with strong attribution and editorial standards",
    ],
}

RELIABILITY_TERMS = [
    "according to",
    "reported by",
    "sources",
    "study",
    "research",
    "data",
    "evidence",
    "methodology",
    "correction",
    "fact check",
    "analysis",
    "report",
    "official",
    "documents",
]
UNRELIABLE_TERMS = [
    "shocking",
    "exposed",
    "they don't want you to know",
    "mainstream media",
    "globalist",
    "hoax",
    "conspiracy",
    "secret",
    "censored",
    "cover-up",
    "tyranny",
    "deep state",
    "wake up",
]
SENSATIONAL_TERMS = [
    "breaking",
    "bombshell",
    "shocking",
    "urgent",
    "destroy",
    "crisis",
    "scandal",
    "outrage",
    "warning",
    "threat",
]
TRANSPARENCY_TERMS = [
    "about us",
    "contact",
    "editorial policy",
    "corrections",
    "ethics",
    "standards",
    "authors",
    "staff",
]

SOURCE_TYPES = [
    "news_outlet",
    "local_news",
    "research_or_academic",
    "government_or_public_institution",
    "fact_checking",
    "advocacy_or_activist",
    "opinion_or_blog",
    "health_or_science",
    "business_or_finance",
    "entertainment_or_lifestyle",
    "single_issue_site",
    "other_or_unclear",
]
CONTENT_FORMATS = [
    "reported_news",
    "analysis",
    "research_reports",
    "opinion_commentary",
    "aggregated_content",
    "advocacy_campaign",
    "institutional_information",
    "mixed_or_unclear",
]
EVIDENCE_STYLES = [
    "strong_visible_attribution",
    "some_visible_attribution",
    "mostly_assertive_without_visible_support",
    "unclear",
]
HEADLINE_STYLES = ["straight", "analytical", "promotional", "sensational", "mixed_or_unclear"]
TONE_VALUES = ["neutral", "analytical", "advocacy", "alarmist", "promotional", "mixed_or_unclear"]
TOPICS = [
    "politics_government",
    "health_science",
    "climate_environment",
    "economy_business",
    "crime_security",
    "international",
    "technology",
    "religion_values",
    "local_community",
    "entertainment_lifestyle",
    "sports",
]
NUMERIC_FIELDS = [
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
BOOLEAN_FIELDS = [
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
CATEGORICAL_FIELDS = {
    "source_type": SOURCE_TYPES,
    "main_content_format": CONTENT_FORMATS,
    "evidence_style": EVIDENCE_STYLES,
    "headline_style": HEADLINE_STYLES,
    "overall_tone": TONE_VALUES,
}

MULTICLASS_PROMPT = (
    "Classify the factuality of this website using only the supplied landing-page screenshot. "
    "Do not browse the web and do not use prior knowledge about the outlet. Judge only visible "
    "signals such as sourcing, attribution, evidence, transparency, headline style, and unsupported "
    "or sensational claims. Return exactly one lowercase label with no explanation: "
    "very low, low, high, or very high."
)

SEMANTIC_PROMPT = f"""
Extract structured visible-page features for factuality classification.

Use only the supplied landing-page screenshots. Do not browse the web and do not use prior
knowledge about the outlet. Do not output a factuality label. Describe only visible cues for
a downstream classifier.

Return valid compact JSON with exactly these keys:
- source_type: one of {SOURCE_TYPES}
- main_content_format: one of {CONTENT_FORMATS}
- evidence_style: one of {EVIDENCE_STYLES}
- headline_style: one of {HEADLINE_STYLES}
- overall_tone: one of {TONE_VALUES}
- topics: object with exactly these boolean keys: {TOPICS}
- numeric fields: {NUMERIC_FIELDS}
- boolean fields: {BOOLEAN_FIELDS}

For level fields use integers 0..3: 0 absent, 1 weak, 2 clear, 3 strong.
For count fields use an approximate visible count from 0..10.
Return JSON only, without markdown.
""".strip()


def embedding_path(name: str) -> Path:
    return Path(f"factuality_{name}_embeddings.npz")


def prompt_score_path(name: str) -> Path:
    return Path(f"factuality_{name}_prompt_scores.csv")


def load_jsonl(path: Path, value_key: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows = {}
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                row = json.loads(line)
                rows[row["image"]] = row[value_key]
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_content(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content)


def usage_row(image_name: str, request_type: str, response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {
        "image": image_name,
        "request_type": request_type,
        "model": MODEL,
        "prompt_tokens": getattr(usage, "prompt_tokens", "") if usage else "",
        "completion_tokens": getattr(usage, "completion_tokens", "") if usage else "",
        "total_tokens": getattr(usage, "total_tokens", "") if usage else "",
        "response_id": getattr(response, "id", ""),
    }


def append_usage(row: dict[str, Any]) -> None:
    exists = USAGE_PATH.exists()
    with USAGE_PATH.open("a", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def request_with_retry(llm: OpenRouterLLM, messages: list[dict[str, Any]]) -> Any:
    last_error: Exception | None = None
    for attempt in range(REQUEST_RETRIES + 1):
        try:
            return llm.openr_llm(model=MODEL, messages=messages, timeout=REQUEST_TIMEOUT)
        except Exception as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(2 ** attempt)
    raise last_error or RuntimeError("OpenRouter request failed")


def normalize_label(text: str, labels: list[str]) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower().strip("\"'` ."))
    if normalized in labels:
        return normalized
    matches = [label for label in labels if re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", normalized)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Unexpected label response: {text[:200]}")


def multiclass_messages(image_name: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": MULTICLASS_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": viewport_data_url(image_path(image_name), 0)},
                },
            ],
        }
    ]


def semantic_messages(image_name: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": SEMANTIC_PROMPT}]
    for view_index in range(VIEW_COUNT):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": viewport_data_url(image_path(image_name), view_index)},
            }
        )
    return [{"role": "user", "content": content}]


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found: {text[:200]}")
    return json.loads(match.group(0))


def ensure_gpt_predictions(rows: list[dict[str, str]]) -> pd.DataFrame:
    with EXISTING_BINARY_PREDICTIONS_PATH.open("r", encoding="utf-8-sig", newline="") as input_file:
        binary_by_image = {
            row["image"]: normalize_label(row["openai/gpt-5.5"], BINARY_LABELS)
            for row in csv.DictReader(input_file)
        }
    image_names = [row["image"] for row in rows]
    missing_binary = sorted(set(image_names) - set(binary_by_image))
    if missing_binary:
        raise ValueError(f"Missing cached binary GPT predictions: {missing_binary[:5]}")

    multiclass_by_image = load_jsonl(MULTICLASS_RAW_PATH, "prediction")
    missing = [image_name for image_name in image_names if image_name not in multiclass_by_image]
    if missing:
        llm = OpenRouterLLM()

        def request(image_name: str) -> tuple[str, Any]:
            response = request_with_retry(llm, multiclass_messages(image_name))
            return normalize_label(extract_content(response), MULTICLASS_LABELS), response

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(request, image_name): image_name for image_name in missing}
            completed = len(multiclass_by_image)
            for future in concurrent.futures.as_completed(futures):
                image_name = futures[future]
                prediction, response = future.result()
                multiclass_by_image[image_name] = prediction
                append_jsonl(MULTICLASS_RAW_PATH, {"image": image_name, "prediction": prediction})
                append_usage(usage_row(image_name, "multiclass_baseline", response))
                completed += 1
                print(f"GPT multiclass {completed}/{len(rows)}: {image_name}", flush=True)

    output_rows = [
        {
            "image": image_name,
            "binary:openai/gpt-5.5": binary_by_image[image_name],
            "multiclass:openai/gpt-5.5": multiclass_by_image[image_name],
        }
        for image_name in image_names
    ]
    write_csv(GPT_PREDICTIONS_PATH, output_rows, list(output_rows[0].keys()))
    return pd.DataFrame(output_rows)


def crop_visual_view(path: Path, view_index: int) -> Image.Image:
    return crop_viewport(path, view_index).resize((256, 144)).convert("RGB")


def entropy_from_histogram(histogram: list[int], total: int) -> float:
    entropy = 0.0
    for count in histogram:
        if count:
            probability = count / total
            entropy -= probability * math.log2(probability)
    return entropy


def visual_view_features(image: Image.Image, prefix: str) -> dict[str, float]:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray_image = image.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    height, width = gray.shape
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    top_band = rgb[: max(1, height // 5), :, :]
    row_dark_fraction = (gray < 0.18).mean(axis=1)
    col_dark_fraction = (gray < 0.18).mean(axis=0)
    red_dominance = np.maximum(rgb[:, :, 0] - np.maximum(rgb[:, :, 1], rgb[:, :, 2]), 0)
    blue_dominance = np.maximum(rgb[:, :, 2] - np.maximum(rgb[:, :, 0], rgb[:, :, 1]), 0)
    stats = ImageStat.Stat(image)

    return {
        f"{prefix}_gray_mean": float(gray.mean()),
        f"{prefix}_gray_std": float(gray.std()),
        f"{prefix}_gray_entropy": entropy_from_histogram(gray_image.histogram(), height * width),
        f"{prefix}_edge_density": (float((dx > 0.16).mean()) + float((dy > 0.16).mean())) / 2,
        f"{prefix}_fine_edge_density": (float((dx > 0.08).mean()) + float((dy > 0.08).mean())) / 2,
        f"{prefix}_saturation_mean": float(hsv[:, :, 1].mean()),
        f"{prefix}_saturation_std": float(hsv[:, :, 1].std()),
        f"{prefix}_value_mean": float(hsv[:, :, 2].mean()),
        f"{prefix}_red_mean": float(rgb[:, :, 0].mean()),
        f"{prefix}_green_mean": float(rgb[:, :, 1].mean()),
        f"{prefix}_blue_mean": float(rgb[:, :, 2].mean()),
        f"{prefix}_red_dominance_mean": float(red_dominance.mean()),
        f"{prefix}_blue_dominance_mean": float(blue_dominance.mean()),
        f"{prefix}_dark_fraction": float((gray < 0.18).mean()),
        f"{prefix}_light_fraction": float((gray > 0.92).mean()),
        f"{prefix}_white_fraction": float((rgb.min(axis=2) > 0.94).mean()),
        f"{prefix}_black_fraction": float((rgb.max(axis=2) < 0.08).mean()),
        f"{prefix}_top_dark_fraction": float((top_band.max(axis=2) < 0.18).mean()),
        f"{prefix}_row_dark_peaks": float((row_dark_fraction > 0.35).mean()),
        f"{prefix}_col_dark_peaks": float((col_dark_fraction > 0.25).mean()),
        f"{prefix}_channel_mean_range": float(max(stats.mean) - min(stats.mean)) / 255.0,
    }


def ensure_visual_features(rows: list[dict[str, str]]) -> pd.DataFrame:
    image_names = [row["image"] for row in rows]
    if VISUAL_FEATURES_PATH.exists():
        cached = pd.read_csv(VISUAL_FEATURES_PATH)
        if cached["image"].tolist() == image_names:
            return cached

    output_rows = []
    for index, image_name in enumerate(image_names, start=1):
        path = image_path(image_name)
        with Image.open(path) as image:
            width, height = image.size
        features: dict[str, Any] = {
            "image": image_name,
            "image_width": float(width),
            "image_height": float(height),
            "image_aspect": float(width / height),
            "image_log_height": float(math.log1p(height)),
            "image_file_mb": float(path.stat().st_size / (1024 * 1024)),
        }
        for view_index in range(VIEW_COUNT):
            features.update(visual_view_features(crop_visual_view(path, view_index), f"view{view_index}"))
        output_rows.append(features)
        print(f"visual features {index}/{len(rows)}: {image_name}", flush=True)

    frame = pd.DataFrame(output_rows)
    frame.to_csv(VISUAL_FEATURES_PATH, index=False)
    return frame


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
    uppercase_words = [word for word in words if len(word) > 2 and word.isupper()]
    return {
        "image": image_name,
        "ocr_text": text,
        "ocr_box_count": boxes,
        "ocr_word_count": len(words),
        "ocr_unique_word_count": len({word.lower() for word in words}),
        "ocr_avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "ocr_upper_word_fraction": len(uppercase_words) / max(1, len(words)),
        "ocr_exclamation_count": text.count("!"),
        "ocr_question_count": text.count("?"),
        "ocr_digit_count": len(re.findall(r"\d", text)),
        "ocr_reliability_term_count": count_terms(text, RELIABILITY_TERMS),
        "ocr_unreliable_term_count": count_terms(text, UNRELIABLE_TERMS),
        "ocr_sensational_term_count": count_terms(text, SENSATIONAL_TERMS),
        "ocr_transparency_term_count": count_terms(text, TRANSPARENCY_TERMS),
    }


def ensure_ocr_features(rows: list[dict[str, str]]) -> pd.DataFrame:
    image_names = [row["image"] for row in rows]
    if OCR_FEATURES_PATH.exists():
        cached = pd.read_csv(OCR_FEATURES_PATH)
        if cached["image"].tolist() == image_names:
            return cached

    import easyocr

    raw_rows = load_jsonl(OCR_RAW_PATH, "features")
    missing = [image_name for image_name in image_names if image_name not in raw_rows]
    if missing:
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        for index, image_name in enumerate(missing, start=1):
            results_by_view = []
            for view_index in range(OCR_VIEW_COUNT):
                crop = crop_viewport(image_path(image_name), view_index, max_width=1280)
                results_by_view.append(
                    reader.readtext(
                        np.asarray(crop),
                        detail=1,
                        paragraph=False,
                        text_threshold=0.6,
                        low_text=0.35,
                    )
                )
            features = summarize_ocr(image_name, results_by_view)
            raw_rows[image_name] = features
            append_jsonl(OCR_RAW_PATH, {"image": image_name, "features": features})
            print(f"easyocr {index}/{len(missing)}: {image_name}", flush=True)

    frame = pd.DataFrame([raw_rows[image_name] for image_name in image_names])
    frame.to_csv(OCR_FEATURES_PATH, index=False)
    return frame


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


def pool_view_features(view_features: np.ndarray) -> np.ndarray:
    return np.concatenate([view_features[0], view_features.mean(axis=0), view_features.std(axis=0)])


def ensure_hf_embeddings(rows: list[dict[str, str]], spec: dict[str, str]) -> np.ndarray:
    output_path = embedding_path(spec["name"])
    image_names = [row["image"] for row in rows]
    if output_path.exists():
        cached = np.load(output_path, allow_pickle=True)
        if cached["image_names"].tolist() == image_names:
            return cached["embeddings"]

    from transformers import AutoImageProcessor, AutoModel, AutoProcessor

    processor = (
        AutoImageProcessor.from_pretrained(spec["model_id"])
        if spec["kind"] == "vision_encoder"
        else AutoProcessor.from_pretrained(spec["model_id"])
    )
    model = AutoModel.from_pretrained(spec["model_id"]).to(DEVICE)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for index, image_name in enumerate(image_names, start=1):
            views = [crop_viewport(image_path(image_name), view_index) for view_index in range(VIEW_COUNT)]
            inputs = processor(images=views, return_tensors="pt")
            tensor_inputs = {key: value.to(DEVICE) for key, value in inputs.items() if torch.is_tensor(value)}
            if hasattr(model, "get_image_features"):
                features = tensor_from_model_output(model.get_image_features(**tensor_inputs))
            else:
                features = tensor_from_model_output(model(**tensor_inputs))
            embeddings.append(pool_view_features(features.detach().float().cpu().numpy()).astype(np.float32))
            print(f"{spec['name']} embeddings {index}/{len(rows)}: {image_name}", flush=True)

    matrix = np.vstack(embeddings)
    np.savez_compressed(output_path, image_names=np.asarray(image_names), embeddings=matrix)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return matrix


def mean_embedding_from_pooled(matrix: np.ndarray) -> np.ndarray:
    segment = matrix.shape[1] // 3
    return matrix[:, segment : segment * 2]


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
    for label, prompts in FACTUALITY_PROMPTS.items():
        prompt_texts.extend(prompts)
        prompt_labels.extend([label] * len(prompts))
    text_inputs = processor(text=prompt_texts, padding=True, return_tensors="pt")
    text_inputs = {key: value.to(DEVICE) for key, value in text_inputs.items() if torch.is_tensor(value)}
    with torch.no_grad():
        text_features = tensor_from_model_output(model.get_text_features(**text_inputs)).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    embeddings = np.load(embedding_path(spec["name"]), allow_pickle=True)["embeddings"]
    image_features = torch.tensor(mean_embedding_from_pooled(embeddings), device=DEVICE)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scores = (image_features @ text_features.T).detach().cpu().numpy()

    output_rows = []
    for row_index, image_name in enumerate(image_names):
        output: dict[str, Any] = {"image": image_name}
        means = {}
        for label in MULTICLASS_LABELS:
            indices = [index for index, prompt_label in enumerate(prompt_labels) if prompt_label == label]
            label_scores = scores[row_index, indices]
            output[f"{label}_prompt_mean"] = float(label_scores.mean())
            output[f"{label}_prompt_max"] = float(label_scores.max())
            means[label] = float(label_scores.mean())
        sorted_scores = sorted(means.values(), reverse=True)
        output["prompt_top2_margin"] = sorted_scores[0] - sorted_scores[1]
        output["prompt_high_low_axis"] = (
            means["very high"] + 0.5 * means["high"] - 0.5 * means["low"] - means["very low"]
        )
        output_rows.append(output)

    frame = pd.DataFrame(output_rows)
    frame.to_csv(output_path, index=False)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return frame


def ensure_resnet_embeddings(rows: list[dict[str, str]]) -> np.ndarray:
    image_names = [row["image"] for row in rows]
    if RESNET_EMBEDDINGS_PATH.exists():
        cached = np.load(RESNET_EMBEDDINGS_PATH, allow_pickle=True)
        if cached["image_names"].tolist() == image_names:
            return cached["embeddings"]

    from torchvision import models, transforms

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(DEVICE)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )
    embeddings = []
    with torch.no_grad():
        for index, image_name in enumerate(image_names, start=1):
            batch = torch.stack(
                [
                    transform(crop_viewport(image_path(image_name), view_index))
                    for view_index in range(VIEW_COUNT)
                ]
            ).to(DEVICE)
            features = model(batch).detach().cpu().numpy()
            embeddings.append(pool_view_features(features).astype(np.float32))
            print(f"resnet18 embeddings {index}/{len(rows)}: {image_name}", flush=True)

    matrix = np.vstack(embeddings)
    np.savez_compressed(RESNET_EMBEDDINGS_PATH, image_names=np.asarray(image_names), embeddings=matrix)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return matrix


def coerce_category(value: Any, allowed: list[str]) -> str:
    normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    by_normalized = {item.replace("-", "_"): item for item in allowed}
    return by_normalized.get(normalized, allowed[-1])


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


def flatten_semantic(image_name: str, raw: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"image": image_name}
    for field, allowed in CATEGORICAL_FIELDS.items():
        row[field] = coerce_category(raw.get(field, allowed[-1]), allowed)
    topics = raw.get("topics", {})
    if not isinstance(topics, dict):
        topics = {}
    for topic in TOPICS:
        row[f"topic_{topic}"] = coerce_bool(topics.get(topic, False))
    for field in BOOLEAN_FIELDS:
        row[field] = coerce_bool(raw.get(field, False))
    for field in NUMERIC_FIELDS:
        row[field] = coerce_int(raw.get(field, 0), 10 if field.endswith("_count") else 3)
    return row


def ensure_semantic_features(rows: list[dict[str, str]]) -> pd.DataFrame:
    image_names = [row["image"] for row in rows]
    raw_by_image = load_jsonl(SEMANTIC_RAW_PATH, "features")
    missing = [image_name for image_name in image_names if image_name not in raw_by_image]
    if missing:
        llm = OpenRouterLLM()

        def request(image_name: str) -> tuple[dict[str, Any], Any]:
            response = request_with_retry(llm, semantic_messages(image_name))
            return parse_json_response(extract_content(response)), response

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(request, image_name): image_name for image_name in missing}
            completed = len(raw_by_image)
            for future in concurrent.futures.as_completed(futures):
                image_name = futures[future]
                features, response = future.result()
                raw_by_image[image_name] = features
                append_jsonl(SEMANTIC_RAW_PATH, {"image": image_name, "features": features})
                append_usage(usage_row(image_name, "semantic_features", response))
                completed += 1
                print(f"GPT semantic {completed}/{len(rows)}: {image_name}", flush=True)

    frame = pd.DataFrame([flatten_semantic(image_name, raw_by_image[image_name]) for image_name in image_names])
    frame.to_csv(SEMANTIC_FEATURES_PATH, index=False)
    return frame


def main() -> None:
    rows = build_dataset()
    ensure_gpt_predictions(rows)
    ensure_visual_features(rows)
    ensure_ocr_features(rows)
    for spec in HF_IMAGE_MODELS:
        ensure_hf_embeddings(rows, spec)
        if spec["kind"] == "clip_like":
            ensure_prompt_scores(rows, spec)
    ensure_resnet_embeddings(rows)
    ensure_semantic_features(rows)
    print("factuality feature preparation complete", flush=True)


if __name__ == "__main__":
    main()
