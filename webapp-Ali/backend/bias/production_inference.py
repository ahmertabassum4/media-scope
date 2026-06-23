from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageStat

from .config import Settings
from .evidence import PaddleEvidenceExtractor, candidate_data_url, evidence_regions_to_api, regions_prompt_table
from .image_utils import TARGET_ASPECT_RATIO, crop_to_first_screen, image_to_data_url, resize_for_llm
from .openrouter import OpenRouterVisionClient


BIAS_LABELS = ["left", "left-center", "least biased", "right-center", "right"]
FACTUALITY_LABELS = ["very low", "low", "high", "very high"]
VIEW_COUNT = 4
OCR_VIEW_COUNT = 2

BIAS_PROMPT = (
    "Classify the political bias of this website based only on the supplied landing-page screenshot. "
    "Do not browse the web, do not search online, and do not use outside knowledge about the outlet. "
    "Use only visible cues such as headlines, branding, navigation, language, topics, and layout. "
    "Return exactly one lowercase label with no explanation: left, left-center, least biased, right-center, or right."
)

FACTUALITY_PROMPT = (
    "Classify the factuality of this website using only the supplied landing-page screenshot. "
    "Do not browse the web and do not use prior knowledge about the outlet. Judge only visible "
    "signals such as sourcing, attribution, evidence, transparency, headline style, and unsupported "
    "or sensational claims. Return exactly one lowercase label with no explanation: "
    "very low, low, high, or very high."
)

EVIDENCE_INSTRUCTIONS = """
The first image is the original screenshot crop.
The second image is the same screenshot with candidate evidence regions.
Blue Txxx boxes are OCR text candidates. Orange Rxxx boxes are broader visual/layout candidates.

Return strict JSON only:
{"label": "<one allowed label>", "evidence": [{"id": "T001", "importance": 0.0, "reason": "short visible reason"}]}

Rules:
- choose exactly one allowed label;
- select 2 to 7 evidence items;
- every evidence id must come from the candidate evidence list;
- use only visible screenshot evidence;
- do not use web search, publisher reputation, domain knowledge, or any information outside the screenshot.
""".strip()

FACTUALITY_SOURCE_TYPES = [
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
FACTUALITY_CONTENT_FORMATS = [
    "reported_news",
    "analysis",
    "research_reports",
    "opinion_commentary",
    "aggregated_content",
    "advocacy_campaign",
    "institutional_information",
    "mixed_or_unclear",
]
FACTUALITY_EVIDENCE_STYLES = [
    "strong_visible_attribution",
    "some_visible_attribution",
    "mostly_assertive_without_visible_support",
    "unclear",
]
FACTUALITY_HEADLINE_STYLES = ["straight", "analytical", "promotional", "sensational", "mixed_or_unclear"]
FACTUALITY_TONES = ["neutral", "analytical", "advocacy", "alarmist", "promotional", "mixed_or_unclear"]
FACTUALITY_TOPICS = [
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
FACTUALITY_CATEGORICAL_FIELDS = {
    "source_type": FACTUALITY_SOURCE_TYPES,
    "main_content_format": FACTUALITY_CONTENT_FORMATS,
    "evidence_style": FACTUALITY_EVIDENCE_STYLES,
    "headline_style": FACTUALITY_HEADLINE_STYLES,
    "overall_tone": FACTUALITY_TONES,
}

FACTUALITY_SEMANTIC_PROMPT = f"""
Extract structured visible-page features for factuality classification.

Use only the supplied landing-page screenshots. Do not browse the web and do not use prior
knowledge about the outlet. Do not output a factuality label. Describe only visible cues for
a downstream classifier.

Return valid compact JSON with exactly these keys:
- source_type: one of {FACTUALITY_SOURCE_TYPES}
- main_content_format: one of {FACTUALITY_CONTENT_FORMATS}
- evidence_style: one of {FACTUALITY_EVIDENCE_STYLES}
- headline_style: one of {FACTUALITY_HEADLINE_STYLES}
- overall_tone: one of {FACTUALITY_TONES}
- topics: object with exactly these boolean keys: {FACTUALITY_TOPICS}
- numeric fields: {FACTUALITY_NUMERIC_FIELDS}
- boolean fields: {FACTUALITY_BOOLEAN_FIELDS}

For level fields use integers 0..3: 0 absent, 1 weak, 2 clear, 3 strong.
For count fields use an approximate visible count from 0..10.
Return JSON only, without markdown.
""".strip()


class ProductionClassifier:
    def __init__(self, settings: Settings, model_dir: Path) -> None:
        self.settings = settings
        self.bias_artifact = joblib.load(model_dir / "bias_core_gpt55_ocr_text_siglip.joblib")
        # Factuality is served by Ali's MLP pipeline; this artifact is optional and only
        # loaded if present so the bias path works without Oleg's factuality model.
        factuality_path = model_dir / "factuality_semantic_visual_extra_trees.joblib"
        self.factuality_artifact = joblib.load(factuality_path) if factuality_path.exists() else None
        self.openrouter = (
            OpenRouterVisionClient(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                timeout_seconds=settings.request_timeout_seconds,
                retries=settings.request_retries,
            )
            if settings.openrouter_api_key
            else None
        )
        self.evidence_extractor = PaddleEvidenceExtractor()
        self._siglip: tuple[Any, Any, Any] | None = None
        self._model_lock = threading.Lock()

    def predict(self, image: Image.Image, upload_bytes: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.openrouter:
            error = "OPENROUTER_API_KEY is not configured"
            return self._error_result("bias", error), self._error_result("factuality", error)

        first_screen, crop_info = crop_to_first_screen(image)
        evidence_regions = self.evidence_extractor.build_regions(first_screen)
        regions_by_id = {region.id: region for region in evidence_regions}
        evidence_candidate_url = candidate_data_url(first_screen, evidence_regions, self.settings.max_llm_image_width)
        views = [crop_viewport(image, index) for index in range(VIEW_COUNT)]
        data_urls = [image_to_data_url(view, self.settings.max_llm_image_width) for view in views]

        bias_result = self.predict_bias(
            image,
            views,
            data_urls,
            evidence_regions,
            regions_by_id,
            evidence_candidate_url,
            int(crop_info["left"]),
            int(crop_info["top"]),
        )
        factuality_result = self.predict_factuality(
            image,
            upload_bytes,
            views,
            data_urls,
            evidence_regions,
            regions_by_id,
            evidence_candidate_url,
            int(crop_info["left"]),
            int(crop_info["top"]),
        )
        return bias_result, factuality_result

    def predict_bias(
        self,
        image: Image.Image,
        views: list[Image.Image],
        data_urls: list[str],
        evidence_regions: list[Any],
        regions_by_id: dict[str, Any],
        evidence_candidate_url: str,
        crop_left: int,
        crop_top: int,
    ) -> dict[str, Any]:
        try:
            gpt_label, evidence = self.request_label_evidence(
                task="bias",
                model=self.settings.bias_openrouter_model,
                prompt=BIAS_PROMPT,
                original_data_url=data_urls[0],
                candidate_data_url=evidence_candidate_url,
                allowed_labels=BIAS_LABELS,
                evidence_regions=evidence_regions,
                regions_by_id=regions_by_id,
                crop_left=crop_left,
                crop_top=crop_top,
            )
            ocr_text = self.extract_ocr_text(views[:OCR_VIEW_COUNT])
            siglip_embedding = self.extract_siglip_embedding(views)
            ocr_prediction = str(self.bias_artifact["ocr_model"].predict([ocr_text])[0])
            siglip_prediction = str(self.bias_artifact["siglip_model"].predict(siglip_embedding.reshape(1, -1))[0])
            meta_model = self.bias_artifact["meta_model"]
            meta_x = prediction_one_hot([gpt_label, ocr_prediction, siglip_prediction], BIAS_LABELS)
            label = str(meta_model.predict(meta_x)[0])
            proba = meta_model.predict_proba(meta_x)[0]
            confidences = {str(cls): float(p) for cls, p in zip(meta_model.classes_, proba)}
            return {
                "task": "bias",
                "label": label,
                "confidences": confidences,
                "llm_label": gpt_label,
                "model": self.bias_artifact["name"],
                "error": None,
                "evidence": evidence,
            }
        except Exception as exc:
            return self._error_result("bias", str(exc))

    def predict_factuality(
        self,
        image: Image.Image,
        upload_bytes: int,
        views: list[Image.Image],
        data_urls: list[str],
        evidence_regions: list[Any],
        regions_by_id: dict[str, Any],
        evidence_candidate_url: str,
        crop_left: int,
        crop_top: int,
    ) -> dict[str, Any]:
        try:
            baseline, evidence = self.request_label_evidence(
                task="factuality",
                model=self.settings.factuality_openrouter_model,
                prompt=FACTUALITY_PROMPT,
                original_data_url=data_urls[0],
                candidate_data_url=evidence_candidate_url,
                allowed_labels=FACTUALITY_LABELS,
                evidence_regions=evidence_regions,
                regions_by_id=regions_by_id,
                crop_left=crop_left,
                crop_top=crop_top,
            )
            raw_semantic, _ = self.openrouter.complete_vision(
                model=self.settings.factuality_openrouter_model,
                prompt=FACTUALITY_SEMANTIC_PROMPT,
                data_urls=data_urls,
            )
            semantic = flatten_factuality_semantic(parse_json_response(raw_semantic))
            visual = factuality_visual_features(image, upload_bytes, views)
            matrix = self.factuality_matrix(semantic, baseline, visual)
            label = str(self.factuality_artifact["model"].predict(matrix)[0])
            return {
                "task": "factuality",
                "label": label,
                "llm_label": baseline,
                "model": self.factuality_artifact["name"],
                "error": None,
                "evidence": evidence,
            }
        except Exception as exc:
            return self._error_result("factuality", str(exc))

    def request_label_evidence(
        self,
        *,
        task: str,
        model: str,
        prompt: str,
        original_data_url: str,
        candidate_data_url: str,
        allowed_labels: list[str],
        evidence_regions: list[Any],
        regions_by_id: dict[str, Any],
        crop_left: int,
        crop_top: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        evidence_prompt = (
            f"{prompt}\n\n"
            f"Allowed labels: {', '.join(allowed_labels)}.\n\n"
            f"{EVIDENCE_INSTRUCTIONS}\n\n"
            f"Candidate evidence list:\n{regions_prompt_table(evidence_regions)}"
        )
        raw, _ = self.openrouter.complete_vision(
            model=model,
            prompt=evidence_prompt,
            data_urls=[original_data_url, candidate_data_url],
            temperature=0,
            max_tokens=1800,
            response_format={"type": "json_object"},
        )
        parsed = parse_json_response(raw)
        label = normalize_label(str(parsed.get("label", "")), allowed_labels)
        selected = parsed.get("evidence", [])
        if not isinstance(selected, list):
            selected = []
        evidence = evidence_regions_to_api(selected, regions_by_id, crop_left, crop_top)
        return label, evidence

    def factuality_matrix(self, semantic: dict[str, Any], baseline: str, visual: dict[str, float]) -> np.ndarray:
        artifact = self.factuality_artifact
        category_matrix = artifact["category_vectorizer"].transform(
            [{field: semantic[field] for field in artifact["categorical_fields"]}]
        )
        numeric_matrix = np.asarray([[float(semantic.get(column, 0)) for column in artifact["numeric_columns"]]], dtype=np.float32)
        baseline_matrix = artifact["baseline_vectorizer"].transform([{"baseline": baseline}])
        visual_matrix = np.asarray([[float(visual.get(column, 0.0)) for column in artifact["visual_columns"]]], dtype=np.float32)
        return np.nan_to_num(
            np.hstack([category_matrix, numeric_matrix, baseline_matrix, visual_matrix]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def extract_ocr_text(self, views: list[Image.Image]) -> str:
        return self.evidence_extractor.extract_text([resize_for_llm(view, max_width=1280) for view in views])

    def extract_siglip_embedding(self, views: list[Image.Image]) -> np.ndarray:
        with self._model_lock:
            if self._siglip is None:
                import torch
                from transformers import AutoModel, AutoProcessor

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                processor = AutoProcessor.from_pretrained(self.bias_artifact["siglip_model_id"], use_fast=False)
                model = AutoModel.from_pretrained(self.bias_artifact["siglip_model_id"]).to(device)
                model.eval()
                self._siglip = (processor, model, device)

        import torch

        processor, model, device = self._siglip
        inputs = processor(images=views, return_tensors="pt")
        tensor_inputs = {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}
        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                features = tensor_from_model_output(model.get_image_features(**tensor_inputs))
            else:
                features = tensor_from_model_output(model(**tensor_inputs))
        return pool_view_features(features.detach().float().cpu().numpy()).astype(np.float32)

    def analyze_bias(self, image: Image.Image, upload_bytes: int) -> dict[str, Any]:
        """Bias-only entry point: runs the screenshot prep block then the bias ensemble.

        Mirrors the prep in predict() but skips factuality so Ali's MLP factuality
        path stays the sole factuality source. Returns the predict_bias dict.
        """
        if not self.openrouter:
            return self._error_result("bias", "OPENROUTER_API_KEY is not configured")

        first_screen, crop_info = crop_to_first_screen(image)
        evidence_regions = self.evidence_extractor.build_regions(first_screen)
        regions_by_id = {region.id: region for region in evidence_regions}
        evidence_candidate_url = candidate_data_url(first_screen, evidence_regions, self.settings.max_llm_image_width)
        views = [crop_viewport(image, index) for index in range(VIEW_COUNT)]
        data_urls = [image_to_data_url(view, self.settings.max_llm_image_width) for view in views]

        return self.predict_bias(
            image,
            views,
            data_urls,
            evidence_regions,
            regions_by_id,
            evidence_candidate_url,
            int(crop_info["left"]),
            int(crop_info["top"]),
        )

    def analyze_factuality_evidence(self, image: Image.Image) -> dict[str, Any]:
        """Factuality evidence-overlay entry point: the same first-screen prep as
        analyze_bias, then a single request_label_evidence call for the factuality
        task. Runs only the evidence path (no MLP / semantic / SigLIP) so /analyze
        stays fast and Ali's MLP remains the factuality label on the card.

        The returned label is the LLM's own pick from FACTUALITY_LABELS and may
        differ from the MLP label, exactly as on the bias path. original_data_url
        matches predict_factuality (data_urls[0] == crop_viewport(image, 0)).
        """
        if not self.openrouter:
            return {"task": "factuality", "label": None, "evidence": [], "error": "OPENROUTER_API_KEY is not configured"}

        first_screen, crop_info = crop_to_first_screen(image)
        evidence_regions = self.evidence_extractor.build_regions(first_screen)
        regions_by_id = {region.id: region for region in evidence_regions}
        evidence_candidate_url = candidate_data_url(first_screen, evidence_regions, self.settings.max_llm_image_width)
        original_data_url = image_to_data_url(crop_viewport(image, 0), self.settings.max_llm_image_width)

        try:
            label, evidence = self.request_label_evidence(
                task="factuality",
                model=self.settings.factuality_openrouter_model,
                prompt=FACTUALITY_PROMPT,
                original_data_url=original_data_url,
                candidate_data_url=evidence_candidate_url,
                allowed_labels=FACTUALITY_LABELS,
                evidence_regions=evidence_regions,
                regions_by_id=regions_by_id,
                crop_left=int(crop_info["left"]),
                crop_top=int(crop_info["top"]),
            )
        except Exception as exc:
            return {"task": "factuality", "label": None, "evidence": [], "error": str(exc)}
        return {"task": "factuality", "label": label, "evidence": evidence, "error": None}

    def _error_result(self, task: str, error: str) -> dict[str, Any]:
        return {"task": task, "label": None, "confidences": {}, "llm_label": None, "model": None, "error": error, "evidence": []}


def crop_viewport(image: Image.Image, viewport_index: int) -> Image.Image:
    width, height = image.size
    viewport_height = min(height, round(width / TARGET_ASPECT_RATIO))
    if viewport_height < height:
        top = min(viewport_index * viewport_height, max(0, height - viewport_height))
        return image.crop((0, top, width, top + viewport_height)).convert("RGB")

    viewport_width = min(width, round(height * TARGET_ASPECT_RATIO))
    left = (width - viewport_width) // 2
    return image.crop((left, 0, left + viewport_width, height)).convert("RGB")


def prediction_one_hot(predictions: list[str], labels: list[str]) -> np.ndarray:
    return np.asarray([[float(prediction == label) for prediction in predictions for label in labels]], dtype=np.float32)


def pool_view_features(view_features: np.ndarray) -> np.ndarray:
    return np.concatenate([view_features[0], view_features.mean(axis=0), view_features.std(axis=0)])


def tensor_from_model_output(output: Any) -> Any:
    for attribute in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attribute, None)
        if value is not None:
            return value
    value = getattr(output, "last_hidden_state", None)
    if value is not None:
        return value[:, 0]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def normalize_label(response_text: str, allowed_labels: list[str]) -> str:
    label = response_text.strip().lower().strip("\"'` .")
    label = re.sub(r"\s+", " ", label)
    label = re.sub(r"\s*-\s*", "-", label)
    compact_label = re.sub(r"[\s-]+", "", label)
    if label in allowed_labels:
        return label
    label_with_spaces = label.replace("-", " ")
    for allowed_label in allowed_labels:
        if label_with_spaces == allowed_label.replace("-", " "):
            return allowed_label
        if compact_label == allowed_label.replace("-", "").replace(" ", ""):
            return allowed_label
    matches = [
        allowed_label
        for allowed_label in allowed_labels
        if re.search(rf"(?<![a-z]){re.escape(allowed_label)}(?![a-z])", label)
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Unexpected label response: {response_text[:200]}")


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found: {text[:200]}")
    return json.loads(match.group(0))


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


def flatten_factuality_semantic(raw: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field, allowed in FACTUALITY_CATEGORICAL_FIELDS.items():
        row[field] = coerce_category(raw.get(field, allowed[-1]), allowed)
    topics = raw.get("topics", {})
    if not isinstance(topics, dict):
        topics = {}
    for topic in FACTUALITY_TOPICS:
        row[f"topic_{topic}"] = coerce_bool(topics.get(topic, False))
    for field in FACTUALITY_BOOLEAN_FIELDS:
        row[field] = coerce_bool(raw.get(field, False))
    for field in FACTUALITY_NUMERIC_FIELDS:
        row[field] = coerce_int(raw.get(field, 0), 10 if field.endswith("_count") else 3)
    return row


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


def factuality_visual_features(image: Image.Image, upload_bytes: int, views: list[Image.Image]) -> dict[str, float]:
    width, height = image.size
    features: dict[str, float] = {
        "image_width": float(width),
        "image_height": float(height),
        "image_aspect": float(width / height),
        "image_log_height": float(math.log1p(height)),
        "image_file_mb": float(upload_bytes / (1024 * 1024)),
    }
    for view_index in range(VIEW_COUNT):
        resized = views[view_index].resize((256, 144), Image.Resampling.BILINEAR).convert("RGB")
        features.update(visual_view_features(resized, f"view{view_index}"))
    return features
