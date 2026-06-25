from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image

from app.config import Settings
from app.evidence import PaddleEvidenceExtractor, candidate_data_url, evidence_regions_to_api, regions_prompt_table
from app.factuality_features import FactualityFeatures
from app.factuality_graph import GraphVisionFactuality
from app.factuality_llm import explain, joint_verdict
from app.image_utils import TARGET_ASPECT_RATIO, crop_to_first_screen, image_to_data_url, resize_for_llm
from app.openrouter import OpenRouterVisionClient


BIAS_LABELS = ["left", "left-center", "least biased", "right-center", "right"]
# The MLP factuality head is 5-class (includes "mixed").
FACTUALITY_LABELS = ["very low", "low", "mixed", "high", "very high"]
# The evidence-region call only needs a label to comply with; its label is discarded
# (the MLP wins), so we keep the original 4-class prompt wording.
FACTUALITY_EVIDENCE_LABELS = ["very low", "low", "high", "very high"]
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


class ProductionClassifier:
    def __init__(self, settings: Settings, model_dir: Path) -> None:
        self.settings = settings
        self.bias_artifact = joblib.load(model_dir / "bias_core_gpt55_ocr_text_siglip.joblib")
        config = json.loads((model_dir / "factuality_config.json").read_text(encoding="utf-8"))
        self.factuality = GraphVisionFactuality(model_dir)
        self.factuality_features = FactualityFeatures(model_dir, config)
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

    def predict(
        self,
        image: Image.Image,
        upload_bytes: int,
        provenance: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
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
            data_urls,
            evidence_regions,
            regions_by_id,
            evidence_candidate_url,
            int(crop_info["left"]),
            int(crop_info["top"]),
            provenance,
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
            meta_x = prediction_one_hot([gpt_label, ocr_prediction, siglip_prediction], BIAS_LABELS)
            label = str(self.bias_artifact["meta_model"].predict(meta_x)[0])
            return {
                "task": "bias",
                "label": label,
                "llm_label": gpt_label,
                "model": self.bias_artifact["name"],
                "error": None,
                "evidence": evidence,
                "confidences": None,
            }
        except Exception as exc:
            return self._error_result("bias", str(exc))

    def predict_factuality(
        self,
        image: Image.Image,
        data_urls: list[str],
        evidence_regions: list[Any],
        regions_by_id: dict[str, Any],
        evidence_candidate_url: str,
        crop_left: int,
        crop_top: int,
        provenance: dict[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            # Bounding-box evidence for the overlay; the label this call returns is discarded.
            try:
                _, evidence = self.request_label_evidence(
                    task="factuality",
                    model=self.settings.factuality_openrouter_model,
                    prompt=FACTUALITY_PROMPT,
                    original_data_url=data_urls[0],
                    candidate_data_url=evidence_candidate_url,
                    allowed_labels=FACTUALITY_EVIDENCE_LABELS,
                    evidence_regions=evidence_regions,
                    regions_by_id=regions_by_id,
                    crop_left=crop_left,
                    crop_top=crop_top,
                )
            except Exception:
                evidence = []

            # Joint verdict over the FULL screenshot -> verdict one-hot + rationale.
            full_data_url = image_to_data_url(image, self.settings.max_llm_image_width)
            verdicts, rationale = joint_verdict(
                self.openrouter, self.settings.factuality_openrouter_model, full_data_url
            )
            # Vision-only graph model: raw DINOv2 + verdict one-hot + rationale embedding,
            # diffused over the training corpus' visual@0.80 graph, then XGBoost. Provenance
            # is unused, so URL captures and image uploads share this path.
            raw_vis = self.factuality_features.visual_embedding(image)
            verdict_oh = self.factuality_features.verdict_onehot(verdicts)
            rat = self.factuality_features.rationale_embedding(rationale)
            label, confidences = self.factuality.predict(raw_vis, verdict_oh, rat)
            llm_label = (verdicts.get("factuality") or "").lower()
            return {
                "task": "factuality",
                "label": label,
                "llm_label": llm_label or None,
                "model": self.factuality.name,
                "error": None,
                "evidence": evidence,
                "confidences": confidences,
            }
        except Exception as exc:
            return self._error_result("factuality", str(exc))

    def reason(
        self,
        image: Image.Image,
        factuality_label: str | None,
        bias_label: str | None,
        evidence: list[dict[str, Any]] | None,
    ) -> str:
        if not self.openrouter:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        data_url = image_to_data_url(image, self.settings.max_llm_image_width)
        return explain(
            self.openrouter,
            self.settings.factuality_openrouter_model,
            data_url,
            factuality_label,
            bias_label,
            evidence,
        )

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

    def _error_result(self, task: str, error: str) -> dict[str, Any]:
        return {
            "task": task,
            "label": None,
            "llm_label": None,
            "model": None,
            "error": error,
            "evidence": [],
            "confidences": None,
        }


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
