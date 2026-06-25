from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.image_utils import image_to_data_url


MAX_TEXT_CANDIDATES = 45
MAX_LAYOUT_CANDIDATES = 12
MIN_OCR_CONFIDENCE = 0.25

SIGNAL_TERMS = {
    "bias": [
        "liberal",
        "conservative",
        "democrat",
        "republican",
        "progressive",
        "patriot",
        "freedom",
        "woke",
        "maga",
        "left",
        "right",
        "trump",
        "biden",
        "election",
        "immigration",
        "abortion",
        "socialist",
    ],
    "factuality": [
        "fact",
        "fact-check",
        "science",
        "evidence",
        "study",
        "research",
        "truth",
        "hoax",
        "conspiracy",
        "exposed",
        "shocking",
        "breaking",
        "censored",
        "secret",
        "miracle",
    ],
}


@dataclass
class EvidenceRegion:
    id: str
    kind: str
    role: str
    bbox: list[int]
    text: str = ""
    confidence: float | None = None
    score: float = 0.0


class PaddleEvidenceExtractor:
    def __init__(self) -> None:
        self._ocr: Any | None = None
        self._lock = threading.Lock()

    def extract_text(self, images: list[Image.Image]) -> str:
        pieces: list[str] = []
        for image in images:
            rows = self._ocr_rows(image)
            pieces.extend(row["text"] for row in rows)
        return " ".join(pieces)

    def build_regions(self, image: Image.Image) -> list[EvidenceRegion]:
        rows = self._ocr_rows(image)
        text_regions = choose_text_regions(rows, image.size, MAX_TEXT_CANDIDATES)
        layout_regions = standard_layout_regions(*image.size)
        layout_regions.extend(
            detected_layout_regions(
                image,
                layout_regions,
                max(0, MAX_LAYOUT_CANDIDATES - len(layout_regions)),
            )
        )
        return text_regions + layout_regions

    def _ocr_rows(self, image: Image.Image) -> list[dict[str, Any]]:
        ocr = self._reader()
        array = np.asarray(image.convert("RGB"))

        if hasattr(ocr, "predict"):
            raw = list(ocr.predict(array))
        elif hasattr(ocr, "ocr"):
            try:
                raw = ocr.ocr(array, cls=True)
            except TypeError:
                raw = ocr.ocr(array)
        else:
            raise RuntimeError("PaddleOCR object exposes neither predict() nor ocr()")

        return parse_paddle_results(raw, image.size)

    def _reader(self) -> Any:
        with self._lock:
            if self._ocr is not None:
                return self._ocr

            from paddleocr import PaddleOCR

            constructor_attempts = [
                {
                    "lang": "en",
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": False,
                },
                {"lang": "en", "use_angle_cls": True, "show_log": False},
                {"lang": "en"},
            ]
            errors: list[str] = []
            for kwargs in constructor_attempts:
                try:
                    self._ocr = PaddleOCR(**kwargs)
                    return self._ocr
                except TypeError as exc:
                    errors.append(str(exc))
                    continue

            raise RuntimeError("Could not initialize PaddleOCR: " + " | ".join(errors))


def parse_paddle_results(raw: Any, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    width, height = image_size
    rows: list[dict[str, Any]] = []
    for item in flatten_result_items(raw):
        data = result_to_dict(item)
        if data:
            rows.extend(parse_result_dict(data, width, height))
            continue
        rows.extend(parse_legacy_line(item, width, height))
    return rows


def flatten_result_items(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)):
        return []
    if isinstance(raw, dict):
        return [raw]
    if hasattr(raw, "json") or hasattr(raw, "res"):
        return [raw]
    if isinstance(raw, (list, tuple)):
        items: list[Any] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and item and not looks_like_legacy_line(item):
                items.extend(flatten_result_items(item))
            else:
                items.append(item)
        return items
    return [raw]


def looks_like_legacy_line(item: Any) -> bool:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return False
    second = item[1]
    return isinstance(second, (list, tuple)) and len(second) >= 2 and isinstance(second[0], str)


def result_to_dict(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        return item
    value = getattr(item, "json", None)
    if callable(value):
        try:
            value = value()
        except TypeError:
            value = None
    if isinstance(value, dict):
        if "res" in value and isinstance(value["res"], dict):
            return value["res"]
        return value
    value = getattr(item, "res", None)
    if isinstance(value, dict):
        return value
    return None


def parse_result_dict(data: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    texts = first_present(data, ["rec_texts", "texts"], [])
    scores = first_present(data, ["rec_scores", "scores"], [])
    polys = first_present(data, ["rec_polys", "dt_polys", "polys"], None)
    boxes = first_present(data, ["rec_boxes", "boxes"], None)
    rows: list[dict[str, Any]] = []
    texts_list = list(texts)
    scores_list = list(scores)
    for index, text in enumerate(texts_list):
        confidence = float(scores_list[index]) if index < len(scores_list) else 1.0
        text = clean_text(str(text))
        if confidence < MIN_OCR_CONFIDENCE or len(text) < 2:
            continue
        if polys is not None and index < len(polys):
            bbox = polygon_to_bbox(polys[index], width, height)
        elif boxes is not None and index < len(boxes):
            bbox = box_to_bbox(boxes[index], width, height)
        else:
            continue
        if bbox_area(bbox) >= 20:
            rows.append({"text": text, "bbox": bbox, "confidence": confidence})
    return rows


def first_present(data: dict[str, Any], keys: list[str], default: Any) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def parse_legacy_line(item: Any, width: int, height: int) -> list[dict[str, Any]]:
    if not looks_like_legacy_line(item):
        return []
    points = item[0]
    text = clean_text(str(item[1][0]))
    confidence = float(item[1][1])
    if confidence < MIN_OCR_CONFIDENCE or len(text) < 2:
        return []
    bbox = polygon_to_bbox(points, width, height)
    return [{"text": text, "bbox": bbox, "confidence": confidence}] if bbox_area(bbox) >= 20 else []


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def box_to_bbox(box: Any, width: int, height: int) -> list[int]:
    values = np.asarray(box, dtype=np.float32).reshape(-1).tolist()
    if len(values) >= 4:
        return clamp_bbox(values[:4], width, height)
    return polygon_to_bbox(box, width, height)


def polygon_to_bbox(points: Any, width: int, height: int) -> list[int]:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 2)
    x1 = float(np.min(array[:, 0]))
    y1 = float(np.min(array[:, 1]))
    x2 = float(np.max(array[:, 0]))
    y2 = float(np.max(array[:, 1]))
    return clamp_bbox([x1, y1, x2, y2], width, height)


def clamp_bbox(box: list[float] | tuple[float, ...], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]


def bbox_area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def bbox_iou(a: list[int], b: list[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / union if union else 0.0


def text_signal_score(text: str) -> float:
    lowered = text.lower()
    score = 0.0
    for group in SIGNAL_TERMS.values():
        score += sum(1.0 for term in group if term in lowered)
    score += min(1.5, len(text) / 80)
    if text.isupper() and len(text) > 8:
        score += 0.4
    if any(mark in text for mark in ["!", "?", ":"]):
        score += 0.25
    return score


def score_text_region(text: str, box: list[int], confidence: float, image_size: tuple[int, int]) -> float:
    width, height = image_size
    area_ratio = bbox_area(box) / max(1, width * height)
    top_bonus = 1.0 - min(1.0, box[1] / max(1, height)) * 0.45
    return confidence + math.log1p(area_ratio * 4000) * 0.35 + text_signal_score(text) + top_bonus


def choose_text_regions(rows: list[dict[str, Any]], image_size: tuple[int, int], limit: int) -> list[EvidenceRegion]:
    seen: set[str] = set()
    scored: list[EvidenceRegion] = []
    for row in rows:
        text = clean_text(str(row["text"]))
        norm = re.sub(r"[^a-z0-9]+", "", text.lower())
        if len(norm) < 2 or norm in seen:
            continue
        seen.add(norm)
        box = list(row["bbox"])
        confidence = float(row["confidence"])
        score = score_text_region(text, box, confidence, image_size)
        scored.append(EvidenceRegion("", "text", "ocr_text", box, text=text, confidence=confidence, score=score))

    scored = sorted(scored, key=lambda item: item.score, reverse=True)[:limit]
    scored = sorted(scored, key=lambda item: (item.bbox[1], item.bbox[0]))
    for index, item in enumerate(scored, start=1):
        item.id = f"T{index:03d}"
    return scored


def standard_layout_regions(width: int, height: int) -> list[EvidenceRegion]:
    specs = [
        ("header_navigation", [0, 0, width, round(height * 0.15)]),
        ("top_hero_area", [0, round(height * 0.10), width, round(height * 0.58)]),
        ("lower_content_grid", [0, round(height * 0.50), width, height]),
        ("left_content_column", [0, round(height * 0.15), round(width * 0.52), height]),
        ("right_content_column", [round(width * 0.48), round(height * 0.15), width, height]),
    ]
    return [
        EvidenceRegion(f"R{index:03d}", "layout", role, clamp_bbox(box, width, height), score=1.0)
        for index, (role, box) in enumerate(specs, start=1)
    ]


def detected_layout_regions(image: Image.Image, existing: list[EvidenceRegion], limit: int) -> list[EvidenceRegion]:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = min(1.0, 900 / max(width, height))
    small = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 60, 150)
    colored = ((hsv[:, :, 1] > 35) & (hsv[:, :, 2] < 248)).astype(np.uint8) * 255
    dark = (gray < 235).astype(np.uint8) * 255
    mask = cv2.bitwise_or(edges, cv2.bitwise_or(colored, dark))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[list[int], float]] = []
    min_area = width * height * 0.015
    max_area = width * height * 0.55
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = clamp_bbox([x / scale, y / scale, (x + w) / scale, (y + h) / scale], width, height)
        area = bbox_area(box)
        if area < min_area or area > max_area:
            continue
        if box[2] - box[0] < 90 or box[3] - box[1] < 45:
            continue
        if any(bbox_iou(box, region.bbox) > 0.72 for region in existing):
            continue
        boxes.append((box, area))

    boxes = sorted(boxes, key=lambda item: item[1], reverse=True)[:limit]
    boxes = sorted(boxes, key=lambda item: (item[0][1], item[0][0]))
    start = len(existing) + 1
    return [
        EvidenceRegion(f"R{start + index:03d}", "layout", "detected_visual_block", box, score=area)
        for index, (box, area) in enumerate(boxes)
    ]


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ["DejaVuSans.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, box: list[int], label: str, color: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    x1, y1, _, _ = box
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    y = y1 - text_height - 5 if y1 > text_height + 8 else y1 + 3
    draw.rectangle([x1, y, x1 + text_width + 8, y + text_height + 6], fill=color)
    draw.text((x1 + 4, y + 3), label, fill=(255, 255, 255), font=font)


def render_candidate_image(image: Image.Image, regions: list[EvidenceRegion]) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    font = load_font(max(12, round(image.width / 120)))
    for region in regions:
        fill = (37, 99, 235, 18) if region.kind == "text" else (234, 88, 12, 22)
        overlay_draw.rectangle(region.bbox, fill=fill)

    canvas.alpha_composite(overlay)
    draw = ImageDraw.Draw(canvas)
    for region in regions:
        color = (37, 99, 235) if region.kind == "text" else (234, 88, 12)
        draw.rectangle(region.bbox, outline=color, width=2)
        draw_label(draw, region.bbox, region.id, color, font)
    return canvas.convert("RGB")


def candidate_data_url(image: Image.Image, regions: list[EvidenceRegion], max_width: int) -> str:
    return image_to_data_url(render_candidate_image(image, regions), max_width)


def regions_prompt_table(regions: list[EvidenceRegion]) -> str:
    lines = []
    for region in regions:
        if region.kind == "text":
            lines.append(
                f'{region.id}: OCR text, confidence={region.confidence:.2f}, text="{truncate(region.text, 150)}"'
            )
        else:
            lines.append(f"{region.id}: layout/visual region, role={region.role}, bbox={region.bbox}")
    return "\n".join(lines)


def truncate(value: str, limit: int) -> str:
    value = clean_text(value)
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "..."


def evidence_regions_to_api(
    selected: list[dict[str, Any]],
    regions_by_id: dict[str, EvidenceRegion],
    crop_left: int,
    crop_top: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in selected:
        region_id = clean_text(str(item.get("id", ""))).upper()
        if region_id in seen or region_id not in regions_by_id:
            continue
        seen.add(region_id)
        region = regions_by_id[region_id]
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        bbox = [
            region.bbox[0] + crop_left,
            region.bbox[1] + crop_top,
            region.bbox[2] + crop_left,
            region.bbox[3] + crop_top,
        ]
        items.append(
            {
                "id": region.id,
                "kind": region.kind,
                "role": region.role,
                "bbox": bbox,
                "text": region.text or None,
                "importance": max(0.0, min(1.0, importance)),
                "reason": truncate(str(item.get("reason", "")), 220),
            }
        )
    return items
