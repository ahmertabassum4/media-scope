from experiment_bootstrap import activate_project_root

activate_project_root()

import argparse
import base64
import csv
import io
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from llm_client import OpenRouterLLM


Image.MAX_IMAGE_PIXELS = None

DEFAULT_MODEL = os.getenv("EVIDENCE_OPENROUTER_MODEL", "openai/gpt-5.5")
DEFAULT_IMAGE_DIR = Path("application/data/images")
DEFAULT_OUTPUT_DIR = Path("experiments/evidence_highlights")

TARGET_ASPECT_RATIO = 16 / 9
MAX_LLM_IMAGE_WIDTH = 1600
REQUEST_TIMEOUT = 180

BIAS_LABELS = ["left", "left-center", "least biased", "right-center", "right"]
FACTUALITY_LABELS = ["very low", "low", "high", "very high"]
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

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


def normalize_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    if image.mode in {"RGBA", "LA"}:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def crop_to_first_screen(image: Image.Image) -> tuple[Image.Image, dict[str, int | float]]:
    width, height = image.size
    target_height = min(height, round(width / TARGET_ASPECT_RATIO))
    if target_height < height:
        box = (0, 0, width, target_height)
    else:
        target_width = min(width, round(height * TARGET_ASPECT_RATIO))
        left = (width - target_width) // 2
        box = (left, 0, left + target_width, height)

    crop = image.crop(box)
    left, top, right, bottom = box
    return crop, {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": crop.width,
        "height": crop.height,
        "aspect_ratio": round(crop.width / crop.height, 6),
    }


def image_to_data_url(image: Image.Image, max_width: int = MAX_LLM_IMAGE_WIDTH) -> str:
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return normalize_rgb(image)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unreadable image: {path}") from exc


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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


def polygon_to_bbox(points: Any, width: int, height: int) -> list[int]:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 2)
    x1 = float(np.min(array[:, 0]))
    y1 = float(np.min(array[:, 1]))
    x2 = float(np.max(array[:, 0]))
    y2 = float(np.max(array[:, 1]))
    return clamp_bbox([x1, y1, x2, y2], width, height)


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


def extract_paddleocr_regions(image: Image.Image, min_confidence: float) -> tuple[list[dict[str, Any]], str]:
    from paddleocr import PaddleOCR

    width, height = image.size
    rows: list[dict[str, Any]] = []
    array = np.asarray(image)

    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        raw = ocr.ocr(array, cls=True)
    except TypeError:
        ocr = PaddleOCR(lang="en")
        raw = ocr.ocr(array)

    pages = raw if isinstance(raw, list) else [raw]
    for page in pages:
        if isinstance(page, dict):
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            polys = page.get("rec_polys") or page.get("dt_polys") or []
            for text, confidence, poly in zip(texts, scores, polys):
                text = clean_text(str(text))
                confidence = float(confidence)
                if confidence >= min_confidence and len(text) >= 2:
                    rows.append({"text": text, "bbox": polygon_to_bbox(poly, width, height), "confidence": confidence})
            continue

        for line in page or []:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            points = line[0]
            text_info = line[1]
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                text, confidence = clean_text(str(text_info[0])), float(text_info[1])
            else:
                continue
            if confidence >= min_confidence and len(text) >= 2:
                rows.append({"text": text, "bbox": polygon_to_bbox(points, width, height), "confidence": confidence})

    return rows, "paddleocr"


def extract_ocr_regions(image: Image.Image, engine: str, min_confidence: float) -> tuple[list[dict[str, Any]], str, list[str]]:
    errors: list[str] = []
    if engine == "paddleocr":
        try:
            rows, name = extract_paddleocr_regions(image, min_confidence)
            return rows, name, errors
        except Exception as exc:
            errors.append(f"paddleocr failed: {exc}")
            raise RuntimeError(errors[-1]) from exc

    raise ValueError(f"Unsupported OCR engine: {engine}")


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
    regions = []
    for index, (role, box) in enumerate(specs, start=1):
        regions.append(EvidenceRegion(f"R{index:03d}", "layout", role, clamp_bbox(box, width, height), score=1.0))
    return regions


def detected_layout_regions(image: Image.Image, existing: list[EvidenceRegion], limit: int) -> list[EvidenceRegion]:
    rgb = np.asarray(image)
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


def build_regions(
    image: Image.Image,
    engine: str,
    min_confidence: float,
    max_text_candidates: int,
    max_layout_candidates: int,
) -> tuple[list[EvidenceRegion], dict[str, Any]]:
    ocr_rows, ocr_engine, ocr_errors = extract_ocr_regions(image, engine, min_confidence)
    text_regions = choose_text_regions(ocr_rows, image.size, max_text_candidates)
    layout_regions = standard_layout_regions(*image.size)
    layout_regions.extend(detected_layout_regions(image, layout_regions, max(0, max_layout_candidates - len(layout_regions))))
    regions = text_regions + layout_regions
    meta = {
        "ocr_engine": ocr_engine,
        "ocr_errors": ocr_errors,
        "raw_ocr_boxes": len(ocr_rows),
        "text_candidates": len(text_regions),
        "layout_candidates": len(layout_regions),
    }
    return regions, meta


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
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
        if region.kind == "text":
            color = (37, 99, 235)
            fill = (37, 99, 235, 18)
        else:
            color = (234, 88, 12)
            fill = (234, 88, 12, 22)
        overlay_draw.rectangle(region.bbox, fill=fill)

    canvas.alpha_composite(overlay)
    draw = ImageDraw.Draw(canvas)
    for region in regions:
        color = (37, 99, 235) if region.kind == "text" else (234, 88, 12)
        draw.rectangle(region.bbox, outline=color, width=2)
        draw_label(draw, region.bbox, region.id, color, font)

    return canvas.convert("RGB")


def render_selected_highlight(
    image: Image.Image,
    regions_by_id: dict[str, EvidenceRegion],
    task: str,
    label: str,
    evidence: list[dict[str, Any]],
) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    font = load_font(max(14, round(image.width / 95)))
    title_font = load_font(max(18, round(image.width / 70)))
    color = (37, 99, 235) if task == "bias" else (15, 118, 110)

    for item in evidence:
        region = regions_by_id.get(str(item.get("id", "")))
        if not region:
            continue
        alpha = 44 + int(70 * max(0.0, min(1.0, float(item.get("importance", 0.5)))))
        overlay_draw.rectangle(region.bbox, fill=(*color, alpha))

    canvas.alpha_composite(overlay)
    draw = ImageDraw.Draw(canvas)
    title = f"{task.capitalize()}: {label}"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.rectangle([12, 12, title_box[2] + 32, title_box[3] + 28], fill=(20, 25, 34, 220))
    draw.text((22, 20), title, fill=(255, 255, 255), font=title_font)

    for index, item in enumerate(evidence, start=1):
        region = regions_by_id.get(str(item.get("id", "")))
        if not region:
            continue
        draw.rectangle(region.bbox, outline=color, width=5)
        draw_label(draw, region.bbox, f"{index}. {region.id}", color, font)

    return canvas.convert("RGB")


def truncate(value: str, limit: int) -> str:
    value = clean_text(value)
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "..."


def candidate_prompt_table(regions: list[EvidenceRegion]) -> str:
    lines = []
    for region in regions:
        if region.kind == "text":
            lines.append(
                f'{region.id}: OCR text, confidence={region.confidence:.2f}, text="{truncate(region.text, 150)}"'
            )
        else:
            lines.append(f"{region.id}: layout/visual region, role={region.role}, bbox={region.bbox}")
    return "\n".join(lines)


def task_schema(task: str) -> str:
    if task == "bias":
        return '"bias": {"label": one of the bias labels, "evidence": [...] }'
    if task == "factuality":
        return '"factuality": {"label": one of the factuality labels, "evidence": [...] }'
    return (
        '"bias": {"label": one of the bias labels, "evidence": [...] }, '
        '"factuality": {"label": one of the factuality labels, "evidence": [...] }'
    )


def build_prompt(task: str, regions: list[EvidenceRegion]) -> str:
    return f"""
You are classifying a media landing-page screenshot using only visible evidence in the provided screenshot.
Do not use web search, publisher reputation, domain knowledge, or any information outside the screenshot.

The first image is the original screenshot.
The second image is the same screenshot with candidate evidence regions.
Blue Txxx boxes are OCR text candidates. Orange Rxxx boxes are broader visual/layout candidates.

Allowed bias labels: {", ".join(BIAS_LABELS)}.
Allowed factuality labels: {", ".join(FACTUALITY_LABELS)}.

Return strict JSON only, with this structure:
{{{task_schema(task)}}}

For every requested task:
- choose exactly one label from the allowed labels;
- select 2 to 7 evidence items;
- each evidence item must use an id from the candidate list below;
- each evidence item must include "id", "importance" from 0 to 1, and a short "reason";
- do not invent IDs or quote text that is not visible.

Candidate evidence list:
{candidate_prompt_table(regions)}
""".strip()


def response_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return json.dumps(content, ensure_ascii=False)


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "response_id": getattr(response, "id", None),
    }


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_label(label: Any, valid_labels: list[str]) -> str:
    text = clean_text(str(label)).lower().replace("_", " ")
    text = text.replace("center", "center")
    for valid in valid_labels:
        if text == valid:
            return valid
    for valid in valid_labels:
        if valid in text:
            return valid
    return text


def normalize_evidence(items: Any, regions_by_id: dict[str, EvidenceRegion]) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    cleaned = []
    used: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        region_id = clean_text(str(item.get("id", ""))).upper()
        if region_id not in regions_by_id or region_id in used:
            continue
        used.add(region_id)
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        cleaned.append(
            {
                "id": region_id,
                "importance": max(0.0, min(1.0, importance)),
                "reason": truncate(str(item.get("reason", "")), 220),
            }
        )
    return cleaned


def normalize_llm_result(data: dict[str, Any], task: str, regions_by_id: dict[str, EvidenceRegion]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    tasks = ["bias", "factuality"] if task == "both" else [task]
    for current in tasks:
        raw = data.get(current, {})
        if not isinstance(raw, dict):
            raw = {}
        labels = BIAS_LABELS if current == "bias" else FACTUALITY_LABELS
        label = normalize_label(raw.get("label", ""), labels)
        normalized[current] = {
            "label": label,
            "evidence": normalize_evidence(raw.get("evidence", []), regions_by_id),
        }
    return normalized


def request_evidence(
    llm: OpenRouterLLM,
    model: str,
    task: str,
    original: Image.Image,
    candidates: Image.Image,
    regions: list[EvidenceRegion],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt(task, regions)},
                {"type": "image_url", "image_url": {"url": image_to_data_url(original)}},
                {"type": "image_url", "image_url": {"url": image_to_data_url(candidates)}},
            ],
        }
    ]
    try:
        response = llm.openr_llm(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=2200,
            response_format={"type": "json_object"},
            timeout=REQUEST_TIMEOUT,
        )
    except Exception:
        response = llm.openr_llm(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=2200,
            timeout=REQUEST_TIMEOUT,
        )
    text = response_text(response)
    parsed = parse_json_response(text)
    return parsed, text, usage_dict(response)


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "image"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def process_image(
    image_path: Path,
    run_dir: Path,
    llm: OpenRouterLLM | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image = load_image(image_path)
    crop, crop_meta = crop_to_first_screen(image)
    regions, region_meta = build_regions(
        crop,
        engine=args.ocr_engine,
        min_confidence=args.min_ocr_confidence,
        max_text_candidates=args.max_text_candidates,
        max_layout_candidates=args.max_layout_candidates,
    )
    regions_by_id = {region.id: region for region in regions}
    candidates_image = render_candidate_image(crop, regions)
    stem = safe_stem(image_path)

    crop_path = run_dir / f"{stem}_crop.png"
    candidates_path = run_dir / f"{stem}_candidates.png"
    result_path = run_dir / f"{stem}_result.json"
    crop.save(crop_path)
    candidates_image.save(candidates_path)

    raw_result: dict[str, Any] = {}
    raw_text = ""
    usage: dict[str, Any] = {}
    normalized: dict[str, Any] = {}
    errors: list[str] = []

    if llm is not None:
        try:
            raw_result, raw_text, usage = request_evidence(llm, args.model, args.task, crop, candidates_image, regions)
            normalized = normalize_llm_result(raw_result, args.task, regions_by_id)
        except Exception as exc:
            errors.append(f"llm failed: {exc}")

    highlights: dict[str, str] = {}
    for task, task_result in normalized.items():
        highlight = render_selected_highlight(
            crop,
            regions_by_id,
            task,
            str(task_result.get("label", "")),
            list(task_result.get("evidence", [])),
        )
        highlight_path = run_dir / f"{stem}_{task}_highlight.png"
        highlight.save(highlight_path)
        highlights[task] = str(highlight_path)

    payload = {
        "image": str(image_path),
        "model": args.model,
        "task": args.task,
        "crop": crop_meta,
        "region_meta": region_meta,
        "regions": [asdict(region) for region in regions],
        "result": normalized,
        "raw_result": raw_result,
        "raw_text": raw_text,
        "usage": usage,
        "errors": errors,
        "artifacts": {
            "crop": str(crop_path),
            "candidates": str(candidates_path),
            "highlights": highlights,
        },
    }
    write_json(result_path, payload)
    return payload


def select_images(args: argparse.Namespace) -> list[Path]:
    if args.image:
        paths = [Path(value) for value in args.image]
    else:
        image_dir = Path(args.image_dir)
        paths = [path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in SUPPORTED_EXTENSIONS]

    if args.limit >= 0:
        paths = paths[: args.limit]
    return paths


def write_summary(run_dir: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Evidence Highlighting Run",
        "",
        f"Images: {len(results)}",
        "",
    ]
    for result in results:
        lines.append(f"## {Path(result['image']).name}")
        lines.append("")
        lines.append(f"- OCR engine: `{result['region_meta']['ocr_engine']}`")
        lines.append(f"- Text candidates: `{result['region_meta']['text_candidates']}`")
        lines.append(f"- Layout candidates: `{result['region_meta']['layout_candidates']}`")
        for task, task_result in result.get("result", {}).items():
            lines.append(f"- {task}: `{task_result.get('label', '')}`")
            for index, item in enumerate(task_result.get("evidence", []), start=1):
                lines.append(f"  - {index}. `{item['id']}` importance `{item['importance']:.2f}`: {item['reason']}")
        lines.append("")
        lines.append(f"Candidates: `{Path(result['artifacts']['candidates']).name}`")
        for task, path in result["artifacts"]["highlights"].items():
            lines.append(f"{task.capitalize()} highlight: `{Path(path).name}`")
        if result.get("errors"):
            lines.append(f"Errors: `{'; '.join(result['errors'])}`")
        lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_usage(run_dir: Path, results: list[dict[str, Any]]) -> None:
    with (run_dir / "usage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "model", "prompt_tokens", "completion_tokens", "total_tokens", "response_id"],
        )
        writer.writeheader()
        for result in results:
            usage = result.get("usage", {})
            writer.writerow(
                {
                    "image": Path(result["image"]).name,
                    "model": result["model"],
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "response_id": usage.get("response_id"),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ground LLM labels in OCR/layout evidence and render highlight overlays.")
    parser.add_argument("--image", action="append", help="Image path. Can be passed multiple times.")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR), help="Directory used when --image is not provided.")
    parser.add_argument("--limit", type=int, default=1, help="Number of images to process from --image-dir. Use -1 for all.")
    parser.add_argument("--task", choices=["bias", "factuality", "both"], default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ocr-engine", choices=["paddleocr"], default="paddleocr")
    parser.add_argument("--min-ocr-confidence", type=float, default=0.25)
    parser.add_argument("--max-text-candidates", type=int, default=45)
    parser.add_argument("--max-layout-candidates", type=int, default=12)
    parser.add_argument("--no-llm", action="store_true", help="Only render OCR/layout candidate boxes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = select_images(args)
    if not images:
        raise SystemExit("No images found")

    run_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    llm = None if args.no_llm else OpenRouterLLM()

    results = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path}", flush=True)
        results.append(process_image(image_path, run_dir, llm, args))

    write_summary(run_dir, results)
    write_usage(run_dir, results)
    print(f"wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
