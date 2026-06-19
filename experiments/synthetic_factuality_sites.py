from experiment_bootstrap import activate_project_root

activate_project_root()

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from factuality_common import TARGET_ASPECT_RATIO, crop_viewport, image_path, load_dataset, write_csv


Image.MAX_IMAGE_PIXELS = None

OUTPUT_DIR = Path("synthetic-factuality-samples")
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
MANIFEST_PATH = OUTPUT_DIR / "synthetic_factuality_manifest.csv"
OCR_BOX_CACHE_PATH = OUTPUT_DIR / "synthetic_factuality_ocr_boxes.jsonl"
DEFAULT_LIMIT = 10
REPLACEMENTS_PER_IMAGE = 4
RANDOM_STATE = 171

HIGH_BASE_PRIORITY = [
    "FactCheck.org",
    "Pew_Research",
    "Haaretz",
    "Boston.com",
    "Bleeping_Computer",
    "Investopedia",
    "Air_Force_Times",
    "ABC_News_Washington_Post_Polling",
    "100_Mile_Free_Press",
    "PA_Media",
]

HIGH_DONOR_PRIORITY = [
    "FactCheck.org",
    "Pew_Research",
    "Boston.com",
    "Bleeping_Computer",
    "100_Mile_Free_Press",
    "Investopedia",
    "Air_Force_Times",
    "Haaretz",
    "ABC_News_Washington_Post_Polling",
    "PA_Media",
]

LOW_BASE_PRIORITY = [
    "The_Gateway_Pundit",
    "RT_News",
    "Zerohedge",
    "American_Thinker",
    "Banned.News",
    "Biased.News",
    "Sputnik",
    "Natural_News",
    "MedicalTyranny",
    "VaccineDamage.News",
    "CaliforniaCollapse.News",
    "Conspiracy_Daily_Update",
    "American_Intelligence_Media",
    "Adams.News",
]

LOW_DONOR_PRIORITY = [
    "RT_News",
    "Zerohedge",
    "American_Thinker",
    "Banned.News",
    "Biased.News",
    "Sputnik",
    "The_Gateway_Pundit",
    "CaliforniaCollapse.News",
    "Conspiracy_Daily_Update",
    "American_Intelligence_Media",
]

STOP_PATTERNS = [
    r"^(home|menu|search|login|join|subscribe|newsletter|advertise|contact)$",
    r"^(news|sports|opinion|about|shop|watch|live|video|podcasts)$",
    r"^\d+\s*(min|minute|minutes)\s*read$",
    r"^(follow|share|sign in|log in)$",
    r"^[\W\d_]+$",
]

UTILITY_TERMS = [
    "accept all cookies",
    "accept only necessary",
    "accept & close",
    "advertisement",
    "ad by",
    "hide this ad",
    "cookie",
    "privacy notice",
    "privacy policy",
    "sponsored",
    "subscribe",
    "newsletter",
    "donate",
    "login",
    "log in",
    "register",
    "open account",
    "find out more",
    "contact us",
    "for journalists",
    "for pr and comms",
    "metadata",
    "about us",
    "hello, do you have any questions",
    "get it here",
    "need a coupon",
    "unlock",
    "read differently",
    "comments",
    "pasta al limone",
    "pew research center",
    "read our research on",
    "site to work",
    "some are necessary",
    "daily brief",
    "opinion security",
    "podcast",
    "in the news:",
    "move with the markets",
    "zero trust",
    "stop ai-attacks",
    "stop malware",
    "can't stop",
    "deepseek",
    "try it now",
    "stunning presentations",
    "shop tax-free",
    "click here to learn more",
]


@dataclass(frozen=True)
class TextBox:
    text: str
    confidence: float
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(1, self.right - self.left)

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height


def normalize_image_name(name: str) -> str:
    return name.strip()


def load_rows() -> pd.DataFrame:
    rows = pd.DataFrame(load_dataset())
    rows["image"] = rows["image"].map(normalize_image_name)
    return rows


def preferred_rows(rows: pd.DataFrame, names: list[str], binary_label: str) -> list[str]:
    available = rows[rows["factuality_2"] == binary_label].set_index("image")
    selected = [name for name in names if name in available.index]
    remaining = [
        row.image
        for row in rows[rows["factuality_2"] == binary_label].itertuples()
        if row.image not in selected
    ]
    return selected + remaining


def build_pairs(rows: pd.DataFrame, limit: int) -> list[dict[str, str]]:
    high_bases = preferred_rows(rows, HIGH_BASE_PRIORITY, "high")
    high_donors = preferred_rows(rows, HIGH_DONOR_PRIORITY, "high")
    low_bases = preferred_rows(rows, LOW_BASE_PRIORITY, "low")
    low_donors = preferred_rows(rows, LOW_DONOR_PRIORITY, "low")
    if not high_bases or not high_donors or not low_bases or not low_donors:
        raise ValueError("Need at least one high and one low factuality source")

    pairs = []
    for index in range(limit):
        if index % 2 == 0:
            base = low_bases[(index // 2) % len(low_bases)]
            donor = high_donors[(index // 2) % len(high_donors)]
            variant = "low_layout_high_headlines"
        else:
            base = high_bases[(index // 2) % len(high_bases)]
            donor = low_donors[(index // 2) % len(low_donors)]
            variant = "high_layout_low_headlines"
        pairs.append({"base": base, "donor": donor, "variant": variant})
    return pairs


def load_box_cache() -> dict[str, list[TextBox]]:
    if not OCR_BOX_CACHE_PATH.exists():
        return {}
    cache: dict[str, list[TextBox]] = {}
    with OCR_BOX_CACHE_PATH.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[row["image"]] = [TextBox(**item) for item in row["boxes"]]
    return cache


def append_box_cache(image_name: str, boxes: list[TextBox]) -> None:
    OCR_BOX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OCR_BOX_CACHE_PATH.open("a", encoding="utf-8") as output_file:
        output_file.write(
            json.dumps(
                {
                    "image": image_name,
                    "boxes": [box.__dict__ for box in boxes],
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def easyocr_boxes(image_name: str) -> list[TextBox]:
    import easyocr

    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    crop = crop_viewport(image_path(image_name), 0)
    results = reader.readtext(
        np.asarray(crop),
        detail=1,
        paragraph=False,
        text_threshold=0.55,
        low_text=0.3,
    )
    boxes = []
    for item in results:
        points, text, confidence = item[0], str(item[1]), float(item[2])
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        boxes.append(
            TextBox(
                text=clean_text(text),
                confidence=confidence,
                left=max(0, int(min(xs))),
                top=max(0, int(min(ys))),
                right=max(0, int(max(xs))),
                bottom=max(0, int(max(ys))),
            )
        )
    return boxes


def ensure_boxes(image_names: list[str]) -> dict[str, list[TextBox]]:
    cache = load_box_cache()
    missing = [name for name in image_names if name not in cache]
    for index, image_name in enumerate(missing, start=1):
        boxes = easyocr_boxes(image_name)
        cache[image_name] = boxes
        append_box_cache(image_name, boxes)
        print(f"OCR boxes {index}/{len(missing)}: {image_name}", flush=True)
    return cache


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("|", " ")).strip()
    text = text.strip("\"'`“”‘’.,;: ")
    return text


def safe_slug(value: str, max_length: int = 44) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug[:max_length].rstrip("._-") or "source"


def is_stop_text(text: str) -> bool:
    normalized = clean_text(text).lower()
    if len(normalized) < 10:
        return True
    if any(term in normalized for term in UTILITY_TERMS):
        return True
    for pattern in STOP_PATTERNS:
        if re.search(pattern, normalized):
            return True
    alpha_count = sum(character.isalpha() for character in normalized)
    return alpha_count < max(6, len(normalized) * 0.45)


def headline_score(box: TextBox, viewport_width: int, viewport_height: int) -> float:
    text = clean_text(box.text)
    word_count = len(re.findall(r"[A-Za-z][A-Za-z'-]+", text))
    y_factor = 1.0 - min(box.top / max(1, viewport_height), 0.85) * 0.45
    width_factor = min(box.width / max(1, viewport_width), 1.0)
    word_factor = min(word_count / 8, 1.4)
    confidence_factor = max(0.3, box.confidence)
    return box.area * (0.5 + width_factor) * (0.6 + word_factor) * y_factor * confidence_factor


def headline_candidates(
    boxes: list[TextBox],
    viewport_width: int,
    viewport_height: int,
    count: int,
) -> list[TextBox]:
    candidates = []
    for box in boxes:
        text = clean_text(box.text)
        word_count = len(re.findall(r"[A-Za-z][A-Za-z'-]+", text))
        if is_stop_text(text):
            continue
        if word_count < 3 or word_count > 18:
            continue
        if box.confidence < 0.25:
            continue
        if box.width < viewport_width * 0.12 or box.height < 12:
            continue
        if box.top < 35 and box.height < 24:
            continue
        if box.top > viewport_height * 0.95:
            continue
        candidates.append(box)

    candidates.sort(
        key=lambda box: headline_score(box, viewport_width, viewport_height),
        reverse=True,
    )
    selected: list[TextBox] = []
    for box in candidates:
        if all(intersection_over_union(box, previous) < 0.18 for previous in selected):
            selected.append(box)
        if len(selected) == count:
            break
    return selected


def intersection_over_union(left: TextBox, right: TextBox) -> float:
    x1 = max(left.left, right.left)
    y1 = max(left.top, right.top)
    x2 = min(left.right, right.right)
    y2 = min(left.bottom, right.bottom)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = left.area + right.area - intersection
    return intersection / max(1, union)


def fallback_phrases(image_name: str, max_count: int) -> list[str]:
    ocr_features = pd.read_csv("factuality_ocr_features.csv").set_index("image")
    if image_name not in ocr_features.index:
        return []
    text = str(ocr_features.loc[image_name, "ocr_text"])
    chunks = re.split(r"\s{2,}|[.!?]\s+", text)
    phrases = []
    for chunk in chunks:
        words = re.findall(r"[A-Za-z][A-Za-z'-]+", chunk)
        if 4 <= len(words) <= 14:
            phrase = clean_text(" ".join(words))
            if not is_stop_text(phrase):
                phrases.append(phrase)
        if len(phrases) >= max_count:
            break
    return phrases


def donor_texts(
    donor_name: str,
    boxes: list[TextBox],
    viewport_width: int,
    viewport_height: int,
    count: int,
) -> list[str]:
    candidates = headline_candidates(boxes, viewport_width, viewport_height, count * 2)
    texts = []
    seen = set()
    for box in candidates:
        text = clean_text(box.text)
        key = text.lower()
        if key not in seen:
            texts.append(text)
            seen.add(key)
        if len(texts) >= count:
            break
    if len(texts) < count:
        for text in fallback_phrases(donor_name, count - len(texts)):
            key = text.lower()
            if key not in seen:
                texts.append(text)
                seen.add(key)
            if len(texts) >= count:
                break
    return texts


def font_paths() -> tuple[str | None, str | None]:
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    return (
        str(regular) if regular.exists() else None,
        str(bold) if bold.exists() else None,
    )


def load_font(size: int, bold: bool) -> ImageFont.ImageFont:
    regular_path, bold_path = font_paths()
    path = bold_path if bold and bold_path else regular_path
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def region_background(image: Image.Image, box: TextBox, padding: int) -> tuple[int, int, int]:
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    left = max(0, box.left - padding * 2)
    top = max(0, box.top - padding * 2)
    right = min(width, box.right + padding * 2)
    bottom = min(height, box.bottom + padding * 2)
    outer = array[top:bottom, left:right]
    inner_left = max(0, box.left - left)
    inner_top = max(0, box.top - top)
    inner_right = min(outer.shape[1], box.right - left)
    inner_bottom = min(outer.shape[0], box.bottom - top)
    mask = np.ones(outer.shape[:2], dtype=bool)
    mask[inner_top:inner_bottom, inner_left:inner_right] = False
    pixels = outer[mask]
    if len(pixels) < 16:
        pixels = outer.reshape(-1, 3)
    median = np.median(pixels, axis=0)
    return tuple(int(value) for value in median)


def readable_text_color(background: tuple[int, int, int]) -> tuple[int, int, int]:
    luminance = 0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
    return (20, 22, 24) if luminance > 145 else (245, 247, 250)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if text_width(draw, candidate, font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def text_height(draw: ImageDraw.ImageDraw, lines: list[str], font: ImageFont.ImageFont) -> int:
    if not lines:
        return 0
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])
    return sum(line_heights) + max(0, len(lines) - 1) * 4


def draw_replacement(
    image: Image.Image,
    box: TextBox,
    text: str,
) -> dict[str, Any]:
    draw = ImageDraw.Draw(image)
    padding = max(5, min(14, int(box.height * 0.35)))
    left = max(0, box.left - padding)
    top = max(0, box.top - padding)
    right = min(image.width, box.right + padding)
    bottom = min(image.height, box.bottom + padding)
    background = region_background(image, box, padding)
    draw.rectangle((left, top, right, bottom), fill=background)

    max_width = max(24, right - left - padding * 2)
    max_height = max(14, bottom - top - padding * 2)
    font_size = max(10, min(34, int(max_height * 0.78)))
    chosen_lines: list[str] = []
    chosen_font = load_font(font_size, bold=True)
    while font_size >= 9:
        chosen_font = load_font(font_size, bold=True)
        lines = wrap_text(draw, text, chosen_font, max_width)
        if text_height(draw, lines[:3], chosen_font) <= max_height:
            chosen_lines = lines[:3]
            break
        font_size -= 1
    if not chosen_lines:
        chosen_lines = wrap_text(draw, text, chosen_font, max_width)[:2]

    color = readable_text_color(background)
    y = top + padding
    for line in chosen_lines:
        draw.text((left + padding, y), line, fill=color, font=chosen_font)
        bbox = draw.textbbox((0, 0), line, font=chosen_font)
        y += (bbox[3] - bbox[1]) + 4

    return {
        "box": [left, top, right, bottom],
        "text": text,
        "font_size": font_size,
        "background": background,
        "color": color,
    }


def generate_one(
    index: int,
    pair: dict[str, str],
    rows_by_image: dict[str, dict[str, str]],
    box_cache: dict[str, list[TextBox]],
) -> dict[str, Any]:
    base = pair["base"]
    donor = pair["donor"]
    base_path = image_path(base)
    with Image.open(base_path) as source:
        image = source.convert("RGB")
    viewport_height = min(image.height, round(image.width / TARGET_ASPECT_RATIO))
    viewport_width = image.width

    base_boxes = headline_candidates(
        box_cache[base],
        viewport_width,
        viewport_height,
        REPLACEMENTS_PER_IMAGE,
    )
    if not base_boxes:
        raise RuntimeError(f"No replaceable headline boxes found for {base}")

    texts = donor_texts(
        donor,
        box_cache[donor],
        viewport_width,
        viewport_height,
        len(base_boxes),
    )
    if not texts:
        raise RuntimeError(f"No donor headline texts found for {donor}")

    replacements = []
    for region, text in zip(base_boxes, texts):
        replacements.append(draw_replacement(image, region, text))

    output_name = (
        f"{index:03d}_{pair['variant']}__base_{safe_slug(base)}"
        f"__donor_{safe_slug(donor)}.png"
    )
    output_path = IMAGE_OUTPUT_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    base_row = rows_by_image[base]
    donor_row = rows_by_image[donor]
    return {
        "synthetic_image": output_path.stem,
        "output_path": output_path.as_posix(),
        "variant": pair["variant"],
        "base_image": base,
        "base_media_name": base_row["media_name"],
        "base_url": base_row["url"],
        "base_factuality_4": base_row["factuality_4"],
        "base_factuality_2": base_row["factuality_2"],
        "donor_image": donor,
        "donor_media_name": donor_row["media_name"],
        "donor_url": donor_row["url"],
        "donor_factuality_4": donor_row["factuality_4"],
        "donor_factuality_2": donor_row["factuality_2"],
        "replacement_count": str(len(replacements)),
        "replacement_json": json.dumps(replacements, ensure_ascii=False),
        "method": "easyocr_headline_box_swap",
        "modified_region": "first_16_9_viewport",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.overwrite and OUTPUT_DIR.exists():
        for path in IMAGE_OUTPUT_DIR.glob("*.png"):
            path.unlink()
        if MANIFEST_PATH.exists():
            MANIFEST_PATH.unlink()

    rows = load_rows()
    rows_by_image = {row["image"]: row for row in rows.to_dict(orient="records")}
    pairs = build_pairs(rows, args.limit * 5)
    box_cache = load_box_cache()
    manifest_rows = []
    skipped = []
    for pair in pairs:
        if len(manifest_rows) >= args.limit:
            break
        box_cache.update(ensure_boxes([pair["base"], pair["donor"]]))
        output_index = len(manifest_rows) + 1
        try:
            manifest_rows.append(generate_one(output_index, pair, rows_by_image, box_cache))
            print(
                f"synthetic {output_index}/{args.limit}: {pair['variant']} "
                f"{pair['base']} <- {pair['donor']}",
                flush=True,
            )
        except RuntimeError as exc:
            skipped.append(
                {
                    "base": pair["base"],
                    "donor": pair["donor"],
                    "variant": pair["variant"],
                    "reason": str(exc),
                }
            )
            print(f"skip {pair['base']} <- {pair['donor']}: {exc}", flush=True)

    if len(manifest_rows) < args.limit:
        raise RuntimeError(f"Only generated {len(manifest_rows)} of {args.limit} requested samples")

    fieldnames = [
        "synthetic_image",
        "output_path",
        "variant",
        "base_image",
        "base_media_name",
        "base_url",
        "base_factuality_4",
        "base_factuality_2",
        "donor_image",
        "donor_media_name",
        "donor_url",
        "donor_factuality_4",
        "donor_factuality_2",
        "replacement_count",
        "replacement_json",
        "method",
        "modified_region",
    ]
    write_csv(MANIFEST_PATH, manifest_rows, fieldnames)
    if skipped:
        write_csv(
            OUTPUT_DIR / "synthetic_factuality_skipped.csv",
            skipped,
            ["base", "donor", "variant", "reason"],
        )
    print(f"wrote {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
