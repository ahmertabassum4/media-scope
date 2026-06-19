from experiment_bootstrap import activate_project_root

activate_project_root()

import base64
import concurrent.futures
import csv
import io
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from llm_client import OpenRouterLLM


Image.MAX_IMAGE_PIXELS = None

IMAGE_FOLDER = Path("bias-samples")
IMAGE_LIMIT = 250
MODELS = [
    "openai/gpt-5.5",
    "anthropic/claude-sonnet-4.6",
    "moonshotai/kimi-k2.6",
    "qwen/qwen3.7-plus",
]
RUN_MODES = ["bias_3", "bias_5"]
PREDICTIONS_PATH = Path("bias_predictions.csv")
METRICS_PATH = Path("bias_metrics.csv")
BIAS_3_SOURCE_BREAKDOWN_PATH = Path("bias_3_source_breakdown.csv")
TARGET_ASPECT_RATIO = 16 / 9
REQUEST_TIMEOUT = 120
REQUEST_MAX_TOKENS = None
REQUEST_RETRIES = 2
MAX_WORKERS = 4
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

BIAS_5_LABELS = ["left", "left-center", "least biased", "right-center", "right"]
BIAS_3_LABELS = ["left biased", "least biased", "right biased"]
BIAS_3_SOURCE_LIMITS = {
    "left": 25,
    "left-center": 25,
    "least biased": 50,
    "right-center": 25,
    "right": 25,
}
BIAS_5_TO_3 = {
    "left": "left biased",
    "left-center": "left biased",
    "least biased": "least biased",
    "right-center": "right biased",
    "right": "right biased",
}
BIAS_JSON_TO_LABEL = {
    "LEFT": "left",
    "LEFT-CENTER": "left-center",
    "LEAST BIASED": "least biased",
    "RIGHT-CENTER": "right-center",
    "RIGHT": "right",
}
MODE_LABELS = {
    "bias_3": BIAS_3_LABELS,
    "bias_5": BIAS_5_LABELS,
}
PROMPTS = {
    "bias_3": (
        "You are classifying the political bias of a website based only on the provided "
        "landing page screenshot. Do not browse the web, do not search online, and do not "
        "use outside knowledge about the outlet. Use only visible landing-page cues such "
        "as headlines, branding, navigation, language, topics, and layout. Your entire "
        "response must be exactly one lowercase label and no explanation. Choose from: "
        "left biased, least biased, right biased."
    ),
    "bias_5": (
        "You are classifying the political bias of a website based only on the provided "
        "landing page screenshot. Do not browse the web, do not search online, and do not "
        "use outside knowledge about the outlet. Use only visible landing-page cues such "
        "as headlines, branding, navigation, language, topics, and layout. Your entire "
        "response must be exactly one lowercase label and no explanation. Choose from: "
        "left, left-center, least biased, right-center, right."
    ),
}


def crop_to_first_screen(image_path: Path) -> bytes:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        target_height = min(height, round(width / TARGET_ASPECT_RATIO))

        if target_height < height:
            crop_box = (0, 0, width, target_height)
        else:
            target_width = min(width, round(height * TARGET_ASPECT_RATIO))
            left = (width - target_width) // 2
            crop_box = (left, 0, left + target_width, height)

        cropped = image.crop(crop_box)
        output = io.BytesIO()
        cropped.save(output, format="PNG")
        return output.getvalue()


def image_to_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(crop_to_first_screen(image_path)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_messages(prompt: str, data_url: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ],
        }
    ]


def iter_images(folder: Path, limit: int) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Image path is not a folder: {folder}")

    images = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if limit == -1:
        return images
    if limit < -1:
        raise ValueError("IMAGE_LIMIT must be -1 or a non-negative integer")
    return images[:limit]


def metadata_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def load_ground_truth(image_path: Path) -> tuple[str, str]:
    metadata_path = metadata_path_for_image(image_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_bias = data.get("bias")
    if raw_bias not in BIAS_JSON_TO_LABEL:
        raise ValueError(f"Unexpected bias label in {metadata_path}: {raw_bias}")

    bias_5 = BIAS_JSON_TO_LABEL[raw_bias]
    return BIAS_5_TO_3[bias_5], bias_5


def extract_content(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


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
    if matches:
        longest_match = max(matches, key=len)
        if sum(1 for match in matches if len(match) == len(longest_match)) == 1:
            return longest_match

    raise ValueError(f"Unexpected label response: {response_text}")


def prediction_column(mode: str, model: str) -> str:
    return f"{mode}:{model}"


def load_existing_results(output_path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    if not output_path.exists():
        return {}, []

    with output_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            return {}, []

        rows = {
            row["image"]: {field: row.get(field, "") for field in reader.fieldnames}
            for row in reader
            if row.get("image")
        }
        return rows, reader.fieldnames


def write_csv(output_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_fieldnames(existing_fieldnames: list[str]) -> list[str]:
    fixed_fields = ["image", "true_bias_3", "true_bias_5", "bias_3_included"]
    prediction_fields = [
        prediction_column(mode, model)
        for mode in RUN_MODES
        for model in MODELS
    ]
    extra_fields = [
        field for field in existing_fieldnames
        if field not in fixed_fields and field not in prediction_fields
    ]
    return [*fixed_fields, *extra_fields, *prediction_fields]


def select_bias_3_image_names(image_paths: list[Path]) -> set[str]:
    selected: set[str] = set()
    counts: Counter[str] = Counter()

    for image_path in image_paths:
        _, true_bias_5 = load_ground_truth(image_path)
        if counts[true_bias_5] >= BIAS_3_SOURCE_LIMITS[true_bias_5]:
            continue
        selected.add(image_path.stem)
        counts[true_bias_5] += 1

    missing = {
        label: limit - counts[label]
        for label, limit in BIAS_3_SOURCE_LIMITS.items()
        if counts[label] < limit
    }
    if missing:
        raise ValueError(f"Not enough images for bias_3 balanced sample: {missing}")

    return selected


def process_images() -> list[dict[str, str]]:
    llm = OpenRouterLLM()
    image_paths = iter_images(IMAGE_FOLDER, IMAGE_LIMIT)
    bias_3_image_names = select_bias_3_image_names(image_paths)
    existing_rows, existing_fieldnames = load_existing_results(PREDICTIONS_PATH)
    fieldnames = build_fieldnames(existing_fieldnames)
    results: list[dict[str, str]] = []

    for image_path in image_paths:
        image_name = image_path.stem
        true_bias_3, true_bias_5 = load_ground_truth(image_path)
        result = existing_rows.get(image_name, {"image": image_name})
        result["image"] = image_name
        result["true_bias_3"] = true_bias_3
        result["true_bias_5"] = true_bias_5
        result["bias_3_included"] = "1" if image_name in bias_3_image_names else "0"
        pending_requests = []

        for mode in RUN_MODES:
            if mode == "bias_3" and image_name not in bias_3_image_names:
                for model in MODELS:
                    result[prediction_column(mode, model)] = ""
                continue

            for model in MODELS:
                column = prediction_column(mode, model)
                current_value = result.get(column, "")
                if current_value and not current_value.startswith("error:"):
                    continue
                pending_requests.append((mode, model, column))

        data_url = image_to_data_url(image_path) if pending_requests else ""
        if pending_requests:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(request_normalized_label, llm, model, mode, data_url): column
                    for mode, model, column in pending_requests
                }
                for future in concurrent.futures.as_completed(futures):
                    column = futures[future]
                    try:
                        result[column] = future.result()
                    except Exception as exc:
                        result[column] = f"error: {exc}"

        existing_rows[image_name] = result
        results.append({field: result.get(field, "") for field in fieldnames})
        write_csv(
            PREDICTIONS_PATH,
            [{field: existing_rows[name].get(field, "") for field in fieldnames} for name in sorted(existing_rows)],
            fieldnames,
        )
        print(f"processed {len(existing_rows)} rows; latest={image_name}", flush=True)

    write_metrics(PREDICTIONS_PATH, METRICS_PATH)
    write_bias_3_source_breakdown(PREDICTIONS_PATH, BIAS_3_SOURCE_BREAKDOWN_PATH)
    return results


def request_normalized_label(llm: OpenRouterLLM, model: str, mode: str, data_url: str) -> str:
    last_error: Exception | None = None
    messages = build_messages(PROMPTS[mode], data_url)

    for attempt in range(REQUEST_RETRIES + 1):
        try:
            response = request_label(llm, model, messages)
            return normalize_label(extract_content(response), MODE_LABELS[mode])
        except Exception as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(2 ** attempt)

    raise last_error or RuntimeError("LLM label normalization failed")


def request_label(llm: OpenRouterLLM, model: str, messages: list[dict[str, Any]]) -> Any:
    last_error: Exception | None = None

    for attempt in range(REQUEST_RETRIES + 1):
        try:
            return llm.openr_llm(
                model=model,
                messages=messages,
                max_tokens=REQUEST_MAX_TOKENS,
                timeout=REQUEST_TIMEOUT,
            )
        except Exception as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(2 ** attempt)

    raise last_error or RuntimeError("LLM request failed")


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_metrics(rows: list[dict[str, str]], mode: str, model: str) -> dict[str, str]:
    labels = MODE_LABELS[mode]
    truth_column = "true_bias_3" if mode == "bias_3" else "true_bias_5"
    pred_column = prediction_column(mode, model)
    counts = {
        label: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for label in labels
    }
    evaluated = 0
    correct = 0
    invalid_or_error = 0

    for row in rows:
        if mode == "bias_3" and row.get("bias_3_included") != "1":
            continue

        truth = row.get(truth_column, "")
        prediction = row.get(pred_column, "")
        if truth not in labels:
            continue
        if prediction not in labels:
            invalid_or_error += 1
            continue

        evaluated += 1
        counts[truth]["support"] += 1
        if prediction == truth:
            correct += 1

        for label in labels:
            if prediction == label and truth == label:
                counts[label]["tp"] += 1
            elif prediction == label and truth != label:
                counts[label]["fp"] += 1
            elif prediction != label and truth == label:
                counts[label]["fn"] += 1

    precision_values = []
    recall_values = []
    f1_values = []
    weighted_f1_sum = 0.0

    for label in labels:
        label_counts = counts[label]
        precision = safe_divide(label_counts["tp"], label_counts["tp"] + label_counts["fp"])
        recall = safe_divide(label_counts["tp"], label_counts["tp"] + label_counts["fn"])
        f1 = safe_divide(2 * precision * recall, precision + recall)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        weighted_f1_sum += f1 * label_counts["support"]

    return {
        "mode": mode,
        "model": model,
        "total": str(sum(1 for row in rows if mode != "bias_3" or row.get("bias_3_included") == "1")),
        "evaluated": str(evaluated),
        "invalid_or_error": str(invalid_or_error),
        "accuracy": f"{safe_divide(correct, evaluated):.6f}",
        "macro_precision": f"{safe_divide(sum(precision_values), len(labels)):.6f}",
        "macro_recall": f"{safe_divide(sum(recall_values), len(labels)):.6f}",
        "macro_f1": f"{safe_divide(sum(f1_values), len(labels)):.6f}",
        "weighted_f1": f"{safe_divide(weighted_f1_sum, evaluated):.6f}",
    }


def write_metrics(predictions_path: Path, metrics_path: Path) -> None:
    if not predictions_path.exists():
        return

    with predictions_path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    metric_rows = [
        calculate_metrics(rows, mode, model)
        for mode in RUN_MODES
        for model in MODELS
    ]
    write_csv(
        metrics_path,
        metric_rows,
        [
            "mode",
            "model",
            "total",
            "evaluated",
            "invalid_or_error",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
        ],
    )


def calculate_bias_3_source_breakdown(rows: list[dict[str, str]], model: str) -> list[dict[str, str]]:
    pred_column = prediction_column("bias_3", model)
    breakdown_rows = []

    for source_label in BIAS_5_LABELS:
        selected_rows = [
            row for row in rows
            if row.get("bias_3_included") == "1" and row.get("true_bias_5") == source_label
        ]
        evaluated = 0
        correct = 0
        invalid_or_error = 0

        for row in selected_rows:
            truth = row.get("true_bias_3", "")
            prediction = row.get(pred_column, "")
            if prediction not in BIAS_3_LABELS:
                invalid_or_error += 1
                continue
            evaluated += 1
            if prediction == truth:
                correct += 1

        breakdown_rows.append(
            {
                "mode": "bias_3",
                "model": model,
                "source_bias_5": source_label,
                "true_bias_3": BIAS_5_TO_3[source_label],
                "total": str(len(selected_rows)),
                "evaluated": str(evaluated),
                "correct": str(correct),
                "incorrect": str(evaluated - correct),
                "invalid_or_error": str(invalid_or_error),
                "accuracy": f"{safe_divide(correct, evaluated):.6f}",
            }
        )

    return breakdown_rows


def write_bias_3_source_breakdown(predictions_path: Path, output_path: Path) -> None:
    if not predictions_path.exists():
        return

    with predictions_path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    breakdown_rows = [
        row
        for model in MODELS
        for row in calculate_bias_3_source_breakdown(rows, model)
    ]
    write_csv(
        output_path,
        breakdown_rows,
        [
            "mode",
            "model",
            "source_bias_5",
            "true_bias_3",
            "total",
            "evaluated",
            "correct",
            "incorrect",
            "invalid_or_error",
            "accuracy",
        ],
    )


if __name__ == "__main__":
    processed = process_images()
    print(json.dumps(processed, ensure_ascii=False, indent=2))
