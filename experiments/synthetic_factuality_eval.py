from experiment_bootstrap import activate_project_root

activate_project_root()

import base64
import concurrent.futures
import csv
import io
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from factuality_common import MULTICLASS_LABELS, MULTICLASS_TO_BINARY, TARGET_ASPECT_RATIO, write_csv
from factuality_prepare_features import MULTICLASS_PROMPT
from llm_client import OpenRouterLLM


MANIFEST_PATH = Path("synthetic-factuality-samples/synthetic_factuality_manifest.csv")
RAW_PATH = Path("synthetic-factuality-samples/synthetic_factuality_llm_raw.jsonl")
PREDICTIONS_PATH = Path("synthetic-factuality-samples/synthetic_factuality_llm_predictions.csv")
SUMMARY_PATH = Path("synthetic-factuality-samples/synthetic_factuality_llm_summary.csv")
REPORT_PATH = Path("synthetic-factuality-samples/synthetic_factuality_llm_report.md")
USAGE_PATH = Path("synthetic-factuality-samples/synthetic_factuality_openrouter_usage.csv")

MODELS = [
    "openai/gpt-5.5",
    "anthropic/claude-sonnet-4.6",
]
REQUEST_TIMEOUT = 180
REQUEST_RETRIES = 2
MAX_WORKERS = 4


def crop_first_viewport_bytes(path: Path) -> bytes:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        viewport_height = min(height, round(width / TARGET_ASPECT_RATIO))
        crop = image.crop((0, 0, width, viewport_height))
        output = io.BytesIO()
        crop.save(output, format="PNG", optimize=True)
        return output.getvalue()


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(crop_first_viewport_bytes(path)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def messages_for_image(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": MULTICLASS_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(path)},
                },
            ],
        }
    ]


def normalize_label(response_text: str) -> str:
    text = re.sub(r"\s+", " ", response_text.strip().lower().strip("\"'` ."))
    if text in MULTICLASS_LABELS:
        return text
    matches = [
        label
        for label in MULTICLASS_LABELS
        if re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", text)
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Unexpected label response: {response_text[:200]}")


def extract_content(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content)


def load_raw_cache() -> dict[tuple[str, str], dict[str, Any]]:
    if not RAW_PATH.exists():
        return {}
    rows = {}
    with RAW_PATH.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[(row["synthetic_image"], row["model"])] = row
    return rows


def append_raw(row: dict[str, Any]) -> None:
    with RAW_PATH.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def usage_row(synthetic_image: str, model: str, response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {
        "synthetic_image": synthetic_image,
        "model": model,
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


def request_prediction(
    llm: OpenRouterLLM,
    synthetic_image: str,
    model: str,
    output_path: Path,
) -> dict[str, Any]:
    last_error: Exception | None = None
    messages = messages_for_image(output_path)
    for attempt in range(REQUEST_RETRIES + 1):
        try:
            response = llm.openr_llm(model=model, messages=messages, timeout=REQUEST_TIMEOUT)
            raw_text = extract_content(response)
            prediction = normalize_label(raw_text)
            append_usage(usage_row(synthetic_image, model, response))
            return {
                "synthetic_image": synthetic_image,
                "model": model,
                "prediction_4": prediction,
                "prediction_2": MULTICLASS_TO_BINARY[prediction],
                "raw_text": raw_text,
            }
        except Exception as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(2**attempt)
    raise last_error or RuntimeError("LLM request failed")


def response_relation(prediction_2: str, base_2: str, donor_2: str) -> str:
    if prediction_2 == base_2 and prediction_2 == donor_2:
        return "same_binary_base_and_donor"
    if prediction_2 == base_2:
        return "stayed_with_base"
    if prediction_2 == donor_2:
        return "shifted_to_donor"
    return "other"


def build_predictions(manifest: pd.DataFrame, cache: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in manifest.to_dict(orient="records"):
        for model in MODELS:
            cached = cache[(item["synthetic_image"], model)]
            prediction_4 = cached["prediction_4"]
            prediction_2 = cached["prediction_2"]
            rows.append(
                {
                    "synthetic_image": item["synthetic_image"],
                    "variant": item["variant"],
                    "model": model,
                    "prediction_4": prediction_4,
                    "prediction_2": prediction_2,
                    "relation_to_swap": response_relation(
                        prediction_2,
                        item["base_factuality_2"],
                        item["donor_factuality_2"],
                    ),
                    "exact_base_4": str(prediction_4 == item["base_factuality_4"]),
                    "exact_donor_4": str(prediction_4 == item["donor_factuality_4"]),
                    "base_image": item["base_image"],
                    "base_factuality_4": item["base_factuality_4"],
                    "base_factuality_2": item["base_factuality_2"],
                    "donor_image": item["donor_image"],
                    "donor_factuality_4": item["donor_factuality_4"],
                    "donor_factuality_2": item["donor_factuality_2"],
                    "replacement_count": item["replacement_count"],
                    "output_path": item["output_path"],
                }
            )
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    groups = [
        ("all", predictions),
        *[(variant, frame) for variant, frame in predictions.groupby("variant")],
    ]
    for scope, frame in groups:
        for model, model_frame in frame.groupby("model"):
            total = len(model_frame)
            counts = model_frame["relation_to_swap"].value_counts().to_dict()
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "n": total,
                    "shifted_to_donor": counts.get("shifted_to_donor", 0),
                    "stayed_with_base": counts.get("stayed_with_base", 0),
                    "other": counts.get("other", 0),
                    "shifted_to_donor_rate": counts.get("shifted_to_donor", 0) / max(1, total),
                    "stayed_with_base_rate": counts.get("stayed_with_base", 0) / max(1, total),
                    "exact_base_4_rate": (model_frame["exact_base_4"] == "True").mean(),
                    "exact_donor_4_rate": (model_frame["exact_donor_4"] == "True").mean(),
                    "prediction_4_counts": json.dumps(
                        model_frame["prediction_4"].value_counts().to_dict(),
                        ensure_ascii=False,
                    ),
                }
            )
    return rows


def write_report(predictions: pd.DataFrame, summary_rows: list[dict[str, Any]]) -> None:
    summary = pd.DataFrame(summary_rows)
    lines = [
        "# Synthetic Factuality LLM Probe",
        "",
        f"Samples: `{predictions['synthetic_image'].nunique()}` synthetic screenshots.",
        f"Models: `{', '.join(MODELS)}`.",
        "",
        "Prompt: unchanged factuality 4-label prompt from the factuality baseline.",
        "Image input: first 16:9 viewport of each synthetic full-page screenshot.",
        "",
        "## Summary",
        "",
        "| Scope | Model | N | Shifted to donor | Stayed with base | Shift rate | Base exact 4 | Donor exact 4 | Prediction counts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['scope']} | `{row['model']}` | {row['n']} | "
            f"{row['shifted_to_donor']} | {row['stayed_with_base']} | "
            f"{row['shifted_to_donor_rate']:.3f} | {row['exact_base_4_rate']:.3f} | "
            f"{row['exact_donor_4_rate']:.3f} | `{row['prediction_4_counts']}` |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{PREDICTIONS_PATH}`",
            f"- `{SUMMARY_PATH}`",
            f"- `{RAW_PATH}`",
            f"- `{USAGE_PATH}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    cache = load_raw_cache()
    missing = [
        (row["synthetic_image"], model, Path(row["output_path"]))
        for row in manifest.to_dict(orient="records")
        for model in MODELS
        if (row["synthetic_image"], model) not in cache
    ]
    if missing:
        llm = OpenRouterLLM()
        total_expected = len(cache) + len(missing)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(request_prediction, llm, synthetic_image, model, output_path): (
                    synthetic_image,
                    model,
                )
                for synthetic_image, model, output_path in missing
            }
            completed = len(cache)
            for future in concurrent.futures.as_completed(futures):
                synthetic_image, model = futures[future]
                row = future.result()
                cache[(synthetic_image, model)] = row
                append_raw(row)
                completed += 1
                print(f"LLM synthetic {completed}/{total_expected}: {model} {synthetic_image}", flush=True)

    predictions = build_predictions(manifest, cache)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    summary_rows = summarize(predictions)
    write_csv(SUMMARY_PATH, summary_rows, list(summary_rows[0].keys()))
    write_report(predictions, summary_rows)
    print(f"wrote {PREDICTIONS_PATH}", flush=True)
    print(f"wrote {SUMMARY_PATH}", flush=True)
    print(f"wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
