from experiment_bootstrap import activate_project_root

activate_project_root()

import csv
import json
from pathlib import Path

import bias5_image_feature_experiments as image_features


FULL_PREFIX = "bias5_image_feature_full"
PREDICTIONS_INPUT_PATH = Path("bias_predictions.csv")

OCR_SOURCE_PATH = Path("bias5_image_feature_ocr_raw.jsonl")
OCR_FULL_PATH = Path(f"{FULL_PREFIX}_ocr_raw.jsonl")


def load_full_rows() -> list[dict[str, str]]:
    with PREDICTIONS_INPUT_PATH.open("r", encoding="utf-8", newline="") as input_file:
        rows = [
            {"image": row["image"], "true_bias_5": row["true_bias_5"]}
            for row in csv.DictReader(input_file)
            if row.get("image") and row.get("true_bias_5")
        ]
    return sorted(rows, key=lambda row: row["image"])


def seed_jsonl_cache(source_path: Path, target_path: Path) -> None:
    if not source_path.exists():
        return

    existing_images = set()
    if target_path.exists():
        with target_path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                if line.strip():
                    existing_images.add(json.loads(line)["image"])

    with source_path.open("r", encoding="utf-8") as input_file, target_path.open("a", encoding="utf-8") as output_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["image"] not in existing_images:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing_images.add(row["image"])


def full_embedding_cache_path(name: str) -> Path:
    return Path(f"{FULL_PREFIX}_{name}_embeddings.npz")


def skip_zero_shot(rows: list[dict[str, str]], spec: dict[str, str]):
    raise RuntimeError("Zero-shot skipped in full run because it was not a top method")


def write_full_report() -> None:
    metrics_text = image_features.METRICS_PATH.read_text(encoding="utf-8").strip()
    metric_rows = list(csv.DictReader(metrics_text.splitlines()))
    best = max(metric_rows, key=lambda row: float(row["macro_f1"]))
    report = (
        "# Bias 5 Image Feature Full Dataset Experiment Report\n\n"
        f"Device: `{image_features.DEVICE}`.\n\n"
        "Dataset: full balanced 250-site bias dataset.\n\n"
        "This run avoids new paid LLM calls. It tests OCR, local vision embeddings, handcrafted visual features, "
        "and local ensembles on all images.\n\n"
        "## Best Method\n\n"
        f"`{best['method']}`: accuracy `{best['accuracy']}`, macro-F1 `{best['macro_f1']}`.\n\n"
        "## Metrics\n\n"
        "```csv\n"
        f"{metrics_text}\n"
        "```\n"
    )
    image_features.REPORT_PATH.write_text(report, encoding="utf-8")


def configure_full_run() -> None:
    image_features.OCR_RAW_PATH = OCR_FULL_PATH
    image_features.OCR_FEATURES_PATH = Path(f"{FULL_PREFIX}_ocr_features.csv")
    image_features.METRICS_PATH = Path(f"{FULL_PREFIX}_metrics.csv")
    image_features.PREDICTIONS_PATH = Path(f"{FULL_PREFIX}_predictions.csv")
    image_features.CONFUSIONS_PATH = Path(f"{FULL_PREFIX}_confusions.csv")
    image_features.REPORT_PATH = Path(f"{FULL_PREFIX}_report.md")
    image_features.load_sample_rows = load_full_rows
    image_features.cache_path_for_model = full_embedding_cache_path
    image_features.ensure_clip_zero_shot = skip_zero_shot


def main() -> None:
    seed_jsonl_cache(OCR_SOURCE_PATH, OCR_FULL_PATH)
    configure_full_run()
    image_features.main()
    write_full_report()


if __name__ == "__main__":
    main()
