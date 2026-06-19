from experiment_bootstrap import activate_project_root

activate_project_root()

import csv
from pathlib import Path

import bias5_image_feature_advanced_experiments as advanced
import bias5_image_feature_experiments as image_features
from bias5_image_feature_full_experiments import (
    FULL_PREFIX,
    full_embedding_cache_path,
    load_full_rows,
)


ADVANCED_PREFIX = "bias5_image_feature_full_advanced"


def full_prompt_score_path(name: str) -> Path:
    return Path(f"{FULL_PREFIX}_{name}_prompt_scores.csv")


def write_full_report() -> None:
    metrics_text = advanced.METRICS_PATH.read_text(encoding="utf-8").strip()
    metric_rows = list(csv.DictReader(metrics_text.splitlines()))
    sorted_rows = sorted(metric_rows, key=lambda row: float(row["macro_f1"]), reverse=True)
    best_overall = sorted_rows[0]
    best_new = next(row for row in sorted_rows if not row["method"].startswith("reference_non_nested_"))
    report = (
        "# Bias 5 Advanced Image Feature Full Dataset Experiment Report\n\n"
        f"Device: `{advanced.DEVICE}`.\n\n"
        "Dataset: full balanced 250-site bias dataset.\n\n"
        "This run reuses full OCR, CLIP, SigLIP, DINOv2, and handcrafted visual features. "
        "Nested stacking is included to reduce optimistic leakage.\n\n"
        "## Best New Method\n\n"
        f"`{best_new['method']}`: accuracy `{best_new['accuracy']}`, macro-F1 `{best_new['macro_f1']}`.\n\n"
        "## Best Overall Reference\n\n"
        f"`{best_overall['method']}`: accuracy `{best_overall['accuracy']}`, macro-F1 `{best_overall['macro_f1']}`.\n\n"
        "## Metrics\n\n"
        "```csv\n"
        f"{metrics_text}\n"
        "```\n"
    )
    advanced.REPORT_PATH.write_text(report, encoding="utf-8")


def configure_full_run() -> None:
    image_features.OCR_RAW_PATH = Path(f"{FULL_PREFIX}_ocr_raw.jsonl")
    image_features.OCR_FEATURES_PATH = Path(f"{FULL_PREFIX}_ocr_features.csv")
    image_features.cache_path_for_model = full_embedding_cache_path

    advanced.METRICS_PATH = Path(f"{ADVANCED_PREFIX}_metrics.csv")
    advanced.PREDICTIONS_PATH = Path(f"{ADVANCED_PREFIX}_predictions.csv")
    advanced.CONFUSIONS_PATH = Path(f"{ADVANCED_PREFIX}_confusions.csv")
    advanced.REPORT_PATH = Path(f"{ADVANCED_PREFIX}_report.md")
    advanced.BASE_IMAGE_FEATURE_PREDICTIONS_PATH = Path(f"{FULL_PREFIX}_predictions.csv")
    advanced.load_sample_rows = load_full_rows
    advanced.cache_path_for_model = full_embedding_cache_path
    advanced.prompt_score_path = full_prompt_score_path
    advanced.ensure_ocr_features = image_features.ensure_ocr_features
    advanced.handcrafted_visual_matrix = image_features.handcrafted_visual_matrix


def main() -> None:
    configure_full_run()
    advanced.main()
    write_full_report()


if __name__ == "__main__":
    main()
