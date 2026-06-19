from experiment_bootstrap import activate_project_root

activate_project_root()

import csv
import json
from pathlib import Path

import bias5_semantic_feature_experiments as semantic


FULL_PREFIX = "bias5_semantic_full"
PREDICTIONS_INPUT_PATH = Path("bias_predictions.csv")

RAW_SOURCE_PATH = Path("bias5_semantic_features_raw.jsonl")
RAW_FULL_PATH = Path(f"{FULL_PREFIX}_features_raw.jsonl")


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


def write_full_report() -> None:
    metrics_text = semantic.METRICS_PATH.read_text(encoding="utf-8").strip()
    importance_df = semantic.pd.read_csv(semantic.IMPORTANCE_PATH)
    top_features = "\n".join(
        f"- `{row.feature}`: {row.importance:.6f}"
        for row in importance_df.head(25).itertuples(index=False)
    )
    metric_rows = list(csv.DictReader(metrics_text.splitlines()))
    best = max(metric_rows, key=lambda row: float(row["macro_f1"]))
    report = (
        "# Bias 5 Semantic Full Dataset Experiment Report\n\n"
        f"Model used for feature extraction: `{semantic.MODEL}`.\n\n"
        "Dataset: full balanced 250-site bias dataset.\n\n"
        "The LLM extracts structured visible-page features from the first four 16:9 viewport screenshots. "
        "Downstream classifiers predict the 5-label bias class from those features with stratified 5-fold CV.\n\n"
        "## Best Method\n\n"
        f"`{best['method']}`: accuracy `{best['accuracy']}`, macro-F1 `{best['macro_f1']}`.\n\n"
        "## Metrics\n\n"
        "```csv\n"
        f"{metrics_text}\n"
        "```\n\n"
        "## Top ExtraTrees Feature Importances\n\n"
        f"{top_features}\n"
    )
    semantic.REPORT_PATH.write_text(report, encoding="utf-8")


def configure_full_run() -> None:
    semantic.RAW_FEATURES_PATH = RAW_FULL_PATH
    semantic.FEATURES_PATH = Path(f"{FULL_PREFIX}_features.csv")
    semantic.PREDICTIONS_PATH = Path(f"{FULL_PREFIX}_ml_predictions.csv")
    semantic.METRICS_PATH = Path(f"{FULL_PREFIX}_ml_metrics.csv")
    semantic.CONFUSIONS_PATH = Path(f"{FULL_PREFIX}_ml_confusions.csv")
    semantic.IMPORTANCE_PATH = Path(f"{FULL_PREFIX}_feature_importance.csv")
    semantic.HYBRID_IMPORTANCE_PATH = Path(f"{FULL_PREFIX}_hybrid_feature_importance.csv")
    semantic.GROUP_ABLATION_PATH = Path(f"{FULL_PREFIX}_group_ablation.csv")
    semantic.SINGLE_FEATURE_ABLATION_PATH = Path(f"{FULL_PREFIX}_single_feature_ablation.csv")
    semantic.PAIR_FEATURE_ABLATION_PATH = Path(f"{FULL_PREFIX}_pair_feature_ablation.csv")
    semantic.REPORT_PATH = Path(f"{FULL_PREFIX}_feature_report.md")
    semantic.load_sample_rows = load_full_rows


def main() -> None:
    seed_jsonl_cache(RAW_SOURCE_PATH, RAW_FULL_PATH)
    configure_full_run()
    semantic.run()
    write_full_report()


if __name__ == "__main__":
    main()
