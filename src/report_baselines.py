"""Consolidate all active baseline metrics into task-specific result tables."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "current"
REPORT_SCHEMA = "all_baselines_report_v1"

SOURCES = {
    "factuality": [
        ("OCR multimodal", RESULTS / "metrics.csv"),
        ("OCR text-only", RESULTS / "text_only_llm_metrics.csv"),
        ("Group A/C", RESULTS / "mgm_groupA_factuality.csv"),
    ],
    "bias": [
        ("OCR multimodal", RESULTS / "multimodal_bias_metrics.csv"),
        ("OCR text-only", RESULTS / "text_only_llm_bias_metrics.csv"),
        ("Group A/C", RESULTS / "mgm_groupA_bias.csv"),
    ],
}

LABEL_MAPPINGS = {
    "factuality": "LOW=-1; MIXED=0; HIGH=1",
    "bias": "left=-2; left-center=-1; center=0; right-center=1; right=2",
}

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "mae",
    "mse",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_task_report(task):
    rows = []
    expected_n_test = None
    for family, path in SOURCES[task]:
        if not path.exists():
            raise FileNotFoundError(f"missing baseline metrics: {path}")
        frame = pd.read_csv(path)
        missing = set([
            "model", "experiment", "components", "n_test", *METRIC_COLUMNS,
        ]) - set(frame)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        values = frame[METRIC_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path} contains non-finite metrics")
        n_test_values = set(frame["n_test"].astype(int))
        if len(n_test_values) != 1:
            raise ValueError(f"{path} contains inconsistent test sizes")
        current_n_test = n_test_values.pop()
        if expected_n_test is None:
            expected_n_test = current_n_test
        elif current_n_test != expected_n_test:
            raise ValueError(
                f"{task} baselines use different test sizes: "
                f"{expected_n_test} and {current_n_test}"
            )
        for record in frame.to_dict("records"):
            rows.append({
                "task": task,
                "family": family,
                "model": record["model"],
                "experiment": record["experiment"],
                "components": record["components"],
                "n_test": int(record["n_test"]),
                "accuracy": float(record["accuracy"]),
                "macro_accuracy": float(record["balanced_accuracy"]),
                "macro_precision": float(record["macro_precision"]),
                "macro_recall": float(record["macro_recall"]),
                "macro_f1": float(record["macro_f1"]),
                "mae": float(record["mae"]),
                "mse": float(record["mse"]),
                "label_mapping": LABEL_MAPPINGS[task],
                "source_file": path.name,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    source_hashes = {}
    for task in SOURCES:
        report = build_task_report(task)
        path = args.out_dir / f"all_baselines_{task}.csv"
        report.to_csv(path, index=False)
        outputs[task] = path
        source_hashes[task] = {
            source.name: sha256_file(source) for _, source in SOURCES[task]
        }
        print(f"\n{task.upper()} ({LABEL_MAPPINGS[task]})")
        print(report[[
            "family", "experiment", "components", "accuracy", "macro_accuracy",
            "macro_precision", "macro_recall", "macro_f1", "mae", "mse",
        ]].round(2).to_string(index=False))

    manifest = {
        "schema": REPORT_SCHEMA,
        "label_mappings": LABEL_MAPPINGS,
        "sources": source_hashes,
        "outputs": {
            task: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for task, path in outputs.items()
        },
        "row_counts": {
            task: int(len(pd.read_csv(path))) for task, path in outputs.items()
        },
    }
    manifest_path = args.out_dir / "all_baselines_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
