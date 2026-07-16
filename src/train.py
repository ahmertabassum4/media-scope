import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import metrics as metric_utils
from dataset import (
    CLEAN_MANIFEST,
    RESULTS_DIR,
    TASKS,
    TEST_CSV,
    TRAIN_CSV,
    embeddings_for,
    load_embeddings,
    load_features,
    load_split,
    validate_clean_dataset,
)
from features import feature_frame, labels, texts
from models import (
    BENCHMARK_ORDER,
    CONFIG_COMPONENTS,
    CONFIG_LABELS,
    fit_config,
    fit_experts,
    majority_baseline,
)


OUTPUT_FILES = {
    "factuality": {
        "metrics": "metrics.csv",
        "predictions": "predictions.csv",
        "report": "classification_report.csv",
        "confusion": "confusion_matrix.csv",
        "plot": "accuracy_f1.png",
        "manifest": "multimodal_run_manifest.json",
    },
    "bias": {
        "metrics": "multimodal_bias_metrics.csv",
        "predictions": "multimodal_bias_predictions.csv",
        "report": "multimodal_bias_classification_report.csv",
        "confusion": "multimodal_bias_confusion_matrix.csv",
        "plot": "multimodal_bias_accuracy_f1.png",
        "manifest": "multimodal_bias_run_manifest.json",
    },
}


def build_data(train_rows, test_rows, embeddings, task):
    task_config = TASKS[task]
    return {
        "X_train": feature_frame(train_rows),
        "X_test": feature_frame(test_rows),
        "y_train": labels(train_rows),
        "train_text": texts(train_rows),
        "test_text": texts(test_rows),
        "train_emb": embeddings_for(train_rows, embeddings),
        "test_emb": embeddings_for(test_rows, embeddings),
        "class_order": task_config["classes"],
        "selection_label": "MIXED" if task == "factuality" else None,
    }


def class_counts(rows, classes):
    counts = pd.Series([r["label"] for r in rows]).value_counts().to_dict()
    return {class_name: int(counts.get(class_name, 0)) for class_name in classes}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Run multimodal outlet classification ablations.")
    parser.add_argument("--task", choices=sorted(TASKS), default="factuality")
    parser.add_argument("--train", type=Path, default=TRAIN_CSV)
    parser.add_argument("--test", type=Path, default=TEST_CSV)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--inner-splits", type=int, default=3)
    args = parser.parse_args()

    task_config = TASKS[args.task]
    classes = task_config["classes"]
    ordinal = task_config["ordinal"]
    filenames = OUTPUT_FILES[args.task]

    features = load_features()
    embeddings = load_embeddings()
    train_rows = load_split(args.train, features, args.task)
    test_rows = load_split(args.test, features, args.task)
    audit = validate_clean_dataset(train_rows, test_rows, embeddings, classes)
    data = build_data(train_rows, test_rows, embeddings, args.task)
    y_test = [r["label"] for r in test_rows]
    keys = [r["key"] for r in test_rows]

    print(f"task: {args.task}")
    print(f"train rows: {len(train_rows)}  {class_counts(train_rows, classes)}")
    print(f"test rows:  {len(test_rows)}  {class_counts(test_rows, classes)}")
    print("clean dataset audit: PASS")

    print("fitting shared feature, OCR, and DINO experts")
    experts = fit_experts(data, args.inner_splits)
    all_metrics, all_preds, all_reports, all_confusions = [], [], [], []
    for name in BENCHMARK_ORDER:
        label = CONFIG_LABELS[name]
        print(f"fitting {name} ({label})")
        if name == "majority":
            preds = majority_baseline(data["y_train"], len(y_test))
        else:
            preds = fit_config(name, data, args.inner_splits, experts)
        all_metrics.append(metric_utils.scores(
            name, label, CONFIG_COMPONENTS[name], y_test, preds, classes, ordinal,
        ))
        all_reports.extend(metric_utils.report_rows(name, label, y_test, preds, classes))
        all_confusions.extend(metric_utils.confusion_rows(name, label, y_test, preds, classes))
        for key, truth, pred in zip(keys, y_test, preds):
            all_preds.append({"model": name, "key": key, "truth": truth, "pred": pred})

    metric_utils.write(
        args.out_dir, all_metrics, all_preds, all_reports, all_confusions,
        filenames, args.task,
    )
    dataset_manifest = CLEAN_MANIFEST if args.train == TRAIN_CSV and args.test == TEST_CSV else None
    output_hashes = {
        name: {
            "path": str((args.out_dir / filename).resolve()),
            "sha256": sha256_file(args.out_dir / filename),
        }
        for name, filename in filenames.items()
        if name != "manifest"
    }
    run_manifest = {
        "schema": "multimodal_run_v3",
        "task": args.task,
        "classes": list(classes),
        "ordinal": ordinal,
        "train_csv": str(args.train.resolve()),
        "test_csv": str(args.test.resolve()),
        "inner_splits": args.inner_splits,
        "dataset_audit": audit,
        "dataset_manifest": str(dataset_manifest.resolve()) if dataset_manifest else None,
        "dataset_manifest_sha256": sha256_file(dataset_manifest) if dataset_manifest else None,
        "models": BENCHMARK_ORDER,
        "n_predictions": len(all_preds),
        "metrics": all_metrics,
        "outputs": output_hashes,
    }
    (args.out_dir / filenames["manifest"]).write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(pd.DataFrame(all_metrics)[["experiment", "accuracy", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
