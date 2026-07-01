import argparse
from pathlib import Path

import pandas as pd

import metrics as metric_utils
from dataset import (
    RESULTS_DIR,
    TEST_CSV,
    TRAIN_CSV,
    duplicate_test_keys,
    embeddings_for,
    load_embeddings,
    load_features,
    load_split,
)
from features import feature_frame, labels, texts
from models import CONFIG_COMPONENTS, CONFIG_LABELS, MODEL_CONFIGS, fit_config


def build_data(train_rows, test_rows, embeddings):
    return {
        "X_train": feature_frame(train_rows),
        "X_test": feature_frame(test_rows),
        "y_train": labels(train_rows),
        "train_text": texts(train_rows),
        "test_text": texts(test_rows),
        "train_emb": embeddings_for(train_rows, embeddings),
        "test_emb": embeddings_for(test_rows, embeddings),
    }


def class_counts(rows):
    counts = pd.Series([r["label"] for r in rows]).value_counts().to_dict()
    return {c: int(counts.get(c, 0)) for c in ("LOW", "MIXED", "HIGH")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=TRAIN_CSV)
    ap.add_argument("--test", type=Path, default=TEST_CSV)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--inner-splits", type=int, default=3)
    args = ap.parse_args()

    features = load_features()
    embeddings = load_embeddings()
    train_rows = load_split(args.train, features)
    test_rows = load_split(args.test, features)
    data = build_data(train_rows, test_rows, embeddings)
    y_test = [r["label"] for r in test_rows]
    keys = [r["key"] for r in test_rows]
    dupes = duplicate_test_keys(train_rows, test_rows)

    print(f"train rows: {len(train_rows)}  {class_counts(train_rows)}")
    print(f"test rows:  {len(test_rows)}  {class_counts(test_rows)}")
    print(f"cross-split duplicate images: {len(dupes)}")

    all_metrics, all_preds, all_reports, all_confusions = [], [], [], []
    for name in MODEL_CONFIGS:
        label = CONFIG_LABELS[name]
        print(f"fitting {name} ({label})")
        preds = fit_config(name, data, args.inner_splits)
        all_metrics.append(metric_utils.scores(name, label, CONFIG_COMPONENTS[name], y_test, preds))
        all_reports.extend(metric_utils.report_rows(name, label, y_test, preds))
        all_confusions.extend(metric_utils.confusion_rows(name, label, y_test, preds))
        for key, truth, pred in zip(keys, y_test, preds):
            all_preds.append({"model": name, "key": key, "truth": truth, "pred": pred})

    metric_utils.write(args.out_dir, all_metrics, all_preds, all_reports, all_confusions)
    print(pd.DataFrame(all_metrics)[["experiment", "accuracy", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
