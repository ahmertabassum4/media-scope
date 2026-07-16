"""Fine-tune a text-only classifier on screenshot OCR without split leakage."""

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Reuse the project's dataset loaders from src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dataset import (  # noqa: E402
    CLEAN_MANIFEST,
    RESULTS_DIR,
    TEST_CSV,
    TRAIN_CSV,
    load_features,
    normalize_key,
)


RUN_SCHEMA = "text_only_llm_run_v3"
INPUT_FILTER_SCHEMA = "text_only_llm_model_input_dedup_v1"
DEFAULT_SEED = 42
RUNS_DIR = Path(__file__).resolve().parent / "models"

# 5-level MBFC bias, extremes merged into left/right (matches baselines/MGM);
# UNRATED is excluded because it is not a target class.
BIAS_MAP = {
    "LEFT": "left", "EXTREME LEFT": "left",
    "LEFT-CENTER": "left-center",
    "LEAST BIASED": "center",
    "RIGHT-CENTER": "right-center",
    "RIGHT": "right", "EXTREME RIGHT": "right",
}

TASKS = {
    "factuality": {
        "csv_column": "label_3class",
        "mapper": lambda value: value if value in ("LOW", "MIXED", "HIGH") else None,
        "classes": ["LOW", "MIXED", "HIGH"],
        "ordinal": {"LOW": -1.0, "MIXED": 0.0, "HIGH": 1.0},
        "out": "text_only_llm_metrics.csv",
    },
    "bias": {
        "csv_column": "bias_rating",
        "mapper": lambda value: BIAS_MAP.get(value),
        "classes": ["left", "left-center", "center", "right-center", "right"],
        "ordinal": {
            "left": -2.0,
            "left-center": -1.0,
            "center": 0.0,
            "right-center": 1.0,
            "right": 2.0,
        },
        "out": "text_only_llm_bias_metrics.csv",
    },
}


def load_task_split(path, features, task):
    """Load labeled OCR rows without silently deduplicating source records."""
    cfg = TASKS[task]
    rows = []
    for source_index, record in enumerate(pd.read_csv(path).to_dict("records")):
        image_path = Path(str(record.get("image_path", ""))).name
        key = normalize_key(image_path)
        label = cfg["mapper"](str(record.get(cfg["csv_column"], "")).strip().upper())
        if label is None or key not in features:
            continue
        text = str(features[key].get("ocr_text") or "")
        if not text.strip():
            continue
        rows.append({
            "key": key,
            "label": label,
            "text": text,
            "media_name": str(record.get("media_name", "")),
            "image_path": image_path,
            "source_key": str(features[key].get("source_key") or key),
            "source_index": source_index,
        })
    if not rows:
        raise ValueError(f"no usable rows for task={task} from {path}")
    return rows


def add_model_input_fingerprints(rows, tokenizer, max_len, batch_size=128):
    """Fingerprint the exact truncated token sequence visible to the model."""
    if max_len < 1:
        raise ValueError("max_len must be positive")
    prepared = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        encoded = tokenizer(
            [row["text"] for row in chunk],
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        )
        for row, token_ids in zip(chunk, encoded["input_ids"]):
            updated = dict(row)
            updated["input_fingerprint"] = hashlib.sha256(
                np.asarray(token_ids, dtype=np.int32).tobytes()
            ).hexdigest()
            prepared.append(updated)
    return prepared


def _row_sort_key(row):
    return (
        str(row["key"]),
        str(row.get("image_path", "")),
        str(row.get("media_name", "")),
        int(row.get("source_index", -1)),
    )


def _dropped_row(row, partition, reason, canonical=None):
    result = {
        "partition": partition,
        "reason": reason,
        "key": row["key"],
        "label": row["label"],
        "media_name": row.get("media_name", ""),
        "image_path": row.get("image_path", ""),
        "source_key": row.get("source_key", ""),
        "source_index": int(row.get("source_index", -1)),
        "input_fingerprint": row["input_fingerprint"],
    }
    if canonical is not None:
        result["canonical_key"] = canonical["key"]
        result["canonical_media_name"] = canonical.get("media_name", "")
    return result


def deduplicate_partition(rows, partition, reject_conflicting_labels=False):
    """Keep one deterministic copy per model input, rejecting ambiguous test labels."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["input_fingerprint"]].append(row)

    kept, dropped = [], []
    duplicate_groups = 0
    conflicting_groups = 0
    conflicting_rows = 0
    for _, group in sorted(groups.items()):
        group = sorted(group, key=_row_sort_key)
        labels = sorted({row["label"] for row in group})
        if len(group) > 1:
            duplicate_groups += 1
        if len(labels) > 1:
            conflicting_groups += 1
            conflicting_rows += len(group)
            if reject_conflicting_labels:
                examples = ", ".join(f"{row['key']}={row['label']}" for row in group)
                raise ValueError(
                    f"{partition} has identical model inputs with conflicting labels: {examples}"
                )
            dropped.extend(
                _dropped_row(row, partition, "conflicting_labels_for_identical_input")
                for row in group
            )
            continue
        canonical = group[0]
        kept.append(canonical)
        dropped.extend(
            _dropped_row(row, partition, "duplicate_model_input", canonical)
            for row in group[1:]
        )

    report = {
        "rows_in": len(rows),
        "rows_kept": len(kept),
        "duplicate_input_groups": duplicate_groups,
        "duplicate_rows_removed": sum(
            1 for row in dropped if row["reason"] == "duplicate_model_input"
        ),
        "conflicting_label_groups": conflicting_groups,
        "conflicting_rows_removed": conflicting_rows,
    }
    return kept, dropped, report


def remove_training_inputs_seen_in_test(train_rows, test_rows):
    """Keep the held-out test copy and remove matching train inputs."""
    test_by_input = {row["input_fingerprint"]: row for row in test_rows}
    kept, dropped = [], []
    label_conflicts = 0
    for row in train_rows:
        test_row = test_by_input.get(row["input_fingerprint"])
        if test_row is None:
            kept.append(row)
            continue
        if row["label"] != test_row["label"]:
            label_conflicts += 1
        dropped.append(_dropped_row(row, "train", "matches_held_out_test_input", test_row))
    return kept, dropped, {
        "rows_in": len(train_rows),
        "rows_kept": len(kept),
        "rows_removed": len(dropped),
        "label_conflicts_removed": label_conflicts,
    }


def stratified_subset(rows, limit, seed):
    """Take a deterministic, label-stratified subset for a smoke test."""
    if not limit or limit >= len(rows):
        return rows
    labels = [row["label"] for row in rows]
    if limit < len(set(labels)):
        raise ValueError(
            f"--limit={limit} is too small to retain all {len(set(labels))} classes"
        )
    try:
        selected, _ = train_test_split(
            np.arange(len(rows)),
            train_size=limit,
            stratify=labels,
            random_state=seed,
        )
    except ValueError as exc:
        raise ValueError(f"cannot build a stratified --limit subset: {exc}") from exc
    return [rows[index] for index in sorted(selected)]


def validate_partitions(partitions):
    """Fail closed if any actual model input appears in multiple partitions."""
    names = tuple(partitions)
    for name, rows in partitions.items():
        fingerprints = [row["input_fingerprint"] for row in rows]
        keys = [row["key"] for row in rows]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError(f"{name} still has duplicate model inputs")
        if len(keys) != len(set(keys)):
            raise ValueError(f"{name} still has duplicate normalized outlet keys")
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            left_inputs = {row["input_fingerprint"] for row in partitions[left]}
            right_inputs = {row["input_fingerprint"] for row in partitions[right]}
            if left_inputs & right_inputs:
                raise ValueError(f"model-input leakage between {left} and {right}")
            left_keys = {row["key"] for row in partitions[left]}
            right_keys = {row["key"] for row in partitions[right]}
            if left_keys & right_keys:
                raise ValueError(f"outlet-key leakage between {left} and {right}")


def class_counts(rows, classes):
    counts = Counter(row["label"] for row in rows)
    return {label: int(counts[label]) for label in classes}


def partition_manifest(rows, classes):
    records = [
        {
            "key": row["key"],
            "label": row["label"],
            "media_name": row.get("media_name", ""),
            "image_path": row.get("image_path", ""),
            "source_key": row.get("source_key", ""),
            "input_fingerprint": row["input_fingerprint"],
        }
        for row in rows
    ]
    payload = json.dumps(records, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "n_rows": len(records),
        "class_counts": class_counts(rows, classes),
        "fingerprint": hashlib.sha256(payload).hexdigest(),
        "records": records,
    }


def prepare_task_rows(
    task,
    train_path,
    test_path,
    features,
    tokenizer,
    max_len,
    val_frac,
    seed,
    limit=0,
):
    """Load, filter, and split rows so no model input crosses a partition."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("--val-frac must be between 0 and 1")
    raw_train = add_model_input_fingerprints(
        load_task_split(train_path, features, task), tokenizer, max_len
    )
    raw_test = add_model_input_fingerprints(
        load_task_split(test_path, features, task), tokenizer, max_len
    )
    test_rows, dropped_test, test_report = deduplicate_partition(
        raw_test, "test", reject_conflicting_labels=True
    )
    train_rows, dropped_train, train_report = deduplicate_partition(raw_train, "train")
    train_rows, dropped_cross_split, cross_split_report = remove_training_inputs_seen_in_test(
        train_rows, test_rows
    )
    before_limit = len(train_rows)
    train_rows = stratified_subset(train_rows, limit, seed)

    classes = TASKS[task]["classes"]
    labels = [row["label"] for row in train_rows]
    try:
        fit_indices, validation_indices = train_test_split(
            np.arange(len(train_rows)),
            test_size=val_frac,
            stratify=labels,
            random_state=seed,
        )
    except ValueError as exc:
        raise ValueError(f"cannot create a stratified validation split: {exc}") from exc
    partitions = {
        "train": [train_rows[index] for index in sorted(fit_indices)],
        "validation": [train_rows[index] for index in sorted(validation_indices)],
        "test": test_rows,
    }
    validate_partitions(partitions)

    audit = {
        "schema": INPUT_FILTER_SCHEMA,
        "raw_rows": {"train": len(raw_train), "test": len(raw_test)},
        "within_partition": {"train": train_report, "test": test_report},
        "train_inputs_seen_in_test": cross_split_report,
        "limit": {
            "requested": int(limit),
            "rows_before_limit": before_limit,
            "rows_after_limit": len(train_rows),
        },
        "dropped_rows": dropped_train + dropped_cross_split + dropped_test,
        "partitions": {
            name: partition_manifest(rows, classes)
            for name, rows in partitions.items()
        },
    }
    return partitions, audit


def scores(model, experiment, components, y_true, y_pred, task):
    """One metrics row (same columns as src/metrics.py) for an arbitrary task."""
    classes = TASKS[task]["classes"]
    ordinal = TASKS[task]["ordinal"]
    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    truth = np.array([ordinal[label] for label in y_true], dtype=float)
    predictions = np.array([ordinal[label] for label in y_pred], dtype=float)
    row = {
        "model": model,
        "experiment": experiment,
        "components": components,
        "n_test": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred) * 100.0,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) * 100.0,
        "macro_precision": precision_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ) * 100.0,
        "macro_recall": recall_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ) * 100.0,
        "macro_f1": f1_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ) * 100.0,
    }
    for label in classes:
        row[f"{label}_f1"] = report[label]["f1-score"] * 100.0
    row["mae"] = float(np.mean(np.abs(truth - predictions)))
    row["mse"] = float(np.mean((truth - predictions) ** 2))
    return row


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def make_macro_f1_metric(num_labels):
    labels = list(range(num_labels))

    def compute_metrics(eval_pred):
        logits, truth = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        return {
            "macro_f1": f1_score(
                truth,
                np.asarray(logits).argmax(axis=-1),
                labels=labels,
                average="macro",
                zero_division=0,
            )
        }

    return compute_metrics


def build_dataset(rows, tokenizer, max_len):
    dataset = Dataset.from_dict({
        "text": [row["text"] for row in rows],
        "labels": [row["label_id"] for row in rows],
    })
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_len),
        batched=True,
        remove_columns=["text"],
    )


def validate_max_len(tokenizer, max_len):
    if max_len < 1:
        raise ValueError("--max-len must be positive")
    tokenizer_limit = int(getattr(tokenizer, "model_max_length", 0) or 0)
    if 0 < tokenizer_limit < 1_000_000 and max_len > tokenizer_limit:
        raise ValueError(
            f"--max-len={max_len} exceeds tokenizer limit {tokenizer_limit}"
        )


def ensure_empty_run_dir(run_dir):
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(
            f"{run_dir} is not empty; choose a new --run-dir to avoid overwriting a saved model"
        )
    run_dir.mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_preprocessing_summary(task, partitions, audit):
    classes = TASKS[task]["classes"]
    print(f"input filter: {audit['schema']}")
    print(
        "dedup: "
        f"train raw={audit['raw_rows']['train']} -> "
        f"{audit['train_inputs_seen_in_test']['rows_kept']} before validation; "
        f"test raw={audit['raw_rows']['test']} -> {len(partitions['test'])}"
    )
    for name, rows in partitions.items():
        print(f"{name}={len(rows)} labels={class_counts(rows, classes)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS), default="factuality")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--train", type=Path, default=TRAIN_CSV)
    parser.add_argument("--test", type=Path, default=TEST_CSV)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap deduplicated train-source rows with a stratified subset (smoke tests)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="directory for the best model, exact split manifest, and predictions",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="metrics CSV (default results/current task-specific file)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate joins, filtering, and splits without fine-tuning",
    )
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")

    set_seed(args.seed)
    classes = TASKS[args.task]["classes"]
    label2id = {label: index for index, label in enumerate(classes)}
    id2label = {index: label for label, index in label2id.items()}
    features = load_features()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    validate_max_len(tokenizer, args.max_len)
    partitions, audit = prepare_task_rows(
        args.task,
        args.train,
        args.test,
        features,
        tokenizer,
        args.max_len,
        args.val_frac,
        args.seed,
        args.limit,
    )
    for rows in partitions.values():
        for row in rows:
            row["label_id"] = label2id[row["label"]]
    print_preprocessing_summary(args.task, partitions, audit)
    if args.dry_run:
        print("dry run: preprocessing and leakage checks passed")
        return

    run_dir = args.run_dir or RUNS_DIR / args.task
    ensure_empty_run_dir(run_dir)
    train_ds = build_dataset(partitions["train"], tokenizer, args.max_len)
    validation_ds = build_dataset(partitions["validation"], tokenizer, args.max_len)
    test_ds = build_dataset(partitions["test"], tokenizer, args.max_len)
    counts = np.bincount(
        [row["label_id"] for row in partitions["train"]],
        minlength=len(classes),
    )
    class_weights = torch.tensor(
        len(partitions["train"]) / (len(classes) * np.maximum(counts, 1)),
        dtype=torch.float,
    )
    print(
        f"task={args.task} model={args.model} max_len={args.max_len} "
        f"train={len(partitions['train'])} val={len(partitions['validation'])} "
        f"test={len(partitions['test'])} classes={classes}"
    )
    print(
        f"train counts={dict(zip(classes, counts.tolist()))} "
        f"weights={[round(weight, 2) for weight in class_weights.tolist()]}"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id,
    )
    with tempfile.TemporaryDirectory(prefix=f"text_only_{args.task}_ckpt_") as checkpoint_dir:
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=max(2, args.batch_size),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=20,
            disable_tqdm=False,
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
            dataloader_pin_memory=not torch.backends.mps.is_available(),
        )
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=validation_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=make_macro_f1_metric(len(classes)),
            class_weights=class_weights,
        )
        trainer.train()
        prediction_output = trainer.predict(test_ds)
        trainer.save_model(run_dir)
        tokenizer.save_pretrained(run_dir)

    logits = prediction_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    prediction_ids = np.asarray(logits).argmax(axis=-1)
    predicted_labels = [id2label[int(index)] for index in prediction_ids]
    true_labels = [row["label"] for row in partitions["test"]]
    row = scores(
        "text_only_llm",
        "Text-only LLM",
        (
            f"fine-tuned {args.model} on OCR text (max_len={args.max_len}); "
            f"{INPUT_FILTER_SCHEMA}"
        ),
        true_labels,
        predicted_labels,
        args.task,
    )
    row["run_schema"] = RUN_SCHEMA
    row["input_filter_schema"] = INPUT_FILTER_SCHEMA
    row["run_dir"] = str(run_dir)

    manifest = {
        "schema": RUN_SCHEMA,
        "task": args.task,
        "model": args.model,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "validation_fraction": args.val_frac,
        "seed": args.seed,
        "classes": classes,
        "train_csv": str(args.train),
        "test_csv": str(args.test),
        "canonical_dataset_manifest": (
            str(CLEAN_MANIFEST.resolve())
            if args.train == TRAIN_CSV and args.test == TEST_CSV else None
        ),
        "canonical_dataset_manifest_sha256": (
            sha256_file(CLEAN_MANIFEST)
            if args.train == TRAIN_CSV and args.test == TEST_CSV else None
        ),
        "input_filter": audit,
        "test_metrics": row,
    }
    write_json(run_dir / "run_manifest.json", manifest)
    pd.DataFrame(
        {
            "key": [record["key"] for record in partitions["test"]],
            "media_name": [record["media_name"] for record in partitions["test"]],
            "truth": true_labels,
            "pred": predicted_labels,
            "input_fingerprint": [
                record["input_fingerprint"] for record in partitions["test"]
            ],
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)

    metrics_out = args.metrics_out or RESULTS_DIR / TASKS[args.task]["out"]
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(metrics_out, index=False)
    print(f"\n=== Text-only LLM ({args.task}) test metrics ===")
    metric_keys = (
        ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"]
        + [f"{label}_f1" for label in classes]
        + ["mae", "mse"]
    )
    for key in metric_keys:
        print(f"  {key}: {row[key]:.4f}")
    print(f"wrote {metrics_out}")
    print(f"saved best model and split manifest to {run_dir}")


if __name__ == "__main__":
    main()
