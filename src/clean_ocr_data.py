"""Build the canonical, leakage-safe OCR dataset used by OCR-based models.

The source split, feature, screenshot, and embedding files remain unchanged.
This command writes aligned clean copies under ``data/clean/ocr`` and records
every removed row in a machine-readable manifest.
"""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"
RAW_TEST_CSV = ROOT / "data" / "splits" / "test.csv"
RAW_FEATURES = ROOT / "data" / "features" / "features.jsonl"
SAFE_EMBEDDINGS = ROOT / "data" / "embeddings" / "visual_embeddings.safe.npz"
SCREENSHOTS = ROOT / "data" / "screenshots"
OUT_DIR = ROOT / "data" / "clean" / "ocr"

SCHEMA = "ocr_canonical_dataset_v1"
FEATURE_SCHEMA = "nollm_raw12_v3_lx9_v8"
_TIMESTAMP = re.compile(r"_\d{8}_\d{6}$")

BIAS_MAP = {
    "LEFT": "left",
    "EXTREME LEFT": "left",
    "LEFT-CENTER": "left-center",
    "LEAST BIASED": "center",
    "RIGHT-CENTER": "right-center",
    "RIGHT": "right",
    "EXTREME RIGHT": "right",
}


def normalize_key(name):
    stem = _TIMESTAMP.sub("", Path(str(name)).stem.lower())
    return re.sub(r"[^a-z0-9]", "", stem)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def normalized_text(text):
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(value.split())


def load_feature_records(path):
    records = {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = str(record.get("_key", ""))
        if not key:
            raise ValueError(f"{path}:{line_number}: feature row has no _key")
        if key in records:
            raise ValueError(f"{path}:{line_number}: duplicate feature key {key!r}")
        if not record.get("ok") or record.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError(f"{path}:{line_number}: unusable feature row for {key!r}")
        if not isinstance(record.get("nollm_features"), dict):
            raise ValueError(f"{path}:{line_number}: invalid feature mapping for {key!r}")
        records[key] = record
    return records


def _string_keys(array, name):
    if array.dtype.kind not in "SU":
        raise ValueError(
            f"{name} uses unsafe object serialization; migrate the key arrays to Unicode first"
        )
    keys = [str(value) for value in array.tolist()]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} contains duplicate keys")
    return keys


def load_embedding_records(path):
    with np.load(path, allow_pickle=False) as archive:
        files = set(archive.files)
        required = {
            "top_square_embeddings",
            "top_square_keys",
            "top_fold_embeddings",
            "top_fold_keys",
        }
        if not required <= files:
            raise ValueError(f"{path} does not contain the expected two-view embeddings")
        square_keys = _string_keys(archive["top_square_keys"], "top_square_keys")
        fold_keys = _string_keys(archive["top_fold_keys"], "top_fold_keys")
        if square_keys != fold_keys:
            raise ValueError("square and fold embedding keys are not identically ordered")
        square = np.asarray(archive["top_square_embeddings"], dtype=np.float32)
        fold = np.asarray(archive["top_fold_embeddings"], dtype=np.float32)
    if square.shape != fold.shape or square.shape[0] != len(square_keys):
        raise ValueError("embedding arrays and keys have inconsistent shapes")
    mean = (square + fold) / 2.0
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    normalized = mean / np.maximum(norms, 1e-12)
    return square_keys, mean, {
        key: hash_bytes(np.ascontiguousarray(row).view(np.uint8))
        for key, row in zip(square_keys, normalized)
    }


def load_rows(train_path, test_path, features, embedding_hashes, screenshots_dir):
    rows = []
    columns = None
    for split, path in (("train", train_path), ("test", test_path)):
        frame = pd.read_csv(path)
        if columns is None:
            columns = list(frame.columns)
        elif list(frame.columns) != columns:
            raise ValueError("train and test CSV schemas differ")
        for source_index, record in enumerate(frame.to_dict("records")):
            image_file = Path(str(record.get("image_path", ""))).name
            key = normalize_key(image_file)
            if key not in features:
                raise ValueError(f"{split}[{source_index}] has no feature row for {key!r}")
            if key not in embedding_hashes:
                raise ValueError(f"{split}[{source_index}] has no embedding for {key!r}")
            image = screenshots_dir / image_file
            if not image.is_file():
                raise ValueError(f"{split}[{source_index}] is missing screenshot {image_file!r}")
            feature = features[key]
            feature_payload = json.dumps(
                feature["nollm_features"], sort_keys=True, separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            ocr = normalized_text(feature.get("ocr_text", ""))
            identifiers = {
                "outlet_key": key,
                "ocr": hash_bytes(ocr.encode("utf-8")) if ocr else None,
                "features": hash_bytes(feature_payload),
                "screenshot": sha256_file(image),
                "embedding": embedding_hashes[key],
            }
            factuality = str(record.get("label_3class", "")).strip().upper()
            factuality = factuality if factuality in {"LOW", "MIXED", "HIGH"} else None
            bias = BIAS_MAP.get(str(record.get("bias_rating", "")).strip().upper())
            rows.append({
                "row_id": len(rows),
                "split": split,
                "source_index": source_index,
                "key": key,
                "media_name": str(record.get("media_name", "")),
                "image_file": image_file,
                "factuality": factuality,
                "bias": bias,
                "identifiers": identifiers,
                "csv_record": record,
            })
    return rows, columns


def duplicate_components(rows):
    parent = list(range(len(rows)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left, right):
        left, right = root(left), root(right)
        if left != right:
            parent[right] = left

    first = {}
    for index, row in enumerate(rows):
        for kind, value in row["identifiers"].items():
            if value is None:
                continue
            identifier = (kind, value)
            previous = first.setdefault(identifier, index)
            join(index, previous)

    components = defaultdict(list)
    for index in range(len(rows)):
        components[root(index)].append(index)
    return sorted(components.values(), key=lambda group: min(group))


def component_match_types(rows, component):
    matches = []
    for kind in rows[component[0]]["identifiers"]:
        values = defaultdict(int)
        for index in component:
            value = rows[index]["identifiers"][kind]
            if value is not None:
                values[value] += 1
        if any(count > 1 for count in values.values()):
            matches.append(kind)
    return matches


def clean_rows(rows):
    kept = set()
    dropped = []
    component_reports = []
    for component in duplicate_components(rows):
        if len(component) == 1:
            kept.add(component[0])
            continue
        labels = {
            task: sorted({rows[index][task] for index in component if rows[index][task]})
            for task in ("factuality", "bias")
        }
        conflicting_tasks = sorted(task for task, values in labels.items() if len(values) > 1)
        matches = component_match_types(rows, component)
        winner = None
        if not conflicting_tasks:
            winner = min(
                component,
                key=lambda index: (
                    0 if rows[index]["split"] == "test" else 1,
                    rows[index]["key"],
                    rows[index]["image_file"],
                    rows[index]["media_name"],
                    rows[index]["source_index"],
                ),
            )
            kept.add(winner)
        for index in component:
            if index == winner:
                continue
            row = rows[index]
            dropped.append({
                "split": row["split"],
                "source_index": row["source_index"],
                "key": row["key"],
                "media_name": row["media_name"],
                "image_file": row["image_file"],
                "factuality": row["factuality"],
                "bias": row["bias"],
                "reason": "conflicting_labels" if conflicting_tasks else "duplicate_model_input",
                "conflicting_tasks": ",".join(conflicting_tasks),
                "match_types": ",".join(matches),
                "canonical_key": rows[winner]["key"] if winner is not None else "",
                "canonical_media_name": rows[winner]["media_name"] if winner is not None else "",
            })
        component_reports.append({
            "size": len(component),
            "match_types": matches,
            "conflicting_tasks": conflicting_tasks,
            "labels": labels,
            "winner": None if winner is None else {
                "split": rows[winner]["split"],
                "source_index": rows[winner]["source_index"],
                "key": rows[winner]["key"],
                "media_name": rows[winner]["media_name"],
            },
            "rows": [{
                "split": rows[index]["split"],
                "source_index": rows[index]["source_index"],
                "key": rows[index]["key"],
                "media_name": rows[index]["media_name"],
                "factuality": rows[index]["factuality"],
                "bias": rows[index]["bias"],
            } for index in component],
        })
    clean = [rows[index] for index in sorted(kept)]
    return clean, dropped, component_reports


def validate_clean_rows(rows):
    seen = {kind: {} for kind in rows[0]["identifiers"]}
    for row in rows:
        for kind, value in row["identifiers"].items():
            if value is None:
                continue
            if value in seen[kind]:
                other = seen[kind][value]
                raise ValueError(
                    f"clean rows still share {kind}: {other['key']!r} and {row['key']!r}"
                )
            seen[kind][value] = row
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    if not train or not test:
        raise ValueError("clean dataset must contain both train and test rows")


def write_outputs(out_dir, rows, columns, dropped, components, features,
                  embedding_keys, embedding_matrix, sources):
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split in ("train", "test"):
        frame = pd.DataFrame(
            [row["csv_record"] for row in rows if row["split"] == split],
            columns=columns,
        )
        path = out_dir / f"{split}.csv"
        frame.to_csv(path, index=False)
        outputs[split] = path

    selected_keys = sorted({row["key"] for row in rows})
    feature_path = out_dir / "features.jsonl"
    feature_path.write_text(
        "".join(json.dumps(features[key], ensure_ascii=False) + "\n" for key in selected_keys),
        encoding="utf-8",
    )
    outputs["features"] = feature_path

    embedding_index = {key: index for index, key in enumerate(embedding_keys)}
    selected_embeddings = np.stack([embedding_matrix[embedding_index[key]] for key in selected_keys])
    embedding_path = out_dir / "visual_embeddings.npz"
    np.savez_compressed(
        embedding_path,
        embeddings=selected_embeddings.astype(np.float32),
        keys=np.asarray(selected_keys, dtype=str),
        dataset_schema=np.asarray(SCHEMA),
    )
    outputs["embeddings"] = embedding_path

    dropped_path = out_dir / "dropped_rows.csv"
    pd.DataFrame(dropped).to_csv(dropped_path, index=False)
    outputs["dropped_rows"] = dropped_path

    manifest = {
        "schema": SCHEMA,
        "policy": {
            "duplicate_identifiers": [
                "normalized outlet key",
                "normalized exact OCR",
                "exact compact feature vector",
                "exact screenshot bytes",
                "exact normalized two-view visual embedding",
            ],
            "conflicting_duplicate_labels": "drop the complete connected component",
            "same_label_duplicates": "keep one deterministic row, preferring test",
            "raw_sources_mutated": False,
        },
        "sources": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
        "counts": {
            "raw_train": sum(row["split"] == "train" for row in rows) +
                         sum(item["split"] == "train" for item in dropped),
            "raw_test": sum(row["split"] == "test" for row in rows) +
                        sum(item["split"] == "test" for item in dropped),
            "clean_train": sum(row["split"] == "train" for row in rows),
            "clean_test": sum(row["split"] == "test" for row in rows),
            "dropped_train": sum(item["split"] == "train" for item in dropped),
            "dropped_test": sum(item["split"] == "test" for item in dropped),
            "duplicate_components": len(components),
            "conflicting_components": sum(bool(item["conflicting_tasks"]) for item in components),
            "clean_feature_rows": len(selected_keys),
            "clean_embedding_rows": len(selected_keys),
        },
        "class_counts": {
            split: {
                task: dict(sorted(pd.Series([
                    row[task] for row in rows
                    if row["split"] == split and row[task]
                ]).value_counts().astype(int).to_dict().items()))
                for task in ("factuality", "bias")
            }
            for split in ("train", "test")
        },
        "duplicate_components": components,
    }
    manifest["outputs"] = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in outputs.items()
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build(args):
    features = load_feature_records(args.features)
    embedding_keys, embedding_matrix, embedding_hashes = load_embedding_records(args.embeddings)
    rows, columns = load_rows(
        args.train, args.test, features, embedding_hashes, args.screenshots,
    )
    clean, dropped, components = clean_rows(rows)
    validate_clean_rows(clean)
    manifest = write_outputs(
        args.out_dir,
        clean,
        columns,
        dropped,
        components,
        features,
        embedding_keys,
        embedding_matrix,
        {
            "train": args.train,
            "test": args.test,
            "features": args.features,
            "embeddings": args.embeddings,
        },
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(f"wrote canonical OCR dataset -> {args.out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=RAW_TRAIN_CSV)
    parser.add_argument("--test", type=Path, default=RAW_TEST_CSV)
    parser.add_argument("--features", type=Path, default=RAW_FEATURES)
    parser.add_argument("--embeddings", type=Path, default=SAFE_EMBEDDINGS)
    parser.add_argument("--screenshots", type=Path, default=SCREENSHOTS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
