import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"
RAW_TEST_CSV = ROOT / "data" / "splits" / "test.csv"
RAW_FEATURES_CACHE = ROOT / "data" / "features" / "features.jsonl"
RAW_EMBEDDINGS_NPZ = ROOT / "data" / "embeddings" / "visual_embeddings.npz"
CLEAN_DATA_DIR = ROOT / "data" / "clean" / "ocr"
TRAIN_CSV = CLEAN_DATA_DIR / "train.csv"
TEST_CSV = CLEAN_DATA_DIR / "test.csv"
FEATURES_CACHE = CLEAN_DATA_DIR / "features.jsonl"
EMBEDDINGS_NPZ = CLEAN_DATA_DIR / "visual_embeddings.npz"
CLEAN_MANIFEST = CLEAN_DATA_DIR / "manifest.json"
SCREENSHOTS_DIR = ROOT / "data" / "screenshots"
RESULTS_DIR = ROOT / "results" / "current"

FACTUALITY_CLASSES = ("LOW", "MIXED", "HIGH")
BIAS_CLASSES = ("left", "left-center", "center", "right-center", "right")
BIAS_MAP = {
    "LEFT": "left",
    "EXTREME LEFT": "left",
    "LEFT-CENTER": "left-center",
    "LEAST BIASED": "center",
    "RIGHT-CENTER": "right-center",
    "RIGHT": "right",
    "EXTREME RIGHT": "right",
}
TASKS = {
    "factuality": {
        "classes": FACTUALITY_CLASSES,
        "ordinal": {"LOW": -1.0, "MIXED": 0.0, "HIGH": 1.0},
    },
    "bias": {
        "classes": BIAS_CLASSES,
        "ordinal": {
            "left": -2.0,
            "left-center": -1.0,
            "center": 0.0,
            "right-center": 1.0,
            "right": 2.0,
        },
    },
}

# Backward-compatible factuality aliases used by the original pipeline.
CLASS_ORDER = list(FACTUALITY_CLASSES)
ORDINAL = TASKS["factuality"]["ordinal"]

FEATURE_SCHEMA = "nollm_raw12_v3_lx9_v8"
_TIMESTAMP = re.compile(r"_\d{8}_\d{6}$")


def normalize_key(name):
    stem = _TIMESTAMP.sub("", Path(str(name)).stem.lower())
    return re.sub(r"[^a-z0-9]", "", stem)


def image_path(value):
    return SCREENSHOTS_DIR / Path(str(value)).name


def load_features(path=FEATURES_CACHE):
    if not path.exists():
        raise FileNotFoundError(
            f"canonical feature cache missing at {path}; run `python src/clean_ocr_data.py`"
        )
    cache = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        feats = row.get("nollm_features")
        if row.get("ok") and row.get("feature_schema") == FEATURE_SCHEMA and isinstance(feats, dict):
            key = str(row["_key"])
            if key in cache:
                raise ValueError(f"{path}:{line_number}: duplicate feature key {key!r}")
            cache[key] = row
    return cache


def record_label(record, task):
    if task == "factuality":
        label = str(record.get("label_3class", "")).strip().upper()
        return label if label in FACTUALITY_CLASSES else None
    if task == "bias":
        return BIAS_MAP.get(str(record.get("bias_rating", "")).strip().upper())
    raise ValueError(f"unknown task: {task!r}")


def load_split(path, features, task="factuality"):
    rows = []
    for record in pd.read_csv(path).to_dict("records"):
        name = Path(str(record.get("image_path", ""))).name
        key = normalize_key(name)
        label = record_label(record, task)
        if label is None or key not in features:
            continue
        cached = features[key]
        rows.append({
            "key": key,
            "label": label,
            "path": str(image_path(name)),
            "media_name": str(record.get("media_name", "")),
            "image_file": name,
            "features": cached["nollm_features"],
            "text": str(cached.get("ocr_text") or ""),
        })
    if not rows:
        raise SystemExit(f"No usable rows loaded from {path}")
    return rows


def _l2(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def load_embeddings(path=EMBEDDINGS_NPZ):
    if not path.exists():
        raise FileNotFoundError(
            f"canonical embeddings missing at {path}; run `python src/clean_ocr_data.py`"
        )
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        if {"top_square_embeddings", "top_square_keys", "top_fold_embeddings", "top_fold_keys"} <= files:
            key_array = z["top_square_keys"]
            fold_key_array = z["top_fold_keys"]
            if key_array.dtype.kind not in "SU" or fold_key_array.dtype.kind not in "SU":
                raise ValueError(f"{path}: embedding keys must use safe Unicode/string arrays")
            keys = [str(k) for k in key_array.tolist()]
            fold_keys = [str(k) for k in fold_key_array.tolist()]
            if keys != fold_keys:
                raise ValueError(f"{path}: square and fold embedding keys are misaligned")
            square = np.asarray(z["top_square_embeddings"], dtype=np.float32)
            fold = np.asarray(z["top_fold_embeddings"], dtype=np.float32)
            matrix = _l2((square + fold) / 2.0)
        elif {"embeddings", "keys"} <= files:
            key_array = z["keys"]
            if key_array.dtype.kind not in "SU":
                raise ValueError(f"{path}: embedding keys must use safe Unicode/string arrays")
            keys = [str(k) for k in key_array.tolist()]
            matrix = _l2(np.asarray(z["embeddings"], dtype=np.float32))
        else:
            raise ValueError(f"{path}: unrecognized embedding arrays")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate embedding keys")
    if matrix.ndim != 2 or matrix.shape[0] != len(keys):
        raise ValueError(f"{path}: embedding shape does not match keys")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: embeddings contain non-finite values")
    return pd.DataFrame(matrix, index=pd.Index(keys, name="key"))


def embeddings_for(rows, embeddings):
    keys = [row["key"] for row in rows]
    missing = [k for k in keys if k not in embeddings.index]
    if missing:
        raise SystemExit(f"{len(missing)} rows missing embeddings, e.g. {missing[:10]}")
    return embeddings.loc[keys].reset_index(drop=True)


def _stable_hash(value):
    return hashlib.sha256(value).hexdigest()


def _row_fingerprints(row):
    feature_payload = json.dumps(
        row["features"], sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    fingerprints = {
        "key": row["key"],
        "features": _stable_hash(feature_payload),
        "screenshot": _sha256(Path(row["path"])),
    }
    normalized_ocr = " ".join(row["text"].casefold().split())
    if normalized_ocr:
        fingerprints["ocr"] = _stable_hash(normalized_ocr.encode("utf-8"))
    return fingerprints


def validate_clean_dataset(train_rows, test_rows, embeddings, classes=CLASS_ORDER):
    """Fail closed if any exact model input is repeated or crosses partitions."""
    partitions = {"train": train_rows, "test": test_rows}
    summary = {}
    fingerprint_sets = {}
    for name, rows in partitions.items():
        if not rows:
            raise ValueError(f"{name} partition is empty")
        fingerprints = {kind: [] for kind in ("key", "ocr", "features", "screenshot")}
        for row in rows:
            for kind, value in _row_fingerprints(row).items():
                fingerprints[kind].append(value)
        keys = [row["key"] for row in rows]
        embedding_matrix = embeddings.loc[keys].to_numpy(dtype=np.float32)
        fingerprints["embedding"] = [
            _stable_hash(np.ascontiguousarray(vector).view(np.uint8))
            for vector in embedding_matrix
        ]
        for kind, values in fingerprints.items():
            if len(values) != len(set(values)):
                raise ValueError(f"{name} still contains duplicate {kind} model inputs")
        fingerprint_sets[name] = {kind: set(values) for kind, values in fingerprints.items()}
        summary[name] = {"rows": len(rows), "class_counts": {
            label: sum(row["label"] == label for row in rows) for label in classes
        }}
    for kind in fingerprint_sets["train"]:
        overlap = fingerprint_sets["train"][kind] & fingerprint_sets["test"][kind]
        if overlap:
            raise ValueError(f"train and test share {len(overlap)} {kind} model inputs")
    return summary


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def duplicate_test_keys(train_rows, test_rows):
    train_hashes = set()
    for row in train_rows:
        p = Path(row["path"])
        if p.exists():
            train_hashes.add(_sha256(p))
    dupes = set()
    for row in test_rows:
        p = Path(row["path"])
        if p.exists() and _sha256(p) in train_hashes:
            dupes.add(row["key"])
    return dupes
