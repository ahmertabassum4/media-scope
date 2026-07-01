import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"
TEST_CSV = ROOT / "data" / "splits" / "test.csv"
FEATURES_CACHE = ROOT / "data" / "features" / "features.jsonl"
EMBEDDINGS_NPZ = ROOT / "data" / "embeddings" / "visual_embeddings.npz"
SCREENSHOTS_DIR = ROOT / "data" / "screenshots"
RESULTS_DIR = ROOT / "results" / "current"

CLASS_ORDER = ["LOW", "MIXED", "HIGH"]
ORDINAL = {"LOW": -1.0, "MIXED": 0.0, "HIGH": 1.0}

FEATURE_SCHEMA = "nollm_raw12_v3_lx9_v8"
_TIMESTAMP = re.compile(r"_\d{8}_\d{6}$")


def normalize_key(name):
    stem = _TIMESTAMP.sub("", Path(str(name)).stem.lower())
    return re.sub(r"[^a-z0-9]", "", stem)


def image_path(value):
    return SCREENSHOTS_DIR / Path(str(value)).name


def load_features(path=FEATURES_CACHE):
    cache = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        feats = row.get("nollm_features")
        if row.get("ok") and row.get("feature_schema") == FEATURE_SCHEMA and isinstance(feats, dict):
            cache[str(row["_key"])] = row
    return cache


def load_split(path, features):
    rows = []
    for record in pd.read_csv(path).to_dict("records"):
        name = Path(str(record.get("image_path", ""))).name
        key = normalize_key(name)
        label = str(record.get("label_3class", "")).upper()
        if label not in CLASS_ORDER or key not in features:
            continue
        cached = features[key]
        rows.append({
            "key": key,
            "label": label,
            "path": str(image_path(name)),
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
    z = np.load(path, allow_pickle=True)
    files = set(z.files)
    if {"top_square_embeddings", "top_square_keys", "top_fold_embeddings", "top_fold_keys"} <= files:
        keys = [str(k) for k in z["top_square_keys"].tolist()]
        square = np.asarray(z["top_square_embeddings"], dtype=np.float32)
        fold = np.asarray(z["top_fold_embeddings"], dtype=np.float32)
        matrix = _l2((square + fold) / 2.0)
    elif {"embeddings", "keys"} <= files:
        keys = [str(k) for k in z["keys"].tolist()]
        matrix = _l2(np.asarray(z["embeddings"], dtype=np.float32))
    else:
        raise ValueError(f"{path}: unrecognized embedding arrays")
    return pd.DataFrame(matrix, index=pd.Index(keys, name="key"))


def embeddings_for(rows, embeddings):
    keys = [row["key"] for row in rows]
    missing = [k for k in keys if k not in embeddings.index]
    if missing:
        raise SystemExit(f"{len(missing)} rows missing embeddings, e.g. {missing[:10]}")
    return embeddings.loc[keys].reset_index(drop=True)


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
