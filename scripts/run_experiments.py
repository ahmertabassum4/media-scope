#!/usr/bin/env python3
"""Feature-group ablation: does color / metadata help over the Sonnet visual flags?

Runs a matrix of feature-group combinations for both targets (factuality, bias) on a
FIXED row set (outlets that have every group present), so each comparison is
apples-to-apples and the marginal value of color / metadata is unambiguous.

Feature groups:
  sonnet    s001-s100 (categorical)        + 12 derived numeric
  bias      d/i bias features (categorical)
  metadata  m* (categorical)               + 41 derived numeric  (site/HTML/HTTP)
  color     16 numeric                      (contrast / color theme / clutter)

Validation: StratifiedKFold(5), out-of-fold predictions. Metrics: accuracy, macro-F1,
weighted-F1, macro precision, macro recall.

Usage:
  /opt/anaconda3/bin/python3.12 scripts/run_experiments.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "features" / "merged_dataset.jsonl"

# group -> (categorical-dict-field, numeric-dict-field)
GROUPS = {
    "sonnet": ("sonnet_features", "sonnet_derived"),
    "bias": ("bias_features", None),
    "metadata": ("metadata_features", "metadata_derived"),
    "color": (None, "color_features"),
}


def load_rows() -> list[dict]:
    return [json.loads(l) for l in open(DATA)]


def build_frame(rows: list[dict], groups: list[str]):
    """Flatten selected groups into one DataFrame; return (df, cat_cols, num_cols)."""
    cat_cols, num_cols = [], []
    records = []
    # discover columns from the first fully-populated row per field
    cat_keys, num_keys = {}, {}
    for g in groups:
        cfield, nfield = GROUPS[g]
        if cfield:
            for r in rows:
                if r.get(cfield):
                    cat_keys[cfield] = list(r[cfield].keys())
                    break
        if nfield:
            for r in rows:
                if r.get(nfield):
                    num_keys[nfield] = list(r[nfield].keys())
                    break
    for field, keys in cat_keys.items():
        cat_cols.extend(f"{field}:{k}" for k in keys)
    for field, keys in num_keys.items():
        num_cols.extend(f"{field}:{k}" for k in keys)

    for r in rows:
        rec = {}
        for field, keys in cat_keys.items():
            src = r.get(field) or {}
            for k in keys:
                rec[f"{field}:{k}"] = src.get(k, "MISSING")
        for field, keys in num_keys.items():
            src = r.get(field) or {}
            for k in keys:
                v = src.get(k, np.nan)
                try:
                    rec[f"{field}:{k}"] = float(v)
                except (TypeError, ValueError):
                    rec[f"{field}:{k}"] = np.nan
        records.append(rec)
    return pd.DataFrame(records), cat_cols, num_cols


def make_pipeline(model, cat_cols, num_cols):
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])
    num = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    transformers = []
    if cat_cols:
        transformers.append(("cat", cat, cat_cols))
    if num_cols:
        transformers.append(("num", num, num_cols))
    pre = ColumnTransformer(transformers)
    return Pipeline([("pre", pre), ("clf", model)])


def evaluate(df, y, cat_cols, num_cols, model_name):
    if model_name == "LR":
        model = LogisticRegression(max_iter=3000, class_weight="balanced")
    else:
        model = HistGradientBoostingClassifier(random_state=42, class_weight="balanced",
                                                max_iter=300, learning_rate=0.05)
    pipe = make_pipeline(model, cat_cols, num_cols)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(pipe, df, y, cv=skf, n_jobs=-1)
    return {
        "acc": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "macro_prec": precision_score(y, pred, average="macro", zero_division=0),
        "macro_rec": recall_score(y, pred, average="macro", zero_division=0),
    }


def run_target(rows, target_field, label, combos):
    # fixed row set: has every group used across all combos + a label
    needed = sorted({g for combo in combos for g in combo})
    flags = {"sonnet": "sonnet_features", "bias": "has_bias",
             "metadata": "has_metadata", "color": "has_color"}
    sub = []
    for r in rows:
        if not r.get(target_field):
            continue
        ok = True
        for g in needed:
            if g == "sonnet":
                ok = ok and bool(r.get("sonnet_features"))
            else:
                ok = ok and bool(r.get(flags[g]))
        if ok:
            sub.append(r)
    y = np.array([r[target_field] for r in sub])

    print(f"\n{'='*78}\nTARGET: {label}   (n={len(sub)} outlets, fixed across all rows)")
    from collections import Counter
    print("class dist:", dict(Counter(y).most_common()))
    print(f"{'='*78}")
    header = f"{'features':<34}{'model':<6}{'acc':>8}{'macroF1':>9}{'wF1':>8}{'mPrec':>8}{'mRec':>8}"
    print(header)
    print("-" * len(header))

    for combo in combos:
        df, cat_cols, num_cols = build_frame(sub, combo)
        name = "+".join(combo)
        nfeat = len(cat_cols) + len(num_cols)
        for model_name in ("LR", "HGB"):
            m = evaluate(df, y, cat_cols, num_cols, model_name)
            print(f"{name+' ('+str(nfeat)+'f)':<34}{model_name:<6}"
                  f"{m['acc']*100:>7.1f}%{m['macro_f1']*100:>8.1f}{m['weighted_f1']*100:>8.1f}"
                  f"{m['macro_prec']*100:>8.1f}{m['macro_rec']*100:>8.1f}")


def main():
    rows = load_rows()

    fac_combos = [
        ["sonnet"],
        ["color"],
        ["metadata"],
        ["sonnet", "color"],
        ["sonnet", "metadata"],
        ["metadata", "color"],
        ["sonnet", "metadata", "color"],
    ]
    run_target(rows, "factuality", "FACTUALITY (5-class)", fac_combos)

    bias_combos = [
        ["bias"],
        ["color"],
        ["metadata"],
        ["bias", "color"],
        ["bias", "metadata"],
        ["metadata", "color"],
        ["bias", "metadata", "color"],
    ]
    run_target(rows, "bias", "BIAS (5-class)", bias_combos)


if __name__ == "__main__":
    main()
