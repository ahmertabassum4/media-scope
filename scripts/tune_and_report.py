#!/usr/bin/env python3
"""Tune the winning configs, report driving features, render confusion matrices.

Winning configs from the ablation:
  factuality = sonnet + metadata
  bias       = bias   + metadata

For each:
  1. Small HGB hyperparameter grid (5-fold CV, macro-F1) + f_classif top-k variants.
  2. Mutual-information feature ranking at the ORIGINAL-feature level (which signals
     actually drive the label).
  3. Confusion-matrix PNG from out-of-fold predictions of the best model.

Usage:
  /opt/anaconda3/bin/python3.12 scripts/tune_and_report.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import load_rows, build_frame, GROUPS  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "combined"
OUTDIR.mkdir(parents=True, exist_ok=True)

FAC_ORDER = ["VERY LOW", "LOW", "MIXED", "HIGH", "VERY HIGH"]
BIAS_ORDER = ["LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT"]

GRID = [
    {"learning_rate": lr, "max_leaf_nodes": ml, "max_iter": mi, "l2_regularization": l2}
    for lr in (0.05, 0.1)
    for ml in (31, 63)
    for mi in (300, 500)
    for l2 in (1.0,)
]


def make_pipeline(model, cat_cols, num_cols, k=None):
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    trs = []
    if cat_cols:
        trs.append(("cat", cat, cat_cols))
    if num_cols:
        trs.append(("num", num, num_cols))
    steps = [("pre", ColumnTransformer(trs))]
    if k is not None:
        steps.append(("sel", SelectKBest(f_classif, k=k)))
    steps.append(("clf", model))
    return Pipeline(steps)


def cv_scores(df, y, cat_cols, num_cols, params, k=None):
    model = HistGradientBoostingClassifier(random_state=42, class_weight="balanced", **params)
    pipe = make_pipeline(model, cat_cols, num_cols, k=k)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    pred = cross_val_predict(pipe, df, y, cv=skf, n_jobs=-1)
    return pred, {
        "acc": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "macro_prec": precision_score(y, pred, average="macro", zero_division=0),
        "macro_rec": recall_score(y, pred, average="macro", zero_division=0),
    }


def mutual_info_ranking(df, y, cat_cols, num_cols, top=20):
    """MI at original-feature level. Categorical ordinal-coded (discrete)."""
    cols = list(cat_cols) + list(num_cols)
    X = np.zeros((len(df), len(cols)), dtype=float)
    discrete = np.zeros(len(cols), dtype=bool)
    for j, c in enumerate(cols):
        if c in cat_cols:
            codes = pd.Series(df[c].fillna("MISSING").astype(str)).astype("category").cat.codes
            X[:, j] = codes.to_numpy()
            discrete[j] = True
        else:
            v = pd.to_numeric(df[c], errors="coerce")
            med = v.median()
            if not np.isfinite(med):
                med = 0.0
            X[:, j] = np.nan_to_num(v.fillna(med).to_numpy(dtype=float),
                                    nan=0.0, posinf=0.0, neginf=0.0)
    mi = mutual_info_classif(X, y, discrete_features=discrete, random_state=42)
    order = np.argsort(mi)[::-1][:top]
    return [(cols[i].split(":", 1)[-1], cols[i].split(":", 1)[0], float(mi[i])) for i in order]


def run(target_field, label, combo, class_order, png_name):
    rows = load_rows()
    flags = {"sonnet": "sonnet_features", "bias": "has_bias",
             "metadata": "has_metadata", "color": "has_color"}
    sub = [r for r in rows if r.get(target_field) and all(
        bool(r.get("sonnet_features")) if g == "sonnet" else bool(r.get(flags[g])) for g in combo)]
    y = np.array([r[target_field] for r in sub])
    df, cat_cols, num_cols = build_frame(sub, combo)
    nfeat = len(cat_cols) + len(num_cols)

    print(f"\n{'#'*78}\n# {label}: {'+'.join(combo)}  (n={len(sub)}, {nfeat} feats)\n{'#'*78}")

    # ---- tuning ----
    best = None
    print("\nTuning HGB (grid + f_classif top-k):")
    configs = [("full", None, p) for p in GRID]
    configs += [(f"top{k}", k, GRID[0]) for k in (40, 80, 120)]
    for tag, k, params in configs:
        pred, m = cv_scores(df, y, cat_cols, num_cols, params, k=k)
        flag = ""
        if best is None or m["macro_f1"] > best["m"]["macro_f1"]:
            best = {"tag": tag, "k": k, "params": params, "m": m, "pred": pred}
            flag = "  <- best"
        print(f"  {tag:<6} lr={params['learning_rate']} leaf={params['max_leaf_nodes']} "
              f"iter={params['max_iter']}  acc={m['acc']*100:.1f}%  macroF1={m['macro_f1']*100:.1f}{flag}")

    bm = best["m"]
    print(f"\nBEST: {best['tag']} {best['params']}  "
          f"acc={bm['acc']*100:.1f}%  macroF1={bm['macro_f1']*100:.1f}  "
          f"wF1={bm['weighted_f1']*100:.1f}  mPrec={bm['macro_prec']*100:.1f}  mRec={bm['macro_rec']*100:.1f}")

    # ---- feature importance (MI) ----
    print("\nTop 20 driving features (mutual information):")
    for name, grp, score in mutual_info_ranking(df, y, cat_cols, num_cols, top=20):
        print(f"  {score:.4f}  [{grp:<17}] {name}")

    # ---- confusion matrix ----
    labels = [c for c in class_order if c in set(y)]
    cm = confusion_matrix(y, best["pred"], labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{label}  ({'+'.join(combo)}, HGB)\nacc={bm['acc']*100:.1f}%  macro-F1={bm['macro_f1']*100:.1f}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    out = OUTDIR / png_name
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nSaved confusion matrix -> {out}")
    return best


if __name__ == "__main__":
    run("factuality", "FACTUALITY (5-class)", ["sonnet", "metadata"], FAC_ORDER, "factuality_combined_cm.png")
    run("bias", "BIAS (5-class)", ["bias", "metadata"], BIAS_ORDER, "bias_combined_cm.png")
