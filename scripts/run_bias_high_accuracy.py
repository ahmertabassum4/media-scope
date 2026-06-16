#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "features" / "merged_dataset.jsonl"
BIAS_FEATURES = ROOT / "data" / "feature_definitions" / "bias_features.json"
BIAS_JSONL = ROOT / "data" / "features" / "sonnet_bias_features.jsonl"
FACT_JSONL = ROOT / "data" / "features" / "sonnet_factuality_features.jsonl"
META_JSONL = ROOT / "data" / "features" / "site_metadata_features.jsonl"
OUTDIR = ROOT / "results" / "bias"
RESULTS_OUT = OUTDIR / "bias_high_accuracy_results.csv"
PER_CLASS_OUT = OUTDIR / "bias_high_accuracy_per_class.csv"
FEATURES_OUT = OUTDIR / "bias_high_accuracy_best_features.csv"
PLOT_OUT = OUTDIR / "bias_high_accuracy_results.png"

LABEL_ORDER = ["LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT"]
STATE_VALUES = ("PRESENT", "ABSENT", "UNCLEAR", "MISSING")


def norm_key(name: str) -> str:
    name = re.sub(r"\.(png|jpg|jpeg|metadata|json)$", "", str(name), flags=re.I)
    name = re.sub(r"_\d{8}_\d{6}$", "", name)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def outlet_type_lookup(path: Path) -> dict[str, str]:
    out = {}
    for row in load_jsonl(path):
        outlet_type = ((row.get("parsed") or {}).get("outlet_type") or "").strip()
        if outlet_type:
            out[norm_key(row.get("filename", ""))] = outlet_type
    return out


def metadata_outlet_lookup(path: Path) -> dict[str, str]:
    out = {}
    for row in load_jsonl(path):
        outlet_type = ((row.get("parsed") or {}).get("outlet_type") or "").strip()
        if outlet_type:
            out[norm_key(row.get("filename", ""))] = outlet_type
    return out


def flatten_numeric(prefix: str, obj: object, rec: dict, group_cols: dict[str, set], group: str):
    if isinstance(obj, dict):
        for key, value in obj.items():
            flatten_numeric(f"{prefix}:{key}", value, rec, group_cols, group)
        return
    try:
        value = float(obj)
    except (TypeError, ValueError):
        return
    if not math.isfinite(value):
        return
    rec[prefix] = value
    group_cols[group].add(prefix)
    if value >= 0:
        col = f"{prefix}__log1p"
        rec[col] = math.log1p(value)
        group_cols[group].add(col)


def add_state_features(rec: dict, prefix: str, features: dict, group_cols: dict[str, set], group: str):
    for key, raw in (features or {}).items():
        state = str(raw or "MISSING").upper().strip()
        if state not in STATE_VALUES:
            state = "MISSING"
        for value in STATE_VALUES:
            col = f"{prefix}:{key}__{value.lower()}"
            rec[col] = int(state == value)
            group_cols[group].add(col)


def add_one_hot(rec: dict, prefix: str, value: str, group_cols: dict[str, set], group: str):
    value = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "missing"
    col = f"{prefix}:{value}"
    rec[col] = 1
    group_cols[group].add(col)


def add_bias_aggregates(rec: dict, features: dict, info: dict, group_cols: dict[str, set]):
    buckets = defaultdict(list)
    for key, raw in (features or {}).items():
        state = str(raw or "MISSING").upper().strip()
        meta = info.get(key, {})
        axis = meta.get("axis", "unknown")
        lean = meta.get("lean", "unknown")
        verdict = meta.get("verdict", "unknown")
        prefix = key[:1]
        for bucket in (
            "all",
            f"prefix_{prefix}",
            f"axis_{axis}",
            f"lean_{lean}",
            f"axis_{axis}_lean_{lean}",
            f"verdict_{verdict}",
        ):
            buckets[bucket].append(state)

    for bucket, states in buckets.items():
        total = max(len(states), 1)
        counts = Counter(states)
        for state in ("PRESENT", "ABSENT", "UNCLEAR"):
            base = f"bias_agg:{bucket}_{state.lower()}"
            rec[f"{base}_count"] = counts[state]
            rec[f"{base}_rate"] = counts[state] / total
            group_cols["bias"].update({f"{base}_count", f"{base}_rate"})

    right = buckets.get("lean_right", [])
    left = buckets.get("lean_left", [])
    right_rate = right.count("PRESENT") / max(len(right), 1)
    left_rate = left.count("PRESENT") / max(len(left), 1)
    derived = {
        "bias_agg:right_present_minus_left_present_count": right.count("PRESENT") - left.count("PRESENT"),
        "bias_agg:right_present_minus_left_present_rate": right_rate - left_rate,
        "bias_agg:right_absent_minus_left_absent_count": right.count("ABSENT") - left.count("ABSENT"),
    }
    for col, value in derived.items():
        rec[col] = value
        group_cols["bias"].add(col)


def build_frame():
    rows = load_jsonl(DATA)
    bias_info = {item["key"]: item for item in json.loads(BIAS_FEATURES.read_text())}
    bias_types = outlet_type_lookup(BIAS_JSONL)
    fact_types = outlet_type_lookup(FACT_JSONL)
    meta_types = metadata_outlet_lookup(META_JSONL)
    group_cols = defaultdict(set)
    records = []

    for row in rows:
        label = str(row.get("bias") or "").upper().strip()
        if label not in LABEL_ORDER:
            continue
        key = row.get("key") or norm_key(row.get("filename", ""))
        rec = {"key": key, "label": label}

        add_state_features(rec, "bias", row.get("bias_features") or {}, group_cols, "bias")
        add_bias_aggregates(rec, row.get("bias_features") or {}, bias_info, group_cols)

        add_state_features(rec, "fact", row.get("sonnet_features") or {}, group_cols, "factuality")
        flatten_numeric("fact_derived", row.get("sonnet_derived") or {}, rec, group_cols, "factuality")

        add_state_features(rec, "meta", row.get("metadata_features") or {}, group_cols, "metadata")
        flatten_numeric("meta_derived", row.get("metadata_derived") or {}, rec, group_cols, "metadata")
        flatten_numeric("color", row.get("color_features") or {}, rec, group_cols, "color")

        rec["source:has_metadata"] = int(bool(row.get("has_metadata")))
        rec["source:has_color"] = int(bool(row.get("has_color")))
        rec["source:has_bias"] = int(bool(row.get("has_bias")))
        group_cols["source"].update({"source:has_metadata", "source:has_color", "source:has_bias"})

        if key in bias_types:
            add_one_hot(rec, "outlet_bias_type", bias_types[key], group_cols, "outlet")
        if key in fact_types:
            add_one_hot(rec, "outlet_fact_type", fact_types[key], group_cols, "outlet")
        if key in meta_types:
            add_one_hot(rec, "outlet_meta_type", meta_types[key], group_cols, "outlet")

        records.append(rec)

    df = pd.DataFrame(records).fillna(0)
    return df, {key: sorted(value) for key, value in group_cols.items()}


def make_model(name: str, seed: int):
    if name == "xgb_d2":
        return XGBClassifier(n_estimators=450, max_depth=2, learning_rate=0.03, subsample=0.9,
                             colsample_bytree=0.9, reg_lambda=3.0, min_child_weight=1,
                             objective="multi:softprob", eval_metric="mlogloss",
                             random_state=seed, n_jobs=1)
    if name == "xgb_d3":
        return XGBClassifier(n_estimators=360, max_depth=3, learning_rate=0.035, subsample=0.9,
                             colsample_bytree=0.85, reg_lambda=2.5, min_child_weight=1,
                             objective="multi:softprob", eval_metric="mlogloss",
                             random_state=seed, n_jobs=1)
    if name == "xgb_d4":
        return XGBClassifier(n_estimators=260, max_depth=4, learning_rate=0.04, subsample=0.85,
                             colsample_bytree=0.8, reg_lambda=3.0, min_child_weight=1,
                             objective="multi:softprob", eval_metric="mlogloss",
                             random_state=seed, n_jobs=1)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=800, max_features="sqrt", min_samples_leaf=1,
                                    class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "extra_trees_log2":
        return ExtraTreesClassifier(n_estimators=900, max_features="log2", min_samples_leaf=1,
                                    class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "extra_trees_035":
        return ExtraTreesClassifier(n_estimators=800, max_features=0.35, min_samples_leaf=1,
                                    class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "extra_trees_leaf2":
        return ExtraTreesClassifier(n_estimators=900, max_features="sqrt", min_samples_leaf=2,
                                    class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "extra_trees_entropy2":
        return ExtraTreesClassifier(n_estimators=1000, criterion="entropy", max_features="sqrt",
                                    min_samples_leaf=2, class_weight="balanced",
                                    random_state=seed, n_jobs=-1)
    if name == "extra_trees_entropy1":
        return ExtraTreesClassifier(n_estimators=1000, criterion="entropy", max_features="sqrt",
                                    min_samples_leaf=1, class_weight="balanced",
                                    random_state=seed, n_jobs=-1)
    if name == "rf":
        return RandomForestClassifier(n_estimators=700, max_features="sqrt", min_samples_leaf=1,
                                      class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=0.5, class_weight="balanced",
                                              random_state=seed)
    if name == "lr":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=6000, C=0.7, class_weight="balanced",
                                                    solver="saga", n_jobs=-1, random_state=seed))])
    if name == "linear_svc":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("clf", LinearSVC(C=0.08, class_weight="balanced", random_state=seed))])
    if name == "rbf_svc":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("clf", SVC(C=1.4, gamma="scale", class_weight="balanced"))])
    if name == "mlp":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(160, 60), alpha=0.02,
                                               learning_rate_init=0.001, early_stopping=True,
                                               max_iter=700, random_state=seed))])
    if name == "lgbm" and LGBMClassifier is not None:
        return LGBMClassifier(n_estimators=420, learning_rate=0.035, num_leaves=31,
                              subsample=0.9, colsample_bytree=0.85, reg_lambda=2.0,
                              class_weight="balanced", random_state=seed, n_jobs=1,
                              objective="multiclass", verbosity=-1)
    if name == "catboost" and CatBoostClassifier is not None:
        return CatBoostClassifier(iterations=420, depth=4, learning_rate=0.035, l2_leaf_reg=4.0,
                                  loss_function="MultiClass", random_seed=seed, verbose=False,
                                  allow_writing_files=False)
    raise ValueError(name)


def maybe_sample_weight(name: str, y_train: np.ndarray):
    if name not in {"xgb_d2", "xgb_d3", "xgb_d4", "catboost"}:
        return None
    counts = np.bincount(y_train)
    weights = {i: len(y_train) / (len(counts) * count) for i, count in enumerate(counts) if count}
    return np.array([weights[int(label)] for label in y_train], dtype=float)


def cv_predict(name: str, X: pd.DataFrame, y: np.ndarray, seed: int):
    pred = np.empty_like(y)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for train, test in cv.split(X, y):
        model = clone(make_model(name, seed))
        imputer = None
        x_train = X.iloc[train]
        x_test = X.iloc[test]
        if not isinstance(model, Pipeline):
            imputer = SimpleImputer(strategy="median")
            x_train = imputer.fit_transform(x_train)
            x_test = imputer.transform(x_test)
        weight = maybe_sample_weight(name, y[train])
        if weight is None:
            model.fit(x_train, y[train])
        else:
            model.fit(x_train, y[train], sample_weight=weight)
        pred[test] = np.asarray(model.predict(x_test)).reshape(-1)
    return pred


def evaluate(name: str, X: pd.DataFrame, y: np.ndarray, seeds: list[int]):
    metrics = []
    preds = []
    for seed in seeds:
        pred = cv_predict(name, X, y, seed)
        preds.append(pred)
        metrics.append((accuracy_score(y, pred), f1_score(y, pred, average="macro", zero_division=0),
                        f1_score(y, pred, average="weighted", zero_division=0)))
    arr = np.array(metrics)
    return {
        "accuracy_mean": arr[:, 0].mean(),
        "accuracy_std": arr[:, 0].std(),
        "macro_f1_mean": arr[:, 1].mean(),
        "macro_f1_std": arr[:, 1].std(),
        "weighted_f1_mean": arr[:, 2].mean(),
        "weighted_f1_std": arr[:, 2].std(),
        "preds": preds,
    }


def plot_results(results: pd.DataFrame):
    top = results.sort_values(["accuracy_mean", "macro_f1_mean"], ascending=False).head(8).copy()
    labels = [f"{r.feature_set} / {r.model} ({int(r.features)}f)" for r in top.itertuples()]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(14, 8))
    height = 0.36
    acc = top["accuracy_mean"].to_numpy() * 100
    f1 = top["macro_f1_mean"].to_numpy() * 100
    ax.barh(y - height / 2, acc, height, label="Accuracy", color="#3367d6")
    ax.barh(y + height / 2, f1, height, label="Macro-F1", color="#1f9d4c")
    for ypos, value in zip(y - height / 2, acc):
        ax.text(value + 0.6, ypos, f"{value:.1f}", ha="left", va="center", fontsize=9)
    for ypos, value in zip(y + height / 2, f1):
        ax.text(value + 0.6, ypos, f"{value:.1f}", ha="left", va="center", fontsize=9)
    ax.set_title("Bias Results")
    ax.set_xlabel("Score (%)")
    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    df, groups = build_frame()
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X_all = df.drop(columns=["key", "label"])

    sets = {
        "bias": groups["bias"],
        "bias_metadata": groups["bias"] + groups["metadata"],
        "bias_color": groups["bias"] + groups["color"],
        "bias_metadata_color": groups["bias"] + groups["metadata"] + groups["color"],
        "everything_no_outlet": groups["bias"] + groups["factuality"] + groups["metadata"] + groups["color"] + groups["source"],
        "everything": groups["bias"] + groups["factuality"] + groups["metadata"] + groups["color"] + groups["source"] + groups["outlet"],
        "metadata_color": groups["metadata"] + groups["color"],
    }

    jobs = None
    if args.quick:
        model_names = ["xgb_d2", "xgb_d3", "extra_trees", "extra_trees_log2", "extra_trees_035", "extra_trees_leaf2", "extra_trees_entropy2", "extra_trees_entropy1", "lgbm", "catboost"]
        seeds = [42]
        sets = {k: sets[k] for k in ("bias_metadata", "bias_metadata_color", "everything_no_outlet", "everything")}
    elif args.final:
        model_names = ["extra_trees", "extra_trees_entropy2", "extra_trees_entropy1", "extra_trees_log2", "extra_trees_leaf2"]
        seeds = [1, 2, 3, 4, 5, 13, 21, 42, 77, 101]
        sets = {k: sets[k] for k in ("bias_metadata_color", "everything_no_outlet", "everything")}
        jobs = [
            ("everything", "extra_trees"),
            ("everything", "extra_trees_entropy2"),
            ("everything", "extra_trees_entropy1"),
            ("everything", "extra_trees_log2"),
            ("everything", "extra_trees_leaf2"),
            ("everything_no_outlet", "extra_trees"),
            ("everything_no_outlet", "extra_trees_entropy2"),
            ("bias_metadata_color", "extra_trees"),
            ("bias_metadata_color", "extra_trees_entropy2"),
        ]
    else:
        model_names = ["xgb_d2", "xgb_d3", "xgb_d4", "extra_trees", "rf", "hgb", "lgbm", "catboost", "lr", "linear_svc", "rbf_svc", "mlp"]
        seeds = [42]

    model_names = [m for m in model_names if not (m == "lgbm" and LGBMClassifier is None)]
    model_names = [m for m in model_names if not (m == "catboost" and CatBoostClassifier is None)]

    rows = []
    best_pred = None
    best_score = (-1, -1)
    best_labels = None

    print(f"Rows={len(df)} labels={dict(zip(le.classes_, np.bincount(y)))}")
    if jobs is None:
        jobs = [(set_name, model_name) for set_name in sets for model_name in model_names]

    for set_name, model_name in jobs:
        cols = sets[set_name]
        cols = [c for c in cols if c in X_all.columns]
        X = X_all[cols]
        scores = evaluate(model_name, X, y, seeds)
        row = {
            "feature_set": set_name,
            "model": model_name,
            "features": len(cols),
                **{k: v for k, v in scores.items() if k != "preds"},
            }
        rows.append(row)
        key = (row["accuracy_mean"], row["macro_f1_mean"])
        if key > best_score:
                best_score = key
                best_pred = scores["preds"]
                best_labels = (set_name, model_name, cols)
        print(f"{set_name:<22} {model_name:<17} f={len(cols):4d} "
              f"acc={row['accuracy_mean']:.4f} macro_f1={row['macro_f1_mean']:.4f}", flush=True)

    results = pd.DataFrame(rows).sort_values(["accuracy_mean", "macro_f1_mean"], ascending=False)
    results.to_csv(RESULTS_OUT, index=False)
    plot_results(results)

    set_name, model_name, cols = best_labels
    pd.DataFrame({"feature": cols}).to_csv(FEATURES_OUT, index=False)
    per_seed = []
    for pred in best_pred:
        precision, recall, f1, support = precision_recall_fscore_support(
            y, pred, labels=np.arange(len(le.classes_)), zero_division=0
        )
        per_seed.append(pd.DataFrame({
            "class": le.classes_,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }))
    per_seed = pd.concat(per_seed, ignore_index=True)
    per_class = pd.DataFrame({
        "class": le.classes_,
        "precision_mean": per_seed.groupby("class", sort=False)["precision"].mean().to_numpy(),
        "precision_std": per_seed.groupby("class", sort=False)["precision"].std().fillna(0).to_numpy(),
        "recall_mean": per_seed.groupby("class", sort=False)["recall"].mean().to_numpy(),
        "recall_std": per_seed.groupby("class", sort=False)["recall"].std().fillna(0).to_numpy(),
        "f1_mean": per_seed.groupby("class", sort=False)["f1"].mean().to_numpy(),
        "f1_std": per_seed.groupby("class", sort=False)["f1"].std().fillna(0).to_numpy(),
        "support": per_seed.groupby("class", sort=False)["support"].first().to_numpy(),
    })
    per_class.to_csv(PER_CLASS_OUT, index=False)

    print("\nTop results:")
    print(results.head(12).to_string(index=False))
    print(f"\nBest: {set_name} / {model_name} / {len(cols)} features")
    print(per_class.to_string(index=False))
    print(f"\nWrote {RESULTS_OUT}")
    print(f"Wrote {FEATURES_OUT}")
    print(f"Wrote {PER_CLASS_OUT}")
    print(f"Wrote {PLOT_OUT}")


if __name__ == "__main__":
    main()
