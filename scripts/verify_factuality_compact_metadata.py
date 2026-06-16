import json
import re
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SONNET = ROOT / "data" / "features" / "sonnet_factuality_features.jsonl"
METADATA = ROOT / "data" / "features" / "site_metadata_features.jsonl"
COLOR = ROOT / "data" / "features" / "color_features.jsonl"
OUTDIR = ROOT / "results" / "factuality"
RESULTS_OUT = OUTDIR / "factuality_compact_metadata_results.csv"
BEST_OUT = OUTDIR / "factuality_compact_metadata_best_features.csv"

RAW14 = [
    "s008_neutral_headlines__present",
    "s076_conspiracy_tropes__present",
    "s010_scare_quotes__present",
    "s072_loaded_category_tags__present",
    "s066_cluttered_blog_look__present",
    "s004_sensational_verbs__present",
    "s078_absolutist_framing__present",
    "s082_ideological_mission__present",
    "s060_templated_imagery__present",
    "s023_no_owner_identified__present",
    "s047_no_sourcing__present",
    "s067_alarmist_colors__present",
    "s085_standard_sections__present",
    "s097_suspicious_url__present",
]
EASY6 = [
    "s010_scare_quotes__present",
    "s072_loaded_category_tags__present",
    "s004_sensational_verbs__present",
    "s023_no_owner_identified__present",
    "s085_standard_sections__present",
    "s097_suspicious_url__present",
]
LABELS4 = {"VERY LOW", "LOW", "HIGH", "VERY HIGH"}
COUNT_META = {
    "bytes_read",
    "links_total",
    "internal_links",
    "external_links",
    "image_count",
    "stylesheet_count",
    "meta_tag_count",
    "h2_count",
    "word_count",
    "heading_count",
    "tracker_script_count",
    "ad_script_count",
    "article_tag_count",
    "nav_tag_count",
}
RAW_META = {"external_link_rate", "avg_heading_length"}
FLAG_META = [
    "m013_has_schema_org",
    "m014_has_news_schema",
    "m015_has_rss_or_atom",
    "m016_has_about_link",
    "m017_has_contact_link",
    "m018_has_privacy_link",
    "m021_has_masthead_or_staff_link",
    "m022_has_corrections_or_policy_link",
    "m029_google_tag_manager_detected",
    "m031_tracker_detected",
    "m035_publisher_schema_hint",
]
COLOR_KEEP = [
    "c_brightness_mean",
    "c_brightness_std",
    "c_whitespace_ratio",
    "c_dark_ratio",
    "c_saturation_mean",
    "c_colorfulness",
    "c_palette_size",
    "c_hue_entropy",
    "c_warm_ratio",
    "c_red_ratio",
    "c_edge_density",
    "c_edge_mean",
]


def primary_key(name):
    stem = re.sub(r"\.(png|jpg|jpeg|json|metadata)$", "", str(name), flags=re.I)
    stem = re.sub(r"_\d{8}_\d{6}$", "", stem)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def fallback_key(name):
    stem = re.sub(r"\.(png|jpg|jpeg|json|metadata)$", "", str(name), flags=re.I)
    stem = re.sub(r"_\d{8}_\d{6}$", "", stem)
    stem = re.sub(r"_\d+$", "", stem)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def load_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def indexed(rows, name_func, parsed_ok=None):
    out = {}
    fallback = {}
    for row in rows:
        if parsed_ok and not parsed_ok(row):
            continue
        filename = row.get("filename", "")
        pk = primary_key(name_func(row) or filename)
        fk = fallback_key(name_func(row) or filename)
        if pk and pk not in out:
            out[pk] = row
        if fk and fk not in fallback:
            fallback[fk] = row
    return out, fallback


def get_match(pk, fk, primary, fallback):
    return primary.get(pk) or fallback.get(fk) or {}


def add_sonnet_features(rec, features):
    for key, value in (features or {}).items():
        value = str(value).upper().strip()
        rec[f"{key}__present"] = int(value == "PRESENT")


def add_metadata_features(rec, row):
    parsed = row.get("parsed") or {}
    flags = parsed.get("features") or {}
    derived = parsed.get("derived") or {}
    for key in FLAG_META:
        value = str(flags.get(key, "UNCLEAR")).upper().strip()
        rec[f"meta:{key}__present"] = int(value == "PRESENT")
    for key in COUNT_META:
        try:
            rec[f"meta:log1p_{key}"] = np.log1p(float(derived.get(key, 0)))
        except (TypeError, ValueError):
            rec[f"meta:log1p_{key}"] = 0.0
    for key in RAW_META:
        try:
            rec[f"meta:{key}"] = float(derived.get(key, 0))
        except (TypeError, ValueError):
            rec[f"meta:{key}"] = 0.0


def add_color_features(rec, row):
    features = (row.get("parsed") or {}).get("features") or {}
    for key in COLOR_KEEP:
        try:
            rec[f"color:{key}"] = float(features.get(key, 0))
        except (TypeError, ValueError):
            rec[f"color:{key}"] = 0.0


def build_frame():
    sonnet_rows = load_jsonl(SONNET)
    metadata_rows = load_jsonl(METADATA)
    color_rows = load_jsonl(COLOR) if COLOR.exists() else []
    meta_primary, meta_fallback = indexed(metadata_rows, lambda r: r.get("filename", ""))
    color_primary, color_fallback = indexed(color_rows, lambda r: r.get("filename", ""), lambda r: r.get("ok"))

    records = []
    for row in sonnet_rows:
        pk = primary_key(row.get("filename", ""))
        fk = fallback_key(row.get("filename", ""))
        meta = get_match(pk, fk, meta_primary, meta_fallback)
        label = str(meta.get("ground_truth", "")).upper().strip()
        if label not in LABELS4:
            continue
        rec = {"key": pk, "label": label}
        add_sonnet_features(rec, (row.get("parsed") or {}).get("features") or {})
        add_metadata_features(rec, meta)
        add_color_features(rec, get_match(pk, fk, color_primary, color_fallback))
        records.append(rec)
    return pd.DataFrame(records).fillna(0)


def models(seed):
    return {
        "xgb": XGBClassifier(
            n_estimators=240,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=2.5,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=1,
        ),
        "xgb_deep": XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        ),
    }


def predict_cv(model, X, y, groups, seed):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    pred = np.empty_like(y)
    for train, test in cv.split(X, y, groups):
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(X.iloc[train])
        x_test = imputer.transform(X.iloc[test])
        model.fit(x_train, y[train])
        pred[test] = model.predict(x_test)
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--best-only", action="store_true")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = build_frame()
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    groups = df["key"].to_numpy()
    X = df.drop(columns=["key", "label"])
    meta_cols = [c for c in X.columns if c.startswith("meta:")]
    color_cols = [c for c in X.columns if c.startswith("color:")]

    sets = {
        "easy6": EASY6,
        "raw14": RAW14,
        "easy6_meta": EASY6 + meta_cols,
        "raw14_meta": RAW14 + meta_cols,
        "easy6_meta_color": EASY6 + meta_cols + color_cols,
        "raw14_meta_color": RAW14 + meta_cols + color_cols,
        "meta_only": meta_cols,
        "meta_color_only": meta_cols + color_cols,
    }
    if args.best_only:
        sets = {k: sets[k] for k in ("easy6", "easy6_meta", "easy6_meta_color", "raw14", "raw14_meta", "raw14_meta_color")}
        model_names = ("xgb",)
        seeds = [1, 2, 3, 4, 5, 13, 21, 42, 77, 101]
    elif args.fast:
        sets = {k: sets[k] for k in ("easy6", "raw14", "easy6_meta", "raw14_meta", "raw14_meta_color", "meta_only")}
        model_names = ("xgb", "hgb")
        seeds = [1, 2, 42]
    else:
        model_names = ("xgb", "xgb_deep", "extra_trees", "hgb")
        seeds = [1, 2, 3, 4, 5, 13, 21, 42, 77, 101]
    rows = []
    for set_name, cols in sets.items():
        cols = [c for c in cols if c in X.columns]
        for model_name in model_names:
            metrics = []
            for seed in seeds:
                pred = predict_cv(models(seed)[model_name], X[cols], y, groups, seed)
                metrics.append((accuracy_score(y, pred), f1_score(y, pred, average="macro", zero_division=0)))
            arr = np.array(metrics)
            rows.append({
                "feature_set": set_name,
                "model": model_name,
                "features": len(cols),
                "accuracy_mean": arr[:, 0].mean(),
                "accuracy_std": arr[:, 0].std(),
                "macro_f1_mean": arr[:, 1].mean(),
                "macro_f1_std": arr[:, 1].std(),
            })
            print(f"{set_name:<18} {model_name:<12} f={len(cols):2d} acc={arr[:,0].mean():.4f} f1={arr[:,1].mean():.4f}", flush=True)

    results = pd.DataFrame(rows).sort_values(["accuracy_mean", "macro_f1_mean"], ascending=False)
    results.to_csv(RESULTS_OUT, index=False)
    best = results.iloc[0]
    best_cols = [c for c in sets[best.feature_set] if c in X.columns]
    pd.DataFrame({"feature": best_cols}).to_csv(BEST_OUT, index=False)
    print("\nTop results:")
    print(results.head(12).to_string(index=False))
    print(f"\nWrote {RESULTS_OUT}")
    print(f"Wrote {BEST_OUT}")
    print(f"Rows={len(df)} labels={dict(zip(le.classes_, np.bincount(y)))}")


if __name__ == "__main__":
    main()
