import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES_FILE = PROJECT_ROOT / "data" / "features" / "sonnet_factuality_features.jsonl"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata" / "media_metadata"
VALID_LABELS = {"VERY HIGH", "HIGH", "LOW", "VERY LOW"}
ORDER = ["VERY LOW", "LOW", "HIGH", "VERY HIGH"]
FEATURE_VALUES = {"PRESENT", "ABSENT", "UNCLEAR"}

GEMINI_SIGNAL_KEYS = [
    "s01_named_bylines", "s02_personal_brand", "s03_editorial_hierarchy",
    "s04_loaded_headlines", "s05_accusatory_questions", "s06_breaking_misuse",
    "s07_biased_categories", "s08_opinion_news_blurred", "s09_standard_sections",
    "s10_incoherent_mixing", "s11_merchandise_section", "s12_timestamps",
    "s13_local_features", "s14_ads_labeled", "s15_sponsored_distinguished",
    "s16_fear_based_ads", "s17_persecution_donation", "s18_ideological_mission",
    "s19_fringe_platforms", "s20_distrust_tagline",
]

DERIVED_NUMERIC_KEYS = [
    "n_present", "n_absent", "n_unclear", "observability_ratio",
    "n_red_flags", "n_trust_signals", "red_flag_ratio", "trust_signal_ratio",
    "hard_to_manipulate_net", "easy_to_manipulate_net",
]
CATEGORY_NUMERIC_PREFIXES = ("category_red_flags", "category_trust_signals")
CORE_CHECK_PREFIX = "core__"


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _normalize_feature(value: object) -> str:
    value = str(value).upper().strip()
    return value if value in FEATURE_VALUES else "UNCLEAR"


def _normalize_label(value: object) -> str:
    value = str(value).upper().strip()
    return value if value in VALID_LABELS else ""


def _feature_sort_key(key: str) -> tuple[int, str]:
    match = re.match(r"s0*(\d+)_", key)
    if match:
        return int(match.group(1)), key
    return 10_000, key


def _build_metadata_lookup(metadata_dir: Path) -> dict:
    lookup = {}
    if not metadata_dir.exists():
        return lookup

    for jf in metadata_dir.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        factuality = _normalize_label(data.get("factuality", ""))
        if not factuality:
            continue

        names = {jf.stem, jf.stem.replace("_", " "), data.get("media name", "")}
        for name in names:
            name = str(name).strip()
            if not name:
                continue
            lookup[name] = factuality
            lookup[name.replace("_", " ")] = factuality
            lookup[_normalize_name(name)] = factuality

    return lookup


def _lookup_ground_truth(filename: str, metadata_lookup: dict) -> str:
    stem = Path(filename).stem
    base = re.sub(r"_\d+$", "", stem)
    for candidate in (stem, base, stem.replace("_", " "), base.replace("_", " "),
                      _normalize_name(stem), _normalize_name(base)):
        truth = metadata_lookup.get(candidate)
        if truth:
            return truth
    return ""


def _feature_columns(columns: list[str]) -> list[str]:
    found = [c for c in columns if re.match(r"s\d+_", c)]
    if found:
        return sorted(found, key=_feature_sort_key)
    return list(GEMINI_SIGNAL_KEYS)


def _verdict_column(df: pd.DataFrame) -> str:
    for col in ("model_verdict", "gemini_verdict", "llm_weak_label", "verdict"):
        if col in df.columns:
            return col
    df["model_verdict"] = ""
    return "model_verdict"


def _binary_labels(labels: np.ndarray | pd.Series) -> np.ndarray:
    labels = np.asarray(labels)
    if np.isin(labels, ["reliable", "unreliable"]).all():
        return labels
    return np.where(np.isin(labels, ["VERY HIGH", "HIGH"]), "reliable", "unreliable")


def load_data(path: Path, metadata_dir: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str], str]:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.")

    if path.suffix.lower() == ".jsonl":
        return load_jsonl_data(path, metadata_dir)
    return load_csv_data(path)


def load_csv_data(path: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str], str]:
    df = pd.read_csv(path, dtype=str).fillna("")
    if "ground_truth" not in df.columns:
        sys.exit(f"ERROR: {path} has no ground_truth column.")

    df = df[df["ground_truth"].isin(VALID_LABELS)].copy()
    signal_keys = _feature_columns(list(df.columns))

    for k in signal_keys:
        if k not in df.columns:
            df[k] = "UNCLEAR"
        df[k] = df[k].map(_normalize_feature)

    count_keys = [k for k in DERIVED_NUMERIC_KEYS if k in df.columns]
    if not count_keys:
        count_keys = ["n_red_flags", "n_trust_signals"]

    for k in count_keys:
        if k not in df.columns:
            df[k] = 0
        df[k] = pd.to_numeric(df[k], errors="coerce").fillna(0)

    verdict_col = _verdict_column(df)
    df[verdict_col] = df[verdict_col].map(_normalize_label)

    return df.reset_index(drop=True), signal_keys, count_keys, [], verdict_col


def load_jsonl_data(path: Path, metadata_dir: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str], str]:
    metadata_lookup = _build_metadata_lookup(metadata_dir)
    rows = []
    signal_keys = set()
    core_keys = set()
    numeric_keys = set()
    skipped_no_truth = 0
    skipped_bad_parse = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped_bad_parse += 1
                continue

            parsed = row.get("parsed")
            if not isinstance(parsed, dict):
                skipped_bad_parse += 1
                continue

            features = parsed.get("features")
            if not isinstance(features, dict) or not features:
                skipped_bad_parse += 1
                continue

            filename = str(row.get("filename", "")).strip()
            truth = _normalize_label(row.get("ground_truth", ""))
            if not truth:
                truth = _lookup_ground_truth(filename, metadata_lookup)
            if not truth:
                skipped_no_truth += 1
                continue

            record = {
                "filename": filename,
                "ground_truth": truth,
                "outlet_type": str(parsed.get("outlet_type", "")).strip(),
                "model_verdict": _normalize_label(parsed.get("llm_weak_label", "")),
            }

            for key, value in features.items():
                record[key] = _normalize_feature(value)
                signal_keys.add(key)

            core_checks = parsed.get("article_core_checks", {})
            if isinstance(core_checks, dict):
                for key, value in core_checks.items():
                    core_key = f"{CORE_CHECK_PREFIX}{key}"
                    record[core_key] = str(value).upper().strip() or "MISSING"
                    core_keys.add(core_key)

            derived = parsed.get("derived", {})
            if isinstance(derived, dict):
                for key, value in derived.items():
                    if key in DERIVED_NUMERIC_KEYS:
                        record[key] = value
                        numeric_keys.add(key)
                    elif key in CATEGORY_NUMERIC_PREFIXES and isinstance(value, dict):
                        for category, category_value in value.items():
                            category_key = f"{key}__{category}"
                            record[category_key] = category_value
                            numeric_keys.add(category_key)

            rows.append(record)

    if not rows:
        sys.exit(f"ERROR: no labelled rows could be loaded from {path}.")

    if skipped_bad_parse:
        print(f"Skipped {skipped_bad_parse} malformed JSONL rows")
    if skipped_no_truth:
        print(f"Skipped {skipped_no_truth} rows without metadata factuality labels")

    df = pd.DataFrame(rows).fillna("")
    signal_keys = sorted(signal_keys, key=_feature_sort_key)
    core_keys = sorted(core_keys)
    numeric_keys = sorted(numeric_keys)

    for key in signal_keys:
        df[key] = df[key].map(_normalize_feature)
    for key in core_keys:
        df[key] = df[key].replace("", "MISSING")

    for key in numeric_keys:
        df[key] = pd.to_numeric(df[key], errors="coerce").fillna(0)

    return df.reset_index(drop=True), signal_keys, numeric_keys, core_keys, "model_verdict"


def build_pipeline(signal_keys: list[str], count_keys: list[str], core_keys: list[str], c_value: float) -> Pipeline:
    cat_cols = list(signal_keys) + list(core_keys)

    transformers = [("sig", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    if count_keys:
        transformers.append(("num", StandardScaler(), count_keys))

    pre = ColumnTransformer(transformers=transformers)
    clf = LogisticRegression(class_weight="balanced", max_iter=3000, C=c_value)
    return Pipeline([("pre", pre), ("clf", clf)])


def safe_n_splits(y: pd.Series, requested: int = 5) -> int:
    min_count = int(y.value_counts().min())
    if min_count < 2:
        sys.exit("ERROR: each class needs at least 2 rows for stratified CV.")
    return min(requested, min_count)


def report(title: str, y_true: np.ndarray, y_pred: np.ndarray, binary: bool = False) -> dict:
    print("\n" + "#" * 64)
    print(f"# {title}")
    print("#" * 64)

    if binary:
        yt = _binary_labels(y_true)
        yp = _binary_labels(y_pred)
        labels = ["reliable", "unreliable"]
    else:
        yt, yp, labels = y_true, y_pred, ORDER

    acc= accuracy_score(yt, yp)
    prec= precision_score(yt, yp, labels=labels, average="macro", zero_division=0)
    rec= recall_score(yt, yp, labels=labels, average="macro", zero_division=0)
    f1= f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
    w_prec= precision_score(yt, yp, labels=labels, average="weighted", zero_division=0)
    w_rec= recall_score(yt, yp, labels=labels, average="weighted", zero_division=0)
    w_f1= f1_score(yt, yp, labels=labels, average="weighted", zero_division=0)

    print(f"\n  Accuracy             : {acc*100:5.1f}%")
    print(f"  Macro    P / R / F1  : {prec*100:5.1f}% / {rec*100:5.1f}% / {f1*100:5.1f}%")
    print(f"  Weighted P / R / F1  : {w_prec*100:5.1f}% / {w_rec*100:5.1f}% / {w_f1*100:5.1f}%")
    print("\n  Per-class breakdown:")
    print(classification_report(yt, yp, labels=labels, zero_division=0))
    _print_cm(yt, yp, labels)

    return {"model": title, "acc": acc, "precision": prec, "recall": rec, "f1": f1, "w_f1": w_f1}


def _print_cm(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    w = max(len(l) for l in labels) + 2
    print("  confusion matrix (rows=true, cols=pred):")
    print("  " + " " * w + "".join(f"{l:>{w}}" for l in labels))
    for i, lab in enumerate(labels):
        print("  " + f"{lab:>{w}}" + "".join(f"{cm[i][j]:>{w}}" for j in range(len(labels))))


def print_top_coefficients(pipe: Pipeline, top_k: int = 8) -> None:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    names = pre.get_feature_names_out()
    print("\n" + "=" * 64)
    print("TOP SIGNAL COEFFICIENTS  (LR features-only, fit on all data)")
    print("=" * 64)
    classes = clf.classes_
    coefs = clf.coef_
    if coefs.shape[0] == 1:
        classes = [classes[1]]
    for ci, cls in enumerate(classes):
        row = coefs[ci]
        order = np.argsort(row)
        print(f"\n  -> {cls}")
        print("     most POSITIVE (push toward this label):")
        for j in order[::-1][:top_k]:
            print(f"        {row[j]:+6.2f}  {names[j]}")
        print("     most NEGATIVE (push away):")
        for j in order[:top_k]:
            print(f"        {row[j]:+6.2f}  {names[j]}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate logistic regression on extracted factuality features."
    )
    parser.add_argument(
        "features_file",
        nargs="?",
        type=Path,
        default=DEFAULT_FEATURES_FILE,
        help="CSV or JSONL feature file to use.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=METADATA_DIR,
        help="Directory of outlet metadata JSON files, used when JSONL lacks ground_truth.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Requested stratified CV folds.")
    parser.add_argument(
        "--target",
        choices=["four-class", "binary"],
        default="four-class",
        help="Train LR on four factuality classes or directly on reliable/unreliable.",
    )
    parser.add_argument(
        "--include-category-counts",
        action="store_true",
        help="Include derived per-category red-flag/trust-signal counts.",
    )
    parser.add_argument(
        "--include-core-checks",
        action="store_true",
        help="Include Sonnet article_core_checks as categorical LR features.",
    )
    parser.add_argument("--c", type=float, default=1.0, help="LogisticRegression C value.")
    args = parser.parse_args()

    df, signal_keys, count_keys, core_keys, verdict_col = load_data(args.features_file, args.metadata_dir)
    source_name = "Sonnet" if "sonnet" in args.features_file.name.lower() else "Gemini"
    if not args.include_category_counts:
        count_keys = [k for k in count_keys if not k.startswith(CATEGORY_NUMERIC_PREFIXES)]
    if not args.include_core_checks:
        core_keys = []

    print(f"Loaded {len(df)} labelled rows from {args.features_file}")
    print(f"Feature columns: {len(signal_keys)} signal categorical, {len(core_keys)} core categorical, {len(count_keys)} numeric")
    print(f"Target: {args.target}; C={args.c}")
    print("Ground-truth distribution:")
    for lab in ORDER:
        print(f"  {lab:9s}: {(df['ground_truth'] == lab).sum()}")

    if len(df) < 10:
        print("\nWARNING: very few rows.")

    mask = df[verdict_col].isin(VALID_LABELS)
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"{n_dropped} rows have no valid {source_name} weak label")

    y_four = df["ground_truth"].to_numpy()
    y = _binary_labels(y_four) if args.target == "binary" else y_four
    cv = StratifiedKFold(
        n_splits=safe_n_splits(pd.Series(y), args.folds),
        shuffle=True,
        random_state=42,
    )
    print(f"\nCross-validation: stratified {cv.n_splits}-fold\n")

    summaries = []

    if mask.any():
        df_verdict = df[mask].reset_index(drop=True)
        y_verdict = df_verdict["ground_truth"].to_numpy()
        model_pred = df_verdict[verdict_col].to_numpy()
        summaries.append(report(f"{source_name} weak label", y_verdict, model_pred, binary=args.target == "binary"))

    pred_lr = cross_val_predict(
        build_pipeline(signal_keys, count_keys, core_keys, args.c),
        df,
        y,
        cv=cv,
    )
    summaries.append(report("Logistic Regression", y, pred_lr, binary=args.target == "binary"))
    if args.target == "four-class":
        summaries.append(report("Logistic Regression combined", y, pred_lr, binary=True))

    if mask.any() and args.target == "four-class":
        summaries.append(report(f"{source_name} weak label combined", y_verdict, model_pred, binary=True))

    full = build_pipeline(signal_keys, count_keys, core_keys, args.c).fit(df, y)
    print_top_coefficients(full)


    print("\n" + "=" * 72)
    print("SUMMARY  (Macro = all classes equal; Weighted = accounts for imbalance)")
    print("=" * 72)
    print(f"{'Model':<32}{'Accuracy':>9}{'Precision':>10}{'Recall':>8}{'MacroF1':>9}{'WtdF1':>8}")
    print("-" * 72)
    for s in summaries:
        print(f"{s['model']:<32}{s['acc']*100:7.1f}% {s['precision']*100:8.1f}% "
              f"{s['recall']*100:6.1f}% {s['f1']*100:7.1f}% {s['w_f1']*100:6.1f}%")

   
if __name__ == "__main__":
    main()
