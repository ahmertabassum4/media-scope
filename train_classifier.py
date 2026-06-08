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

FEATURES_FILE = Path("gemini_features.csv")
VALID_LABELS = {"VERY HIGH", "HIGH", "LOW", "VERY LOW"}
ORDER = ["VERY LOW", "LOW", "HIGH", "VERY HIGH"]

SIGNAL_KEYS = [
    "s01_named_bylines", "s02_personal_brand", "s03_editorial_hierarchy",
    "s04_loaded_headlines", "s05_accusatory_questions", "s06_breaking_misuse",
    "s07_biased_categories", "s08_opinion_news_blurred", "s09_standard_sections",
    "s10_incoherent_mixing", "s11_merchandise_section", "s12_timestamps",
    "s13_local_features", "s14_ads_labeled", "s15_sponsored_distinguished",
    "s16_fear_based_ads", "s17_persecution_donation", "s18_ideological_mission",
    "s19_fringe_platforms", "s20_distrust_tagline",
]
COUNT_KEYS = ["n_red_flags", "n_trust_signals"]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run evaluate_with_gemini.py first.")

    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df["ground_truth"].isin(VALID_LABELS)].copy()

    for k in SIGNAL_KEYS:
        if k not in df.columns:
            df[k] = "UNCLEAR"
        df[k] = (
            df[k].str.upper().str.strip()
            .where(lambda s: s.isin(["PRESENT", "ABSENT", "UNCLEAR"]), "UNCLEAR")
        )

    for k in COUNT_KEYS:
        if k not in df.columns:
            df[k] = 0
        df[k] = pd.to_numeric(df[k], errors="coerce").fillna(0)

    if "gemini_verdict" not in df.columns:
        df["gemini_verdict"] = ""
    df["gemini_verdict"] = df["gemini_verdict"].str.upper().str.strip()

    return df.reset_index(drop=True)


def build_pipeline(use_verdict: bool) -> Pipeline:
    cat_cols = list(SIGNAL_KEYS)
    if use_verdict:
        cat_cols = cat_cols + ["gemini_verdict"]

    pre = ColumnTransformer(transformers=[
        ("sig", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), COUNT_KEYS),
    ])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
    return Pipeline([("pre", pre), ("clf", clf)])


def safe_n_splits(y: pd.Series, requested: int = 5) -> int:
    return max(2, min(requested, int(y.value_counts().min())))


def report(title: str, y_true: np.ndarray, y_pred: np.ndarray, binary: bool = False) -> dict:
    print("\n" + "#" * 64)
    print(f"# {title}")
    print("#" * 64)

    if binary:
        yt = np.where(np.isin(y_true, ["VERY HIGH", "HIGH"]), "reliable", "unreliable")
        yp = np.where(np.isin(y_pred, ["VERY HIGH", "HIGH"]), "reliable", "unreliable")
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
    print("TOP SIGNAL COEFFICIENTS  (LR signals-only, fit on all data)")
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
    df = load_data(FEATURES_FILE)
    print(f"Loaded {len(df)} labelled rows from {FEATURES_FILE}")
    print("Ground-truth distribution:")
    for lab in ORDER:
        print(f"  {lab:9s}: {(df['ground_truth'] == lab).sum()}")

    if len(df) < 10:
        print("\nWARNING: very few rows — run evaluate_with_gemini.py first.")

    mask = df["gemini_verdict"].isin(VALID_LABELS)
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"Dropping {n_dropped} rows with UNCLEAR Gemini verdict")
    df = df[mask].reset_index(drop=True)
    y = df["ground_truth"].to_numpy()
    cv = StratifiedKFold(n_splits=safe_n_splits(df["ground_truth"], 5), shuffle=True, random_state=42)
    print(f"\nCross-validation: stratified {cv.n_splits}-fold\n")

    gemini_pred = df["gemini_verdict"].to_numpy()
    summaries = []

    summaries.append(report("Gemini", y, gemini_pred))


    pred_lr = cross_val_predict(build_pipeline(use_verdict=False), df, y, cv=cv)
    summaries.append(report("Gemini + Classifier", y, pred_lr))


    summaries.append(report("Gemini Combined", y, gemini_pred, binary=True))


    pred_gv = cross_val_predict(build_pipeline(use_verdict=True), df, y, cv=cv)
    summaries.append(report("Gemini + Classifier Combined", y, pred_gv, binary=True))



    full = build_pipeline(use_verdict=False).fit(df, y)
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
