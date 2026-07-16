import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

# name -> (use_features, use_ocr, use_dino)
MODEL_CONFIGS = {
    "full": (True, True, True),
    "no_ocr": (True, False, True),
    "no_features": (False, True, True),
    "ocr_only": (False, True, False),
    "dino_only": (False, False, True),
    "features_only": (True, False, False),
}

CONFIG_LABELS = {
    "full": "Full",
    "no_ocr": "Removed OCR",
    "no_features": "Removed 51 features",
    "ocr_only": "Only OCR",
    "dino_only": "Only DINOv2",
    "features_only": "Only 51 features",
}

CONFIG_COMPONENTS = {
    "full": "51 features + OCR expert + DINO knn15",
    "no_ocr": "51 features + DINO knn15",
    "no_features": "OCR expert + DINO knn15",
    "ocr_only": "OCR expert only",
    "dino_only": "DINO knn15 only",
    "features_only": "51 compact features only",
}

# baseline: no model, always predicts the majority train class
CONFIG_LABELS["majority"] = "Majority class"
CONFIG_COMPONENTS["majority"] = "predicts the majority train class for every row"
BENCHMARK_ORDER = ["majority"] + list(MODEL_CONFIGS)


def majority_baseline(y_train, n_test):
    majority = y_train.value_counts().idxmax()
    return np.full(n_test, majority, dtype=object)


def aligned_scores(model, matrix, class_order):
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(matrix), dtype=float)
    else:
        raw = np.asarray(model.decision_function(matrix), dtype=float)
        if raw.ndim == 1:
            raw = np.column_stack([-raw, raw])
    classes = list(model.classes_)
    out = np.zeros((matrix.shape[0], len(class_order)), dtype=float)
    for j, label in enumerate(class_order):
        if label in classes:
            out[:, j] = raw[:, classes.index(label)]
    return out


# ── 51-feature tree expert ────────────────────────────────────────────────────
def _estimator(spec, seed):
    kind = spec["kind"]
    if kind == "hgb":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_iter=spec.get("max_iter", 220),
                learning_rate=spec.get("learning_rate", 0.06),
                max_leaf_nodes=spec.get("max_leaf_nodes", 15),
                min_samples_leaf=spec.get("min_samples_leaf", 12),
                l2_regularization=spec.get("l2_regularization", 0.5),
                class_weight="balanced",
                random_state=seed,
            )),
        ])
    if kind == "extra_trees":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=spec.get("n_estimators", 500),
                min_samples_leaf=spec.get("min_samples_leaf", 2),
                max_features=spec.get("max_features", "sqrt"),
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ])
    if kind == "logreg":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=spec.get("C", 1.0), class_weight="balanced",
                                       max_iter=5000, random_state=seed)),
        ])
    if kind == "linearsvc":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(C=spec.get("C", 1.0), class_weight="balanced",
                              max_iter=10000, random_state=seed)),
        ])
    raise ValueError(f"Unknown model kind: {kind}")


_SPECS = [
    {"kind": "hgb", "max_iter": 220, "learning_rate": 0.06, "max_leaf_nodes": 15, "min_samples_leaf": 10},
    {"kind": "extra_trees", "n_estimators": 420, "min_samples_leaf": 2},
    {"kind": "logreg", "C": 0.5},
    {"kind": "linearsvc", "C": 0.5},
]


def _tune(X, y, inner_splits, class_order, selection_label=None, seed=3042):
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    yy = y.to_numpy()
    best = None
    for spec in _SPECS:
        preds = np.empty(len(y), dtype=object)
        for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
            model = _estimator(spec, seed + 100 * fold)
            model.fit(X.iloc[tr], yy[tr])
            preds[va] = np.asarray(model.predict(X.iloc[va]), dtype=object).ravel()
        rep = classification_report(
            yy, preds, labels=class_order, zero_division=0, output_dict=True,
        )
        if selection_label in rep:
            selection_f1 = float(rep[selection_label]["f1-score"])
        else:
            selection_f1 = min(float(rep[label]["f1-score"]) for label in class_order)
        key = (float(f1_score(yy, preds, average="macro", zero_division=0)),
               float(accuracy_score(yy, preds)),
               selection_f1, spec)
        if best is None or key[:3] > best[:3]:
            best = key
    return dict(best[3])


def feature_expert(
    X_train, y, X_test, inner_splits, class_order, selection_label=None, seed=4042,
):
    """Return leakage-safe OOF scores and a full-data test prediction.

    Hyperparameters for each OOF fold are selected using only that fold's
    training rows. Selecting one specification on all labels before creating
    OOF predictions would let each held-out label influence model selection.
    """
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    yy = y.to_numpy()
    oof = np.zeros((len(y), len(class_order)), dtype=float)
    for fold, (tr, va) in enumerate(cv.split(X_train, y), start=1):
        fold_y = y.iloc[tr].reset_index(drop=True)
        spec = _tune(
            X_train.iloc[tr].reset_index(drop=True),
            fold_y,
            inner_splits,
            class_order,
            selection_label,
            seed=seed + 10_000 * fold,
        )
        model = _estimator(spec, seed + 100 * fold)
        model.fit(X_train.iloc[tr], yy[tr])
        oof[va] = aligned_scores(model, X_train.iloc[va], class_order)
    spec = _tune(
        X_train, y, inner_splits, class_order, selection_label, seed=seed + 90_000,
    )
    final = _estimator(spec, seed + 999)
    final.fit(X_train, yy)
    return oof, aligned_scores(final, X_test, class_order)


# ── OCR text expert ───────────────────────────────────────────────────────────
def _tfidf():
    return TfidfVectorizer(
        lowercase=True, strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{2,}\b",
        ngram_range=(1, 2), min_df=2, max_df=0.85, max_features=5000, sublinear_tf=True,
    )


def ocr_expert(train_text, test_text, y, inner_splits, class_order, seed=4542):
    yy = y.to_numpy()
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), len(class_order)), dtype=float)
    for fold, (tr, va) in enumerate(cv.split(train_text, y), start=1):
        vec = _tfidf()
        Xtr = vec.fit_transform(train_text.iloc[tr].fillna(""))
        Xva = vec.transform(train_text.iloc[va].fillna(""))
        clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, random_state=seed + 1000 * fold)
        clf.fit(Xtr, yy[tr])
        oof[va] = aligned_scores(clf, Xva, class_order)
    vec = _tfidf()
    Xfull = vec.fit_transform(train_text.fillna(""))
    Xtest = vec.transform(test_text.fillna(""))
    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, random_state=seed + 9000)
    clf.fit(Xfull, yy)
    return oof, aligned_scores(clf, Xtest, class_order)


# ── DINOv2 visual expert (mean view, PCA64, cosine kNN-15) ────────────────────
def _dino_model(X_train, seed):
    n = min(64, X_train.shape[0] - 1, X_train.shape[1])
    solver = "randomized" if n < min(X_train.shape) else "full"
    return Pipeline([
        ("pca", PCA(n_components=n, svd_solver=solver, random_state=seed)),
        ("norm", Normalizer(norm="l2")),
        ("knn", KNeighborsClassifier(n_neighbors=15, weights="distance", metric="cosine", algorithm="brute")),
    ])


def dino_expert(train_emb, test_emb, y, inner_splits, class_order, seed=5042):
    yy = y.to_numpy()
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), len(class_order)), dtype=float)
    for fold, (tr, va) in enumerate(cv.split(train_emb, y), start=1):
        model = _dino_model(train_emb.iloc[tr], seed + 1000 * fold)
        model.fit(train_emb.iloc[tr], yy[tr])
        oof[va] = aligned_scores(model, train_emb.iloc[va], class_order)
    final = _dino_model(train_emb, seed + 9000)
    final.fit(train_emb, yy)
    return oof, aligned_scores(final, test_emb, class_order)


# ── balanced logistic-regression meta stack ──────────────────────────────────
def meta_predict(Z_train, y, Z_test, class_order, selection_label=None, seed=6042):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    yy = y.to_numpy()
    best_key, best_c = None, 0.3
    for c in (0.1, 0.3, 1.0):
        preds = np.empty(len(y), dtype=object)
        for fold, (tr, va) in enumerate(cv.split(Z_train, y), start=1):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=seed + fold)),
            ])
            model.fit(Z_train[tr], yy[tr])
            preds[va] = model.predict(Z_train[va])
        rep = classification_report(
            yy, preds, labels=class_order, zero_division=0, output_dict=True,
        )
        if selection_label in rep:
            selection_f1 = float(rep[selection_label]["f1-score"])
        else:
            selection_f1 = min(float(rep[label]["f1-score"]) for label in class_order)
        key = (float(f1_score(yy, preds, average="macro", zero_division=0)),
               float(accuracy_score(yy, preds)),
               selection_f1)
        if best_key is None or key > best_key:
            best_key, best_c = key, c
    final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=best_c, class_weight="balanced", max_iter=5000, random_state=seed + 999)),
    ])
    final.fit(Z_train, yy)
    scores = aligned_scores(final, Z_test, class_order)
    return np.asarray(class_order, dtype=object)[np.argmax(scores, axis=1)]


def fit_experts(data, inner_splits):
    """Fit each base expert once so all ablations use identical predictions."""
    class_order = data["class_order"]
    selection_label = data.get("selection_label")
    return {
        "features": feature_expert(
            data["X_train"], data["y_train"], data["X_test"], inner_splits,
            class_order, selection_label,
        ),
        "ocr": ocr_expert(
            data["train_text"], data["test_text"], data["y_train"], inner_splits,
            class_order,
        ),
        "dino": dino_expert(
            data["train_emb"], data["test_emb"], data["y_train"], inner_splits,
            class_order,
        ),
    }


def fit_config(name, data, inner_splits, experts=None):
    use_features, use_ocr, use_dino = MODEL_CONFIGS[name]
    if experts is None:
        experts = fit_experts(data, inner_splits)
    train_blocks, test_blocks = [], []
    if use_features:
        oof, test = experts["features"]
        train_blocks.append(oof)
        test_blocks.append(test)
    if use_ocr:
        oof, test = experts["ocr"]
        train_blocks.append(oof)
        test_blocks.append(test)
    if use_dino:
        oof, test = experts["dino"]
        train_blocks.append(oof)
        test_blocks.append(test)
    return meta_predict(
        np.hstack(train_blocks), data["y_train"], np.hstack(test_blocks),
        data["class_order"], data.get("selection_label"),
    )
