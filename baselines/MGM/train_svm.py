"""Train/evaluate the Baly ACL'20 classifier on the article features.

Mirrors the paper's released pipeline: MinMaxScaler
+ grid-searched RBF-SVM (gamma in logspace(-6,1,8), C in logspace(-2,2,5), 5-fold
CV, macro-F1) refit with probability=True. Trains on the `svm_train` outlet slice
only and evaluates on the held-out test outlets. All article feature archives
must have been extracted after global cross-outlet de-duplication. Reports their Tables 2/3 Group-A
rows: NELA (row 3), BERT representations (row 4), BERT probabilities (row 5),
Articles ALL concatenated (row 11) and as a weighted-posterior ensemble (row 12),
plus the majority-class baseline. When the complete task-agnostic Wikipedia
archive is present, it also appends Wikipedia BERT (row 24) and the A+C
concatenated and posterior-ensemble rows (29/30).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (  # noqa: E402
    ARTICLES_DIR,
    RESULTS_DIR,
    TASKS,
    build_content_filter,
    load_outlets,
    load_roles,
    outlet_label,
    scores,
)

# Parameter grid used by the paper's released implementation.
PARAMS_SVM = [dict(kernel=["rbf"], gamma=np.logspace(-6, 1, 8), C=np.logspace(-2, 2, 5))]
PARAMS_FAST = [dict(kernel=["rbf"], gamma=[1e-3, 1e-1], C=[1.0, 10.0])]
WIKI_FEATURE_SCHEMA = "mgm_wiki_features_v3"
RUN_SCHEMA = "mgm_group_ac_run_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_matching_content_filter(archive, path, content_filter):
    """Reject stale feature archives that include raw syndicated content."""
    try:
        schema = str(archive["content_filter_schema"].item())
        fingerprint = str(archive["content_filter_fingerprint"].item())
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"{path.name} has no strict content-filter metadata — re-extract features") from exc
    if schema != content_filter["schema"] or fingerprint != content_filter["fingerprint"]:
        raise SystemExit(f"{path.name} does not match the current strict content filter — re-extract features")


def require_complete_wiki_features(archive, path):
    """Do not silently evaluate Group C on a partial Wikipedia cache."""
    try:
        schema = str(archive["wiki_feature_schema"].item())
        complete = bool(archive["wiki_complete"].item())
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"{path.name} lacks Group-C cache metadata — re-run extract_wiki.py") from exc
    if schema != WIKI_FEATURE_SCHEMA or not complete:
        raise SystemExit(f"{path.name} is stale or partial — finish the Wikipedia fetch and re-encode")


def load_feature_sets(task, feat_dir, content_filter):
    base_path = feat_dir / f"feats_{task}.npz"
    if not base_path.exists():
        raise SystemExit(f"{base_path.name} missing — extract Group-A article features before training")
    with np.load(base_path, allow_pickle=False) as archive:
        require_matching_content_filter(archive, base_path, content_filter)
        keys = [str(key) for key in archive["keys"].tolist()]
        feature_sets = {
            "bert_repr": archive["bert_repr"],
            "bert_prob": archive["bert_prob"],
        }
    optional_features = (
        ("nela_features.npz", "nela", "NELA rows skipped"),
        ("feats_wiki.npz", "wiki", "Wikipedia (Group C) rows skipped"),
    )
    for filename, column, missing_note in optional_features:
        path = feat_dir / filename
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                if column == "nela":
                    require_matching_content_filter(archive, path, content_filter)
                elif column == "wiki":
                    require_complete_wiki_features(archive, path)
                if [str(key) for key in archive["keys"].tolist()] != keys:
                    raise SystemExit(f"{filename} keys differ from the task features — re-extract")
                feature_sets[column] = archive[column]
        else:
            print(f"note: {filename} missing — {missing_note}")
    index = {k: i for i, k in enumerate(keys)}
    return feature_sets, index


def matrices(feature_sets, index, keys):
    rows = [index[k] for k in keys]
    return {name: values[rows] for name, values in feature_sets.items()}


def aligned_proba(clf, X, classes):
    raw = clf.predict_proba(X)
    out = np.zeros((X.shape[0], len(classes)))
    have = list(clf.classes_)
    for j, c in enumerate(classes):
        if c in have:
            out[:, j] = raw[:, have.index(c)]
    return out


def run_svm(X_train, y_train, X_test, grid, seed=16):
    """The repo's pipeline: MinMaxScaler -> GridSearchCV(SVC rbf) -> refit w/ probs."""
    np.random.seed(seed)  # matches their np.random.seed(16)
    scaler = MinMaxScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    cv = GridSearchCV(SVC(), scoring="f1_macro", cv=5, n_jobs=-1, param_grid=grid)
    cv.fit(Xtr, y_train)
    best = cv.best_estimator_
    clf = SVC(kernel=best.kernel, gamma=best.gamma, C=best.C, probability=True)
    clf.fit(Xtr, y_train)
    return clf, Xte, float(cv.best_score_), dict(gamma=float(best.gamma), C=float(best.C))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--features-dir", type=Path, default=ARTICLES_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fast", action="store_true", help="tiny grid for smoke tests")
    args = parser.parse_args()
    out_path = args.out or RESULTS_DIR / f"mgm_groupA_{args.task}.csv"
    grid = PARAMS_FAST if args.fast else PARAMS_SVM
    classes = TASKS[args.task]["classes"]

    outlets = load_outlets()
    content_filter = build_content_filter(outlets)
    roles = load_roles(args.task, outlets)
    train_keys, test_keys = roles["svm_train"], roles["test"]
    assert not set(train_keys) & set(roles["bert_tune"]), "leakage: svm_train overlaps bert_tune"
    assert not set(test_keys) & (set(train_keys) | set(roles["bert_tune"])), "leakage: test overlaps train"
    y_train = np.array([outlet_label(outlets[k], args.task) for k in train_keys])
    y_test = np.array([outlet_label(outlets[k], args.task) for k in test_keys])

    print(f"content filter: {content_filter['summary']}")
    feature_sets, index = load_feature_sets(args.task, args.features_dir, content_filter)
    missing = [k for k in train_keys + test_keys if k not in index]
    if missing:
        raise SystemExit(f"{len(missing)} outlets missing from features, e.g. {missing[:5]}")
    train_features = matrices(feature_sets, index, train_keys)
    test_features = matrices(feature_sets, index, test_keys)
    print(f"task={args.task}  svm_train={len(train_keys)}  test={len(test_keys)}  "
          f"features={{{', '.join(f'{name}:{values.shape[1]}' for name, values in train_features.items())}}}")

    labels = {"nela": "Articles: NELA", "bert_repr": "Articles: BERT representations",
              "bert_prob": "Articles: BERT probabilities"}
    order = [name for name in ("nela", "bert_repr", "bert_prob") if name in feature_sets]
    rows, probas, weights = [], {}, {}

    majority = pd.Series(y_train).value_counts().idxmax()
    rows.append(scores("majority", "Majority class",
                       "predicts the majority svm_train class", y_test,
                       np.full(len(y_test), majority, dtype=object), args.task))

    for name in order:
        clf, scaled_test, cv_f1, best = run_svm(
            train_features[name], y_train, test_features[name], grid,
        )
        pred = clf.predict(scaled_test)
        probas[name] = aligned_proba(clf, scaled_test, classes)
        weights[name] = cv_f1
        rows.append(scores(name, labels[name],
                           f"SVM(rbf) on {name} [{train_features[name].shape[1]}d], cv_f1={cv_f1:.3f}, "
                           f"gamma={best['gamma']:.2g}, C={best['C']:.2g}", y_test, pred, args.task))
        print(f"  {labels[name]:34s} macro_f1={rows[-1]['macro_f1']:.2f}  acc={rows[-1]['accuracy']:.2f}")

    # Articles: ALL (c) — single SVM on the concatenated features (their row 11)
    cat_tr = np.hstack([train_features[name] for name in order])
    cat_te = np.hstack([test_features[name] for name in order])
    clf, Xte_s, cv_f1, best = run_svm(cat_tr, y_train, cat_te, grid)
    rows.append(scores("all_concat", "Articles: ALL (c)",
                       f"SVM(rbf) on concatenated {'+'.join(order)} [{cat_tr.shape[1]}d], cv_f1={cv_f1:.3f}",
                       y_test, clf.predict(Xte_s), args.task))
    print(f"  {'Articles: ALL (c)':34s} macro_f1={rows[-1]['macro_f1']:.2f}  acc={rows[-1]['accuracy']:.2f}")

    # Articles: ALL (en) — posteriors averaged, weighted by each model's CV macro-F1 (their row 12)
    w = np.array([weights[n] for n in order])
    ens = sum(wi * probas[n] for wi, n in zip(w / w.sum(), order))
    pred = np.asarray(classes, dtype=object)[ens.argmax(axis=1)]
    rows.append(scores("all_ensemble", "Articles: ALL (en)",
                       f"posterior ensemble of {'+'.join(order)}, weights=CV macro-F1", y_test, pred, args.task))
    print(f"  {'Articles: ALL (en)':34s} macro_f1={rows[-1]['macro_f1']:.2f}  acc={rows[-1]['accuracy']:.2f}")

    # Group C — Wikipedia alone (their row 24) and Articles+Wikipedia A+C (rows 29/30)
    if "wiki" in feature_sets:
        clf, Xte_s, cv_f1, best = run_svm(
            train_features["wiki"], y_train, test_features["wiki"], grid,
        )
        probas["wiki"] = aligned_proba(clf, Xte_s, classes)
        weights["wiki"] = cv_f1
        rows.append(scores("wiki", "Wikipedia: BERT",
                           f"SVM(rbf) on wiki [{train_features['wiki'].shape[1]}d], cv_f1={cv_f1:.3f}, "
                           f"gamma={best['gamma']:.2g}, C={best['C']:.2g}",
                           y_test, clf.predict(Xte_s), args.task))
        print(f"  {'Wikipedia: BERT':34s} macro_f1={rows[-1]['macro_f1']:.2f}  acc={rows[-1]['accuracy']:.2f}")

        ac_order = order + ["wiki"]
        ac_tr = np.hstack([train_features[name] for name in ac_order])
        ac_te = np.hstack([test_features[name] for name in ac_order])
        clf, Xte_s, cv_f1, best = run_svm(ac_tr, y_train, ac_te, grid)
        rows.append(scores("articles_wiki_concat", "Articles+Wikipedia (c)",
                           f"SVM(rbf) on concatenated {'+'.join(ac_order)} [{ac_tr.shape[1]}d], cv_f1={cv_f1:.3f}",
                           y_test, clf.predict(Xte_s), args.task))
        print(f"  {'Articles+Wikipedia (c)':34s} macro_f1={rows[-1]['macro_f1']:.2f}  acc={rows[-1]['accuracy']:.2f}")

        w = np.array([weights[n] for n in ac_order])
        ens = sum(wi * probas[n] for wi, n in zip(w / w.sum(), ac_order))
        pred = np.asarray(classes, dtype=object)[ens.argmax(axis=1)]
        rows.append(scores("articles_wiki_ensemble", "Articles+Wikipedia (en)",
                           f"posterior ensemble of {'+'.join(ac_order)}, weights=CV macro-F1",
                           y_test, pred, args.task))
        print(f"  {'Articles+Wikipedia (en)':34s} macro_f1={rows[-1]['macro_f1']:.2f}  acc={rows[-1]['accuracy']:.2f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    feature_paths = [
        args.features_dir / f"feats_{args.task}.npz",
        args.features_dir / "nela_features.npz",
        args.features_dir / "feats_wiki.npz",
    ]
    feature_paths = [path for path in feature_paths if path.exists()]
    manifest = {
        "schema": RUN_SCHEMA,
        "task": args.task,
        "grid": "fast" if args.fast else "paper",
        "role_schema": roles.get("schema"),
        "role_seed": roles.get("seed"),
        "role_bert_fraction": roles.get("bert_frac"),
        "role_outlet_fingerprint": roles.get("outlet_fingerprint"),
        "content_filter_schema": content_filter["schema"],
        "content_filter_fingerprint": content_filter["fingerprint"],
        "content_filter_summary": content_filter["summary"],
        "n_svm_train": len(train_keys),
        "n_test": len(test_keys),
        "feature_files": {
            path.name: sha256_file(path) for path in feature_paths
        },
        "result_csv": str(out_path.resolve()),
        "result_csv_sha256": sha256_file(out_path),
        "experiments": result.to_dict("records"),
    }
    manifest_path = out_path.with_name(f"{out_path.stem}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out_path}")
    print(result[["experiment", "accuracy", "balanced_accuracy", "macro_f1", "mae", "mse"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
