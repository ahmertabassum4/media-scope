#Show which OCR words the text classifier leans on, per class.

"""Run:
    python src/top_ocr_words.py --task factuality --out results/current/ocr_top_words_factuality.csv
    python src/top_ocr_words.py --task bias       --out results/current/ocr_top_words_bias.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))  
from dataset import (  
    FEATURE_SCHEMA,
    FEATURES_CACHE,
    TASKS,
    TRAIN_CSV,
    load_split,
)
from features import labels, texts  
from models import _tfidf  

def load_features_utf8(path=FEATURES_CACHE):
    #UTF-8-safe feature loader.
    # I had problems loading it on a Windows pc so i made it inclusive
    cache = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        feats = row.get("nollm_features")
        if row.get("ok") and row.get("feature_schema") == FEATURE_SCHEMA and isinstance(feats, dict):
            key = str(row["_key"])
            if key in cache:
                raise ValueError(f"{path}:{n}: duplicate feature key {key!r}")
            cache[key] = row
    return cache


def extract(task, topn):
    feats = load_features_utf8()
    rows = load_split(TRAIN_CSV, feats, task)
    text = texts(rows).fillna("")
    y = labels(rows).to_numpy()

    vec = _tfidf()                       
    X = vec.fit_transform(text)
    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, random_state=13542)
    clf.fit(X, y)                        

    vocab = np.asarray(vec.get_feature_names_out())
    fitted = list(clf.classes_)
    coef = clf.coef_
    if coef.shape[0] == 1:               
        coef = np.vstack([-coef[0], coef[0]])

    ordered = [c for c in TASKS[task]["classes"] if c in fitted]  
    out = {}
    for cls in ordered:
        w = coef[fitted.index(cls)]
        top = np.argsort(w)[::-1][:topn]
        out[cls] = [(vocab[j], float(w[j])) for j in top]
    return out, len(vocab), len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=sorted(TASKS), default="factuality")
    ap.add_argument("--topn", type=int, default=25)
    ap.add_argument("--out", type=Path, help="optional CSV output path")
    args = ap.parse_args()

    result, vocab_size, n_rows = extract(args.task, args.topn)
    print(f"task={args.task}  train_rows={n_rows}  tfidf_vocab={vocab_size}")
    for cls, words in result.items():
        print(f"\n=== {cls}: top {args.topn} OCR terms it leans on ===")
        print("  " + ", ".join(term for term, _ in words))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["task", "class", "rank", "term", "weight"])
            for cls, words in result.items():
                for rank, (term, weight) in enumerate(words, 1):
                    wr.writerow([args.task, cls, rank, term, f"{weight:.6f}"])
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
