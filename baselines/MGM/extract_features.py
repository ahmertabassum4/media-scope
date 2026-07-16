"""Extract the Group-A article features for every outlet (Baly ACL'20 §3.1.1).

Per article, using the frozen fine-tuned BERT from finetune_bert.py:
- BERT representations: first 510 WordPieces -> second-to-last hidden layer,
  mean over tokens (768-d)  [their Tables 2/3 row 4]
- BERT probabilities: softmax posteriors from the fine-tuned head [row 5]
Per article, task-independent (computed once and shared by both tasks):
- NELA linguistic features [row 3]

Article vectors are averaged per outlet; outlets with no scraped articles get
zero vectors (the paper's missing-media handling). One canonical copy of each
globally syndicated or duplicate article is retained before any feature is
computed. Output: data/articles/feats_<task>.npz and
data/articles/nela_features.npz.
"""

import argparse
import json
import sys
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (  # noqa: E402
    ARTICLES_DIR,
    MODELS_DIR,
    TASKS,
    build_content_filter,
    filtered_articles,
    load_outlets,
)


# Scraped pages occasionally contain books, archives, or site-wide text rather
# than a news article. Keep NELA bounded like the BERT path so one malformed
# scrape cannot monopolize all feature workers. This only affects the 0.6% of
# retained articles longer than 10,000 words in the current corpus.
NELA_MAX_WORDS = 10_000


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def mean_wordpiece_hidden(hidden_states, attention_mask, special_tokens_mask):
    """Average only article WordPieces, excluding padding, [CLS], and [SEP]."""
    token_mask = attention_mask.bool() & ~special_tokens_mask.bool()
    counts = token_mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("cannot average an article with no non-special WordPieces")
    weights = token_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * weights).sum(dim=1) / counts.unsqueeze(-1).to(hidden_states.dtype)


@torch.no_grad()
def bert_outlet_features(outlets, keys, model_dir, max_len, batch_size, content_filter):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = pick_device()
    model.to(device).eval()
    n_classes = model.config.num_labels
    hidden = model.config.hidden_size

    reprs = np.zeros((len(keys), hidden), dtype=np.float32)
    probs = np.zeros((len(keys), n_classes), dtype=np.float32)
    for i, key in enumerate(tqdm(keys, unit="outlet", desc="BERT features")):
        texts = [a["text"] for a in filtered_articles(outlets, key, content_filter)]
        if not texts:
            continue  # zero vector = missing media, as in the paper
        art_repr, art_prob = [], []
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], truncation=True,
                              max_length=max_len, padding=True,
                              return_special_tokens_mask=True, return_tensors="pt")
            special_tokens_mask = batch.pop("special_tokens_mask").to(device)
            batch = batch.to(device)
            out = model(**batch, output_hidden_states=True)
            hs = out.hidden_states[-2]                      # second-to-last layer
            art_repr.append(mean_wordpiece_hidden(
                hs, batch["attention_mask"], special_tokens_mask).float().cpu().numpy())
            art_prob.append(torch.softmax(out.logits, dim=-1).float().cpu().numpy())
        reprs[i] = np.concatenate(art_repr).mean(axis=0)
        probs[i] = np.concatenate(art_prob).mean(axis=0)
    return reprs, probs


_NELA = None


def _nela_init():
    global _NELA
    from nela_features.nela_features import NELAFeatureExtractor
    _NELA = NELAFeatureExtractor()


def _nela_outlet(texts):
    """Mean NELA vector over one outlet's articles; (vector|None, names|None)."""
    vecs, names = [], None
    for text in texts:
        try:
            vec, names = _NELA.extract_all(text)
            vecs.append(np.asarray(vec, dtype=np.float32))
        except Exception:
            continue
    return (np.mean(vecs, axis=0) if vecs else None), names


def truncate_nela_text(text, max_words):
    """Keep a deterministic leading article window for bounded NELA extraction."""
    if not max_words:
        return str(text)
    words = str(text).split(None, max_words)
    return " ".join(words[:max_words])


def nela_outlet_features(outlets, keys, workers, content_filter, max_words):
    from multiprocessing import Pool
    jobs = [
        [truncate_nela_text(a["text"], max_words)
         for a in filtered_articles(outlets, k, content_filter)]
        for k in keys
    ]
    with Pool(workers, initializer=_nela_init) as pool:
        results = list(tqdm(pool.imap(_nela_outlet, jobs, chunksize=4),
                            total=len(jobs), unit="outlet", desc="NELA features"))
    names = next((n for _, n in results if n), None)
    if names is None:
        raise SystemExit("NELA produced no features at all")
    X = np.zeros((len(keys), len(names)), dtype=np.float32)
    for i, (row, _) in enumerate(results):
        if row is not None:
            X[i] = row
    return X, list(names)


def matching_bert_cache(path, keys, content_filter):
    """Whether an existing complete BERT archive matches the curated articles."""
    try:
        with np.load(path, allow_pickle=False) as cached:
            if not (
                str(cached["content_filter_schema"].item()) == content_filter["schema"]
                and str(cached["content_filter_fingerprint"].item()) == content_filter["fingerprint"]
            ):
                return False
            cached_keys = [str(key) for key in cached["keys"].tolist()]
            reprs = cached["bert_repr"]
            probs = cached["bert_prob"]
            return (
                cached_keys == list(keys)
                and reprs.ndim == 2
                and probs.ndim == 2
                and reprs.shape[0] == len(keys)
                and probs.shape[0] == len(keys)
                and np.isfinite(reprs).all()
                and np.isfinite(probs).all()
            )
    except (BadZipFile, EOFError, KeyError, OSError, TypeError, ValueError):
        return False


def matching_nela_cache(path, keys, content_filter, max_words=NELA_MAX_WORDS):
    """Whether a complete NELA cache was built from the same curated articles."""
    try:
        with np.load(path, allow_pickle=False) as cached:
            if not (
                str(cached["content_filter_schema"].item()) == content_filter["schema"]
                and str(cached["content_filter_fingerprint"].item()) == content_filter["fingerprint"]
            ):
                return False
            cached_keys = [str(key) for key in cached["keys"].tolist()]
            nela = cached["nela"]
            names = cached["names"]
            return (
                cached_keys == list(keys)
                and nela.ndim == 2
                and nela.shape[0] == len(keys)
                and np.isfinite(nela).all()
                and names.ndim == 1
                and names.dtype.kind in {"U", "S"}
                and int(cached["nela_max_words"].item()) == max_words
            )
    except (BadZipFile, EOFError, KeyError, OSError, TypeError, ValueError):
        return False


def require_matching_model_content_filter(model_dir, content_filter):
    """Prevent a checkpoint trained on unfiltered articles from being reused."""
    split_path = Path(model_dir) / "finetune_split.json"
    try:
        metadata = json.loads(split_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"{split_path} is missing or invalid — re-run finetune_bert.py with the strict content filter"
        ) from exc
    if (metadata.get("content_filter_schema") != content_filter["schema"]
            or metadata.get("content_filter_fingerprint") != content_filter["fingerprint"]):
        raise SystemExit(
            "fine-tuned model does not match the current strict content filter — re-run finetune_bert.py"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="fine-tuned model (default baselines/MGM/models/bert_<task>)")
    parser.add_argument("--out-dir", type=Path, default=ARTICLES_DIR)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap the number of outlets (smoke tests)")
    parser.add_argument("--skip-bert", action="store_true",
                        help="reuse a complete matching BERT archive and only extract NELA")
    parser.add_argument("--skip-nela", action="store_true")
    parser.add_argument("--nela-workers", type=int, default=8)
    parser.add_argument("--nela-max-words", type=int, default=NELA_MAX_WORDS,
                        help="maximum words per article for NELA (0 keeps full text)")
    args = parser.parse_args()
    model_dir = args.model_dir or MODELS_DIR / f"bert_{args.task}"
    if not (Path(model_dir) / "config.json").exists():
        raise SystemExit(f"no fine-tuned model at {model_dir} — run finetune_bert.py --task {args.task} first")
    if args.nela_max_words < 0:
        raise SystemExit("--nela-max-words must be non-negative")

    outlets = load_outlets()
    content_filter = build_content_filter(outlets)
    require_matching_model_content_filter(model_dir, content_filter)
    keys = sorted(outlets)
    if args.limit:
        keys = keys[:args.limit]
    n_zero = sum(1 for k in keys if not filtered_articles(outlets, k, content_filter))
    print(f"content filter: {content_filter['summary']}")
    print(f"task={args.task}  outlets={len(keys)}  (no usable articles -> zero vectors: {n_zero})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"feats_{args.task}.npz"
    if args.skip_bert:
        if not matching_bert_cache(out, keys, content_filter):
            raise SystemExit(f"{out} is missing, partial, or stale — cannot use --skip-bert")
        print(f"BERT cache already present: {out}")
    else:
        reprs, probs = bert_outlet_features(
            outlets, keys, model_dir, args.max_len, args.batch_size, content_filter)
        np.savez_compressed(
            out,
            keys=np.array(keys),
            bert_repr=reprs,
            bert_prob=probs,
            content_filter_schema=np.array(content_filter["schema"]),
            content_filter_fingerprint=np.array(content_filter["fingerprint"]),
        )
        print(f"saved {out}  bert_repr{reprs.shape} bert_prob{probs.shape}  "
              f"zero-repr rows={int((np.abs(reprs).sum(axis=1) == 0).sum())}")

    nela_out = args.out_dir / "nela_features.npz"
    if args.skip_nela:
        print("NELA skipped (--skip-nela)")
    elif not args.limit and matching_nela_cache(
            nela_out, keys, content_filter, args.nela_max_words):
        print(f"NELA cache already present: {nela_out}")
    else:
        if nela_out.exists() and not args.limit:
            print(f"rebuilding stale NELA cache: {nela_out}")
        features, names = nela_outlet_features(
            outlets, keys, args.nela_workers, content_filter, args.nela_max_words)
        np.savez_compressed(
            nela_out,
            keys=np.array(keys),
            nela=features,
            names=np.asarray(names, dtype=str),
            content_filter_schema=np.array(content_filter["schema"]),
            content_filter_fingerprint=np.array(content_filter["fingerprint"]),
            nela_max_words=np.array(args.nela_max_words),
        )
        print(f"saved {nela_out}  nela{features.shape}  nela_max_words={args.nela_max_words}")


if __name__ == "__main__":
    main()
