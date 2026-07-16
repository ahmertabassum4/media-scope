"""DistilBERT textual views F(t) Articles and F(w) Wikipedia (paper §4.2.3).

The paper uses DistilBERT-base "chosen for efficiency, to generate embeddings
from media articles and Wikipedia descriptions", fine-tuned per Table 4
(MBFC column: max_len 256, epochs 6, lr 2e-5, batch 100 — adapted here to
batch 16 x grad-accum 6 for Apple MPS; a documented deviation).

Protocol (leakage-safe): the model is fine-tuned with distant supervision on
TRAIN-split outlets only (articles inherit the outlet label; our global
cross-outlet article dedup filter stays on). Every outlet — train and test —
is then embedded with the frozen encoder: [CLS] last-hidden state per
document, mean over documents per outlet (768-d). Outlets with no usable
document get a zero vector (the paper's missing-view handling).

Outputs: data/graphs/emb_articles_<task>.npz, emb_wiki_<task>.npz.

Run (user, one per view x task; articles is the long one — hours on MPS):
  nohup /opt/anaconda3/bin/python3 baselines/multiview/text_embed.py \
      --view articles --task factuality > data/graphs/text_articles_factuality.log 2>&1 &
Track:  tail -f data/graphs/text_articles_<task>.log
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (  # noqa: E402
    ARTICLES_JSONL,
    GRAPHS_DIR,
    TASKS,
    build_content_filter,
    deduplicate_embedding_rows,
    emb_path,
    exclusive_file_lock,
    filtered_articles,
    load_outlets,
    outlet_label,
    outlet_universe_sha256,
    save_embeddings,
    sha256_file,
    task_split,
)

WIKI_PAGES = Path(__file__).resolve().parents[2] / "data" / "wiki" / "wiki_pages.jsonl"
MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
MAX_LEN, EPOCHS, LR = 256, 6, 2e-5          # paper Table 4, MBFC column
BATCH, GRAD_ACCUM = 16, 6                    # 16 x 6 = 96 ~ their batch 100
SEED = 42


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_wiki_texts(outlets, task):
    """Load unique pages and reject duplicate text with conflicting labels."""
    candidates = {}
    if not WIKI_PAGES.exists():
        raise SystemExit(f"{WIKI_PAGES} missing — run baselines/MGM/extract_wiki.py first")
    with open(WIKI_PAGES) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = str(rec.get("text", "") or "").strip()
            key = rec.get("key")
            if key in outlets and rec.get("found") and text:
                if key in candidates and " ".join(candidates[key].split()) != " ".join(text.split()):
                    raise ValueError(f"{WIKI_PAGES}: conflicting duplicate records for {key}")
                candidates[key] = text

    by_text = defaultdict(list)
    for key, text in candidates.items():
        normalized = " ".join(text.split())
        by_text[hashlib.sha256(normalized.encode("utf-8")).hexdigest()].append(key)

    texts = {}
    duplicate_groups = 0
    conflicting_groups = 0
    removed = []
    for keys in by_text.values():
        if len(keys) == 1:
            key = keys[0]
            texts[key] = candidates[key]
            continue
        duplicate_groups += 1
        labels = {outlet_label(outlets[key], task) for key in keys}
        labels.discard(None)
        if len(labels) > 1:
            conflicting_groups += 1
            removed.extend(keys)
            continue
        winner = min(
            keys,
            key=lambda key: (0 if outlets[key].get("split") == "test" else 1, key),
        )
        texts[winner] = candidates[winner]
        removed.extend(key for key in keys if key != winner)
    report = {
        "found_pages": len(candidates),
        "retained_pages": len(texts),
        "duplicate_groups": duplicate_groups,
        "conflicting_label_groups": conflicting_groups,
        "removed_pages": len(removed),
    }
    return texts, report


def collect_documents(view, task, outlets, content_filter):
    """outlet key -> list of document texts for one view."""
    if view == "articles":
        documents = {
            key: [article["text"] for article in filtered_articles(outlets, key, content_filter)]
            for key in outlets
        }
        return documents, {"content_filter_fingerprint": content_filter["fingerprint"]}
    if view == "wiki":
        wiki, report = load_wiki_texts(outlets, task)
        return {key: ([wiki[key]] if key in wiki else []) for key in outlets}, report
    raise ValueError(f"unknown view {view!r}")


class DocDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        item = {"text": self.texts[i]}
        if self.labels is not None:
            item["label"] = self.labels[i]
        return item


def make_collate(tokenizer, with_labels):
    def collate(batch):
        enc = tokenizer([b["text"] for b in batch], truncation=True,
                        max_length=MAX_LEN, padding=True, return_tensors="pt")
        if with_labels:
            enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return enc
    return collate


def accumulation_divisor(step, n_steps, grad_accum):
    """Scale the final short accumulation group by its actual batch count."""
    remainder = n_steps % grad_accum
    if remainder and step > n_steps - remainder:
        return remainder
    return grad_accum


def fine_tune(texts, labels, n_classes, device, epochs, batch=BATCH,
              grad_accum=GRAD_ACCUM, lr=LR):
    """Fixed-epoch fine-tuning using the adapted Table 4 settings."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, num_labels=n_classes).to(device)
    loader = DataLoader(DocDataset(texts, labels), batch_size=batch, shuffle=True,
                        collate_fn=make_collate(tokenizer, with_labels=True),
                        generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        total, count = 0.0, 0
        optimizer.zero_grad()
        for step, enc in enumerate(loader, start=1):
            enc = {k: v.to(device) for k, v in enc.items()}
            raw_loss = model(**enc).loss
            (raw_loss / accumulation_divisor(step, len(loader), grad_accum)).backward()
            if step % grad_accum == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad()
            total += raw_loss.item()
            count += 1
            if count % 200 == 0:
                print(f"  epoch {epoch} step {count}/{len(loader)} "
                      f"loss {total / count:.4f}", flush=True)
        print(f"epoch {epoch}/{epochs} mean loss {total / max(1, count):.4f}", flush=True)
    return model, tokenizer


@torch.no_grad()
def embed_outlets(model, tokenizer, docs_by_key, keys, device, batch=BATCH * 2):
    """[CLS] last-hidden per document -> mean per outlet; zeros if no docs."""
    model.eval()
    encoder = model.distilbert  # frozen backbone of the fine-tuned classifier
    hidden = model.config.dim
    X = np.zeros((len(keys), hidden), dtype=np.float32)
    for i, key in enumerate(keys):
        texts = docs_by_key.get(key) or []
        if not texts:
            continue
        vecs = []
        for start in range(0, len(texts), batch):
            enc = tokenizer(texts[start:start + batch], truncation=True,
                            max_length=MAX_LEN, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = encoder(**enc).last_hidden_state[:, 0]  # [CLS]
            vecs.append(out.float().cpu().numpy())
        X[i] = np.concatenate(vecs).mean(axis=0)
        if (i + 1) % 200 == 0:
            print(f"  embedded {i + 1}/{len(keys)} outlets", flush=True)
    return X


def embedding_output_path(view, task, epochs, limit):
    canonical = emb_path(view, task=task)
    if not limit and epochs == EPOCHS:
        return canonical
    suffix = f"_limit{limit}" if limit else ""
    if epochs != EPOCHS:
        suffix += f"_epochs{epochs}"
    return canonical.with_name(f"{canonical.stem}{suffix}{canonical.suffix}")


def run(view, task, epochs, limit, device):
    print(f"=== view={view} task={task} epochs={epochs} device={device} ===")
    outlets = load_outlets()
    content_filter = (build_content_filter(outlets) if view == "articles"
                      else {"excluded_indices": {}})
    docs_by_key, source_report = collect_documents(view, task, outlets, content_filter)
    keys = sorted(outlets)
    if limit:
        keys = keys[:limit]
        docs_by_key = {k: docs_by_key.get(k, []) for k in keys}

    train_keys, _ = task_split(outlets, task)
    train_keys = [key for key in train_keys if docs_by_key.get(key)]
    classes = TASKS[task]["classes"]
    label2id = {c: i for i, c in enumerate(classes)}
    texts, labels = [], []
    for key in train_keys:
        label = label2id[outlet_label(outlets[key], task)]
        for text in docs_by_key.get(key, []):
            texts.append(text)
            labels.append(label)
    if set(labels) != set(range(len(classes))):
        raise SystemExit(
            f"training documents do not cover every {task} class: "
            f"present ids={sorted(set(labels))}"
        )
    print(f"fine-tuning on {len(texts)} documents from {len(train_keys)} "
          f"TRAIN outlets ({len(classes)} classes)")
    if not texts:
        raise SystemExit("no training documents — nothing to fine-tune on")

    model, tokenizer = fine_tune(texts, labels, len(classes), device, epochs)
    X = embed_outlets(model, tokenizer, docs_by_key, keys, device)
    X, duplicate_rows = deduplicate_embedding_rows(keys, X, outlets, task=task)
    complete = not limit and epochs == EPOCHS
    out = embedding_output_path(view, task, epochs, limit)
    n_zero = int((np.abs(X).sum(axis=1) == 0).sum())
    save_embeddings(
        out, keys, X,
        schema="multiview_embedding_v2",
        complete=complete,
        view=view,
        task=task,
        model=MODEL_NAME,
        model_revision=MODEL_REVISION,
        epochs=epochs,
        max_len=MAX_LEN,
        n_zero=n_zero,
        duplicate_vectors_zeroed=len(duplicate_rows),
        train_outlets=len(train_keys),
        train_documents=len(texts),
        outlet_universe_sha256=outlet_universe_sha256(outlets),
        source_sha256=sha256_file(WIKI_PAGES if view == "wiki" else ARTICLES_JSONL),
        content_filter_sha256=(source_report["content_filter_fingerprint"]
                               if view == "articles" else ""),
    )
    print(
        f"saved {out}  X{X.shape}  zero-vector outlets={n_zero}  "
        f"duplicate vectors zeroed={len(duplicate_rows)}"
    )
    if view == "wiki":
        print(f"wiki dedup report: {json.dumps(source_report, sort_keys=True)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=["articles", "wiki"], required=True)
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap the number of outlets (smoke tests)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    output = embedding_output_path(args.view, args.task, args.epochs, args.limit)
    try:
        with exclusive_file_lock(output.with_suffix(output.suffix + ".lock")):
            run(args.view, args.task, args.epochs, args.limit, device)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
