"""Fine-tune bert-base-uncased on articles, distant supervision (Baly ACL'20 §3.1.1).

Articles come ONLY from the `bert_tune` outlet slice; every article inherits its
outlet's label. Factuality is trained binary low-vs-high (paper footnote 8 —
MIXED outlets are excluded from fine-tuning but still classified by the SVM
later); bias is trained 5-way. Validation is stratified at the outlet level,
so articles from one outlet cannot appear in both BERT train and validation.
Only one canonical copy of syndicated or duplicated articles is retained
globally before either partition is used. The saved model is the frozen feature extractor used by
extract_features.py.
"""

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (  # noqa: E402
    MODELS_DIR,
    TASKS,
    article_text_hash,
    article_url_key,
    build_content_filter,
    filtered_articles,
    load_outlets,
    load_roles,
    outlet_label,
    split_fine_tune_outlets,
)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss


def macro_f1(eval_pred):
    logits, labels = eval_pred
    return {"macro_f1": f1_score(labels, logits.argmax(-1), average="macro")}


def collect_articles(task, outlet_keys, outlets, content_filter):
    """Return article text/label pairs from a known-disjoint outlet partition."""
    bert_labels = set(TASKS[task]["bert_classes"])
    texts, labels, urls = [], [], []
    for key in sorted(outlet_keys):
        label = outlet_label(outlets[key], task)
        if label not in bert_labels:
            raise ValueError(f"{key} has an invalid BERT label for {task}: {label!r}")
        for art in filtered_articles(outlets, key, content_filter):
            text = str(art.get("text", "")).strip()
            if text:
                texts.append(text)
                labels.append(label)
                urls.append(str(art.get("url", "")))
    return texts, labels, urls


def remove_training_content_seen_in_validation(train_texts, train_labels, train_urls,
                                               val_texts, val_urls):
    """Keep validation documents unseen by BERT training, including syndication."""
    validation_hashes = {article_text_hash(text) for text in val_texts}
    validation_urls = {key for url in val_urls if (key := article_url_key(url))}
    keep = [i for i, (text, url) in enumerate(zip(train_texts, train_urls))
            if article_text_hash(text) not in validation_hashes
            and (article_url_key(url) is None or article_url_key(url) not in validation_urls)]
    removed = len(train_texts) - len(keep)
    return ([train_texts[i] for i in keep], [train_labels[i] for i in keep],
            [train_urls[i] for i in keep], removed)


def limit_pairs(texts, labels, urls, limit, seed):
    """Deterministically cap a single partition for smoke tests."""
    if not limit or len(texts) <= limit:
        return texts, labels, urls
    rng = random.Random(seed)
    idx = rng.sample(range(len(texts)), limit)
    return [texts[i] for i in idx], [labels[i] for i in idx], [urls[i] for i in idx]


def require_all_classes(labels, classes, partition):
    missing = sorted(set(classes) - set(labels))
    if missing:
        raise SystemExit(f"{partition} has no examples for BERT classes: {missing}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="where to save the fine-tuned model (default baselines/MGM/models/bert_<task>)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=512)       # first 510 WordPieces + [CLS]/[SEP]
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap articles in each train/validation partition (smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = args.model_dir or MODELS_DIR / f"bert_{args.task}"

    outlets = load_outlets()
    content_filter = build_content_filter(outlets)
    print(f"content filter: {content_filter['summary']}")
    roles = load_roles(args.task, outlets)
    classes = TASKS[args.task]["bert_classes"]
    train_outlets, val_outlets = split_fine_tune_outlets(
        args.task, roles, outlets, val_frac=args.val_frac, seed=args.seed)
    if set(train_outlets) & set(val_outlets):
        raise SystemExit("leakage: BERT train and validation share outlets")

    train_texts, train_labels, train_urls = collect_articles(
        args.task, train_outlets, outlets, content_filter)
    val_texts, val_labels, val_urls = collect_articles(
        args.task, val_outlets, outlets, content_filter)
    train_texts, train_labels, train_urls, n_removed = remove_training_content_seen_in_validation(
        train_texts, train_labels, train_urls, val_texts, val_urls)
    if n_removed:
        print(f"removed {n_removed} BERT-train articles duplicated in validation outlets")
    if {article_text_hash(text) for text in train_texts} & {article_text_hash(text) for text in val_texts}:
        raise SystemExit("leakage: BERT train and validation share exact article text")
    if ({key for url in train_urls if (key := article_url_key(url))}
            & {key for url in val_urls if (key := article_url_key(url))}):
        raise SystemExit("leakage: BERT train and validation share article URLs")

    train_texts, train_labels, train_urls = limit_pairs(
        train_texts, train_labels, train_urls, args.limit, args.seed)
    val_texts, val_labels, val_urls = limit_pairs(
        val_texts, val_labels, val_urls, args.limit, args.seed + 1)
    require_all_classes(train_labels, classes, "BERT training")
    require_all_classes(val_labels, classes, "BERT validation")

    label2id = {c: i for i, c in enumerate(classes)}
    y_train = np.array([label2id[label] for label in train_labels])
    y_val = np.array([label2id[label] for label in val_labels])

    counts = np.bincount(y_train, minlength=len(classes))
    class_weights = torch.tensor(len(y_train) / (len(classes) * np.maximum(counts, 1)), dtype=torch.float)
    print(f"task={args.task}  classes={classes}  outlets: train={len(train_outlets)} val={len(val_outlets)}")
    print(f"articles: train={len(y_train)} val={len(y_val)}")
    print(f"train label counts={dict(zip(classes, counts.tolist()))}  weights={[round(w, 2) for w in class_weights.tolist()]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def build(texts, labels):
        ds = Dataset.from_dict({"text": texts, "label": labels.tolist()})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_len),
                      batched=True, remove_columns=["text"])

    train_ds, val_ds = build(train_texts, y_train), build(val_texts, y_val)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(classes),
        id2label={i: c for c, i in label2id.items()}, label2id=label2id)

    train_args = TrainingArguments(
        output_dir=tempfile.mkdtemp(prefix=f"bert_{args.task}_ckpt_"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )
    trainer = WeightedTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=macro_f1,
        class_weights=class_weights,
    )
    trainer.train()
    val = trainer.evaluate()
    print(f"best val macro_f1={val['eval_macro_f1']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "finetune_split.json").write_text(json.dumps({
        "task": args.task,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "role_bert_frac": roles["bert_frac"],
        "train_outlets": train_outlets,
        "validation_outlets": val_outlets,
        "train_articles": len(y_train),
        "validation_articles": len(y_val),
        "removed_train_articles_seen_in_validation": n_removed,
        "content_filter_schema": content_filter["schema"],
        "content_filter_fingerprint": content_filter["fingerprint"],
        "content_filter_summary": content_filter["summary"],
    }, indent=1) + "\n")
    print(f"saved fine-tuned model -> {out_dir}")


if __name__ == "__main__":
    main()
