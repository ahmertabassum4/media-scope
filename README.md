# UGRIP 2026 Media Profiling

This repository predicts outlet factuality and political bias from three input
families:

- Screenshot features, OCR text, and DINOv2 embeddings (`src/`)
- Scraped article features (`baselines/MGM/`, Group A)
- Wikipedia features (`baselines/MGM/`, Group C)
- Multi-view graph and text fusion (`baselines/multiview/`)

The active OCR dataset is the cleaned copy under `data/clean/ocr/`. Raw splits
and feature archives remain under `data/splits/`, `data/features/`, and
`data/embeddings/` for reproducibility.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Commands assume the repository root is the working directory.

## OCR Experiments

Build the canonical dataset, run the multimodal factuality and bias
experiments, and train the text-only DistilBERT baselines:

```bash
python3 src/clean_ocr_data.py
python3 src/train.py --task factuality
python3 src/train.py --task bias
python3 baselines/text_only_llm/train.py --task factuality
python3 baselines/text_only_llm/train.py --task bias
```

The cleaning step removes exact duplicate model inputs across normalized outlet
keys, OCR text, compact features, screenshots, and visual embeddings. It writes
a manifest and the removed-row list without changing the raw inputs.

## Group A and C

The article pipeline keeps the BERT fine-tuning outlets separate from the SVM
training outlets. It also removes duplicate article text, canonical URLs, and
long body-prefix matches across all partitions.

```bash
python3 baselines/MGM/scrape_articles.py
python3 baselines/MGM/finetune_bert.py --task factuality
python3 baselines/MGM/finetune_bert.py --task bias
python3 baselines/MGM/extract_features.py --task factuality
python3 baselines/MGM/extract_features.py --task bias
python3 baselines/MGM/extract_wiki.py
python3 baselines/MGM/train_svm.py --task factuality
python3 baselines/MGM/train_svm.py --task bias
```

NELA features are task-independent and stored once in
`data/articles/nela_features.npz`. Group A and C leakage checks can be run with:

```bash
python3 baselines/MGM/audit_group_a.py --task all
```

## Multi-View Baseline

The paper-guided multi-view implementation combines Alexa, hyperlink, and LLM
graphs with article and Wikipedia embeddings. Its graph downloads are pinned,
smoke outputs cannot replace final archives, and final fusion rejects missing
or stale inputs. See [`baselines/multiview/README.md`](baselines/multiview/README.md)
for the run order, fidelity ledger, and leakage assumptions.

## Results and Tests

Experiment outputs are written to `results/current/`. Build the combined metric
tables with:

```bash
python3 src/report_baselines.py
```

Run the complete test suite and import/bytecode checks with:

```bash
python3 -m pytest -q src/tests baselines/text_only_llm/tests baselines/MGM/tests baselines/multiview/tests
python3 -m compileall -q src baselines
```
