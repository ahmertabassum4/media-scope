# Project Memory: UGRIP 2026 Media Profiling

Keep this file in the repository. Update it when the dataset, experiment
protocol, active paths, or validated results change.

Last validated: 2026-07-16.

## Scope

The project predicts outlet factuality and political bias from three input
families:

- Screenshot features, OCR text, and DINOv2 embeddings (`src/`)
- Scraped news articles (`baselines/MGM/`, Group A)
- English Wikipedia pages (`baselines/MGM/`, Group C)

No test labels are used for model fitting or hyperparameter selection.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Run commands from the repository root. Human-facing setup and pipeline commands
are also documented in `README.md`.

## Canonical OCR Dataset

Raw sources are preserved:

- `data/splits/train.csv`: 2,279 rows
- `data/splits/test.csv`: 561 rows
- `data/features/features.jsonl`
- `data/embeddings/visual_embeddings.safe.npz`
- `data/screenshots/`

OCR-based experiments use the cleaned copies in `data/clean/ocr/`:

- Train: 2,259 rows
- Test: 559 rows
- Feature and embedding rows: 2,818
- Removed: 20 train rows and 2 test rows
- Duplicate components found: 18
- Conflicting duplicate components fully removed: 3

The cleaner checks normalized outlet keys, normalized exact OCR, exact compact
feature vectors, exact screenshot bytes, and exact normalized visual
embeddings. Same-label duplicates keep one deterministic copy, preferring test;
conflicting-label components are dropped completely. Raw inputs are not
modified.

Factuality class counts:

- Train: LOW 364 / MIXED 1,212 / HIGH 683
- Test: LOW 93 / MIXED 293 / HIGH 173

Bias class counts for labeled rows:

- Train: left 148 / left-center 320 / center 309 / right-center 911 / right 564
- Test: left 38 / left-center 77 / center 74 / right-center 221 / right 145

Rebuild with:

```bash
python3 src/clean_ocr_data.py
```

## OCR Models

`src/train.py` evaluates factuality and five-class bias using:

- 51 deterministic screenshot features
- OCR TF-IDF text features
- DINOv2 visual embeddings
- All ablations and a majority baseline

The 51-feature expert tunes its model inside each outer fold before producing
out-of-fold predictions. Current multimodal results are:

- Factuality full: 81.93 accuracy / 79.98 macro-F1 (n=559)

| Bias configuration | Accuracy | Macro accuracy | Macro precision | Macro recall | Macro-F1 | MAE | MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 78.02 | 74.24 | 70.66 | 74.24 | 71.11 | 0.4342 | 1.1441 |
| Removed OCR | 69.91 | 62.50 | 63.72 | 62.50 | 61.63 | 0.7459 | 2.2631 |
| Removed 51 features | 78.56 | 74.19 | 70.87 | 74.19 | 71.41 | 0.4234 | 1.1261 |
| Only OCR | 78.92 | 74.66 | 71.06 | 74.66 | 71.75 | 0.4144 | 1.1063 |
| Only DINOv2 | 62.16 | 51.11 | 52.25 | 51.11 | 51.24 | 0.8739 | 2.5207 |
| Only 51 features | 70.45 | 64.74 | 65.06 | 64.74 | 62.94 | 0.6901 | 2.0450 |

Bias excludes the seven unlabeled training outlets and four unlabeled test
outlets. The bias ablations use the same feature, OCR, and DINO pipelines as
factuality, with five aligned output classes.

Run the two tasks with `python3 src/train.py --task factuality` and
`python3 src/train.py --task bias`. Bias artifacts use the
`results/current/multimodal_bias_*` filenames and do not overwrite factuality.

`baselines/text_only_llm/train.py` fine-tunes `distilbert-base-uncased` for four
epochs with a maximum length of 512 and a 15% outlet-level validation split.
Current results:

- Factuality: 76.21 accuracy / 73.70 macro-F1 (n=559)
- Bias: 74.05 accuracy / 66.08 macro-F1 (n=555)

Active checkpoints and prediction files are under
`baselines/text_only_llm/models/{factuality,bias}/`. The obsolete pre-cleaning
`*_v2_raw` checkpoints were removed.

## Group A: Articles

Shared implementation: `baselines/MGM/pipeline.py`.

- One third of training outlets is reserved for BERT fine-tuning.
- Remaining training outlets fit the SVM classifiers.
- Fine-tuning, SVM, validation, and test outlet keys are disjoint.
- Factuality BERT supervision is binary LOW versus HIGH, as in the paper.
- Bias BERT supervision uses five classes.
- SVM evaluation uses NELA, BERT representations, BERT probabilities, their
  concatenation/ensemble, and Group A+C combinations.

The article filter starts with 45,144 usable articles. It removes 4,114
cross-outlet copies and 293 within-outlet duplicates, retaining 40,737
articles. Active partitions have zero overlap by normalized exact text,
canonical URL, or long article-body prefix.

NELA features are task-independent and stored once at
`data/articles/nela_features.npz`.

The split treats dataset outlet keys as outlet identities. A stricter
publisher-family interpretation still finds shared registered domains for a
few differently named/localized outlets (`focusonthefamily.com`,
`news-pravda.com`, and for bias `gaysagainstgroomers.com`). No retained article
content is shared across those partitions.

## Group C: Wikipedia

`baselines/MGM/extract_wiki.py` builds `data/articles/feats_wiki.npz` with one
768-dimensional BERT vector per outlet. The current cache is complete for all
2,831 outlet keys:

- Wikipedia pages found: 852
- Duplicate pages removed: 2
- Unique retained pages: 850
- Missing pages use zero vectors, matching the paper's missing-media policy

## Multi-View Baseline

`baselines/multiview/` is a paper-guided adaptation of arXiv:2605.01336v1,
not an exact reproduction. It combines Alexa, hyperlink, and LLM graphs with
task-specific DistilBERT article and Wikipedia embeddings.

- The canonical cleaned universe is 2,818 outlets.
- Factuality uses 2,259 train / 559 test outlets and three classes.
- Bias uses 2,252 train / 555 test outlets and five classes.
- Alexa matches 719 outlets; hyperlink matches 1,582 outlets.
- Graph mappings contain one outlet per node. URL/host matching does not
  collapse hosted subdomains, and ambiguous registered domains are excluded.
- Source downloads are pinned by commit and SHA-256. Embeddings and result
  manifests carry universe, source, match, model, and output fingerprints.
- Graph DGI uses feature shuffling for Alexa and topology corruption for the
  constant-feature hyperlink/LLM graphs. Graph learning is label-free but
  transductive, since held-out nodes can occur in the unlabeled topology.
- Article and Wikipedia text views use the shared global duplicate filters;
  text labels are read from training outlets only.
- Smoke and non-paper settings always use suffixed filenames. Final fusion
  fails before writing if any required archive is missing, stale, partial, or
  has the wrong outlet keys or embedding width.
- Fusion rows are task-specific MBFC-2025 combinations from Tables 5/6, plus
  all nine graph/encoder SVM ablations. The paper's separate PLM voting rows
  are not mislabeled as multi-view rows.

Pre-audit 2,831-key GNN files were preserved as
`data/graphs/*_pre_audit_invalid.npz`; the old partial result was preserved as
`results/current/multiview_factuality_pre_audit_partial.csv`. Fusion rejects
those archives. Current `multiview_*_partial.csv` files are validation-only
graph SVM reports and are not final tables. The production LLM graph and four
full text embedding archives are still required before final fusion.

The detailed run order, fidelity ledger, and leakage assumptions are in
`baselines/multiview/README.md`.

## Results

Current task reports:

- `results/current/all_baselines_factuality.csv`: 17 rows, n=559
- `results/current/all_baselines_bias.csv`: 17 rows, n=555
- `results/current/all_baselines_manifest.json`: source and output hashes

Regenerate reports with:

```bash
python3 src/report_baselines.py
```

Metric columns are accuracy, balanced accuracy (reported as macro accuracy),
macro precision, macro recall, macro-F1, MAE, and MSE. Ordinal mappings are:

- Factuality: LOW=-1, MIXED=0, HIGH=1
- Bias: left=-2, left-center=-1, center=0, right-center=1, right=2

## Validation State

The latest full validation passed:

- Canonical clean rebuild is deterministic
- No exact OCR, feature, screenshot, embedding, or outlet-key train/test overlap
- Group A article leakage audit passes for factuality and bias
- Group C archive is complete, aligned, and finite
- All saved prediction metrics and result hashes match their manifests
- Multimodal bias prediction coverage is 7 models x 555 outlets (3,885 rows)
- DistilBERT checkpoints and tokenizers reload locally
- 85 tests pass across OCR, text-only, MGM, and multi-view suites
- `compileall`, `pyflakes`, CLI import checks, and runtime dependency imports pass
- No scraper or LLM API process was left running. A user-started production
  hyperlink ResGated GNN job was still active during the 2026-07-16 audit.

Global Anaconda contains unrelated package-version conflicts reported by
`pip check`; the dependencies used by this project import and execute
successfully. Generated caches, logs, duplicate documentation tables, stale
checkpoints, and the incomplete upstream repository snapshot were removed.
