# Multi-View Media Profiling on Mediascope-2800

This directory is a paper-guided reimplementation of **"A Multi-View Media
Profiling Suite: Resources, Evaluation, and Analysis"** (arXiv:2605.01336v1)
on the project's cleaned outlet-level train/test split. It is not an exact
reproduction: the paper does not specify every model and pooling detail, and
our outlets, labels, split, articles, and Wikipedia pages differ from theirs.

The canonical universe has 2,818 outlets. Factuality has 2,259 train and 559
test outlets with LOW/MIXED/HIGH labels. Bias has 2,252 train and 555 test
outlets with left/left-center/center/right-center/right labels.

| view | local source | dim | current matched coverage |
|---|---|---:|---:|
| F(a) Alexa audience graph | pinned level-3 crawl from `marslanm/MGM_code` | 64 | 719 / 2,818 |
| F(h) Hyperlink graph | pinned MBFC-2023 graph from `marslanm/multi-graph-perspective` | 64 | 1,582 / 2,818 |
| F(l) LLM similarity graph | regenerated from one canonical domain per outlet | 64 | generated at run time |
| F(t) Articles | cleaned and deduplicated `data/articles/articles.jsonl` | 768 | data dependent |
| F(w) Wikipedia | deduplicated `data/wiki/wiki_pages.jsonl` | 768 | data dependent |

Unmatched outlets receive zero vectors. Source downloads are pinned to commit
revisions and SHA-256 hashes. Embedding archives include the outlet-universe
hash, source fingerprints, model settings, and a `complete` flag. Fusion
rejects stale, partial, wrong-width, or wrong-universe archives.

## Run Order

Use `/opt/anaconda3/bin/python3` from the repository root.

```bash
PY=/opt/anaconda3/bin/python3

# 1. Download pinned graph sources and rebuild outlet matches.
$PY baselines/multiview/fetch_graphs.py

# 2. Exercise graph expansion without an API key or canonical outputs.
$PY baselines/multiview/llm_graph.py --dry-run --limit-seeds 10

# 3. Build the production LLM graph. This is resumable and incurs API cost.
export OPENAI_API_KEY=...
nohup $PY baselines/multiview/llm_graph.py \
  > data/graphs/llm_graph.log 2>&1 &

# 4. Train all three encoders for each graph. Run these jobs serially unless
# the machine has enough memory for multiple full graphs.
$PY baselines/multiview/gnn_embed.py --graph alexa --encoder all
$PY baselines/multiview/gnn_embed.py --graph hyperlink --encoder all
$PY baselines/multiview/gnn_embed.py --graph llm --encoder all

# 5. Fine-tune DistilBERT separately for each view and task.
nohup $PY baselines/multiview/text_embed.py --view articles --task factuality \
  > data/graphs/text_articles_factuality.log 2>&1 &
# Wait for that process before starting another large text job.

# 6. Produce final tables only after every required archive is complete.
$PY baselines/multiview/fuse.py --task factuality
$PY baselines/multiview/fuse.py --task bias
```

Track long jobs with `ps`, `tail`, and the response count:

```bash
ps -axo pid,etime,%cpu,%mem,command | rg 'multiview|gnn_embed|text_embed'
tail -f data/graphs/llm_graph.log
wc -l data/graphs/llm_responses.jsonl
```

Each long-running script holds a nonblocking job lock. Starting a second
instance for the same output fails immediately. Writes are atomic, so a killed
process cannot replace a valid archive with a partial file.

Smoke settings such as `--epochs 1`, `--limit`, `--limit-seeds`, `--dry-run`,
or a non-paper graph level always use suffixed filenames. They cannot overwrite
canonical archives. `fuse.py` fails before writing a final CSV if any input is
missing or invalid. `--allow-partial` writes
`results/current/multiview_<task>_partial.csv` instead.

## Benchmark Rows

The suite evaluates all nine graph/encoder combinations with a linear SVM.
It then applies the MBFC-2025 fusion combinations from the paper separately by
task:

- Bias: SVM, MLP, self-attention, and PPO combinations from Table 5.
- Factuality: cross-attention, co-attention, and PPO combinations from Table 6.

The paper's separate article-level PLM and hard/soft-voting rows use BERT,
RoBERTa, DistilBERT, and ALBERT classifiers. Those are not relabeled as
multi-view fusion rows here. This implementation uses task-fine-tuned
DistilBERT embeddings for F(t) and F(w).

Final outputs are:

- `results/current/multiview_<task>.csv`
- `results/current/multiview_<task>_predictions.csv`
- `results/current/multiview_<task>_manifest.json`

Metrics are accuracy, balanced accuracy (reported as macro accuracy), macro
precision/recall/F1, per-class F1, MAE, and MSE. Ordinal values are -1/0/1 for
factuality and -2/-1/0/1/2 for bias.

## Fidelity Ledger

Paper-specified details retained:

- The released Alexa and hyperlink graph files and five Alexa attributes.
- GCN/GraphConv, GraphSAGE, and ResGatedGCN with four layers, hidden size 128,
  output size 64, 50 epochs, learning rate 1e-4, and dropout 0.5.
- Constant node features for hyperlink and LLM graphs.
- The published LLM prompt, at most five parsed sites, and level-3 expansion.
- DistilBERT max length 256, six epochs, and learning rate 2e-5.
- Linear SVM with `max_iter=60` and `tol=0.01`.
- PPO contextual-bandit settings: gamma 0, two 128-unit tanh layers, learning
  rate 1e-4, batch 256, rollout 1,024, and true-label-probability reward.

Documented assumptions and deviations:

- The paper names an unsupervised contrastive objective but not its algorithm.
  This code uses Deep Graph Infomax. Alexa corrupts features; constant-feature
  graphs corrupt topology because shuffling identical rows is a no-op.
- Current pinned graphs are optimized full-batch. This differs from the
  paper's reported GNN batch size of 128 and GraphSAGE neighbor sampling.
- Text uses the `[CLS]` last-hidden state and averages documents per outlet.
  The paper does not fully specify outlet embedding pooling.
- Effective text batch size is 96 (`16 x 6` gradient accumulation), close to
  the paper's MBFC DistilBERT batch size of 100.
- Attention modules project each view to a 64-dimensional token. Query groups
  are text-to-graph; for Articles+Wikipedia, Articles is the query.
- The fixed project test split is retained. The 10% development carve is from
  the existing training split, so it is not the paper's global 80/10/10 split.
- The LLM graph defaults to the paper's `gpt-3.5-turbo-0125`. A replacement
  model is allowed only as a documented provenance change in the manifest.
- Our factuality task is three-class; bias is five-class.

## Leakage and Provenance

- Only outlets in the canonical cleaned OCR split are admitted. The raw
  article file cannot reintroduce removed duplicate/conflicting outlets.
- Article content is deduplicated globally before text training. A test copy is
  retained and matching train copies are removed. Wikipedia duplicate pages
  receive the same treatment; conflicting-label duplicate groups are dropped.
- Graph matching uses exact URL/host matches and only unambiguous registered
  domains. One outlet is retained per graph node, preferring the test copy so
  a shared node cannot occur in train and test.
- Text models use labels from training outlets only. Fusion scalers, SVMs,
  neural models, frozen RL classifiers, and PPO policies fit on train/dev only.
- GNN training is label-free but transductive: test nodes may participate in
  the unlabeled graph topology. This is not an inductive graph evaluation.
- The external LLM may encode prior knowledge about outlet reputation. The LLM
  view is therefore structural and label-free in code, but not knowledge-free.

## Validation

```bash
$PY -m pytest -q baselines/multiview/tests
$PY -m compileall -q baselines/multiview
$PY -m pyflakes baselines/multiview
```

As of the audit on 2026-07-16, pre-audit 2,831-outlet GNN archives and the
partial result were preserved with `pre_audit_invalid` / `pre_audit_partial`
suffixes. They are not accepted by fusion. One-epoch Alexa and hyperlink smoke
archives pass the corrected 2,818-outlet contract. Some 50-epoch graph archives
were also regenerated during the audit, but the production LLM graph and full
text archives are still missing, so no final multi-view table exists yet.
