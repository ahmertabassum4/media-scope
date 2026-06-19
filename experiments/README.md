# Experiments Package

The reports in `reports/` preserve the main results and method summaries.

## Layout

| File group | Purpose |
|---|---|
| `llm_client.py`, `image_process.py` | OpenRouter client, image preprocessing, shared bias/factuality helpers. |
| `bias_5_experiments.py` | Early GPT-5.5 prompt and cascade experiments on the 5-label bias task. |
| `bias5_ml_experiments.py` | GPT-label calibration and handcrafted visual-feature baselines for bias. |
| `bias5_image_feature_*` | OCR, CLIP, SigLIP, DINOv2, prompt-score, and nested local-feature bias experiments. |
| `bias5_semantic_*` | GPT-5.5 structured semantic feature extraction and downstream bias classifiers. |
| `bias5_top_ablation_experiments.py` | Ablations for the top nested and semantic+visual bias methods. |
| `bias5_pca_component_fusion_experiments.py` | Direct PCA/SVD component fusion experiments for bias. |
| `bias5_ordinal_metrics.py` | Ordinal metrics for 5-label bias predictions. |
| `bias5_cnn_experiments.py` | ResNet18 embedding and scratch-CNN bias experiments. |
| `factuality_common.py` | Shared factuality dataset, crop, label, metric, and CSV helpers. |
| `factuality_prepare_features.py` | Factuality OCR, visual, embedding, GPT baseline, and semantic feature preparation. |
| `factuality_experiments.py` | Full binary and 4-label factuality experiment grid. |
| `factuality_results_analysis.py` | Factuality result summaries and selected per-class metrics. |
| `synthetic_factuality_sites.py` | Synthetic factuality screenshot generation by headline/layout transfer. |
| `synthetic_factuality_eval.py` | LLM evaluation for synthetic factuality screenshots. |
| `evidence_highlighting.py` | PaddleOCR/layout evidence grounding and highlight overlays for LLM explanations. |
| `validate_experiment_package.py` | Fast package integrity check. |

## Running

Install dependencies from the project root:

```bash
pip install -r experiments/requirements.txt
```

The scripts can be launched from either the project root or from inside `experiments`. Each script activates the project root at startup so existing relative paths to `bias-samples`, `snapshot-samples`, and saved CSV artifacts continue to work.

Examples:

```bash
python experiments/bias5_pca_component_fusion_experiments.py
python experiments/factuality_experiments.py
python experiments/bias5_top_ablation_experiments.py
python experiments/evidence_highlighting.py --image application/data/images/001_247sports.jpg
```

Experiments that call OpenRouter require `OPENROUTER_API_KEY` in the project `.env`.

## Validation

Run the fast package check:

```bash
python experiments/validate_experiment_package.py
```

The validator checks:

- required files are present
- every Python file compiles
- no Cyrillic text is present in packaged Python, Markdown reports, or requirements

It does not run full experiments because those can be expensive, slow, and dependent on GPU/model caches.

## Main Reports

The most useful entry points are:

- `reports/experiment_methods_summary.md`
- `reports/factuality_experiment_summary.md`
- `reports/factuality_all_methods_report.md`
- `reports/bias5_top_ablation_report.md`
- `reports/bias5_pca_component_fusion_report.md`
- `reports/bias5_ordinal_metrics_report.md`
