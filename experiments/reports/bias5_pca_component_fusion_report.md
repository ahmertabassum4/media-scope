# Bias 5 PCA Component Fusion Experiment

Dataset: full balanced 250-site bias dataset.

Each fold independently fits PCA for CLIP, SigLIP, DINOv2 and visual features, plus TF-IDF/TruncatedSVD for OCR text. The resulting continuous components are concatenated with OCR numeric features, prompt scores, and one-hot GPT-5.5 labels and passed directly to one final classifier.

## Best Results

- Best Macro-F1: `medium::core_without_clip::ridge_a3` = `0.771920`; accuracy `0.772000`.
- Best MSE: `compact::core_without_clip::mlp` = `0.400000`; Macro-F1 `0.743125`.
- Best QWK: `compact::core_without_clip::mlp` = `0.898167`.

## Top 15 By Macro-F1

| Method | Accuracy | Macro-F1 | MAE | MSE | Within-1 | QWK |
|---|---:|---:|---:|---:|---:|---:|
| `medium::core_without_clip::ridge_a3` | 0.772000 | 0.771920 | 0.300000 | 0.460000 | 0.936000 | 0.878820 |
| `large::pruned_components::mlp` | 0.772000 | 0.771731 | 0.344000 | 0.688000 | 0.924000 | 0.826263 |
| `large::all_components::extra_trees` | 0.760000 | 0.765435 | 0.332000 | 0.596000 | 0.936000 | 0.844953 |
| `large::pruned_components::ridge_a3` | 0.764000 | 0.764862 | 0.296000 | 0.448000 | 0.956000 | 0.884298 |
| `medium::pruned_components::logreg_c0.2` | 0.760000 | 0.764278 | 0.308000 | 0.476000 | 0.948000 | 0.878447 |
| `large::pruned_components::logreg_c0.2` | 0.760000 | 0.762331 | 0.308000 | 0.484000 | 0.948000 | 0.875386 |
| `medium::pruned_components::ridge_a3` | 0.760000 | 0.762169 | 0.300000 | 0.436000 | 0.948000 | 0.886812 |
| `large::pruned_components::logreg_c1` | 0.756000 | 0.759136 | 0.312000 | 0.488000 | 0.948000 | 0.875000 |
| `large::pruned_components::extra_trees` | 0.760000 | 0.755899 | 0.300000 | 0.428000 | 0.944000 | 0.887487 |
| `large::all_components::logreg_c1` | 0.752000 | 0.754738 | 0.340000 | 0.612000 | 0.940000 | 0.846847 |
| `large::core_without_clip::logreg_c0.2` | 0.752000 | 0.754126 | 0.312000 | 0.456000 | 0.944000 | 0.883436 |
| `large::core_without_clip::mlp` | 0.756000 | 0.754110 | 0.324000 | 0.516000 | 0.932000 | 0.870871 |
| `large::all_components::logreg_c0.2` | 0.752000 | 0.754086 | 0.344000 | 0.624000 | 0.936000 | 0.844311 |
| `large::core_without_clip::ridge_a3` | 0.748000 | 0.751646 | 0.324000 | 0.500000 | 0.944000 | 0.868835 |
| `large::pruned_components::hist_gradient_boosting` | 0.744000 | 0.751577 | 0.384000 | 0.736000 | 0.912000 | 0.802151 |

## Reference

- Best nested label-level stack before this experiment: accuracy `0.772000`, Macro-F1 `0.775729`.
- Best ordinal nested configuration: MSE `0.408000`, QWK `0.893305`.