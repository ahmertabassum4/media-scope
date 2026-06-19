# Bias 5 Full Dataset Top Experiment Summary

Dataset: full balanced 250-site bias dataset.

## Best Results

| Rank | Method | Accuracy | Macro-F1 | Source |
|---:|---|---:|---:|---|
| 1 | `nested_gpt55_plus_local_prediction_stack_logreg_cv` | 0.740000 | 0.746109 | `bias5_image_feature_full_advanced_metrics.csv` |
| 2 | `baseline_plus_semantic_visual_extra_trees_cv` | 0.740000 | 0.745695 | `bias5_semantic_full_ml_metrics.csv` |
| 3 | `baseline_plus_semantic_mlp_cv` | 0.720000 | 0.722934 | `bias5_semantic_full_ml_metrics.csv` |
| 4 | `baseline_plus_semantic_random_forest_cv` | 0.716000 | 0.720067 | `bias5_semantic_full_ml_metrics.csv` |
| 5 | `semantic_only_random_forest_cv` | 0.716000 | 0.718379 | `bias5_semantic_full_ml_metrics.csv` |
| 6 | previous best `all_model_stack_plus_visual_extra_trees_cv` | 0.720000 | 0.717370 | `bias5_ml_metrics.csv` |
| 7 | previous cheap best `gpt55_label_plus_visual_extra_trees_cv` | 0.716000 | 0.716226 | `bias5_ml_metrics.csv` |
| 8 | `gpt55_plus_strong_local_majority_vote` | 0.684000 | 0.683462 | `bias5_image_feature_full_metrics.csv` |
| 9 | GPT-5.5 baseline | 0.532000 | 0.503595 | `bias5_ml_metrics.csv` |

## Method Notes

### `nested_gpt55_plus_local_prediction_stack_logreg_cv`

Uses no new LLM calls beyond the cached GPT-5.5 label. Local features are extracted from the page screenshots:

- EasyOCR text and numeric OCR/lexicon counts from the first two 16:9 viewport crops.
- CLIP ViT-B/32 embeddings from the first four 16:9 viewport crops.
- SigLIP B/16 embeddings from the first four 16:9 viewport crops.
- DINOv2-small embeddings from the first four 16:9 viewport crops.
- Handcrafted visual features: color, brightness, edge density, dark/light fractions, top-band color dominance, image size signals.
- CLIP/SigLIP prompt-score features.

Evaluation uses nested stacking:

- Outer stratified 5-fold CV produces final predictions.
- Inside each outer train fold, base model predictions are generated with inner 4-fold CV.
- A logistic-regression meta-classifier is trained on one-hot base predictions plus the cached GPT-5.5 label.

This is the best full-dataset result by a very small margin: macro-F1 `0.746109`.

### `baseline_plus_semantic_visual_extra_trees_cv`

Uses GPT-5.5 as a structured feature extractor, not only as a labeler. For each site, GPT-5.5 sees the first four 16:9 viewport screenshots and outputs structured visible-page cues:

- source/page type,
- page scope,
- ideological direction and intensity hints,
- headline framing,
- main content format,
- tone,
- topic booleans,
- numeric levels for advocacy, opinion prominence, sensationalism, partisan branding, CTA, emotional language, and political content,
- boolean cues like donation/join CTA, local weather/traffic, wire-service style, balanced framing, loaded labels, research/data cues.

Those features are combined with:

- cached GPT-5.5 baseline label,
- handcrafted visual features.

The final classifier is `ExtraTreesClassifier` with stratified 5-fold CV. This result is effectively tied with the best stack: macro-F1 `0.745695`.

## Semantic Feature Signals

Top full-dataset semantic feature importances:

- `ideological_direction_hint=right`: 0.072738
- `headline_frame=left_framed`: 0.071028
- `ideological_direction_hint=left`: 0.063900
- `headline_frame=right_framed`: 0.055617
- `page_scope=local`: 0.054330
- `source_type=local_news`: 0.045695
- `emotional_language_level`: 0.042404
- `ideological_intensity_hint=strong`: 0.038844

Group ablation confirms that ideology/framing features matter most: removing `ideology_hint` drops semantic-only ExtraTrees macro-F1 by `0.081860`.

## Conclusion

The previous full-dataset ceiling was `0.717370` macro-F1. The new full-dataset ceiling is `0.746109` macro-F1.

The two best approaches are close enough that both are worth keeping:

- Use `nested_gpt55_plus_local_prediction_stack_logreg_cv` when we want to minimize extra paid LLM feature extraction.
- Use `baseline_plus_semantic_visual_extra_trees_cv` when we can afford GPT-5.5 structured feature extraction and want a more interpretable feature set.
