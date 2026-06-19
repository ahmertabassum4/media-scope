# Bias 5 Advanced Image Feature Full Dataset Experiment Report

Device: `cuda`.

Dataset: full balanced 250-site bias dataset.

This run reuses full OCR, CLIP, SigLIP, DINOv2, and handcrafted visual features. Nested stacking is included to reduce optimistic leakage.

## Best New Method

`nested_gpt55_plus_local_prediction_stack_logreg_cv`: accuracy `0.740000`, macro-F1 `0.746109`.

## Best Overall Reference

`nested_gpt55_plus_local_prediction_stack_logreg_cv`: accuracy `0.740000`, macro-F1 `0.746109`.

## Metrics

```csv
method,sample_size,accuracy,macro_f1,weighted_f1
baseline_gpt5.5_cached,250,0.532000,0.503595,0.503595
clip_prompt_scores_logreg_cv,250,0.404000,0.391361,0.391361
siglip_prompt_scores_logreg_cv,250,0.372000,0.362956,0.362956
clip_siglip_prompt_scores_logreg_cv,250,0.480000,0.477381,0.477381
prompt_scores_plus_ocr_visual_extra_trees_cv,250,0.556000,0.553894,0.553894
siglip_ordinal_ridge_cv,250,0.344000,0.275423,0.275423
stack_ordinal_ridge_cv,250,0.348000,0.276894,0.276894
siglip_hierarchical_side_intensity_cv,250,0.572000,0.575422,0.575422
semantic_light_hierarchical_side_intensity_cv,250,0.564000,0.566063,0.566063
full_local_stack_pca_ridge_cv,250,0.592000,0.587766,0.587766
full_local_stack_extra_trees_cv,250,0.568000,0.570771,0.570771
nested_local_prediction_stack_logreg_cv,250,0.628000,0.631206,0.631206
nested_gpt55_plus_local_prediction_stack_logreg_cv,250,0.740000,0.746109,0.746109
reference_non_nested_strong_local_prediction_stack_logreg_cv,250,0.592000,0.593086,0.593086
reference_non_nested_local_embedding_visual_stack_extra_trees_cv,250,0.568000,0.571193,0.571193
reference_non_nested_siglip_b16_pca_ridge_cv,250,0.600000,0.597895,0.597895
```
