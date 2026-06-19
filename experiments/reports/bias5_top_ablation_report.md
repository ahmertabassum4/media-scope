# Bias 5 Top Method Ablation Report

Dataset: full balanced 250-site bias dataset.

## Nested Stack Source Ablation

Full nested stack: accuracy `0.740000`, macro-F1 `0.746109`.

Largest drops when a source is removed:

- `without_gpt55_cached`: macro-F1 `0.631206` (-0.114903)
- `without_ocr_text_tfidf`: macro-F1 `0.729901` (-0.016208)
- `without_siglip_pca_ridge`: macro-F1 `0.738626` (-0.007483)
- `without_clip_pca_ridge`: macro-F1 `0.749343` (+0.003234)
- `without_ocr_numeric_visual_extra_trees`: macro-F1 `0.750652` (+0.004543)
- `without_dino_pca_ridge`: macro-F1 `0.751474` (+0.005365)
- `without_prompt_scores_logreg`: macro-F1 `0.758593` (+0.012484)
- `without_visual_extra_trees`: macro-F1 `0.760843` (+0.014734)

Best nested/source rows:

- `nested_stack::core_gpt55_ocr_text_siglip`: accuracy `0.772000`, macro-F1 `0.775729`
- `nested_stack::pruned_without_visual_prompt_dino_ocrnum`: accuracy `0.768000`, macro-F1 `0.773924`
- `nested_stack::core_gpt55_ocr_text_siglip_clip`: accuracy `0.768000`, macro-F1 `0.773924`
- `nested_stack::pruned_without_visual_prompt_dino`: accuracy `0.768000`, macro-F1 `0.773105`
- `nested_stack::without_visual_extra_trees`: accuracy `0.756000`, macro-F1 `0.760843`
- `nested_stack::without_prompt_scores_logreg`: accuracy `0.752000`, macro-F1 `0.758593`
- `nested_stack::pruned_without_visual_and_prompt`: accuracy `0.752000`, macro-F1 `0.757271`
- `nested_stack::core_gpt55_ocr_text_siglip_ocrnum`: accuracy `0.752000`, macro-F1 `0.756260`
- `nested_stack::without_dino_pca_ridge`: accuracy `0.744000`, macro-F1 `0.751474`
- `nested_stack::without_ocr_numeric_visual_extra_trees`: accuracy `0.744000`, macro-F1 `0.750652`
- `nested_stack::without_clip_pca_ridge`: accuracy `0.744000`, macro-F1 `0.749343`
- `nested_stack::full`: accuracy `0.740000`, macro-F1 `0.746109`

## Semantic + Visual ExtraTrees Ablation

Full semantic+visual model: accuracy `0.740000`, macro-F1 `0.745695`.

Largest drops when a group is removed:

- `without_all_semantic_excluding_baseline`: macro-F1 `0.695806` (-0.049889)
- `without_visual_features`: macro-F1 `0.714633` (-0.031062)
- `without_ideology_hint`: macro-F1 `0.724434` (-0.021261)
- `without_topics_boolean_baseline_and_numeric`: macro-F1 `0.735052` (-0.010643)
- `without_categorical_source_page`: macro-F1 `0.735831` (-0.009864)
- `without_numeric_levels_counts`: macro-F1 `0.739043` (-0.006652)
- `without_topics_boolean_and_numeric`: macro-F1 `0.739666` (-0.006029)
- `without_baseline_label`: macro-F1 `0.748701` (+0.003006)

Best single-feature cuts among tested high/low-importance candidates:

- remove `baseline_gpt5.5=least biased`: accuracy `0.744000`, macro-F1 `0.747594` (+0.001899)
- remove `visual::image_width`: accuracy `0.740000`, macro-F1 `0.746691` (+0.000996)
- remove `ideological_direction_hint=mixed_or_unclear`: accuracy `0.740000`, macro-F1 `0.746484` (+0.000789)
- remove `baseline_gpt5.5=right`: accuracy `0.740000`, macro-F1 `0.745543` (-0.000152)
- remove `ideological_intensity_hint=strong`: accuracy `0.740000`, macro-F1 `0.745168` (-0.000527)
- remove `main_content_format=research_org`: accuracy `0.740000`, macro-F1 `0.744412` (-0.001283)
- remove `overall_tone=neutral`: accuracy `0.736000`, macro-F1 `0.743212` (-0.002483)
- remove `headline_frame=right_framed`: accuracy `0.740000`, macro-F1 `0.742098` (-0.003597)
- remove `main_content_format=aggregator`: accuracy `0.736000`, macro-F1 `0.741565` (-0.004130)
- remove `main_content_format=unclear`: accuracy `0.736000`, macro-F1 `0.740985` (-0.004710)

## Files

- `bias5_top_nested_source_ablation_metrics.csv`
- `bias5_top_nested_source_ablation_predictions.csv`
- `bias5_top_nested_source_ablation_confusions.csv`
- `bias5_top_semantic_visual_group_ablation_metrics.csv`
- `bias5_top_semantic_visual_single_feature_ablation.csv`
- `bias5_top_semantic_visual_group_ablation_predictions.csv`