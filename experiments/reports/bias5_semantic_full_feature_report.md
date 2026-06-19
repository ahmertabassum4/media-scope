# Bias 5 Semantic Full Dataset Experiment Report

Model used for feature extraction: `openai/gpt-5.5`.

Dataset: full balanced 250-site bias dataset.

The LLM extracts structured visible-page features from the first four 16:9 viewport screenshots. Downstream classifiers predict the 5-label bias class from those features with stratified 5-fold CV.

## Best Method

`baseline_plus_semantic_visual_extra_trees_cv`: accuracy `0.740000`, macro-F1 `0.745695`.

## Metrics

```csv
method,sample_size,accuracy,macro_f1,weighted_f1
baseline_gpt5.5,250,0.532000,0.503595,0.503595
semantic_only_logreg_cv,250,0.664000,0.668320,0.668320
semantic_only_random_forest_cv,250,0.716000,0.718379,0.718379
semantic_only_extra_trees_cv,250,0.708000,0.711793,0.711793
semantic_only_mlp_cv,250,0.688000,0.691316,0.691316
baseline_plus_semantic_logreg_cv,250,0.684000,0.688386,0.688386
baseline_plus_semantic_random_forest_cv,250,0.716000,0.720067,0.720067
baseline_plus_semantic_extra_trees_cv,250,0.708000,0.714633,0.714633
baseline_plus_semantic_mlp_cv,250,0.720000,0.722934,0.722934
baseline_plus_semantic_visual_extra_trees_cv,250,0.740000,0.745695,0.745695
```

## Top ExtraTrees Feature Importances

- `ideological_direction_hint=right`: 0.072738
- `headline_frame=left_framed`: 0.071028
- `ideological_direction_hint=left`: 0.063900
- `headline_frame=right_framed`: 0.055617
- `page_scope=local`: 0.054330
- `source_type=local_news`: 0.045695
- `emotional_language_level`: 0.042404
- `ideological_intensity_hint=strong`: 0.038844
- `has_local_weather_or_traffic`: 0.027258
- `has_wire_service_style`: 0.025452
- `headline_frame=mostly_neutral`: 0.023147
- `overall_tone=neutral`: 0.022710
- `ideological_direction_hint=none`: 0.021588
- `ideological_intensity_hint=none`: 0.020956
- `advocacy_language_level`: 0.019528
- `sensationalism_level`: 0.018663
- `topic_local_community`: 0.017974
- `has_loaded_labels_for_opponents`: 0.017363
- `opinion_prominence_level`: 0.016939
- `partisan_branding_level`: 0.016778
- `topic_sports`: 0.016511
- `topic_technology`: 0.015901
- `topic_climate_environment`: 0.014114
- `main_content_format=news_homepage`: 0.014016
- `has_opinion_section`: 0.013844
