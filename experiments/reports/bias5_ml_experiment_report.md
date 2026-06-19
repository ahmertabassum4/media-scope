# Bias 5-Label ML Experiment Report

No new LLM calls were made in this experiment. It reuses cached full-run labels and screenshot pixels.

## Methods

- `visual_*`: classical ML using image-only features from the first four viewport-height screenshots.
- `gpt55_label_*`: calibration/stacking using only the cached GPT-5.5 5-label prediction, optionally plus visual features.
- `gpt55_least_visual_router_cv`: keep GPT-5.5 unless it predicts `least biased`; train a visual router for those ambiguous rows.
- `*_ordinal_ridge_cv`: regress labels on an ordinal left-to-right axis and round back to five labels.
- `all_model_*`: uses cached labels from all four prior models as an upper-bound stacking signal; no new API calls, but not a single-model method.

## Best By Macro-F1

`all_model_stack_plus_visual_extra_trees_cv`: accuracy `0.720000`, macro-F1 `0.717370`.

## Takeaways

- Best single-model method: `gpt55_label_plus_visual_extra_trees_cv`, accuracy `0.716000`, macro-F1 `0.716226`.
- Best overall method using cached labels from all four previous LLMs: `all_model_stack_plus_visual_extra_trees_cv`, accuracy `0.720000`, macro-F1 `0.717370`.
- The gain comes from using GPT-5.5 as a semantic prior and visual screenshot features as a learned calibrator. This directly fixes the baseline's major failure mode: over-predicting `least biased` for `left-center` and `right-center`.
- Visual-only ML is not strong enough by itself (`0.468`-`0.488` accuracy), but it carries useful calibration signal when combined with GPT-5.5.
- Simple stacking on labels alone is weak. `all_model_stack_logreg_cv` only reaches `0.544`; the visual features are the important additional signal.
- Ordinal regression was not competitive. The five labels behave partly ordinal, but the boundary between `least biased` and center-lean labels is not captured well by a single left-to-right scalar.

## Recommended Pipeline

For a single-model setup: run GPT-5.5 with the original 5-label prompt, extract the same four-viewport visual features from the screenshot, then classify with an ExtraTrees calibrator trained on `GPT-5.5 label + visual features`.

This keeps LLM cost to one GPT-5.5 call per website and moves the improvement into cheap local ML.

## Metrics

```csv
method,sample_size,evaluated,invalid_or_error,accuracy,macro_f1,weighted_f1
baseline_gpt5.5,250,250,0,0.532000,0.503595,0.503595
all_models_majority_vote,250,250,0,0.540000,0.514385,0.514385
visual_logreg_cv,250,250,0,0.468000,0.456437,0.456437
visual_random_forest_cv,250,250,0,0.488000,0.478217,0.478217
visual_extra_trees_cv,250,250,0,0.484000,0.471248,0.471248
gpt55_label_logreg_cv,250,250,0,0.532000,0.459257,0.459257
gpt55_label_plus_visual_logreg_cv,250,250,0,0.652000,0.646129,0.646129
gpt55_label_plus_visual_random_forest_cv,250,250,0,0.652000,0.655173,0.655173
gpt55_label_plus_visual_extra_trees_cv,250,250,0,0.716000,0.716226,0.716226
gpt55_least_visual_router_cv,250,250,0,0.652000,0.644886,0.644886
gpt55_least_allmodel_visual_router_cv,250,250,0,0.656000,0.649132,0.649132
gpt55_visual_ordinal_ridge_cv,250,250,0,0.480000,0.489824,0.489824
all_model_stack_logreg_cv,250,250,0,0.544000,0.504681,0.504681
all_model_stack_plus_visual_logreg_cv,250,250,0,0.672000,0.663781,0.663781
all_model_stack_plus_visual_extra_trees_cv,250,250,0,0.720000,0.717370,0.717370
all_model_visual_ordinal_ridge_cv,250,250,0,0.572000,0.583402,0.583402
all_model_stack_random_forest_cv,250,250,0,0.520000,0.498106,0.498106
```

Confusions are saved in `bias5_ml_confusions.csv`.
