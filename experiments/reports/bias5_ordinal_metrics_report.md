# Bias 5 Ordinal Metrics Report

Dataset: full balanced 250-site bias dataset.

Ordinal encoding: `left=0`, `left-center=1`, `least biased=2`, `right-center=3`, `right=4`.

Lower is better for MAE, MSE, RMSE, normalized MSE, and severe-error rates. Higher is better for accuracy, within-N accuracy, MSE closeness, and quadratic weighted kappa.

## Best Results

- Best MSE: `nested_stack::pruned_without_visual_prompt_dino_ocrnum` = `0.408000`.
- Best MAE: `nested_stack::pruned_without_visual_prompt_dino_ocrnum` = `0.288000` label steps.
- Best quadratic weighted kappa: `nested_stack::pruned_without_visual_prompt_dino_ocrnum` = `0.893305`.

## Selected Methods

| Method | Accuracy | Macro-F1 | MAE | MSE | RMSE | Within 1 | Severe >=2 | QWK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `nested_stack::pruned_without_visual_prompt_dino_ocrnum` | 0.768000 | 0.773924 | 0.288000 | 0.408000 | 0.638749 | 0.948000 | 0.052000 | 0.893305 |
| `nested_stack::core_gpt55_ocr_text_siglip` | 0.772000 | 0.775729 | 0.304000 | 0.488000 | 0.698570 | 0.936000 | 0.064000 | 0.871849 |
| `all_model_stack_plus_visual_extra_trees_cv` | 0.720000 | 0.717370 | 0.344000 | 0.496000 | 0.704273 | 0.948000 | 0.052000 | 0.870833 |
| `semantic_visual_extra_trees::without_topics_boolean_and_baseline` | 0.752000 | 0.754188 | 0.328000 | 0.504000 | 0.709930 | 0.928000 | 0.072000 | 0.866242 |
| `bias_5:anthropic/claude-sonnet-4.6` | 0.552000 | 0.529386 | 0.500000 | 0.604000 | 0.777174 | 0.948000 | 0.052000 | 0.825836 |
| `semantic_visual_extra_trees::full` | 0.740000 | 0.745695 | 0.360000 | 0.608000 | 0.779744 | 0.920000 | 0.080000 | 0.839662 |
| `gpt55_label_plus_visual_extra_trees_cv` | 0.716000 | 0.716226 | 0.376000 | 0.616000 | 0.784857 | 0.932000 | 0.068000 | 0.840909 |
| `nested_stack::full` | 0.740000 | 0.746109 | 0.356000 | 0.644000 | 0.802496 | 0.940000 | 0.060000 | 0.833506 |
| `bias_5:openai/gpt-5.5` | 0.532000 | 0.503595 | 0.532000 | 0.660000 | 0.812404 | 0.936000 | 0.064000 | 0.801444 |
| `bias_5:moonshotai/kimi-k2.6` | 0.544000 | 0.508361 | 0.524000 | 0.660000 | 0.812404 | 0.932000 | 0.068000 | 0.806110 |
| `bias_5:qwen/qwen3.7-plus` | 0.512000 | 0.487914 | 0.552000 | 0.680000 | 0.824621 | 0.936000 | 0.064000 | 0.791667 |
| `semantic_only_random_forest_cv` | 0.716000 | 0.718379 | 0.404000 | 0.700000 | 0.836660 | 0.904000 | 0.096000 | 0.814422 |
| `always_least_biased` | 0.200000 | 0.066667 | 1.200000 | 2.000000 | 1.414214 | 0.600000 | 0.400000 | 0.000000 |

## Interpretation

- Exact accuracy and Macro-F1 treat every wrong class as fully wrong.
- MAE reports the average number of label positions missed.
- MSE penalizes distant errors quadratically, so an error of two positions costs four times an adjacent error.
- Within-1 accuracy gives credit when the prediction is exact or one neighboring bias class away.
- Quadratic weighted kappa measures chance-corrected ordinal agreement and penalizes distant disagreements more strongly.
- `mse_closeness_score = 1 - MSE/16` is a convenience normalization for this five-class scale, not a standard research metric.

Ranked Probability Score was not calculated because the saved pipelines expose hard labels rather than calibrated per-class probabilities.

## Research Context

- Gaudette and Japkowicz compare ordinal evaluation methods and emphasize that plain accuracy ignores error severity: https://doi.org/10.1007/978-3-642-01818-3_25
- Cohen's weighted kappa gives partial credit for disagreements of different severity and corrects for chance agreement: https://doi.org/10.1037/h0026256
- Baccianella, Esuli, and Sebastiani discuss MAE and macro-averaged ordinal error measures: https://iris.cnr.it/retrieve/5bcf86c7-cd68-4884-93b5-ff86095082ec/prod_91979-doc_199135.pdf
- Galdran recommends Ranked Probability Score for probabilistic ordinal predictions: https://arxiv.org/abs/2309.08701

## Files

- `bias5_ordinal_metrics.csv`: all 90 evaluated prediction columns.
- `bias5_ordinal_selected_metrics.csv`: selected top methods and baselines.
- `bias5_ordinal_per_class_metrics.csv`: per-class distance errors.