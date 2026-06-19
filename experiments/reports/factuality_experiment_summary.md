# Factuality Experiment Summary

## Dataset

- 200 landing-page screenshots matched exactly to `snapshot_index.csv`.
- Binary target: 100 low (`VERY LOW` + `LOW`) and 100 high (`HIGH` + `VERY HIGH`).
- Four-class target: 50 very low, 50 low, 89 high, and 11 very high.
- The requested multi-label setup is implemented as four-class single-label classification, because each dataset row has exactly one factuality label.

## Evaluation

- 147 binary and 149 four-class methods/configurations were evaluated.
- Learned methods use stratified 5-fold CV.
- PCA, OCR TF-IDF/SVD, nested base models, and final classifiers are fitted inside training folds.
- Primary metric: Macro-F1. Accuracy, balanced accuracy, per-class metrics, MSE, and quadratic weighted kappa are also reported.

## Binary Result

- GPT-5.5 baseline: accuracy `0.965`, Macro-F1 `0.965`.
- Best semantic hybrid: accuracy `0.965`, Macro-F1 `0.965`.
- The semantic hybrid changes two predictions, correcting one and breaking one. It does not improve binary accuracy.
- Best fully local method: `full_local_stack_pca_ridge_cv`, accuracy `0.845`, Macro-F1 `0.845`.

## Four-Class Result

- Baseline GPT-5.5: accuracy `0.725`, Macro-F1 `0.619`.
- Best Macro-F1: `baseline_plus_semantic_visual_extra_trees_cv`, accuracy `0.790`, Macro-F1 `0.719`, MSE `0.240`, QWK `0.847`.
- Nearly tied direct fusion: `pca_fusion::compact::pruned_components::linear_svc_c0.2`, accuracy `0.795`, Macro-F1 `0.719`, MSE `0.220`, QWK `0.856`.
- Best accuracy: `pca_fusion::large::core_without_clip::linear_svc_c0.2`, accuracy `0.820`, Macro-F1 `0.713`.
- Best ordinal errors: `pca_fusion::medium::core_without_clip::extra_trees`, MSE `0.205`, QWK `0.857`.
- Best balanced accuracy and very-high recall: `nested_core_gpt55_ocr_text_siglip_cv`, balanced accuracy `0.758`, very-high recall `0.818`.

## Interpretation

- The four-class gain comes from calibrating the GPT label with structured visible-page features and local OCR/SigLIP signals.
- The best semantic hybrid raises Macro-F1 by about 0.100 and accuracy by 0.065 over the GPT baseline.
- Maximum accuracy is not the best selection criterion: the 0.820-accuracy model recalls only 0.273 of the rare `very high` class.
- The nested core stack recalls 0.818 of `very high`, but over-predicts that class and loses overall accuracy.
- With only 11 `very high` examples, differences involving that class have high variance and should be confirmed on a larger external test set.

## Main Files

- `factuality_dataset.csv`
- `factuality_all_methods_metrics.csv`
- `factuality_selected_per_class_metrics.csv`
- `factuality_binary_all_methods_predictions.csv`
- `factuality_multiclass_all_methods_predictions.csv`
- `factuality_all_methods_confusions.csv`
- `factuality_all_methods_report.md`

## OpenRouter Usage

- `multiclass_baseline`: prompt `265847`, completion `29166`, total `295013` tokens.
- `semantic_features`: prompt `1113312`, completion `182912`, total `1296224` tokens.
- Total measured for the new 400 requests: `1591237` tokens.
- The earlier cached binary GPT-5.5 run is not included because its usage was not recorded.
