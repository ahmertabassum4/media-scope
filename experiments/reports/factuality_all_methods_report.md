# Factuality Classification: All Methods

Dataset: 200 landing-page screenshots.

- Binary task: 100 low (`VERY LOW` + `LOW`) and 100 high (`HIGH` + `VERY HIGH`).
- Multiclass task: 50 very low, 50 low, 89 high, and 11 very high.
- All learned methods use stratified 5-fold CV.
- PCA, SVD, base-model stacking, and final classifiers are fit inside their training folds.

## Binary Results

| Rank | Method | Family | Accuracy | Balanced accuracy | Macro-F1 | MSE | QWK |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `baseline_plus_semantic_visual_extra_trees_cv` | semantic_features | 0.965000 | 0.965000 | 0.964999 | - | - |
| 2 | `baseline_gpt5.5` | gpt_baseline_or_calibration | 0.965000 | 0.965000 | 0.964992 | - | - |
| 3 | `gpt55_label_logreg_cv` | gpt_baseline_or_calibration | 0.965000 | 0.965000 | 0.964992 | - | - |
| 4 | `gpt55_label_plus_visual_extra_trees_cv` | gpt_baseline_or_calibration | 0.965000 | 0.965000 | 0.964992 | - | - |
| 5 | `nested_core_gpt55_ocr_text_siglip_cv` | nested_stacking | 0.965000 | 0.965000 | 0.964992 | - | - |
| 6 | `nested_pruned_gpt55_ocr_text_clip_siglip_cv` | nested_stacking | 0.965000 | 0.965000 | 0.964992 | - | - |
| 7 | `pca_fusion::compact::all_components::ridge_a3` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 8 | `pca_fusion::compact::all_components::extra_trees` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 9 | `pca_fusion::compact::pruned_components::logreg_c0.2` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 10 | `pca_fusion::compact::pruned_components::logreg_c1` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 11 | `pca_fusion::compact::pruned_components::ridge_a3` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 12 | `pca_fusion::compact::pruned_components::extra_trees` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 13 | `pca_fusion::compact::pruned_components::hist_gradient_boosting` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 14 | `pca_fusion::compact::pruned_components::mlp` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 15 | `pca_fusion::compact::core_without_clip::logreg_c0.2` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 16 | `pca_fusion::compact::core_without_clip::logreg_c1` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 17 | `pca_fusion::compact::core_without_clip::ridge_a3` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 18 | `pca_fusion::compact::core_without_clip::linear_svc_c0.2` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 19 | `pca_fusion::compact::core_without_clip::extra_trees` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 20 | `pca_fusion::compact::core_without_clip::mlp` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 21 | `pca_fusion::medium::all_components::ridge_a3` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 22 | `pca_fusion::medium::all_components::extra_trees` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 23 | `pca_fusion::medium::all_components::hist_gradient_boosting` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 24 | `pca_fusion::medium::pruned_components::logreg_c0.2` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |
| 25 | `pca_fusion::medium::pruned_components::ridge_a3` | pca_component_fusion | 0.965000 | 0.965000 | 0.964992 | - | - |

### Best By Family

- `cnn_from_scratch`: `scratch_small_cnn_cv`, accuracy `0.740000`, Macro-F1 `0.735530`.
- `control_or_ensemble`: `majority_class`, accuracy `0.500000`, Macro-F1 `0.333333`.
- `gpt_baseline_or_calibration`: `baseline_gpt5.5`, accuracy `0.965000`, Macro-F1 `0.964992`.
- `handcrafted_visual`: `visual_extra_trees_cv`, accuracy `0.690000`, Macro-F1 `0.689503`.
- `local_feature_fusion`: `full_local_stack_pca_ridge_cv`, accuracy `0.845000`, Macro-F1 `0.844996`.
- `nested_stacking`: `nested_core_gpt55_ocr_text_siglip_cv`, accuracy `0.965000`, Macro-F1 `0.964992`.
- `ocr`: `ocr_text_tfidf_svc_cv`, accuracy `0.810000`, Macro-F1 `0.809829`.
- `pca_component_fusion`: `pca_fusion::compact::all_components::ridge_a3`, accuracy `0.965000`, Macro-F1 `0.964992`.
- `semantic_features`: `baseline_plus_semantic_visual_extra_trees_cv`, accuracy `0.965000`, Macro-F1 `0.964999`.
- `vision_embeddings`: `clip_vit_b32_cosine_centroid_cv`, accuracy `0.835000`, Macro-F1 `0.834996`.

## Multiclass Results

| Rank | Method | Family | Accuracy | Balanced accuracy | Macro-F1 | MSE | QWK |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `baseline_plus_semantic_visual_extra_trees_cv` | semantic_features | 0.790000 | 0.705546 | 0.719293 | 0.240000 | 0.846542 |
| 2 | `pca_fusion::compact::pruned_components::linear_svc_c0.2` | pca_component_fusion | 0.795000 | 0.706164 | 0.718846 | 0.220000 | 0.856327 |
| 3 | `pca_fusion::compact::pruned_components::extra_trees` | pca_component_fusion | 0.795000 | 0.706164 | 0.718621 | 0.220000 | 0.853240 |
| 4 | `pca_fusion::large::core_without_clip::linear_svc_c0.2` | pca_component_fusion | 0.820000 | 0.686946 | 0.712980 | 0.250000 | 0.831723 |
| 5 | `pca_fusion::medium::core_without_clip::extra_trees` | pca_component_fusion | 0.810000 | 0.676946 | 0.709629 | 0.205000 | 0.857183 |
| 6 | `pca_fusion::medium::core_without_clip::logreg_c1` | pca_component_fusion | 0.805000 | 0.671946 | 0.705053 | 0.225000 | 0.847726 |
| 7 | `baseline_plus_semantic_random_forest_cv` | semantic_features | 0.795000 | 0.690628 | 0.704864 | 0.265000 | 0.832723 |
| 8 | `pca_fusion::medium::pruned_components::extra_trees` | pca_component_fusion | 0.805000 | 0.671946 | 0.704339 | 0.210000 | 0.852874 |
| 9 | `nested_core_gpt55_ocr_text_siglip_cv` | nested_stacking | 0.765000 | 0.758029 | 0.701689 | 0.365000 | 0.789819 |
| 10 | `pca_fusion::large::core_without_clip::ridge_a3` | pca_component_fusion | 0.805000 | 0.674137 | 0.701111 | 0.240000 | 0.836004 |
| 11 | `pca_fusion::large::core_without_clip::logreg_c0.2` | pca_component_fusion | 0.800000 | 0.669137 | 0.700026 | 0.230000 | 0.842654 |
| 12 | `baseline_plus_semantic_logreg_cv` | semantic_features | 0.740000 | 0.713110 | 0.699798 | 0.335000 | 0.800832 |
| 13 | `pca_fusion::compact::pruned_components::mlp` | pca_component_fusion | 0.755000 | 0.668355 | 0.696577 | 0.290000 | 0.810946 |
| 14 | `semantic_only_mlp_cv` | semantic_features | 0.745000 | 0.684847 | 0.695921 | 0.360000 | 0.776404 |
| 15 | `pca_fusion::large::core_without_clip::logreg_c1` | pca_component_fusion | 0.795000 | 0.664137 | 0.694917 | 0.235000 | 0.838344 |
| 16 | `pca_fusion::compact::core_without_clip::linear_svc_c0.2` | pca_component_fusion | 0.785000 | 0.678437 | 0.692763 | 0.230000 | 0.849412 |
| 17 | `baseline_plus_semantic_extra_trees_cv` | semantic_features | 0.770000 | 0.696502 | 0.691143 | 0.275000 | 0.832521 |
| 18 | `pca_fusion::compact::all_components::ridge_a3` | pca_component_fusion | 0.740000 | 0.684229 | 0.691098 | 0.320000 | 0.797692 |
| 19 | `pca_fusion::compact::core_without_clip::extra_trees` | pca_component_fusion | 0.780000 | 0.697737 | 0.690797 | 0.280000 | 0.821298 |
| 20 | `pca_fusion::medium::pruned_components::logreg_c0.2` | pca_component_fusion | 0.785000 | 0.656328 | 0.688073 | 0.260000 | 0.823340 |
| 21 | `pca_fusion::medium::pruned_components::logreg_c1` | pca_component_fusion | 0.785000 | 0.656328 | 0.688073 | 0.260000 | 0.823340 |
| 22 | `pca_fusion::medium::pruned_components::linear_svc_c0.2` | pca_component_fusion | 0.785000 | 0.656328 | 0.687598 | 0.275000 | 0.814164 |
| 23 | `pca_fusion::medium::core_without_clip::logreg_c0.2` | pca_component_fusion | 0.785000 | 0.656328 | 0.687598 | 0.245000 | 0.834437 |
| 24 | `pca_fusion::large::pruned_components::ridge_a3` | pca_component_fusion | 0.790000 | 0.663519 | 0.686872 | 0.285000 | 0.810190 |
| 25 | `pca_fusion::large::pruned_components::logreg_c1` | pca_component_fusion | 0.785000 | 0.654137 | 0.685495 | 0.275000 | 0.811334 |

### Best By Family

- `cnn_from_scratch`: `scratch_small_cnn_cv`, accuracy `0.520000`, Macro-F1 `0.469987`.
- `control_or_ensemble`: `majority_class`, accuracy `0.445000`, Macro-F1 `0.153979`.
- `gpt_baseline_or_calibration`: `baseline_gpt5.5`, accuracy `0.725000`, Macro-F1 `0.619330`.
- `handcrafted_visual`: `visual_extra_trees_cv`, accuracy `0.580000`, Macro-F1 `0.464699`.
- `local_feature_fusion`: `full_local_stack_pca_ridge_cv`, accuracy `0.670000`, Macro-F1 `0.568327`.
- `nested_stacking`: `nested_core_gpt55_ocr_text_siglip_cv`, accuracy `0.765000`, Macro-F1 `0.701689`.
- `ocr`: `ocr_text_tfidf_svc_cv`, accuracy `0.700000`, Macro-F1 `0.523452`.
- `pca_component_fusion`: `pca_fusion::compact::pruned_components::linear_svc_c0.2`, accuracy `0.795000`, Macro-F1 `0.718846`.
- `semantic_features`: `baseline_plus_semantic_visual_extra_trees_cv`, accuracy `0.790000`, Macro-F1 `0.719293`.
- `vision_embeddings`: `siglip_b16_cosine_centroid_cv`, accuracy `0.695000`, Macro-F1 `0.571402`.
