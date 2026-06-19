# Bias 5 Image Feature Full Dataset Experiment Report

Device: `cuda`.

Dataset: full balanced 250-site bias dataset.

This run avoids new paid LLM calls. It tests OCR, local vision embeddings, handcrafted visual features, and local ensembles on all images.

## Best Method

`gpt55_plus_strong_local_majority_vote`: accuracy `0.684000`, macro-F1 `0.683462`.

## Metrics

```csv
method,sample_size,accuracy,macro_f1,weighted_f1
baseline_gpt5.5_cached,250,0.532000,0.503595,0.503595
handcrafted_visual_centroid_cv,250,0.304000,0.247634,0.247634
handcrafted_visual_knn5_cv,250,0.332000,0.329465,0.329465
handcrafted_visual_pca_ridge_cv,250,0.440000,0.414756,0.414756
handcrafted_visual_extra_trees_cv,250,0.456000,0.439230,0.439230
ocr_text_tfidf_svc_cv,250,0.624000,0.624144,0.624144
ocr_numeric_centroid_cv,250,0.364000,0.330066,0.330066
ocr_numeric_knn5_cv,250,0.412000,0.414822,0.414822
ocr_numeric_pca_ridge_cv,250,0.528000,0.518647,0.518647
ocr_numeric_extra_trees_cv,250,0.508000,0.490206,0.490206
ocr_numeric_plus_visual_centroid_cv,250,0.328000,0.273902,0.273902
ocr_numeric_plus_visual_knn5_cv,250,0.412000,0.400996,0.400996
ocr_numeric_plus_visual_pca_ridge_cv,250,0.500000,0.469582,0.469582
ocr_numeric_plus_visual_extra_trees_cv,250,0.516000,0.503734,0.503734
clip_vit_b32_centroid_cv,250,0.584000,0.585675,0.585675
clip_vit_b32_knn5_cv,250,0.556000,0.552395,0.552395
clip_vit_b32_pca_ridge_cv,250,0.576000,0.576004,0.576004
siglip_b16_centroid_cv,250,0.580000,0.583032,0.583032
siglip_b16_knn5_cv,250,0.524000,0.522674,0.522674
siglip_b16_pca_ridge_cv,250,0.600000,0.597895,0.597895
dinov2_small_centroid_cv,250,0.476000,0.476856,0.476856
dinov2_small_knn5_cv,250,0.408000,0.382971,0.382971
dinov2_small_pca_ridge_cv,250,0.508000,0.498770,0.498770
local_embedding_visual_stack_pca_ridge_cv,250,0.588000,0.587030,0.587030
local_embedding_visual_stack_extra_trees_cv,250,0.568000,0.571193,0.571193
local_pca_ridge_majority_vote,250,0.604000,0.601000,0.601000
strong_local_majority_vote,250,0.612000,0.611378,0.611378
gpt55_plus_strong_local_majority_vote,250,0.684000,0.683462,0.683462
strong_local_prediction_stack_logreg_cv,250,0.592000,0.593086,0.593086
```
