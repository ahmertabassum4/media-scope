# Bias/Factuality Method Summary

This document summarizes the key full-dataset experiments for 5-label bias and 4-label factuality classification from landing-page screenshots.

Main table rules:

- Bias is the balanced 250-site, 5-label dataset.
- Factuality is the 200-site, 4-label dataset (`very low`, `low`, `high`, `very high`).
- Values are `accuracy / macro-F1`; bold marks the best value for that metric within the task column, using exact saved metrics before rounding.
- Learned methods use stratified 5-fold CV unless the method is a direct LLM baseline.
- `-` means there is no comparable full-dataset run for that task in the saved experiment artifacts.
- Rows are ranked approximately from strongest to weakest, primarily by combined macro-F1 across the two task columns. Single-task rows are placed in the closest comparable performance band.


## Results

| Rank | Method | Bias 5-label | Factuality 4-label |
|---:|---|---:|---:|
| 1 | `nested_core_gpt55_ocr_text_siglip` | **0.772** / **0.776** | 0.765 / 0.702 |
| 2 | `pca_fusion::compact::pruned_components::linear_svc_c0.2` | 0.748 / 0.749 | 0.795 / 0.719 |
| 3 | `baseline_plus_semantic_visual_extra_trees_cv` | 0.740 / 0.746 | 0.790 / **0.719** |
| 4 | `pca_fusion::medium::core_without_clip::ridge_a3` | **0.772** / 0.772 | 0.780 / 0.671 |
| 5 | `pca_fusion::large::core_without_clip::linear_svc_c0.2` | 0.728 / 0.728 | **0.820** / 0.713 |
| 6 | `nested_pruned_gpt55_ocr_text_clip_siglip` | 0.768 / 0.774 | 0.735 / 0.656 |
| 7 | `baseline_plus_semantic_random_forest_cv` | 0.716 / 0.720 | 0.795 / 0.705 |
| 8 | `pca_fusion::compact::core_without_clip::mlp` | 0.744 / 0.743 | 0.755 / 0.662 |
| 9 | `baseline_plus_semantic_mlp_cv` | 0.720 / 0.723 | 0.725 / 0.674 |
| 10 | `semantic_only_random_forest_cv` | 0.716 / 0.718 | 0.765 / 0.676 |
| 11 | `nested_gpt55_plus_local_prediction_stack_logreg_cv` | 0.740 / 0.746 | 0.735 / 0.644 |
| 12 | `semantic_visual_extra_trees_without_topics_boolean_and_baseline` | 0.752 / 0.754 | - |
| 13 | `all_model_stack_plus_visual_extra_trees_cv` | 0.720 / 0.717 | - |
| 14 | `gpt55_label_plus_visual_extra_trees_cv` | 0.716 / 0.716 | 0.750 / 0.594 |
| 15 | `gpt55_plus_strong_local_majority_vote` | 0.684 / 0.683 | 0.695 / 0.555 |
| 16 | `nested_local_prediction_stack_logreg_cv` | 0.628 / 0.631 | 0.650 / 0.549 |
| 17 | `full_local_stack_pca_ridge_cv` | 0.592 / 0.588 | 0.670 / 0.568 |
| 18 | `ocr_text_tfidf_svc_cv` | 0.624 / 0.624 | 0.700 / 0.523 |
| 19 | `siglip_b16_pca_ridge_cv` | 0.600 / 0.598 | 0.655 / 0.534 |
| 20 | `clip_vit_b32_centroid_cv` | 0.584 / 0.586 | 0.655 / 0.537 |
| 21 | `baseline_gpt5.5` | 0.532 / 0.504 | 0.725 / 0.619 |
| 22 | `bias_5:anthropic/claude-sonnet-4.6` | 0.552 / 0.529 | - |
| 23 | `bias_5:moonshotai/kimi-k2.6` | 0.544 / 0.508 | - |
| 24 | `dinov2_small_pca_ridge_cv` | 0.508 / 0.499 | 0.565 / 0.496 |
| 25 | `bias_5:qwen/qwen3.7-plus` | 0.512 / 0.488 | - |
| 26 | `visual_extra_trees_cv` | 0.484 / 0.471 | 0.580 / 0.465 |
| 27 | `scratch_small_cnn_cv` | 0.460 / 0.437 | 0.520 / 0.470 |

## Method Descriptions

`baseline_gpt5.5` is the direct GPT-5.5 screenshot classifier. The model receives landing-page screenshot crops and must output one label without web search or outside outlet knowledge. It is the main proprietary-model baseline: weak for 5-label bias because it overuses the center class, but already strong for 4-label factuality.

`bias_5:anthropic/claude-sonnet-4.6` is the direct Claude Sonnet 4.6 5-label bias run with the same screenshot-only constraint. It was evaluated on the full 250-site bias dataset. It slightly beat direct GPT-5.5 on bias, but we did not run a comparable full 4-label factuality experiment for this direct model.

`bias_5:moonshotai/kimi-k2.6` is the direct Kimi K2.6 5-label bias run. Like the other direct LLM baselines, it uses only screenshot evidence and does not include local ML calibration. It was close to Claude and GPT-5.5, but below the hybrid methods.

`bias_5:qwen/qwen3.7-plus` is the direct Qwen 5-label bias run. It was part of the OpenRouter model comparison for the bias dataset. It did not outperform the other direct LLM baselines and was not carried forward as a main hybrid signal.

`visual_extra_trees_cv` uses only handcrafted visual features from screenshots and an ExtraTrees classifier. The features include image dimensions, brightness, contrast, saturation, edge density, dark/light fractions, color statistics, and top-band layout/color cues. This tests how far pure visible design signals go without OCR, LLMs, or pretrained vision embeddings.

`ocr_text_tfidf_svc_cv` extracts visible text with EasyOCR, converts the OCR text to TF-IDF features, and trains a linear SVM. It is a local text-only baseline: it ignores pixel layout except for what OCR can read. It is notably useful for bias because outlet landing pages often expose ideological words, sections, and topic framing.

`clip_vit_b32_centroid_cv` embeds viewport crops with CLIP ViT-B/32 and classifies by cosine centroids. On factuality the comparable saved row is `clip_vit_b32_cosine_centroid_cv`. This method uses generic image-text pretraining signals without OCR text features or LLM labels.

`siglip_b16_pca_ridge_cv` embeds the first viewport crops with SigLIP B/16, compresses embeddings with PCA inside each fold, and classifies with ridge/logistic-style linear modeling. SigLIP was one of the stronger local visual embedding sources for bias, but by itself it did not match hybrid methods.

`dinov2_small_pca_ridge_cv` uses DINOv2-small visual embeddings, fold-local PCA, and a ridge classifier. DINOv2 is a self-supervised visual representation rather than a text-aligned model. It was weaker than CLIP/SigLIP for these label spaces, which suggests screenshot text/semantics matter more than pure visual structure.

`full_local_stack_pca_ridge_cv` concatenates local non-LLM signals into a single representation: OCR text components, OCR numeric/lexicon features, handcrafted visual features, and pretrained visual embeddings. PCA/SVD reductions are fitted inside each training fold, then a ridge classifier predicts the final label. This is the best "no new proprietary LLM label" family, but still below GPT-hybrid stacks.

`gpt55_label_plus_visual_extra_trees_cv` uses the cached GPT-5.5 label as a categorical feature and adds handcrafted visual screenshot features. ExtraTrees then learns corrections to the GPT baseline. This is a cost-efficient hybrid: one LLM classification call plus cheap local features.

`all_model_stack_plus_visual_extra_trees_cv` stacks cached direct predictions from several LLMs together with handcrafted visual features, then trains ExtraTrees. It was an upper-bound style bias experiment because it relies on multiple proprietary model labels. It improved over single GPT-5.5 calibration but is not the production choice because it is more expensive and was not reproduced as a full factuality method.

`gpt55_plus_strong_local_majority_vote` combines the GPT-5.5 label with strong local model predictions through a simple voting rule. The local voters are OCR/embedding/visual models selected from the stronger full-dataset local runs. It is simpler than learned stacking, but it loses information because all model outputs are treated as coarse votes.

`nested_local_prediction_stack_logreg_cv` trains several local base predictors and feeds their out-of-fold predictions into a logistic-regression meta-classifier. It does not include the GPT-5.5 label. The nested evaluation prevents training-fold leakage: base model predictions for the meta-model are generated inside inner CV splits.

`nested_gpt55_plus_local_prediction_stack_logreg_cv` is the earlier full nested hybrid stack. It uses cached GPT-5.5 labels plus local base-predictor outputs from OCR text, OCR numeric features, CLIP, SigLIP, DINOv2, prompt-score features, and handcrafted visuals. A logistic-regression meta-classifier learns how to combine one-hot base predictions.

`nested_core_gpt55_ocr_text_siglip` is the best bias method after ablation. It keeps only the strongest core signals: the GPT-5.5 label, OCR text TF-IDF/LinearSVC prediction, and SigLIP embedding/PCA/Ridge prediction. For factuality, the comparable implementation is `nested_core_gpt55_ocr_text_siglip_cv`. The result shows that pruning noisy base predictors can beat the larger stack.

`nested_pruned_gpt55_ocr_text_clip_siglip` is a pruned nested stack variant that keeps GPT-5.5, OCR text, SigLIP, and CLIP-style local predictors while removing weaker/noisier feature groups such as handcrafted visual stack pieces, prompt-score classifiers, DINO, or OCR numeric variants depending on the task-specific implementation. It is close to the core nested stack on bias but weaker on factuality.

`semantic_only_random_forest_cv` uses GPT-5.5 as a structured feature extractor instead of a final classifier. The LLM returns visible-page fields such as source type, page scope, ideological hints, headline framing, tone, topics, advocacy level, sensationalism, and boolean cues. A RandomForest classifier predicts labels using only those extracted semantic features.

`baseline_plus_semantic_random_forest_cv` adds the direct GPT-5.5 baseline label to the structured semantic feature table and trains RandomForest. This tests whether the extracted feature schema can calibrate the raw LLM decision. It performs well on factuality and moderately well on bias.

`baseline_plus_semantic_mlp_cv` uses the same GPT-5.5 baseline label plus structured semantic features, but the final classifier is an MLP. It was competitive for bias but less stable than tree ensembles for factuality, likely because the dataset is small and has a rare `very high` class.

`baseline_plus_semantic_visual_extra_trees_cv` is the best overall semantic-feature method. GPT-5.5 extracts structured visible-page features from the first four 16:9 crops; those features are combined with the cached GPT-5.5 baseline label and handcrafted visual features; ExtraTrees performs the final classification. It is the strongest factuality method by macro-F1 and effectively tied with top bias methods before later pruning/PCA experiments.

`semantic_visual_extra_trees_without_topics_boolean_and_baseline` is a bias ablation of the semantic+visual ExtraTrees method. It removes topic/boolean feature groups and the baseline label, leaving a more focused semantic/visual feature set. On bias this unexpectedly improves over the full semantic+visual version, suggesting some topic and baseline-label features add noise for the 5-label bias boundary.

`pca_fusion::medium::core_without_clip::ridge_a3` directly feeds continuous reduced features into one final classifier. The pipeline creates fold-local PCA/SVD components from OCR text, SigLIP, DINOv2, visual features, prompt-score features, and one-hot GPT labels, but excludes CLIP in the `core_without_clip` variant. Ridge with alpha 3 is then trained on all components. This is tied for best bias accuracy and is one of the cleanest alternatives to prediction-level stacking.

`pca_fusion::compact::pruned_components::linear_svc_c0.2` is a compact PCA/SVD fusion method with weaker component groups pruned and a linear SVM final classifier. It is nearly tied for best factuality macro-F1 and has strong ordinal behavior on factuality. On bias it is good but below the top core nested/PCA ridge methods.

`pca_fusion::large::core_without_clip::linear_svc_c0.2` uses a larger component budget in the PCA/SVD fusion pipeline, excludes CLIP, and trains a linear SVM. It has the best factuality accuracy in the saved experiments, but its macro-F1 is lower than the semantic+visual ExtraTrees method because the rare `very high` class is harder.

`pca_fusion::compact::core_without_clip::mlp` is a compact PCA/SVD fusion variant with an MLP final classifier and no CLIP components. For bias it had the best ordinal MSE/QWK among PCA-fusion rows, even though exact accuracy/macro-F1 were not the best. This makes it useful when near-miss ordinal errors matter more than exact labels.

`scratch_small_cnn_cv` trains a small convolutional neural network from scratch on a 2x2 grid of the first four viewport crops. It is the pure image-learning baseline with no pretrained embeddings and no LLM/OCR features. On these small datasets it underperforms, which is expected because 200-250 screenshots are too few to learn robust visual semantics from scratch.

