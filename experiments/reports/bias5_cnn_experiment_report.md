# Bias 5 CNN Experiment Report

Device: `cuda`.

Dataset: full 250-site bias dataset, evaluated with 5-fold stratified CV.

## Methods

- `resnet18_mean_cosine_centroid_cv`: first four viewport crops are passed through frozen ImageNet ResNet18; the mean embedding is classified by nearest cosine centroid.
- `resnet18_meanstd_cosine_centroid_cv`: same frozen ResNet18 embeddings, with mean and standard deviation pooled across the first four viewport crops.
- `resnet18_meanstd_cosine_knn*_cv`: cosine kNN on the same mean/std ResNet18 embedding representation.
- `scratch_small_cnn_cv`: a small 4-layer CNN trained from scratch on a 2x2 grid of the first four viewport crops.

## Best Method

`baseline_gpt5.5`: accuracy `0.532000`, macro-F1 `0.503595`.

## Metrics

```csv
method,sample_size,accuracy,macro_f1,weighted_f1
baseline_gpt5.5,250,0.532000,0.503595,0.503595
resnet18_mean_cosine_centroid_cv,250,0.464000,0.466774,0.466774
resnet18_meanstd_cosine_centroid_cv,250,0.448000,0.449825,0.449825
resnet18_meanstd_cosine_knn5_cv,250,0.484000,0.467249,0.467249
resnet18_meanstd_cosine_knn9_cv,250,0.476000,0.455978,0.455978
scratch_small_cnn_cv,250,0.460000,0.437373,0.437373
```

## Scratch CNN Fold Metrics

```csv
fold,accuracy,macro_f1
1,0.500000,0.476358
2,0.500000,0.467233
3,0.400000,0.380521
4,0.340000,0.305656
5,0.560000,0.542688
```
