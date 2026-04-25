# Project Goals

- Date: `2026-04-24`
- Version: `0.1.0`
- Status: `active`
- Purpose: working context doc for the project motivation, main research questions, and current claim.

## Project Framing

A segmentation model may look strong on average, but average performance can hide an important question: which image regions are consistently hard, and can that difficulty be anticipated in advance?

This project studies whether patch-level segmentation difficulty in H&E pathology images can be explained and predicted from pretrained pathology representations. The workflow has three connected parts:

1. Benchmark recent segmentation models on H&E patches to measure patch-level variability in performance.
2. Test whether GigaPath tile embeddings encode biologically and morphologically meaningful patch structure.
3. Evaluate whether those embeddings can predict segmentation difficulty, so that foundation-model features may be useful not only for visualization but also for reliability estimation and failure-aware pathology workflows.

## Project Summary

This project evaluates whether foundation-model pathology embeddings can help explain and predict segmentation difficulty in H&E image patches.

First, four strong segmentation models are benchmarked at the patch level:

- CellSAM
- Cellpose-SAM
- CellViT-SAMH
- StarDist

The main performance view is patch-level behavior rather than only aggregate averages, with metrics such as panoptic quality (`pq`) used to define comparative performance and patch difficulty.

Next, `1536`-dimensional GigaPath tile embeddings are extracted for each patch and explored with PCA and UMAP. These visualizations are colored by interpretable patch-level descriptors such as:

- foreground fraction
- mean nuclei area
- circularity
- eccentricity
- cell type composition

Finally, simple classifiers are trained on embedding features to predict patch difficulty categories derived from segmentation performance. Taken together, the project asks whether pretrained pathology representations capture the visual properties that govern segmentation success and whether they can support failure-aware segmentation workflows.

## Main Research Questions

### Question 1

How do recent strong segmentation models compare on H&E image patches?

### Question 2

Do GigaPath tile embeddings capture patch properties that matter for segmentation?

### Question 3

Can embedding-based models predict patch difficulty categories derived from segmentation performance?

### Question 4

Which patch properties seem most associated with segmentation success or failure?

## Expanded Question Set

- How do recent segmentation models compare on H&E image patches?
- Do GigaPath tile embeddings organize patches by meaningful morphological or tissue-related properties?
- Can embedding-based models predict patch difficulty categories derived from segmentation performance?
- Which patch properties seem most associated with segmentation success or failure?
- Without seeing the ground truth at inference time, can a patch embedding indicate whether a patch is likely to be easy, medium, or hard for segmentation?

## Working Claim

GigaPath tile embeddings appear to capture patch-level structure that is relevant to morphology and segmentation difficulty, and these embeddings provide a useful signal for predicting which H&E patches are likely to be easy or hard for current segmentation models.

## Interpretation Notes

- The project does not assume that dimensionality reduction preserves all original embedding geometry.
- PCA and UMAP are used as exploratory tools to test whether embedding space reflects patch-level biological or morphological organization.
- The intent is not only to visualize embeddings, but to assess whether those embeddings contain information that is operationally useful for segmentation reliability estimation.
- A central premise is that segmentation failures are not fully random and may relate to measurable patch properties reflected in embedding space.
