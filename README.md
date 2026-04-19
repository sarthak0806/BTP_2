# Attribute Prediction for E-commerce Products using CLIP
https://docs.google.com/document/d/17qbhYI4-6hopTvNnghWJZjipSynI8vk8um5b439OEAU/edit?usp=sharing
https://docs.google.com/document/d/1gCcZDRGSRO2P9javYQTkJtPbUZ5hQ_XKlYW4tksPI4g/edit?usp=sharing

This project focuses on predicting structured product attributes (such as color, pattern, sleeve length, etc.) from product images using deep learning.

## Overview

Modern e-commerce platforms rely heavily on structured product attributes for search, filtering, recommendations, and catalog organization. Manual annotation of these attributes is slow and error-prone. This project builds an automated system to predict product attributes directly from images.

## Problem

Given a product image and its category, the goal is to predict multiple category-specific attributes. This is a multi-attribute classification problem where each category has its own set of attributes.

## Approach

The pipeline consists of:

1. Image preprocessing:
   - Resize images (224×224 / 256×256)
   - Normalize using pretrained model statistics

2. Feature extraction:
   - CLIP Vision Transformer (ViT-B/32)
   - ConvNext (for advanced experiments)

3. Attribute prediction:
   - Category-aware multi-head MLP classifiers
   - One head per attribute

## Models Explored

1. Similarity-based approach:
   - Uses frozen CLIP embeddings
   - Retrieves similar images using cosine similarity
   - Predicts attributes using majority voting

2. Classification-based approach:
   - Uses pretrained visual backbone
   - Learns direct mapping from features to attributes

## Final Model

- Backbone: CLIP ViT-B/32 (frozen)
- Classifier: Multi-head MLP
- Loss: Cross-entropy with masking for missing labels

This setup balances performance and efficiency.

## Results

| Model | Backbone | Score |
|------|--------|------|
| Similarity (CLIP) | ViT-B/32 | 0.723 |
| CLIP Classification | ViT-B/32 | 0.765 |
| CLIP Classification | ViT-L/14 | 0.770 |
| ConvNext Classification | XXLarge | 0.786 |
| Final Model | CLIP + MLP | 0.801 |

## Dataset Structure

dataset/
  train.csv
  test.csv
  category_attributes.parquet
  train_images/
  test_images/

- ~70K training images
- Multiple fashion categories
- Category-specific attributes

## Installation

pip install open_clip_torch pandas scikit-learn tqdm pillow pyarrow

## Usage

Smoke test:
python train.py --mode smoke

Full training:
python train.py --mode full

## ConvNext Training

python train_convnext.py --mode smoke
python train_convnext.py --mode full

## Features

- Category-aware multi-head classification
- Handles missing labels via masking
- Uses pretrained visual representations
- Scalable across categories
- Smoke mode for quick debugging

## Challenges

- Category-specific attribute spaces
- Class imbalance
- Missing labels
- Fine-grained visual differences

## Future Improvements

- Fine-tuning pretrained encoders
- Multimodal learning (image + text)
- Better handling of class imbalance
- Attention-based attribute modeling
