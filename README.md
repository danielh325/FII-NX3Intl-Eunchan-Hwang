# Domain Generalization for TB Detection on Chest X-rays
DenseNet-121 + **MixStyle** + multi-level augmentation for robust, cross-site tuberculosis (TB) screening.

> This repository implements the methods and experiments from the accompanying research paper, including training, external validation (Shenzhen, Pakistan), and single-image inference.

## Why this matters
- **TB burden:** ~10.8M infections and 1.25M deaths in 2023; many cases go undetected in low-resource settings.
- **Gap:** CAD models often degrade across hospitals/devices (domain shift).
- **Goal:** Improve **cross-site generalization** without extra labels/data by mixing feature “styles” and strong image-space augmentations.

## Method 
- **Backbone:** DenseNet-121 (ImageNet pretrained).
- **Head:** GAP + GMP → concat → BN → Dropout → Dense(256) → BN → Dropout → Dense(1, sigmoid).
- **Domain Generalization:**
  - **MixStyle (p=0.5, α=0.3)** injected after backbone features during training to mix channel-wise mean/variance across samples → style-invariant features.
  - **Multi-level augmentations:** horizontal flip, ±3° rotation, ±5–10% zoom/shift, ±10–15% brightness/contrast jitter (training only).
- **Training schedule (20 epochs):**
  - Phase 1: **freeze** backbone, train head.
  - Phase 2: **unfreeze** backbone, lower LR.
- **Imbalance handling:** **class-weighted BCE** (higher FN penalty).
- **Optimizer/Callbacks:** Adam, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint.

## Datasets
- **Train/val/test:** Aggregated TB CXR dataset (≈3500 normal / 700 TB).
- **External tests:** **Shenzhen** (234 normal / 41 TB) and **Pakistan** (2494 TB / 514 normal).
- **Preprocessing:** resize 224×224; ImageNet normalization; grayscale replicated to 3-channel.

## Key results (external, unseen sites)
- **Shenzhen:** Accuracy **0.66 → 0.90**, TB precision **0.31 → 0.60**, TB recall **1.00 → 0.95** (far fewer false positives while keeping very high sensitivity).
- **Pakistan:** Accuracy **0.72 → 0.78**, TB recall **0.71 → 0.79** (many more true positives with modest FP increase).

> Takeaway: The same recipe improves specificity in a low-prevalence set (Shenzhen) and **sensitivity** in a high-prevalence set (Pakistan), which is what screening programs need.

## Repo structure
