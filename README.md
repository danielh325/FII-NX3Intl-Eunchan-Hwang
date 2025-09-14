# TB Chest X-ray Classifier (DenseNet-121 + MixStyle)

End-to-end pipeline to train, fine-tune, and externally validate a tuberculosis (TB) chest X-ray classifier. Includes a plain baseline (DenseNet-121) and an enhanced domain-generalization model (MixStyle + strong augmentations + weighted BCE), with evaluation on two unseen datasets.

## Highlights
- **Models:** DenseNet-121 baseline; enhanced head with **MixStyle**, GAP+GMP **concatenation**, BatchNorm, Dropout.
- **Class imbalance:** custom **weighted BCE** from empirical class frequencies.
- **Augmentations:** flip, rotation, zoom, contrast, brightness, ImageNet-style normalization.
- **Evaluation:** AUROC/PR-AUC, F1, MCC, Brier score, balanced accuracy; **ROC curves**, **confusion matrix**, and training curves.
- **External validation:** Shenzhen & Pakistan CXR datasets (thresholds tunable; 0.9/0.5 shown).

## Data
Downloaded automatically via `kagglehub`:
- `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` (primary train/val/test split)
- `jtiptj/chest-xray-pneumoniacovid19tuberculosis` (Shenzhen) for external test
- `yasserhessein/tuberculosis-chest-x-rays-images` (Pakistan) for external test

> You’ll need Kaggle access on the machine (Kaggle/Colab recommended).

## Environment
- Python 3.10+
- TensorFlow/Keras 2.15+ (GPU recommended), scikit-learn, pandas, numpy, pillow, matplotlib, opencv-python, kagglehub
- One helper script is fetched at runtime:
  - `helper_functions.py` (confusion matrix & training curves) from mrdbourke’s repo

_Minimal install:_
```bash
pip install tensorflow scikit-learn pandas numpy pillow matplotlib opencv-python kagglehub
