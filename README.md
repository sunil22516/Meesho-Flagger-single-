# 👗 Fashion Attribute Classification

A multi-label image classification system that predicts multiple fashion attributes (e.g. color, sleeve length, neckline, pattern) from product images using a fine-tuned EfficientNet-B5 backbone with per-attribute classification heads.

---

## 🧠 Model Architecture

The model uses a **multi-head classifier** built on top of `EfficientNet-B5` (via [timm](https://github.com/huggingface/pytorch-image-models)):

- **Backbone**: EfficientNet-B5 pretrained on ImageNet, with the classification head removed
- **Heads**: One independent MLP head per attribute → `Linear(feat_dim, 512) → ReLU → Dropout(0.3) → Linear(512, num_classes)`
- **Loss**: Focal Loss (γ=2.0) with per-class inverse-frequency weights to handle class imbalance
- **Augmentation**: MixUp (α=0.2) applied during training

```
Input Image
     │
EfficientNet-B5 Backbone (feat_dim = 2048)
     │
     ├── Head 1 → Attribute 1 (e.g. Color)
     ├── Head 2 → Attribute 2 (e.g. Sleeve Length)
     ├── Head 3 → Attribute 3 (e.g. Neckline)
     └── ...
```

---

## ⚙️ Training Details

| Setting | Value |
|---|---|
| Backbone | EfficientNet-B5 |
| Optimizer | AdamW (backbone lr=1e-5, heads lr=1e-4) |
| Scheduler | OneCycleLR (max_lr=5e-4) |
| Mixed Precision | ✅ AMP (GradScaler) |
| Loss | Focal Loss (γ=2.0) |
| Augmentation | MixUp (β=0.2) |
| Batch Size | 16 |
| Max Epochs | 10 |
| Early Stopping | Patience = 5 (Macro-F1) |
| Gradient Clipping | max_norm=1.0 |
| Weight Decay | 1e-3 |

### Gradual Unfreezing Schedule

| Epoch | Backbone State |
|---|---|
| 1–3 | Frozen |
| 4–7 | Last 2 blocks unfrozen (`blocks.6`, `blocks.7`) |
| 8+ | Fully unfrozen |

---

## 📁 Project Structure

```
├── notebooks/
│   ├── cell_1_setup.ipynb        # Device setup, paths
│   ├── cell_2_data.ipynb         # Dataset loading, DataLoaders, label encoding
│   └── cell_3_train.ipynb        # Model definition & training loop  ← (this file)
├── models/
│   └── best_model.pth            # Saved checkpoint (best Macro-F1)
├── data/
│   └── ...                       # Dataset (not included)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision timm scikit-learn numpy
```

### Running the Notebook

Run cells in order:

1. **Cell 1** — Sets up `DEVICE` and `MODEL_PATH`
2. **Cell 2** — Loads dataset, builds `train_loader`, `val_loader`, `led` (label encoder), `attr_cols`, and `df_listed`
3. **Cell 3** — Defines and trains the model

> The training script will automatically save the best checkpoint to `MODEL_PATH` based on validation Macro-F1.

---

## 📊 Evaluation Metric

The model is evaluated using **Macro-averaged F1 score** across all attribute heads:

```
Macro-F1 = mean(F1_attr1, F1_attr2, ..., F1_attrN)
```

This treats all attributes equally regardless of class frequency.

---

## ⚡ Hardware Recommendations

| Environment | Estimated Time per Epoch | Recommended? |
|---|---|---|
| GPU (T4/V100) | 2–5 min | ✅ Recommended |
| Google Colab (free T4) | 3–6 min | ✅ Good option |
| Kaggle Notebook (P100) | 2–4 min | ✅ Good option |
| CPU only (e.g. GitHub Codespaces) | 60–90 min | ❌ Not recommended |

> **Note:** EfficientNet-B5 is compute-heavy. Running on CPU is possible but will take several hours per full training run. Use [Google Colab](https://colab.research.google.com/) or [Kaggle Notebooks](https://www.kaggle.com/code) for practical training times.

---

## 🔧 Quick Experiments / Debugging

To test the pipeline quickly before a full run, reduce the dataset and epochs:

```python
REDUCE_SIZE = 0.02   # Use only 2% of data
NUM_EPOCHS  = 2      # Just 2 epochs to validate pipeline
```

To use a lighter backbone for faster iteration:

```python
self.backbone = timm.create_model(
    'efficientnet_b0',   # ~5x faster than B5
    pretrained=True,
    num_classes=0,
    global_pool='avg'
)
```

---

## 📌 Notes

- Label encoding is handled by a custom `led` (Label Encoder Dict) object defined in Cell 2
- Class weights for Focal Loss are computed from the training set's label distribution
- The model checkpoint is only saved when validation Macro-F1 improves (best model only)
- MixUp is applied at the batch level with a per-batch sampled λ ∼ Beta(0.2, 0.2)

---

## 📄 License

This project is for personal/portfolio use. Dataset and pretrained weights are subject to their respective licenses.
