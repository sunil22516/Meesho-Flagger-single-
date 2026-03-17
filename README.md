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


