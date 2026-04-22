# Landmark Classifier CNN

> UCB MSc Data Science and AI - Deep Learning Project 5
> Guillermo Carvajal Vaca - 2026

## Overview

Automatic landmark classification using CNNs in PyTorch.
Infers geographic location from images without GPS metadata across 50 landmark classes.

| Phase | Approach | Target | Result |
|---|---|---|---|
| Phase 2 | CNN from Scratch (5 conv + BN + GAP) | Test Accuracy >= 40% | 31.84% |
| Phase 3 | Transfer Learning (ResNet18 / ResNet50) | Test Accuracy >= 70% | **79.68%** + bonus |
| Phase 4 | TorchScript inference app | predict_landmarks(img_path, k) | Done |

## Dataset

Subset of Google Landmarks Dataset v2 - 50 classes.
Provided by UCB instructor. Not included in this repository.

Place the dataset at the root of the project directory:

```
./data/landmark_images/
    train/  <- one subfolder per class
    test/   <- same structure
```

## Installation

```bash
git clone https://github.com/guillermocarvajalvaca-dev/landmark-classifier.git
cd landmark-classifier
python -m venv .venv
.venv/Scripts/Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python src/config.py
```

Expected output: `DEVICE: cuda | TRAIN_DIR exists=True | TEST_DIR exists=True`

## Execution

| Node | Role | Notebooks |
|---|---|---|
| PyCharm (local) | src/ development, EDA, inference | 01, 04 |
| Google Colab Pro A100 | Training (30 epochs scratch / 10 epochs transfer) | 02, 03 |

Open notebooks from GitHub directly in Colab. Cell 0A mounts Drive and pulls latest src/ automatically.

## Inference

```python
from src.predictor import predict_landmarks

predictions = predict_landmarks('my_photo.jpg', k=3)
# [('Eiffel_Tower', 0.87), ('Arc_de_Triomphe', 0.09), ('Notre_Dame', 0.04)]
```

## Results

| Experiment | Type | Backbone | Strategy | Test Accuracy | Top-5 |
|---|---|---|---|---|---|
| E1_scratch_baseline | CNN Scratch | - | - | 31.84% | 66.40% |
| E2_scratch_augmented | CNN Scratch | - | - | 31.52% | - |
| E3_scratch_lower_lr | CNN Scratch | - | - | 31.12% | - |
| E4_resnet18_frozen | Transfer Learning | ResNet18 | Frozen | 66.00% | - |
| E5_resnet18_finetune_layer4 | Transfer Learning | ResNet18 | Finetune layer4 | 75.12% | - |
| **E6_resnet50_frozen** | **Transfer Learning** | **ResNet50** | **Frozen** | **79.68%** | **94.00%** |

Best model: `E6_resnet50_frozen` - 79.68% Top-1 / 94.00% Top-5.

## Visualization

BI-grade training curves via plotnine with UCB corporate palette:

- `#003262` UCB Blue - primary series (Train)
- `#FDB515` UCB Gold - optimal checkpoint marker
- `#C4820E` UCB Dark Gold - Validation / overfitting onset

Each experiment generates: narrative PNG + BI confusion matrix (dpi=200) + Executive Report MD.

## Video

YouTube: https://youtu.be/LINK

## Tech Stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.11.9 | Runtime |
| PyTorch | 2.5.1+cu121 | Deep Learning framework |
| torchvision | 0.20.1 | Pretrained ResNet models |
| plotnine | 0.15.3 | Grammar of Graphics visualization |
| scikit-learn | 1.8.0 | Metrics and evaluation |
| nbformat | 5.x | Programmatic notebook generation |

## License

MIT

## Author

Guillermo Carvajal Vaca
MSc Data Science and Applied AI - UCB San Pablo - Santa Cruz de la Sierra, Bolivia
https://github.com/guillermocarvajalvaca-dev
