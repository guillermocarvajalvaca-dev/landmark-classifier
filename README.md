# Landmark Classifier CNN

> UCB MSc Data Science and AI - Deep Learning Project 5
> Guillermo Carvajal Vaca - 2026

## Overview

Automatic landmark classification using CNNs in PyTorch.
Infers geographic location from images without GPS metadata.

| Phase | Approach | Target |
|---|---|---|
| Phase 2 | CNN from Scratch (5 conv + BN + GAP) | Test Accuracy >= 40% |
| Phase 3 | Transfer Learning (ResNet18 / ResNet50) | Test Accuracy >= 70% |
| Phase 4 | TorchScript inference app | predict_landmarks(img_path, k) |

## Dataset

Subset of Google Landmarks Dataset v2 - 50 classes.
Provided by UCB instructor. Not included in this repository.

Place at:

    G:/My Drive/Maestria Ciencia de Datos/DEEP_LEARNING/PROJECT_01/landmark_images/
        train/  <- one subfolder per class
        test/   <- same structure

## Installation

    git clone https://github.com/guillermocarvajalvaca-dev/landmark-classifier.git
    cd landmark-classifier
    python -m venv .venv
    .venv/Scripts/Activate.ps1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    python src/config.py

Expected: DEVICE: cuda | TRAIN_DIR exists=True | TEST_DIR exists=True

## Execution

| Node | Role | Notebooks |
|---|---|---|
| PyCharm Local | src/ development, EDA, inference | 01, 04 |
| Google Colab T4 | Heavy training (>=30 epochs) | 02, 03 |

## Inference

    from src.predictor import predict_landmarks
    predictions = predict_landmarks('my_photo.jpg', k=3)
    # [('Eiffel_Tower', 0.87), ('Arc_de_Triomphe', 0.09), ('Notre_Dame', 0.04)]

## Visualization

BI-grade curves via plotnine with UCB palette:
- UCB Blue #003262 - primary series
- UCB Gold #FDB515 - optimal checkpoint
- UCB Dark Gold #C4820E - overfitting onset

Each run generates: narrative PNG + BI confusion matrix + Executive Report MD.

## Results

| Experiment | Type | Test Accuracy |
|---|---|---|
| E1_scratch_baseline | CNN Scratch | fill after training |
| E2_scratch_augmented | CNN Scratch | fill after training |
| E3_scratch_lower_lr | CNN Scratch | fill after training |
| E4_resnet18_frozen | Transfer Learning | fill after training |
| E5_resnet18_finetune_layer4 | Transfer Learning | fill after training |
| E6_resnet50_frozen | Transfer Learning | fill after training |

## Video

YouTube: https://youtu.be/LINK  (update after uploading)

## Tech Stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.11.9 | Runtime |
| PyTorch | 2.5.1+cu121 | Deep Learning |
| torchvision | 0.20.1 | Pretrained models |
| plotnine | 0.15.3 | Grammar of Graphics |
| scikit-learn | 1.8.0 | Metrics |

## License

MIT

## Author

Guillermo Carvajal Vaca
MSc Data Science and Applied AI - UCB San Pablo - Santa Cruz de la Sierra, Bolivia
https://github.com/guillermocarvajalvaca-dev
