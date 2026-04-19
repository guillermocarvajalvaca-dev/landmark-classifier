# ============================================================
# MODULE: model.py
# PURPOSE: CNN architecture from scratch and Transfer Learning
#          factory. Two separate model families, one common
#          interface for run_experiment() in train.py.
# NORMATIVE BASIS: UCB Project 5 rubric — Phase 2: >=3 conv
#                  layers, pooling, dropout, FC, 50-class output.
#                  Phase 3: pretrained torchvision model, frozen
#                  feature extractor, replaced classifier head.
# AUTHOR: Guillermo Carvajal Vaca — UCB MSc Data Science & AI
# VERSION: 1.0.0
# ============================================================
from __future__ import annotations

# --- stdlib (alphabetical) ---
import logging
from typing import Literal

# --- third-party (alphabetical) ---
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

# --- local (alphabetical) ---
from src.config import DEVICE, NUM_CLASSES

logger = logging.getLogger(__name__)


# ===========================================================================
# BLOCK 1 — CNN From Scratch
# ===========================================================================

class CNNScratch(nn.Module):
    """
    Custom CNN with 5 convolutional blocks for landmark classification.

    Architecture rationale:
        Progressive filter growth (32->64->128->256->512) mirrors how
        biological vision hierarchies work: early layers detect edges and
        textures; deeper layers combine them into semantic structures
        (arches, domes, towers).

        BatchNorm after each Conv:
            Normalizes activations between layers, allowing higher learning
            rates without gradient explosion. Eliminates the need for
            careful weight initialization.

        Dropout2d on intermediate blocks:
            Drops entire feature map channels rather than individual neurons.
            More effective than standard Dropout for spatial data because
            adjacent pixels are highly correlated — dropping one neuron
            leaves its neighbors as effective proxies.

        GlobalAveragePooling (AdaptiveAvgPool2d(1)):
            Collapses 512x7x7 -> 512x1x1, reducing FC parameters by ~25x
            compared to Flatten. Also makes the model input-size agnostic,
            which simplifies TorchScript export.

    Args:
        num_classes: Number of output classes. Defaults to NUM_CLASSES (50).
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        def _conv_block(
            in_ch   : int,
            out_ch  : int,
            dropout : bool = False,
        ) -> nn.Sequential:
            """
            Reusable block: Conv2d -> BatchNorm -> ReLU -> MaxPool [-> Dropout2d].

            Args:
                in_ch:   Input channels.
                out_ch:  Output channels.
                dropout: If True, append Dropout2d after pooling.

            Returns:
                Sequential block ready to be stacked in self.features.

            Why bias=False with BatchNorm:
                BatchNorm has its own learnable bias (beta). Including a
                Conv bias as well is redundant — it gets absorbed into
                BatchNorm's beta and wastes a parameter.
            """
            layers: list[nn.Module] = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            if dropout:
                layers.append(nn.Dropout2d(p=0.25))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            _conv_block(3,   32),                    # 224x224 -> 112x112
            _conv_block(32,  64),                    # 112x112 ->  56x56
            _conv_block(64,  128, dropout=True),     #  56x56  ->  28x28
            _conv_block(128, 256),                   #  28x28  ->  14x14
            _conv_block(256, 512, dropout=True),     #  14x14  ->   7x7
        )

        # Collapses spatial dims to 1x1 — input-size agnostic for TorchScript
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),     # standard dropout before final FC
            nn.Linear(256, num_classes),
            # No Softmax here: CrossEntropyLoss applies LogSoftmax internally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, 3, H, W].

        Returns:
            Raw logits of shape [B, num_classes].
        """
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# ===========================================================================
# BLOCK 2 — Transfer Learning Factory
# ===========================================================================

def get_transfer_model(
    backbone    : Literal["resnet18", "resnet50"] = "resnet18",
    num_classes : int = NUM_CLASSES,
    strategy    : Literal["frozen", "finetune"] = "frozen",
) -> nn.Module:
    """
    Build a pretrained torchvision model adapted to num_classes.

    Args:
        backbone:    Pretrained backbone to use.
                     "resnet18" — 11M params, efficient, strong baseline.
                     "resnet50" — 25M params, higher capacity, slower.
        num_classes: Number of output classes (replaces ImageNet FC head).
        strategy:
            "frozen"   — Freeze entire feature extractor. Only the new FC
                         head is trainable. Use for the first training stage:
                         fast convergence, zero risk of catastrophic forgetting.
            "finetune" — Unfreeze layer4 (last residual block) in addition
                         to the FC head. Use after "frozen" stage: allows the
                         backbone to adapt high-level features to landmarks
                         with a very low backbone LR (TL_LR_BACKBONE=1e-5).

    Returns:
        Model with pretrained weights, frozen backbone (or layer4 unfrozen),
        and a new dropout + FC classifier head for num_classes.

    Raises:
        ValueError: If backbone is not one of the supported options.

    Why ResNet18 as default over VGG16:
        VGG16 has 138M parameters — its FC layers alone (4096->4096) consume
        more VRAM than all of ResNet18. With batch_size=32 and augmentation
        on a 6GB VRAM card, VGG16 causes OOM. ResNet18 achieves comparable
        accuracy on medium-scale datasets at 12x fewer parameters.

    Why residual connections matter for fine-tuning:
        Skip connections allow gradients to bypass layers, preventing
        vanishing gradient even when only the last block is unfrozen.
        VGG16 lacks this — fine-tuning deep VGG layers is unstable.
    """
    _supported = ("resnet18", "resnet50")
    if backbone not in _supported:
        raise ValueError(f"backbone must be one of {_supported}, got: {backbone!r}")

    weights_map = {
        "resnet18": ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": ResNet50_Weights.IMAGENET1K_V2,
    }
    model_fn = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
    }

    model = model_fn[backbone](weights=weights_map[backbone])

    # Freeze all layers — ImageNet features are reused as-is
    for param in model.parameters():
        param.requires_grad = False

    # Replace the ImageNet FC head (Linear(512->1000)) with a landmark head
    in_features : int = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),   # requires_grad=True by default (new module)
    )

    if strategy == "finetune":
        # Unfreeze layer4 — the last residual block captures the highest-level
        # ImageNet features. Adapting it to landmarks lets the model learn
        # domain-specific representations (stone textures, architectural shapes).
        for param in model.layer4.parameters():
            param.requires_grad = True
        logger.info("[%s] strategy=finetune — layer4 + FC head trainable", backbone)
    else:
        logger.info("[%s] strategy=frozen — FC head only trainable", backbone)

    return model


def count_params(model: nn.Module) -> dict[str, int]:
    """
    Count total and trainable parameters in a model.

    Args:
        model: Any nn.Module instance.

    Returns:
        Dict with keys 'total' and 'trainable'.

    Why log this before training:
        Verifying trainable parameter count confirms the freeze/unfreeze
        strategy worked as intended. A frozen ResNet18 should show ~500K
        trainable params (FC only); finetune should show ~2.8M (layer4 + FC).
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct       = 100.0 * trainable / total if total > 0 else 0.0

    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}  ({pct:.1f}%)")
    logger.info("Params — total: %d | trainable: %d (%.1f%%)", total, trainable, pct)

    return {"total": total, "trainable": trainable}
