# ============================================================
# MODULE: predictor.py
# PURPOSE: Top-k landmark inference using a TorchScript model.
#          Zero dependency on model.py — portable and deployable
#          without the full project source tree.
# NORMATIVE BASIS: UCB Project 5 rubric — Phase 4: function
#                  predict_landmarks(img_path, k) returning top-k
#                  landmarks with probabilities. Tested on >=4
#                  external images not from the training set.
# AUTHOR: Guillermo Carvajal Vaca — UCB MSc Data Science & AI
# VERSION: 1.0.0
# ============================================================
from __future__ import annotations

# --- stdlib (alphabetical) ---
import logging
from pathlib import Path
from typing import List, Tuple

# --- third-party (alphabetical) ---
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# --- local (alphabetical) ---
from src.config import (
    DEVICE,
    EXTERNAL_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    MODELS_DIR,
    RESIZE_SIZE,
    TRAIN_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------
def _get_class_names(train_dir: Path = TRAIN_DIR) -> list[str]:
    """
    Infer class names from the ImageFolder directory structure.

    Args:
        train_dir: Root training directory with one subfolder per class.

    Returns:
        Alphabetically sorted list of class names — identical to the order
        used by torchvision.datasets.ImageFolder during training.

    Why alphabetical order:
        ImageFolder sorts subfolders alphabetically and assigns class indices
        in that order. The predictor must use the exact same mapping or
        all predictions will be systematically wrong.

    Raises:
        FileNotFoundError: If train_dir does not exist.
    """
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")

    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])


def _get_inference_transform() -> transforms.Compose:
    """
    Build the deterministic inference transform pipeline.

    Returns:
        Compose pipeline: Resize -> CenterCrop -> ToTensor -> Normalize.

    Why this must be identical to the validation transform:
        If the inference pipeline differs from val/test transforms even
        slightly (e.g. RandomCrop instead of CenterCrop), the model receives
        a pixel distribution it was never evaluated on, causing silent
        accuracy degradation on real-world images.
    """
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _load_scripted_model(scripted_model_path: Path | None) -> torch.jit.ScriptModule:
    """
    Load a TorchScript model from disk.

    Args:
        scripted_model_path: Explicit path to a .pt TorchScript file.
                             If None, auto-selects the latest in MODELS_DIR.

    Returns:
        Loaded ScriptModule in eval mode on DEVICE.

    Raises:
        FileNotFoundError: If no scripted model is found.

    Why TorchScript and not state_dict:
        torch.jit.load() does not require importing model.py or any project
        source. The serialized graph is self-contained — deployable to any
        Python environment with only torch installed.
    """
    if scripted_model_path is None:
        candidates = sorted(MODELS_DIR.glob("*_scripted.pt"))
        if not candidates:
            raise FileNotFoundError(
                "No TorchScript model found in models/. "
                "Run train.py first to generate a scripted checkpoint."
            )
        scripted_model_path = candidates[-1]
        logger.info("Auto-selected model: %s", scripted_model_path.name)

    model = torch.jit.load(str(scripted_model_path), map_location=DEVICE)
    model.eval()
    logger.info("TorchScript model loaded from %s", scripted_model_path)
    return model


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------
def predict_landmarks(
    img_path            : str | Path,
    k                   : int = 3,
    scripted_model_path : str | Path | None = None,
    train_dir           : Path = TRAIN_DIR,
) -> List[Tuple[str, float]]:
    """
    Predict the top-k most probable landmarks for a given image.

    Args:
        img_path:             Path to the input image (JPG, PNG, WEBP).
        k:                    Number of top predictions to return.
        scripted_model_path:  Path to a TorchScript .pt file.
                              If None, auto-selects the latest in models/.
        train_dir:            Training directory used to infer class names.

    Returns:
        List of k tuples (class_name, probability) ordered by descending
        probability. Example:
            [("Eiffel_Tower", 0.87), ("Arc_de_Triomphe", 0.09), ...]

    Raises:
        FileNotFoundError: If img_path does not exist.
        ValueError:        If the image cannot be opened or decoded.
        FileNotFoundError: If no TorchScript model is available.

    Why Softmax on logits and not raw logits:
        Raw logits are unbounded and not comparable across classes.
        Softmax normalizes them to a probability distribution summing to 1,
        making the output directly interpretable as confidence scores.

    Why convert to RGB explicitly:
        User-provided images may be RGBA (PNG with transparency), grayscale,
        or CMYK. The model expects exactly 3 channels — convert() normalizes
        the input without raising cryptic shape errors downstream.
    """
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    if scripted_model_path is not None:
        scripted_model_path = Path(scripted_model_path)

    model       = _load_scripted_model(scripted_model_path)
    transform   = _get_inference_transform()
    class_names = _get_class_names(train_dir)

    try:
        img = Image.open(img_path).convert("RGB")   # normalize to 3-channel RGB
    except Exception as e:
        raise ValueError(f"Cannot open image {img_path}: {e}") from e

    # Add batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits     = model(img_tensor)                  # [1, num_classes]
        probs      = F.softmax(logits, dim=1)           # convert to probabilities
        top_probs, top_indices = probs.topk(k, dim=1)  # [1, k] each

    predictions: List[Tuple[str, float]] = [
        (class_names[idx], float(prob))
        for idx, prob in zip(top_indices[0].tolist(), top_probs[0].tolist())
    ]

    logger.info(
        "Predictions for %s: %s",
        img_path.name,
        [(name, f"{prob:.2%}") for name, prob in predictions],
    )
    return predictions


def predict_and_display(
    img_path            : str | Path,
    k                   : int = 3,
    scripted_model_path : str | Path | None = None,
) -> List[Tuple[str, float]]:
    """
    Run predict_landmarks and render an inline visualization in notebooks.

    Displays the original image alongside a horizontal bar chart of
    top-k class probabilities using UCB corporate palette.

    Args:
        img_path:             Path to the input image.
        k:                    Number of top predictions to display.
        scripted_model_path:  Path to TorchScript model (or None for auto).

    Returns:
        Same list of (class_name, probability) tuples as predict_landmarks.

    Why separate from predict_landmarks:
        predict_landmarks is a pure function with no side effects — usable
        in batch pipelines and APIs. predict_and_display adds matplotlib
        as a dependency only for interactive notebook usage.
    """
    import matplotlib.pyplot as plt

    UCB_BLUE      = "#003262"
    UCB_GOLD      = "#FDB515"

    predictions = predict_landmarks(img_path, k, scripted_model_path)
    img         = Image.open(img_path).convert("RGB")

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("white")

    # --- Original image panel ---
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(Path(img_path).name, fontsize=10, color=UCB_BLUE, fontweight="bold")

    # --- Probability bar chart panel ---
    names  = [p[0].replace("_", " ") for p in predictions]
    probs  = [p[1] for p in predictions]
    colors = [UCB_GOLD if i == 0 else UCB_BLUE for i in range(len(probs))]

    bars = ax_bar.barh(range(len(names)), probs, color=colors, edgecolor="white")
    ax_bar.set_yticks(range(len(names)))
    ax_bar.set_yticklabels(names, fontsize=9)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Probability", fontsize=9, color=UCB_BLUE)
    ax_bar.set_title(f"Top-{k} Predictions", fontsize=10, color=UCB_BLUE, fontweight="bold")
    ax_bar.invert_yaxis()   # highest probability at top
    ax_bar.spines[["top", "right"]].set_visible(False)

    # Probability label at end of each bar
    for bar, prob in zip(bars, probs):
        ax_bar.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center", fontsize=9, color=UCB_BLUE,
        )

    plt.tight_layout()
    plt.show()

    return predictions
