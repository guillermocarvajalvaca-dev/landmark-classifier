# ============================================================
# MODULE: data.py
# PURPOSE: Dataset loading, transforms pipeline, reproducible
#          train/val split, and DataLoader factory.
#          Single entry point for all data access in the project.
# NORMATIVE BASIS: UCB Project 5 rubric — Phase 1 requirements:
#                  ImageFolder, resize 256, crop 224, ImageNet
#                  normalization, augmentation on train only.
# AUTHOR: Guillermo Carvajal Vaca — UCB MSc Data Science & AI
# VERSION: 1.0.0
# ============================================================
from __future__ import annotations

# --- stdlib (alphabetical) ---
import logging
from pathlib import Path
from typing import Tuple

# --- third-party (alphabetical) ---
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# --- local (alphabetical) ---
from src.config import (
    BATCH_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    RESIZE_SIZE,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_SPLIT,
)
from src.utils import set_seed

logger = logging.getLogger(__name__)


def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    Build the image preprocessing pipeline.

    Args:
        augment: If True, apply training augmentation transforms.
                 If False, apply deterministic val/test transforms only.

    Returns:
        Composed transform pipeline ready for ImageFolder.

    Why separate augment flag and not two fixed pipelines:
        Callers decide at runtime whether they need augmentation.
        This avoids duplicating the normalization block and guarantees
        val/test always receive the exact same deterministic transforms.

    Why RandomResizedCrop instead of RandomCrop:
        RandomResizedCrop simulates different zoom levels and framings,
        forcing the model to recognize landmarks regardless of how close
        or far the photographer was — critical for a real-world dataset.

    Why augmentation on train only:
        Applying augmentation to val/test introduces randomness into
        evaluation metrics, making runs non-reproducible and masking
        true generalization performance.

    Why ImageNet normalization is mandatory for Transfer Learning:
        ResNet/VGG backbones were pretrained with this exact pixel
        distribution. Deviating from it shifts the feature space and
        degrades transfer accuracy by 10+ percentage points.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=15),
            transforms.RandomGrayscale(p=0.05),   # robustness to desaturated photos
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        normalize,
    ])


def get_dataloaders(
    train_dir  : Path = TRAIN_DIR,
    test_dir   : Path = TEST_DIR,
    batch_size : int  = BATCH_SIZE,
    val_split  : float = VAL_SPLIT,
    seed       : int  = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build train, validation, and test DataLoaders from ImageFolder structure.

    Args:
        train_dir:  Root directory with one subfolder per class (ImageFolder).
        test_dir:   Root directory for the test set (same structure).
        batch_size: Mini-batch size for all loaders.
        val_split:  Fraction of train data reserved for validation.
        seed:       Reproducibility seed for the split permutation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names).
        class_names is the alphabetically ordered list of landmark labels.

    Raises:
        FileNotFoundError: If train_dir or test_dir do not exist.

    Why two ImageFolder instances for train:
        We need the same image indices for train and val subsets, but with
        different transforms — augmentation on train, none on val.
        random_split cannot do this; Subset over two separate ImageFolder
        instances pointing to the same directory solves it cleanly.

    Why Generator().manual_seed instead of random.shuffle:
        torch.randperm with a seeded Generator produces a deterministic
        permutation that is independent of Python and NumPy random states,
        ensuring the split is stable even if other seeds change.
    """
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"test_dir not found: {test_dir}")

    set_seed(seed)

    # Two views of the same directory — different transforms, same indices
    full_train_aug = datasets.ImageFolder(str(train_dir), transform=get_transforms(augment=True))
    full_train_val = datasets.ImageFolder(str(train_dir), transform=get_transforms(augment=False))
    test_ds        = datasets.ImageFolder(str(test_dir),  transform=get_transforms(augment=False))

    class_names : list[str] = full_train_aug.classes   # alphabetical — matches ImageFolder order

    n_total = len(full_train_aug)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    perm      = torch.randperm(n_total, generator=generator)
    train_idx = perm[:n_train].tolist()
    val_idx   = perm[n_train:].tolist()

    train_subset = Subset(full_train_aug, train_idx)   # augmentation active
    val_subset   = Subset(full_train_val, val_idx)     # augmentation off

    # persistent_workers avoids worker process restart between epochs
    # Only enabled when num_workers > 0 to avoid a PyTorch warning
    _pw = NUM_WORKERS > 0

    train_loader = DataLoader(
        train_subset,
        batch_size       = batch_size,
        shuffle          = True,    # randomize order each epoch for better generalization
        num_workers      = NUM_WORKERS,
        pin_memory       = PIN_MEMORY,
        persistent_workers = _pw,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size       = batch_size,
        shuffle          = False,   # fixed order -> reproducible metrics across runs
        num_workers      = NUM_WORKERS,
        pin_memory       = PIN_MEMORY,
        persistent_workers = _pw,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size       = batch_size,
        shuffle          = False,
        num_workers      = NUM_WORKERS,
        pin_memory       = PIN_MEMORY,
        persistent_workers = _pw,
    )

    logger.info(
        "DataLoaders ready — train: %d | val: %d | test: %d | classes: %d",
        len(train_subset), len(val_subset), len(test_ds), len(class_names),
    )
    return train_loader, val_loader, test_loader, class_names


def verify_dataloaders(
    train_loader : DataLoader,
    val_loader   : DataLoader,
    test_loader  : DataLoader,
    class_names  : list[str],
) -> None:
    """
    Sanity-check one batch from each loader before any training begins.

    Args:
        train_loader: Training DataLoader to inspect.
        val_loader:   Validation DataLoader to inspect.
        test_loader:  Test DataLoader to inspect.
        class_names:  List of class labels from the dataset.

    Why run this before training:
        Shape mismatches and normalization errors are silent — the model
        trains but learns nothing useful. Catching them at batch-zero
        costs seconds and prevents hours of wasted GPU time.
    """
    imgs, labels = next(iter(train_loader))
    logger.info("Train batch shape : %s  -> expected [%d, 3, 224, 224]", imgs.shape, imgs.shape[0])
    logger.info("Labels shape      : %s  -> expected [%d]", labels.shape, labels.shape[0])

    print(f"Train batch shape : {imgs.shape}   -> expected [B, 3, 224, 224]")
    print(f"Labels shape      : {labels.shape} -> expected [B]")
    print(f"Classes detected  : {len(class_names)}")
    print(f"Train batches     : {len(train_loader)}")
    print(f"Val batches       : {len(val_loader)}")
    print(f"Test batches      : {len(test_loader)}")
    print(f"Pixel min / max   : {imgs.min():.3f} / {imgs.max():.3f}  -> expected approx [-2.1, 2.6]")
