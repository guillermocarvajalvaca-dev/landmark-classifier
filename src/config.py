# ============================================================
# MODULE: config.py
# PURPOSE: Single source of truth for paths, hyperparameters,
#          and device configuration across the entire project.
#          Auto-detects runtime environment (PyCharm vs Colab).
# NORMATIVE BASIS: UCB Project 5 rubric — Phase 2 threshold
#                  (>=40%) and Phase 3 threshold (>=70%).
#                  PyTorch 2.x official documentation.
# AUTHOR: Guillermo Carvajal Vaca — UCB MSc Data Science & AI
# VERSION: 1.0.0
# ============================================================
from __future__ import annotations

# --- stdlib (alphabetical) ---
import logging
import os
from pathlib import Path

# --- third-party (alphabetical) ---
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ENVIRONMENT DETECTION
# Why: absolute paths differ between Windows local and Colab Linux.
# Centralizing detection here avoids scattered conditionals in notebooks.
# ---------------------------------------------------------------------------
_IN_COLAB: bool = (
    "COLAB_GPU" in os.environ
    or "COLAB_BACKEND_API_SERVICE" in os.environ
    or os.path.exists("/content")
)

if _IN_COLAB:
    # Drive must be mounted before importing this module in Colab
    BASE_PATH: Path = Path(
        "/content/drive/MyDrive/Maestria Ciencia de Datos/DEEP_LEARNING/PROJECT_01"
    )
else:
    # __file__ = src/config.py -> .parent = src/ -> .parent = project root
    BASE_PATH = Path(__file__).resolve().parent.parent

logger.debug("Runtime: %s | BASE_PATH: %s", "Colab" if _IN_COLAB else "Local", BASE_PATH)

# ---------------------------------------------------------------------------
# DATASET PATHS
# Why separate DATASET_PATH from BASE_PATH: the dataset lives in synced
# Google Drive (G:\) and must NEVER be committed to the repo (rubric: -1 pt).
# BASE_PATH -> repo | DATASET_PATH -> data (unversioned)
# ---------------------------------------------------------------------------
DATASET_PATH    : Path = Path(
    r"G:\My Drive\Maestria Ciencia de Datos\DEEP_LEARNING\PROJECT_01\landmark_images"
)
TRAIN_DIR       : Path = DATASET_PATH / "train"
TEST_DIR        : Path = DATASET_PATH / "test"
EXTERNAL_DIR    : Path = BASE_PATH / "data" / "external"   # own images for Phase 4
EXPERIMENTS_DIR : Path = BASE_PATH / "experiments"          # JSON + PNG artifacts
MODELS_DIR      : Path = BASE_PATH / "models"               # checkpoints + TorchScript
DOCS_DIR        : Path = BASE_PATH / "docs"                 # report + video script

# ---------------------------------------------------------------------------
# GLOBAL HYPERPARAMETERS
# Why here and not in notebooks: changing a single LR across 4 notebooks
# introduces silent inconsistencies. One file -> one change -> all in sync.
# ---------------------------------------------------------------------------
SEED        : int   = 42     # full reproducibility across runs
NUM_CLASSES : int   = 50     # verified: 50 subfolders in train/ (2026-04-19)
BATCH_SIZE  : int   = 32     # optimal balance for Colab T4 (16 GB VRAM)
NUM_WORKERS : int   = 2      # Colab safe maximum — higher values cause DataLoader timeout
PIN_MEMORY  : bool  = True   # async CPU->VRAM transfer for higher throughput
VAL_SPLIT   : float = 0.20   # 80/20 reproducible split controlled by SEED

# --- Phase 2: CNN From Scratch (rubric: >=30 epochs, threshold >=40%) ---
SCRATCH_LR              : float = 1e-3   # standard Adam LR for mid-size CNNs
SCRATCH_EPOCHS          : int   = 30     # rubric minimum
SCRATCH_SCHEDULER_STEP  : int   = 10     # decay LR every N epochs
SCRATCH_SCHEDULER_GAMMA : float = 0.5   # LR *= 0.5 per step — gentle annealing

# --- Phase 3: Transfer Learning (rubric: threshold >=70%, bonus >75%) ---
TL_LR_HEAD         : float = 1e-3   # new FC head: normal LR (randomly initialized)
TL_LR_BACKBONE     : float = 1e-5   # unfrozen backbone: 100x lower LR
                                     # Why: prevents catastrophic forgetting of ImageNet weights
TL_EPOCHS_FROZEN   : int   = 10     # stage 1: train only the new head
TL_EPOCHS_FINETUNE : int   = 10     # stage 2: unfreeze layer4 + fine-tune

# ---------------------------------------------------------------------------
# DEVICE
# Why not hardcode "cuda": the same script must run unmodified on PyCharm
# (CPU debug) and Colab T4 (CUDA training).
# ---------------------------------------------------------------------------
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ImageNet normalization constants ---
# Why these exact values: ResNet/VGG were pretrained with this distribution.
# Using different values degrades transfer learning accuracy by >=10 pp.
IMAGENET_MEAN : list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD  : list[float] = [0.229, 0.224, 0.225]
INPUT_SIZE    : int = 224   # final spatial size required by ResNet/VGG
RESIZE_SIZE   : int = 256   # resize before CenterCrop to preserve context

# ---------------------------------------------------------------------------
# SANITY CHECK — run directly to verify environment before training
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"BASE_PATH   : {BASE_PATH}")
    print(f"DEVICE      : {DEVICE}")
    print(f"IN_COLAB    : {_IN_COLAB}")
    print(f"TRAIN_DIR   : {TRAIN_DIR}  exists={TRAIN_DIR.exists()}")
    print(f"TEST_DIR    : {TEST_DIR}   exists={TEST_DIR.exists()}")
    if not TRAIN_DIR.exists():
        logger.warning("TRAIN_DIR not found — verify Google Drive sync")
    if not TEST_DIR.exists():
        logger.warning("TEST_DIR not found — verify Google Drive sync")
