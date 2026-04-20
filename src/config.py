# ============================================================
# MODULE: config.py
# PURPOSE: Single source of truth for paths, hyperparameters,
#          and device configuration across the entire project.
#          Auto-detects runtime environment (PyCharm vs Colab).
# NORMATIVE BASIS: UCB Project 5 rubric — Phase 2 threshold
#                  (>=40%) and Phase 3 threshold (>=70%).
#                  PyTorch 2.x official documentation.
# AUTHOR: Guillermo Carvajal Vaca — UCB MSc Data Science & AI
# VERSION: 1.1.0
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
    BASE_PATH: Path = Path(
        "/content/landmark-classifier"
    )
    # Why /content/drive/MyDrive: Google Drive mounts at this path in Colab.
    # Windows G:\ path is only valid on local machine with Drive desktop sync.
    DATASET_PATH: Path = Path(
        "/content/drive/MyDrive/Maestria Ciencia de Datos/DEEP_LEARNING/PROJECT_01/landmark_images"
    )
else:
    BASE_PATH = Path(__file__).resolve().parent.parent
    # Why G:\My Drive: local Windows path where Drive desktop syncs the dataset.
    DATASET_PATH = Path(
        r"G:\My Drive\Maestria Ciencia de Datos\DEEP_LEARNING\PROJECT_01\landmark_images"
    )

logger.debug("Runtime: %s | BASE_PATH: %s", "Colab" if _IN_COLAB else "Local", BASE_PATH)

# ---------------------------------------------------------------------------
# DATASET PATHS
# Why separate DATASET_PATH from BASE_PATH: dataset is never committed
# to the repo (rubric: -1 pt). BASE_PATH -> repo | DATASET_PATH -> data.
# ---------------------------------------------------------------------------
TRAIN_DIR       : Path = DATASET_PATH / "train"
TEST_DIR        : Path = DATASET_PATH / "test"
EXTERNAL_DIR    : Path = BASE_PATH / "data" / "external"
EXPERIMENTS_DIR : Path = BASE_PATH / "experiments"
MODELS_DIR      : Path = BASE_PATH / "models"
DOCS_DIR        : Path = BASE_PATH / "docs"

# ---------------------------------------------------------------------------
# GLOBAL HYPERPARAMETERS
# Why here and not in notebooks: one file -> one change -> all in sync.
# ---------------------------------------------------------------------------
SEED        : int   = 42
NUM_CLASSES : int   = 50
BATCH_SIZE  : int   = 32
NUM_WORKERS : int   = 2
PIN_MEMORY  : bool  = True
VAL_SPLIT   : float = 0.20

# --- Phase 2: CNN From Scratch ---
SCRATCH_LR              : float = 1e-3
SCRATCH_EPOCHS          : int   = 30
SCRATCH_SCHEDULER_STEP  : int   = 10
SCRATCH_SCHEDULER_GAMMA : float = 0.5

# --- Phase 3: Transfer Learning ---
TL_LR_HEAD         : float = 1e-3
TL_LR_BACKBONE     : float = 1e-5
TL_EPOCHS_FROZEN   : int   = 10
TL_EPOCHS_FINETUNE : int   = 10

# ---------------------------------------------------------------------------
# DEVICE
# Why not hardcode: same script must run unmodified on PyCharm and Colab T4.
# ---------------------------------------------------------------------------
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ImageNet normalization ---
# Why these exact values: ResNet/VGG pretrained with this distribution.
# Deviating degrades transfer accuracy by >=10 percentage points.
IMAGENET_MEAN : list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD  : list[float] = [0.229, 0.224, 0.225]
INPUT_SIZE    : int = 224
RESIZE_SIZE   : int = 256

# ---------------------------------------------------------------------------
# SANITY CHECK
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
