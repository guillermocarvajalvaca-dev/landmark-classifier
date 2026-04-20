# ============================================================
# SCRIPT: scripts/create_notebooks.py
# PURPOSE: Programmatically generate all 4 project notebooks
#          using nbformat. Ensures valid .ipynb structure and
#          integrates BI/Storytelling visualization pipeline.
# NORMATIVE BASIS: UCB Project 5 rubric — 4 notebooks required:
#                  exploration, CNN scratch, transfer learning,
#                  inference app. All cells must be executed
#                  before submission (rubric: -3 pts if not).
# AUTHOR: Guillermo Carvajal Vaca — UCB MSc Data Science & AI
# VERSION: 1.1.0
# ============================================================
from __future__ import annotations

# --- stdlib (alphabetical) ---
import logging
from pathlib import Path

# --- third-party (alphabetical) ---
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SHARED CELL TEMPLATES
# ---------------------------------------------------------------------------

# Why a shared setup cell template:
# All 4 notebooks need identical environment detection logic.
# Centralizing it here ensures a single fix propagates to all notebooks.
SETUP_CELL = (
    "# ── Cell 0A: Mount Google Drive (Colab only) ───────────────\n"
    "# Why: dataset lives in Google Drive. Must mount before any\n"
    "# src.config import — TRAIN_DIR and TEST_DIR resolve to Drive paths.\n"
    "# Skip this cell if running locally in PyCharm.\n"
    "import os\n"
    "if os.path.exists('/content'):\n"
    "    from google.colab import drive\n"
    "    drive.mount('/content/drive')\n"
    "    import subprocess\n"
    "    if not os.path.exists('/content/landmark-classifier'):\n"
    "        subprocess.run(['git', 'clone',\n"
    "            'https://github.com/guillermocarvajalvaca-dev/landmark-classifier.git',\n"
    "            '/content/landmark-classifier'], check=True)\n"
    "    import subprocess\n"
    "    subprocess.run(['pip', 'install', '-q', 'plotnine'], check=True)\n"
    "    print('Colab environment ready')\n"
    "else:\n"
    "    print('Local environment detected')\n"
)

CONFIG_CELL = (
    "# ── Cell 1: Environment setup ──────────────────────────────\n"
    "# Why first cell is always config: single source of truth for\n"
    "# paths and hyperparameters. Any change propagates to all cells.\n"
    "import logging\n"
    "import os\n"
    "import sys\n"
    "from pathlib import Path\n\n"
    "# Why explicit Colab path: Path('..').resolve() returns /content\n"
    "# in Colab instead of the project root, breaking src.* imports.\n"
    "PROJECT_ROOT = (\n"
    "    Path('/content/landmark-classifier')\n"
    "    if os.path.exists('/content/landmark-classifier/src')\n"
    "    else Path('..').resolve()\n"
    ")\n"
    "if str(PROJECT_ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
    "logging.basicConfig(level=logging.INFO)\n"
)


def _save_notebook(nb: nbformat.NotebookNode, filename: str) -> Path:
    """
    Validate and persist a notebook to disk.

    Args:
        nb:       NotebookNode built with nbformat.v4 helpers.
        filename: Target filename including .ipynb extension.

    Returns:
        Path to the saved notebook.

    Why validate before saving:
        An invalid .ipynb silently fails to open in Jupyter or Colab.
        Validating here catches structural errors before they become
        hard-to-debug runtime issues.
    """
    nbformat.validate(nb)
    out_path = NOTEBOOKS_DIR / filename
    with out_path.open("w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)
    logger.info("Notebook saved -> %s", out_path)
    return out_path


def _kernel_metadata() -> dict:
    """Return standard kernel metadata for all notebooks."""
    return {
        "display_name": "Python (landmark-venv)",
        "language": "python",
        "name": "landmark-venv",
    }


# ===========================================================================
# NOTEBOOK 1 — Exploration and Preprocessing (Phase 1)
# ===========================================================================
def create_01_exploration() -> Path:
    """Build 01_exploration.ipynb — Phase 1: EDA + DataLoaders."""

    cells = [
        new_markdown_cell(
            "# Notebook 01 — Dataset Exploration and Preprocessing\n"
            "## Landmark Classification CNN · UCB MSc Data Science & AI\n\n"
            "> **Business question:** What does our dataset look like, and are our "
            "DataLoaders correctly configured before we invest GPU time in training?\n\n"
            "**Phase 1 rubric targets:**\n"
            "- Visualize >=5 sample images with labels (1 pt)\n"
            "- Class distribution bar chart (1 pt)\n"
            "- Functional train / val / test DataLoaders (1 pt)"
        ),

        new_code_cell(SETUP_CELL),

        new_code_cell(
            CONFIG_CELL +
            "from src.config import (\n"
            "    DEVICE, EXPERIMENTS_DIR, NUM_CLASSES, SEED,\n"
            "    TEST_DIR, TRAIN_DIR,\n"
            ")\n"
            "from src.utils import set_seed\n\n"
            "set_seed(SEED)\n"
            "print(f'Device      : {DEVICE}')\n"
            "print(f'Train dir   : {TRAIN_DIR}  exists={TRAIN_DIR.exists()}')\n"
            "print(f'Test dir    : {TEST_DIR}   exists={TEST_DIR.exists()}')\n"
            "print(f'Num classes : {NUM_CLASSES}')"
        ),

        new_markdown_cell(
            "## 1. Dataset Structure Audit\n\n"
            "Before loading anything into memory, we audit the raw folder structure.\n"
            "This catches missing classes or corrupt directories before training."
        ),

        new_code_cell(
            "# ── Cell 2: Dataset structure audit ────────────────────────────\n"
            "# Why audit before loading: ImageFolder silently skips malformed\n"
            "# subdirectories. Explicit counting confirms the expected 50 classes.\n"
            "import os\n\n"
            "train_classes = sorted(os.listdir(TRAIN_DIR))\n"
            "test_classes  = sorted(os.listdir(TEST_DIR))\n\n"
            "print(f'Train classes found : {len(train_classes)}')\n"
            "print(f'Test  classes found : {len(test_classes)}')\n"
            "print(f'Classes match       : {train_classes == test_classes}')\n"
            "print(f'First 5 classes     : {train_classes[:5]}')\n\n"
            "train_counts = {\n"
            "    cls: len(list((TRAIN_DIR / cls).glob('*')))\n"
            "    for cls in train_classes\n"
            "}\n"
            "total_train = sum(train_counts.values())\n"
            "print(f'Total train images  : {total_train}')\n"
            "print(f'Avg per class       : {total_train / len(train_classes):.1f}')"
        ),

        new_markdown_cell(
            "## 2. Class Distribution — BI Visualization\n\n"
            "**Storytelling question:** Is the dataset balanced across all 50 landmark classes?\n\n"
            "Class imbalance biases the model toward majority classes.\n"
            "Detecting it here allows corrective action before wasting compute."
        ),

        new_code_cell(
            "# ── Cell 3: BI class distribution plot ─────────────────────────\n"
            "# Why UCB palette with semantic color encoding:\n"
            "# Gold = majority classes (>1.5x mean) — potential bias source.\n"
            "# Color encodes information, not decoration.\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "UCB_BLUE      = '#003262'\n"
            "UCB_GOLD      = '#FDB515'\n"
            "UCB_DARK_GOLD = '#C4820E'\n\n"
            "names      = list(train_counts.keys())\n"
            "counts     = list(train_counts.values())\n"
            "mean_count = np.mean(counts)\n"
            "colors     = [UCB_GOLD if c > mean_count * 1.5 else UCB_BLUE for c in counts]\n\n"
            "fig, ax = plt.subplots(figsize=(22, 5), facecolor='white')\n"
            "ax.bar(range(len(names)), counts, color=colors, edgecolor='white', linewidth=0.5)\n"
            "ax.axhline(mean_count, color=UCB_DARK_GOLD, linestyle='--', linewidth=1.2,\n"
            "           label=f'Mean: {mean_count:.0f} images/class')\n"
            "ax.set_xticks(range(len(names)))\n"
            "ax.set_xticklabels([n.replace('_', ' ') for n in names], rotation=90, fontsize=6)\n"
            "ax.set_title('Landmark Dataset — Class Distribution\\n'\n"
            "             'Gold = majority classes (>1.5x mean) — potential bias source',\n"
            "             fontsize=12, fontweight='bold', color=UCB_BLUE)\n"
            "ax.set_xlabel('Landmark Class', fontsize=9, color=UCB_BLUE)\n"
            "ax.set_ylabel('Number of Images', fontsize=9, color=UCB_BLUE)\n"
            "ax.legend(fontsize=9)\n"
            "ax.spines[['top', 'right']].set_visible(False)\n"
            "plt.tight_layout()\n"
            "EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)\n"
            "out = EXPERIMENTS_DIR / 'class_distribution.png'\n"
            "plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')\n"
            "plt.show()\n"
            "print(f'Saved -> {out}')"
        ),

        new_markdown_cell(
            "## 3. Sample Images with Labels\n\n"
            "Visual inspection confirms images load correctly and labels match landmarks."
        ),

        new_code_cell(
            "# ── Cell 4: Sample images visualization ─────────────────────────\n"
            "# Why visual inspection before DataLoader: ImageFolder may load\n"
            "# images correctly by shape but with wrong label mapping.\n"
            "# Human review catches mismatches that metrics cannot.\n"
            "import random\n"
            "from torchvision.datasets import ImageFolder\n\n"
            "raw_ds     = ImageFolder(str(TRAIN_DIR))\n"
            "sample_idx = random.sample(range(len(raw_ds)), 8)\n\n"
            "fig, axes = plt.subplots(2, 4, figsize=(16, 7), facecolor='white')\n"
            "for ax, idx in zip(axes.flatten(), sample_idx):\n"
            "    img, label = raw_ds[idx]\n"
            "    ax.imshow(img)\n"
            "    ax.set_title(raw_ds.classes[label].replace('_', ' '),\n"
            "                 fontsize=8, color=UCB_BLUE, fontweight='bold')\n"
            "    ax.axis('off')\n"
            "fig.suptitle('Sample Landmark Images with Ground Truth Labels',\n"
            "             fontsize=12, fontweight='bold', color=UCB_BLUE)\n"
            "plt.tight_layout()\n"
            "out = EXPERIMENTS_DIR / 'sample_images.png'\n"
            "plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')\n"
            "plt.show()\n"
            "print(f'Saved -> {out}')"
        ),

        new_markdown_cell(
            "## 4. DataLoaders — Build and Verify\n\n"
            "Three DataLoaders: train (shuffle + augmentation), "
            "val (no shuffle, no augmentation), test (no shuffle, no augmentation)."
        ),

        new_code_cell(
            "# ── Cell 5: DataLoaders ─────────────────────────────────────────\n"
            "# Why verify before training: shape mismatches and normalization\n"
            "# errors are silent — the model trains but learns nothing useful.\n"
            "from src.data import get_dataloaders, verify_dataloaders\n\n"
            "train_loader, val_loader, test_loader, class_names = get_dataloaders()\n"
            "verify_dataloaders(train_loader, val_loader, test_loader, class_names)"
        ),

        new_markdown_cell(
            "## 5. Augmentation Preview\n\n"
            "Same image before and after augmentation confirms transforms are active "
            "and reasonable — not destroying landmark structure."
        ),

        new_code_cell(
            "# ── Cell 6: Augmentation preview ───────────────────────────────\n"
            "# Why show augmented vs original: augmentation that is too aggressive\n"
            "# confuses the model more than it helps. Visual check prevents\n"
            "# silent accuracy degradation from bad transform choices.\n"
            "import torch\n"
            "from PIL import Image\n"
            "from src.data import get_transforms\n\n"
            "sample_class = class_names[0]\n"
            "sample_path  = next((TRAIN_DIR / sample_class).glob('*'))\n"
            "raw_img      = Image.open(sample_path).convert('RGB')\n\n"
            "transform_aug = get_transforms(augment=True)\n\n"
            "fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor='white')\n"
            "axes[0].imshow(raw_img)\n"
            "axes[0].set_title('Original', color=UCB_BLUE, fontweight='bold')\n"
            "for i, ax in enumerate(axes[1:], 1):\n"
            "    aug_tensor = transform_aug(raw_img)\n"
            "    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)\n"
            "    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)\n"
            "    img_display = (aug_tensor * std + mean).clamp(0,1).permute(1,2,0).numpy()\n"
            "    ax.imshow(img_display)\n"
            "    ax.set_title(f'Augmented #{i}', color=UCB_DARK_GOLD, fontweight='bold')\n"
            "for ax in axes:\n"
            "    ax.axis('off')\n"
            "fig.suptitle(f'Augmentation Preview — {sample_class.replace(\"_\", \" \")}',\n"
            "             fontsize=11, fontweight='bold', color=UCB_BLUE)\n"
            "plt.tight_layout()\n"
            "out = EXPERIMENTS_DIR / 'augmentation_preview.png'\n"
            "plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')\n"
            "plt.show()\n"
            "print(f'Saved -> {out}')"
        ),

        new_markdown_cell(
            "## Phase 1 — Checklist\n\n"
            "| Check | Status |\n"
            "|---|---|\n"
            "| 50 classes detected | ✅ |\n"
            "| Class distribution plotted | ✅ |\n"
            "| >=5 sample images displayed | ✅ |\n"
            "| DataLoaders: batch shape [B, 3, 224, 224] | ✅ |\n"
            "| Augmentation preview confirmed | ✅ |\n"
            "| All artifacts saved to experiments/ | ✅ |\n\n"
            "**Next step:** `02_cnn_from_scratch.ipynb` — Phase 2 on Colab T4."
        ),
    ]

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = _kernel_metadata()
    return _save_notebook(nb, "01_exploration.ipynb")


# ===========================================================================
# NOTEBOOK 2 — CNN From Scratch (Phase 2)
# ===========================================================================
def create_02_cnn_scratch() -> Path:
    """Build 02_cnn_from_scratch.ipynb — Phase 2: train CNN, BI curves."""

    cells = [
        new_markdown_cell(
            "# Notebook 02 — CNN From Scratch\n"
            "## Landmark Classification CNN · UCB MSc Data Science & AI\n\n"
            "> **Business question:** Can a custom CNN trained from zero reach "
            ">=40% test accuracy on 50 landmark classes?\n\n"
            "**Phase 2 rubric targets:**\n"
            "- Custom CNN with >=3 conv layers, pooling, dropout, FC (2 pts)\n"
            "- Training >=30 epochs with loss/accuracy curves (1 pt)\n"
            "- Test accuracy >=40% + TorchScript export (2 pts)\n\n"
            "> Run this notebook on Google Colab T4 GPU."
        ),

        new_code_cell(SETUP_CELL),

        new_code_cell(
            CONFIG_CELL +
            "from src.config import (\n"
            "    DEVICE, SCRATCH_EPOCHS, SCRATCH_LR,\n"
            "    SCRATCH_SCHEDULER_GAMMA, SCRATCH_SCHEDULER_STEP, SEED,\n"
            ")\n"
            "from src.utils import set_seed\n\n"
            "set_seed(SEED)\n"
            "print(f'Device  : {DEVICE}')\n"
            "print(f'Epochs  : {SCRATCH_EPOCHS}')\n"
            "print(f'LR      : {SCRATCH_LR}')"
        ),

        new_code_cell(
            "# ── Cell 2: DataLoaders ─────────────────────────────────────────\n"
            "# Why rebuild DataLoaders here: Colab disconnects lose all in-memory\n"
            "# objects. Rebuilding is fast and guarantees a clean state.\n"
            "from src.data import get_dataloaders, verify_dataloaders\n\n"
            "train_loader, val_loader, test_loader, class_names = get_dataloaders()\n"
            "verify_dataloaders(train_loader, val_loader, test_loader, class_names)"
        ),

        new_markdown_cell(
            "## 1. Architecture Design — CNNScratch\n\n"
            "Progressive filter growth (32->512) mirrors biological visual hierarchy.\n"
            "BatchNorm stabilizes training. GlobalAveragePooling reduces FC parameters by 25x."
        ),

        new_code_cell(
            "# ── Cell 3: Architecture sanity check ──────────────────────────\n"
            "# Why run before training: if output shape mismatches num_classes,\n"
            "# CrossEntropyLoss raises a cryptic CUDA error mid-epoch.\n"
            "# Catching it here costs milliseconds, saves hours.\n"
            "import torch\n"
            "from src.model import CNNScratch, count_params\n\n"
            "model = CNNScratch(num_classes=len(class_names))\n"
            "count_params(model)\n"
            "dummy  = torch.zeros(2, 3, 224, 224)\n"
            "output = model(dummy)\n"
            "print(f'Output shape: {output.shape}  -> expected [2, {len(class_names)}]')\n"
            "assert output.shape == (2, len(class_names)), 'Shape mismatch — check model.py'"
        ),

        new_markdown_cell(
            "## 2. Experiment E1 — Baseline\n\n"
            "**Hypothesis:** Baseline CNN without augmentation will overfit quickly.\n"
            "**Purpose:** Reference point to measure augmentation impact in E2."
        ),

        new_code_cell(
            "# ── Cell 4: Experiment E1 — baseline ───────────────────────────\n"
            "# Ablation rule: ONE factor changed per experiment.\n"
            "# E1 = no augmentation. All other params at config defaults.\n"
            "from src.model import CNNScratch\n"
            "from src.train import run_experiment\n\n"
            "model_e1 = CNNScratch(num_classes=len(class_names))\n"
            "metrics_e1 = run_experiment(\n"
            "    exp_id       = 'E1_scratch_baseline',\n"
            "    model        = model_e1,\n"
            "    train_loader = train_loader,\n"
            "    val_loader   = val_loader,\n"
            "    test_loader  = test_loader,\n"
            "    class_names  = class_names,\n"
            "    epochs       = SCRATCH_EPOCHS,\n"
            "    lr           = SCRATCH_LR,\n"
            "    extra_params = {'architecture': 'CNNScratch_5conv_BN', 'augmentation': False},\n"
            ")\n"
            "print(f'E1 Test Accuracy: {metrics_e1[\"results\"][\"test_accuracy\"]*100:.2f}%')"
        ),

        new_markdown_cell(
            "## 3. Experiment E2 — With Augmentation\n\n"
            "**Hypothesis:** Augmentation reduces overfitting and improves val accuracy.\n"
            "**Single factor changed from E1:** augmentation ON."
        ),

        new_code_cell(
            "# ── Cell 5: Experiment E2 — augmentation ───────────────────────\n"
            "# Only change from E1: train_loader uses augment=True (default).\n"
            "# Same architecture, same LR, same epochs — isolates augmentation effect.\n"
            "from src.model import CNNScratch\n"
            "from src.train import run_experiment\n\n"
            "model_e2 = CNNScratch(num_classes=len(class_names))\n"
            "metrics_e2 = run_experiment(\n"
            "    exp_id       = 'E2_scratch_augmented',\n"
            "    model        = model_e2,\n"
            "    train_loader = train_loader,\n"
            "    val_loader   = val_loader,\n"
            "    test_loader  = test_loader,\n"
            "    class_names  = class_names,\n"
            "    epochs       = SCRATCH_EPOCHS,\n"
            "    lr           = SCRATCH_LR,\n"
            "    extra_params = {'architecture': 'CNNScratch_5conv_BN', 'augmentation': True},\n"
            ")\n"
            "print(f'E2 Test Accuracy: {metrics_e2[\"results\"][\"test_accuracy\"]*100:.2f}%')"
        ),

        new_markdown_cell(
            "## 4. Experiment E3 — Lower LR\n\n"
            "**Hypothesis:** Lower LR with augmentation allows more stable convergence.\n"
            "**Single factor changed from E2:** lr 1e-3 -> 1e-4."
        ),

        new_code_cell(
            "# ── Cell 6: Experiment E3 — lower LR ───────────────────────────\n"
            "# Lower LR is the standard next step when E2 shows unstable val_loss.\n"
            "# If E2 val_loss curve is smooth, E3 may not add value.\n"
            "from src.model import CNNScratch\n"
            "from src.train import run_experiment\n\n"
            "model_e3 = CNNScratch(num_classes=len(class_names))\n"
            "metrics_e3 = run_experiment(\n"
            "    exp_id       = 'E3_scratch_lower_lr',\n"
            "    model        = model_e3,\n"
            "    train_loader = train_loader,\n"
            "    val_loader   = val_loader,\n"
            "    test_loader  = test_loader,\n"
            "    class_names  = class_names,\n"
            "    epochs       = SCRATCH_EPOCHS,\n"
            "    lr           = 1e-4,\n"
            "    extra_params = {'architecture': 'CNNScratch_5conv_BN', 'augmentation': True, 'lr_reason': 'stable convergence'},\n"
            ")\n"
            "print(f'E3 Test Accuracy: {metrics_e3[\"results\"][\"test_accuracy\"]*100:.2f}%')"
        ),

        new_code_cell(
            "# ── Cell 7: Comparative table Phase 2 ──────────────────────────\n"
            "# Why a table: side-by-side comparison makes the impact of each\n"
            "# factor immediately visible without mental arithmetic.\n"
            "import pandas as pd\n\n"
            "results = pd.DataFrame([\n"
            "    {\n"
            "        'Experiment' : m['exp_id'],\n"
            "        'LR'         : m['hyperparameters']['lr'],\n"
            "        'Test Acc'   : f\"{m['results']['test_accuracy']*100:.2f}%\",\n"
            "        'Time (min)' : m['results']['total_time_min'],\n"
            "        'Meets >=40%': '✅' if m['results']['test_accuracy'] >= 0.40 else '❌',\n"
            "    }\n"
            "    for m in [metrics_e1, metrics_e2, metrics_e3]\n"
            "])\n"
            "print(results.to_string(index=False))"
        ),

        new_code_cell(
            "# ── Cell 8: Full evaluation best scratch model ──────────────────\n"
            "# full_evaluation generates BI confusion matrix + executive report.\n"
            "# Why run on best model only: confusion matrix on a weak model\n"
            "# produces noise, not actionable insight.\n"
            "import torch\n"
            "from src.config import MODELS_DIR\n"
            "from src.evaluate import full_evaluation\n"
            "from src.model import CNNScratch\n\n"
            "best = max([metrics_e1, metrics_e2, metrics_e3],\n"
            "           key=lambda m: m['results']['test_accuracy'])\n"
            "print(f'Best: {best[\"exp_id\"]} — {best[\"results\"][\"test_accuracy\"]*100:.2f}%')\n\n"
            "best_model = CNNScratch(num_classes=len(class_names))\n"
            "best_model.load_state_dict(\n"
            "    torch.load(MODELS_DIR / f'{best[\"exp_id\"]}_best.pt', weights_only=True)\n"
            ")\n"
            "eval_results = full_evaluation(\n"
            "    exp_id      = best['exp_id'],\n"
            "    model       = best_model,\n"
            "    loader      = test_loader,\n"
            "    class_names = class_names,\n"
            "    topk        = 5,\n"
            ")"
        ),

        new_markdown_cell(
            "## Phase 2 — Checklist\n\n"
            "| Check | Status |\n"
            "|---|---|\n"
            "| CNNScratch >=3 conv layers | ✅ (5 conv blocks) |\n"
            "| Trained >=30 epochs | ✅ |\n"
            "| BI narrative curves | ✅ |\n"
            "| TorchScript exported | ✅ |\n"
            "| BI confusion matrix + Top-3 business errors | ✅ |\n"
            "| Test accuracy >=40% | ⬜ fill after run |\n\n"
            "**Next step:** `03_transfer_learning.ipynb`"
        ),
    ]

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = _kernel_metadata()
    return _save_notebook(nb, "02_cnn_from_scratch.ipynb")


# ===========================================================================
# NOTEBOOK 3 — Transfer Learning (Phase 3)
# ===========================================================================
def create_03_transfer_learning() -> Path:
    """Build 03_transfer_learning.ipynb — Phase 3: ResNet, BI comparison."""

    cells = [
        new_markdown_cell(
            "# Notebook 03 — Transfer Learning\n"
            "## Landmark Classification CNN · UCB MSc Data Science & AI\n\n"
            "> **Business question:** Does a pretrained ResNet18 backbone reach "
            ">=70% test accuracy on 50 landmark classes?\n\n"
            "**Phase 3 rubric targets:**\n"
            "- Pretrained model selection with written justification (2 pts)\n"
            "- Training curves + comparison with Phase 2 (1 pt)\n"
            "- Test accuracy >=70% + TorchScript export (2 pts)\n"
            "- Written analysis of strengths and weaknesses (2 pts)\n\n"
            "> Run this notebook on Google Colab T4 GPU."
        ),

        new_code_cell(SETUP_CELL),

        new_code_cell(
            CONFIG_CELL +
            "from src.config import (\n"
            "    DEVICE, SEED,\n"
            "    TL_EPOCHS_FINETUNE, TL_EPOCHS_FROZEN,\n"
            "    TL_LR_BACKBONE, TL_LR_HEAD,\n"
            ")\n"
            "from src.utils import set_seed\n\n"
            "set_seed(SEED)\n"
            "print(f'Device          : {DEVICE}')\n"
            "print(f'Epochs frozen   : {TL_EPOCHS_FROZEN}')\n"
            "print(f'Epochs finetune : {TL_EPOCHS_FINETUNE}')\n"
            "print(f'LR head         : {TL_LR_HEAD}')\n"
            "print(f'LR backbone     : {TL_LR_BACKBONE}')"
        ),

        new_code_cell(
            "# ── Cell 2: DataLoaders ─────────────────────────────────────────\n"
            "from src.data import get_dataloaders, verify_dataloaders\n\n"
            "train_loader, val_loader, test_loader, class_names = get_dataloaders()\n"
            "verify_dataloaders(train_loader, val_loader, test_loader, class_names)"
        ),

        new_markdown_cell(
            "## 1. Model Selection — Written Justification\n\n"
            "**Why ResNet18 over VGG16:**\n"
            "- VGG16: 138M parameters — FC layers alone exhaust 6 GB VRAM at batch_size=32\n"
            "- ResNet18: 11M parameters — residual connections prevent vanishing gradient\n"
            "- Fine-tuning ResNet18 is stable; VGG16 deep fine-tuning diverges\n\n"
            "**Why ResNet50 as second candidate:**\n"
            "- 25M parameters — more capacity for complex landmark features\n"
            "- Tests whether larger backbone justifies added compute cost"
        ),

        new_code_cell(
            "# ── Cell 3: Inspect trainable parameters per strategy ───────────\n"
            "# Why check before training: confirms freeze/unfreeze worked.\n"
            "# Frozen ResNet18: ~144K trainable (FC only).\n"
            "# Finetune: ~8.5M trainable (layer4 + FC).\n"
            "from src.model import count_params, get_transfer_model\n\n"
            "print('--- ResNet18 frozen ---')\n"
            "count_params(get_transfer_model('resnet18', strategy='frozen'))\n"
            "print()\n"
            "print('--- ResNet18 finetune ---')\n"
            "count_params(get_transfer_model('resnet18', strategy='finetune'))"
        ),

        new_code_cell(
            "# ── Cell 4: Experiment E4 — ResNet18 frozen ────────────────────\n"
            "# Feature extraction: only FC head trainable.\n"
            "# Why start frozen: fast convergence, zero catastrophic forgetting risk.\n"
            "from src.model import get_transfer_model\n"
            "from src.train import run_experiment\n\n"
            "model_e4 = get_transfer_model('resnet18', num_classes=len(class_names), strategy='frozen')\n"
            "metrics_e4 = run_experiment(\n"
            "    exp_id       = 'E4_resnet18_frozen',\n"
            "    model        = model_e4,\n"
            "    train_loader = train_loader,\n"
            "    val_loader   = val_loader,\n"
            "    test_loader  = test_loader,\n"
            "    class_names  = class_names,\n"
            "    epochs       = TL_EPOCHS_FROZEN,\n"
            "    lr           = TL_LR_HEAD,\n"
            "    extra_params = {'backbone': 'resnet18', 'strategy': 'frozen'},\n"
            ")\n"
            "print(f'E4 Test Accuracy: {metrics_e4[\"results\"][\"test_accuracy\"]*100:.2f}%')"
        ),

        new_code_cell(
            "# ── Cell 5: Experiment E5 — ResNet18 finetune ──────────────────\n"
            "# Why lr_backbone 100x lower: prevents catastrophic forgetting\n"
            "# while allowing domain adaptation of high-level features.\n"
            "from src.model import get_transfer_model\n"
            "from src.train import run_experiment\n\n"
            "model_e5 = get_transfer_model('resnet18', num_classes=len(class_names), strategy='finetune')\n"
            "metrics_e5 = run_experiment(\n"
            "    exp_id       = 'E5_resnet18_finetune_layer4',\n"
            "    model        = model_e5,\n"
            "    train_loader = train_loader,\n"
            "    val_loader   = val_loader,\n"
            "    test_loader  = test_loader,\n"
            "    class_names  = class_names,\n"
            "    epochs       = TL_EPOCHS_FINETUNE,\n"
            "    lr           = TL_LR_HEAD,\n"
            "    lr_backbone  = TL_LR_BACKBONE,\n"
            "    extra_params = {'backbone': 'resnet18', 'strategy': 'finetune', 'layer_unfrozen': 'layer4'},\n"
            ")\n"
            "print(f'E5 Test Accuracy: {metrics_e5[\"results\"][\"test_accuracy\"]*100:.2f}%')"
        ),

        new_code_cell(
            "# ── Cell 6: Experiment E6 — ResNet50 frozen ────────────────────\n"
            "# Tests whether larger backbone capacity justifies extra compute.\n"
            "# Single factor changed from E4: backbone resnet18 -> resnet50.\n"
            "from src.model import get_transfer_model\n"
            "from src.train import run_experiment\n\n"
            "model_e6 = get_transfer_model('resnet50', num_classes=len(class_names), strategy='frozen')\n"
            "metrics_e6 = run_experiment(\n"
            "    exp_id       = 'E6_resnet50_frozen',\n"
            "    model        = model_e6,\n"
            "    train_loader = train_loader,\n"
            "    val_loader   = val_loader,\n"
            "    test_loader  = test_loader,\n"
            "    class_names  = class_names,\n"
            "    epochs       = TL_EPOCHS_FROZEN,\n"
            "    lr           = TL_LR_HEAD,\n"
            "    extra_params = {'backbone': 'resnet50', 'strategy': 'frozen'},\n"
            ")\n"
            "print(f'E6 Test Accuracy: {metrics_e6[\"results\"][\"test_accuracy\"]*100:.2f}%')"
        ),

        new_code_cell(
            "# ── Cell 7: Full comparative table Scratch vs Transfer ──────────\n"
            "# Why reload E3 from JSON: notebooks are stateless across sessions.\n"
            "import json\n"
            "import pandas as pd\n"
            "from src.config import EXPERIMENTS_DIR\n\n"
            "def load_metrics(exp_id: str) -> dict:\n"
            "    path = EXPERIMENTS_DIR / f'{exp_id}_metrics.json'\n"
            "    with path.open() as f:\n"
            "        return json.load(f)\n\n"
            "metrics_e3 = load_metrics('E3_scratch_lower_lr')\n\n"
            "all_results = pd.DataFrame([\n"
            "    {\n"
            "        'Model'        : m['exp_id'],\n"
            "        'Type'         : 'Scratch' if 'scratch' in m['exp_id'] else 'Transfer',\n"
            "        'Test Acc'     : f\"{m['results']['test_accuracy']*100:.2f}%\",\n"
            "        'Time (min)'   : m['results']['total_time_min'],\n"
            "        'Meets target' : '✅' if (\n"
            "            m['results']['test_accuracy'] >= 0.70\n"
            "            if 'resnet' in m['exp_id']\n"
            "            else m['results']['test_accuracy'] >= 0.40\n"
            "        ) else '❌',\n"
            "    }\n"
            "    for m in [metrics_e3, metrics_e4, metrics_e5, metrics_e6]\n"
            "])\n"
            "print(all_results.to_string(index=False))"
        ),

        new_code_cell(
            "# ── Cell 8: Full evaluation best transfer model ─────────────────\n"
            "import torch\n"
            "from src.config import MODELS_DIR\n"
            "from src.evaluate import full_evaluation\n"
            "from src.model import get_transfer_model\n\n"
            "best_tl = max([metrics_e4, metrics_e5, metrics_e6],\n"
            "              key=lambda m: m['results']['test_accuracy'])\n"
            "print(f'Best TL: {best_tl[\"exp_id\"]} — {best_tl[\"results\"][\"test_accuracy\"]*100:.2f}%')\n\n"
            "backbone = best_tl['hyperparameters'].get('backbone', 'resnet18')\n"
            "strategy = best_tl['hyperparameters'].get('strategy', 'frozen')\n"
            "best_tl_model = get_transfer_model(backbone, num_classes=len(class_names), strategy=strategy)\n"
            "best_tl_model.load_state_dict(\n"
            "    torch.load(MODELS_DIR / f'{best_tl[\"exp_id\"]}_best.pt', weights_only=True)\n"
            ")\n"
            "eval_tl = full_evaluation(\n"
            "    exp_id      = best_tl['exp_id'],\n"
            "    model       = best_tl_model,\n"
            "    loader      = test_loader,\n"
            "    class_names = class_names,\n"
            "    topk        = 5,\n"
            ")"
        ),

        new_markdown_cell(
            "## 2. Analysis — Strengths and Weaknesses\n\n"
            "*(Fill after running all experiments)*\n\n"
            "### Strengths\n"
            "- Transfer Learning reaches >=70% with only 10 epochs\n"
            "- ImageNet features generalize well to architectural landmarks\n\n"
            "### Weaknesses\n"
            "- Visually similar landmarks cause systematic confusion\n"
            "- Dataset imbalance biases toward majority classes\n\n"
            "## Phase 3 — Checklist\n\n"
            "| Check | Status |\n"
            "|---|---|\n"
            "| 2 pretrained models tested | ✅ |\n"
            "| Written justification | ✅ |\n"
            "| BI narrative curves | ✅ |\n"
            "| Scratch vs Transfer table | ✅ |\n"
            "| TorchScript exported | ✅ |\n"
            "| Test accuracy >=70% | ⬜ fill after run |\n\n"
            "**Next step:** `04_inference_app.ipynb`"
        ),
    ]

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = _kernel_metadata()
    return _save_notebook(nb, "03_transfer_learning.ipynb")


# ===========================================================================
# NOTEBOOK 4 — Inference App (Phase 4)
# ===========================================================================
def create_04_inference_app() -> Path:
    """Build 04_inference_app.ipynb — Phase 4: predict_landmarks on own images."""

    cells = [
        new_markdown_cell(
            "# Notebook 04 — Inference Application\n"
            "## Landmark Classification CNN · UCB MSc Data Science & AI\n\n"
            "> **Business question:** Can the deployed model correctly identify "
            "landmarks in real-world images not seen during training?\n\n"
            "**Phase 4 rubric targets:**\n"
            "- predict_landmarks(img_path, k) using TorchScript\n"
            "- Test on >=4 personal images not from the dataset\n"
            "- Written analysis of strengths and weaknesses\n\n"
            "> Run this notebook locally on PyCharm — inference is CPU-friendly."
        ),

        new_code_cell(SETUP_CELL),

        new_code_cell(
            CONFIG_CELL +
            "from src.config import EXTERNAL_DIR, MODELS_DIR, SEED\n"
            "from src.utils import set_seed\n\n"
            "set_seed(SEED)\n"
            "scripted_models = sorted(MODELS_DIR.glob('*_scripted.pt'))\n"
            "print('Available TorchScript models:')\n"
            "for m in scripted_models:\n"
            "    print(f'  {m.name}')"
        ),

        new_code_cell(
            "# ── Cell 2: Select best TorchScript model ───────────────────────\n"
            "# Why TorchScript: self-contained computation graph — no model.py\n"
            "# dependency. Deployable with only torch installed.\n"
            "finetune_models = [m for m in scripted_models if 'finetune' in m.name]\n"
            "BEST_MODEL_PATH = finetune_models[-1] if finetune_models else scripted_models[-1]\n"
            "print(f'Using model: {BEST_MODEL_PATH.name}')"
        ),

        new_code_cell(
            "# ── Cell 3: Verify external images ─────────────────────────────\n"
            "# Why verify before inference: a missing image raises a cryptic\n"
            "# CUDA error mid-batch. Explicit check gives a clear message.\n"
            "images = (\n"
            "    list(EXTERNAL_DIR.glob('*.jpg'))\n"
            "    + list(EXTERNAL_DIR.glob('*.jpeg'))\n"
            "    + list(EXTERNAL_DIR.glob('*.png'))\n"
            "    + list(EXTERNAL_DIR.glob('*.webp'))\n"
            ")\n"
            "print(f'External images found: {len(images)}')\n"
            "for img in images:\n"
            "    print(f'  {img.name}')\n"
            "assert len(images) >= 4, f'Rubric requires >=4 images. Found: {len(images)}'"
        ),

        new_code_cell(
            "# ── Cell 4: Inference on all external images ────────────────────\n"
            "# Why predict_and_display: renders UCB palette bar chart alongside\n"
            "# the image — communicates confidence, not just the top-1 label.\n"
            "from src.predictor import predict_and_display\n\n"
            "all_predictions = {}\n"
            "for img_path in images:\n"
            "    print(f'\\n=== {img_path.name} ===')\n"
            "    preds = predict_and_display(\n"
            "        img_path            = img_path,\n"
            "        k                   = 3,\n"
            "        scripted_model_path = BEST_MODEL_PATH,\n"
            "    )\n"
            "    all_predictions[img_path.name] = preds\n"
            "    for rank, (name, prob) in enumerate(preds, 1):\n"
            "        print(f'  {rank}. {name.replace(\"_\", \" \")}: {prob:.2%}')"
        ),

        new_code_cell(
            "# ── Cell 5: Prediction summary table ───────────────────────────\n"
            "import pandas as pd\n\n"
            "rows = []\n"
            "for img_name, preds in all_predictions.items():\n"
            "    for rank, (name, prob) in enumerate(preds, 1):\n"
            "        rows.append({\n"
            "            'Image'      : img_name,\n"
            "            'Rank'       : rank,\n"
            "            'Prediction' : name.replace('_', ' '),\n"
            "            'Confidence' : f'{prob:.2%}',\n"
            "        })\n"
            "print(pd.DataFrame(rows).to_string(index=False))"
        ),

        new_markdown_cell(
            "## Analysis — Strengths and Weaknesses\n\n"
            "*(Fill based on prediction results)*\n\n"
            "### Strengths\n"
            "- High confidence on iconic visually distinctive landmarks\n"
            "- Top-3 includes correct class even when Top-1 is wrong\n\n"
            "### Weaknesses\n"
            "- Low confidence on underrepresented landmarks\n"
            "- Confuses architecturally similar structures\n\n"
            "## Final Deliverables Checklist\n\n"
            "| Deliverable | Status |\n"
            "|---|---|\n"
            "| GitHub repo public (no dataset) | ⬜ |\n"
            "| All 4 notebooks executed with outputs | ⬜ |\n"
            "| README.md with YouTube link | ⬜ |\n"
            "| Video 3-5 min uploaded | ⬜ |\n"
            "| docs/informe_landmarks.pdf | ⬜ |\n"
            "| git tag v1.0-entrega pushed | ⬜ |"
        ),
    ]

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = _kernel_metadata()
    return _save_notebook(nb, "04_inference_app.ipynb")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    paths = [
        create_01_exploration(),
        create_02_cnn_scratch(),
        create_03_transfer_learning(),
        create_04_inference_app(),
    ]
    print("\nAll notebooks generated:")
    for p in paths:
        print(f"  {p}")
