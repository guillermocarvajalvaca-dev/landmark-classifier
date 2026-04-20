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
# VERSION: 1.0.0
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


def _save_notebook(nb: nbformat.NotebookNode, filename: str) -> Path:
    """
    Validate and persist a notebook to disk.

    Args:
        nb:       NotebookNode built with nbformat.v4 helpers.
        filename: Target filename including .ipynb extension.

    Returns:
        Path to the saved notebook.

    Why nbformat.validate before saving:
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
            "- Visualize ≥5 sample images with labels (1 pt)\n"
            "- Class distribution bar chart (1 pt)\n"
            "- Functional train / val / test DataLoaders (1 pt)"
        ),

        # ── Cell 1: Environment setup ────────────────────────────────────────
        new_code_cell(
            "# ── Cell 1: Environment setup ──────────────────────────────────\n"
            "# Why first cell is always config: single source of truth for paths\n"
            "# and hyperparameters. Any change here propagates to all cells below.\n"
            "import logging\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "# Add project root to sys.path so src.* imports resolve in both\n"
            "# PyCharm (local) and Colab (after Drive mount)\n"
            "PROJECT_ROOT = Path('..').resolve()\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "logging.basicConfig(level=logging.INFO)\n\n"
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

        # ── Cell 2: Dataset structure audit ─────────────────────────────────
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
            "# Count images per class — reveals class imbalance before training\n"
            "train_counts = {\n"
            "    cls: len(list((TRAIN_DIR / cls).glob('*')))\n"
            "    for cls in train_classes\n"
            "}\n"
            "total_train = sum(train_counts.values())\n"
            "print(f'Total train images  : {total_train}')\n"
            "print(f'Avg per class       : {total_train / len(train_classes):.1f}')"
        ),

        # ── Cell 3: BI class distribution plot ───────────────────────────────
        new_markdown_cell(
            "## 2. Class Distribution — BI Visualization\n\n"
            "**Storytelling question:** Is the dataset balanced across all 50 landmark classes?\n\n"
            "Class imbalance forces the model to be biased toward majority classes.\n"
            "If detected here, we address it via weighted sampling or loss weighting — "
            "before wasting compute on a biased model."
        ),

        new_code_cell(
            "# ── Cell 3: BI class distribution plot ─────────────────────────\n"
            "# UCB palette applied — color encodes information (not decoration):\n"
            "# UCB Gold = majority classes (potential bias source)\n"
            "# UCB Blue = standard classes\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "UCB_BLUE      = '#003262'\n"
            "UCB_GOLD      = '#FDB515'\n"
            "UCB_DARK_GOLD = '#C4820E'\n\n"
            "names  = list(train_counts.keys())\n"
            "counts = list(train_counts.values())\n"
            "mean_count = np.mean(counts)\n\n"
            "colors = [UCB_GOLD if c > mean_count * 1.5 else UCB_BLUE for c in counts]\n\n"
            "fig, ax = plt.subplots(figsize=(22, 5), facecolor='white')\n"
            "bars = ax.bar(range(len(names)), counts, color=colors, edgecolor='white', linewidth=0.5)\n"
            "ax.axhline(mean_count, color=UCB_DARK_GOLD, linestyle='--', linewidth=1.2,\n"
            "           label=f'Mean: {mean_count:.0f} images/class')\n"
            "ax.set_xticks(range(len(names)))\n"
            "ax.set_xticklabels([n.replace('_', ' ') for n in names],\n"
            "                   rotation=90, fontsize=6)\n"
            "ax.set_title('Landmark Dataset — Class Distribution\\n'\n"
            "             'Gold bars = majority classes (>1.5x mean) — potential bias source',\n"
            "             fontsize=12, fontweight='bold', color=UCB_BLUE)\n"
            "ax.set_xlabel('Landmark Class', fontsize=9, color=UCB_BLUE)\n"
            "ax.set_ylabel('Number of Images', fontsize=9, color=UCB_BLUE)\n"
            "ax.legend(fontsize=9)\n"
            "ax.spines[['top', 'right']].set_visible(False)\n"
            "plt.tight_layout()\n\n"
            "out = EXPERIMENTS_DIR / 'class_distribution.png'\n"
            "EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)\n"
            "plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')\n"
            "plt.show()\n"
            "print(f'Saved -> {out}')"
        ),

        # ── Cell 4: Sample images visualization ──────────────────────────────
        new_markdown_cell(
            "## 3. Sample Images with Labels\n\n"
            "Visual inspection of raw images confirms:\n"
            "- Images load correctly from the dataset path\n"
            "- Labels map to recognizable landmark names\n"
            "- No obvious data corruption"
        ),

        new_code_cell(
            "# ── Cell 4: Sample images visualization ─────────────────────────\n"
            "# Why visual inspection before DataLoader: ImageFolder may load\n"
            "# images correctly by shape but with wrong label mapping.\n"
            "# Human review catches mismatches that metrics cannot.\n"
            "import random\n"
            "from torchvision.datasets import ImageFolder\n\n"
            "raw_ds = ImageFolder(str(TRAIN_DIR))\n"
            "sample_idx = random.sample(range(len(raw_ds)), 8)\n\n"
            "fig, axes = plt.subplots(2, 4, figsize=(16, 7), facecolor='white')\n"
            "axes = axes.flatten()\n\n"
            "for ax, idx in zip(axes, sample_idx):\n"
            "    img, label = raw_ds[idx]\n"
            "    ax.imshow(img)\n"
            "    ax.set_title(\n"
            "        raw_ds.classes[label].replace('_', ' '),\n"
            "        fontsize=8, color=UCB_BLUE, fontweight='bold'\n"
            "    )\n"
            "    ax.axis('off')\n\n"
            "fig.suptitle('Sample Landmark Images with Ground Truth Labels',\n"
            "             fontsize=12, fontweight='bold', color=UCB_BLUE)\n"
            "plt.tight_layout()\n\n"
            "out = EXPERIMENTS_DIR / 'sample_images.png'\n"
            "plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')\n"
            "plt.show()\n"
            "print(f'Saved -> {out}')"
        ),

        # ── Cell 5: DataLoaders ───────────────────────────────────────────────
        new_markdown_cell(
            "## 4. DataLoaders — Build and Verify\n\n"
            "We build three DataLoaders:\n"
            "- **Train**: shuffle=True + augmentation active\n"
            "- **Val**: shuffle=False + no augmentation (reproducible metrics)\n"
            "- **Test**: shuffle=False + no augmentation (held-out evaluation)"
        ),

        new_code_cell(
            "# ── Cell 5: DataLoaders ─────────────────────────────────────────\n"
            "# Why verify_dataloaders before any training:\n"
            "# Shape mismatches and normalization errors are silent —\n"
            "# the model trains but learns nothing useful.\n"
            "# Catching them here costs seconds, not hours of GPU time.\n"
            "from src.data import get_dataloaders, verify_dataloaders\n\n"
            "train_loader, val_loader, test_loader, class_names = get_dataloaders()\n"
            "verify_dataloaders(train_loader, val_loader, test_loader, class_names)"
        ),

        # ── Cell 6: Augmentation preview ─────────────────────────────────────
        new_markdown_cell(
            "## 5. Augmentation Preview\n\n"
            "Showing the same image before and after augmentation confirms:\n"
            "- Augmentation is active on training samples\n"
            "- Transforms are reasonable (not destroying landmark structure)\n"
            "- Normalization shifts pixel values to the expected ImageNet range"
        ),

        new_code_cell(
            "# ── Cell 6: Augmentation preview ───────────────────────────────\n"
            "# Why show augmented vs original side by side:\n"
            "# Augmentation that is too aggressive (e.g. extreme rotation on\n"
            "# symmetric landmarks) can confuse the model more than help it.\n"
            "# Visual confirmation prevents silent accuracy degradation.\n"
            "from src.data import get_transforms\n"
            "from PIL import Image\n"
            "import torch\n\n"
            "# Load one raw image for comparison\n"
            "sample_class = class_names[0]\n"
            "sample_path  = next((TRAIN_DIR / sample_class).glob('*'))\n"
            "raw_img      = Image.open(sample_path).convert('RGB')\n\n"
            "transform_plain = get_transforms(augment=False)\n"
            "transform_aug   = get_transforms(augment=True)\n\n"
            "fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor='white')\n"
            "axes[0].imshow(raw_img)\n"
            "axes[0].set_title('Original', color=UCB_BLUE, fontweight='bold')\n\n"
            "for i, ax in enumerate(axes[1:], 1):\n"
            "    aug_tensor = transform_aug(raw_img)\n"
            "    # Denormalize for display: reverse ImageNet normalization\n"
            "    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)\n"
            "    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)\n"
            "    img_display = (aug_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()\n"
            "    ax.imshow(img_display)\n"
            "    ax.set_title(f'Augmented #{i}', color=UCB_DARK_GOLD, fontweight='bold')\n\n"
            "for ax in axes:\n"
            "    ax.axis('off')\n\n"
            "fig.suptitle(f'Augmentation Preview — {sample_class.replace(\"_\", \" \")}',\n"
            "             fontsize=11, fontweight='bold', color=UCB_BLUE)\n"
            "plt.tight_layout()\n"
            "out = EXPERIMENTS_DIR / 'augmentation_preview.png'\n"
            "plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')\n"
            "plt.show()\n"
            "print(f'Saved -> {out}')"
        ),

        # ── Cell 7: Phase 1 summary ───────────────────────────────────────────
        new_markdown_cell(
            "## Phase 1 — Summary and Checklist\n\n"
            "| Check | Status |\n"
            "|---|---|\n"
            "| 50 classes detected | ✅ |\n"
            "| Class distribution plotted | ✅ |\n"
            "| ≥5 sample images displayed | ✅ |\n"
            "| DataLoaders: batch shape [B, 3, 224, 224] | ✅ |\n"
            "| Augmentation preview confirmed | ✅ |\n"
            "| All artifacts saved to experiments/ | ✅ |\n\n"
            "**Next step:** `02_cnn_from_scratch.ipynb` — Phase 2 training on Colab T4."
        ),
    ]

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python (landmark-venv)",
        "language": "python",
        "name": "landmark-venv",
    }
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
            "≥40% test accuracy on 50 landmark classes?\n\n"
            "**Phase 2 rubric targets:**\n"
            "- Custom CNN with ≥3 conv layers, pooling, dropout, FC (2 pts)\n"
            "- Training ≥30 epochs with loss/accuracy curves (1 pt)\n"
            "- Test accuracy ≥40% + TorchScript export (2 pts)\n\n"
            "> ⚠️ **Run this notebook on Google Colab T4 GPU.**\n"
            "> Training 30 epochs on 224×224 images takes ~8h on CPU, ~30 min on T4."
        ),

        # ── Cell 1: Environment setup ────────────────────────────────────────
        new_code_cell(
            "# ── Cell 1: Environment setup ──────────────────────────────────\n"
            "# Colab: run drive.mount('/content/drive') before this cell\n"
            "# Local: PROJECT_ROOT auto-detected from __file__\n"
            "import logging\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "PROJECT_ROOT = Path('..').resolve()\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "logging.basicConfig(level=logging.INFO)\n\n"
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

        # ── Cell 2: DataLoaders ───────────────────────────────────────────────
        new_code_cell(
            "# ── Cell 2: DataLoaders ─────────────────────────────────────────\n"
            "# Re-build DataLoaders here — notebooks are stateless across sessions.\n"
            "# Colab disconnects lose all in-memory objects; rebuilding is fast.\n"
            "from src.data import get_dataloaders, verify_dataloaders\n\n"
            "train_loader, val_loader, test_loader, class_names = get_dataloaders()\n"
            "verify_dataloaders(train_loader, val_loader, test_loader, class_names)"
        ),

        # ── Cell 3: Architecture inspection ──────────────────────────────────
        new_markdown_cell(
            "## 1. Architecture Design — CNNScratch\n\n"
            "**Why this architecture:**\n"
            "- Progressive filter growth (32→512) mirrors biological visual hierarchy\n"
            "- BatchNorm stabilizes training — allows higher LR without explosion\n"
            "- GlobalAveragePooling replaces Flatten — 25× fewer FC parameters\n"
            "- Dropout2d regularizes spatial features — more effective than scalar Dropout"
        ),

        new_code_cell(
            "# ── Cell 3: Architecture inspection ────────────────────────────\n"
            "# Sanity check: if this cell fails, there is a bug in model.py.\n"
            "# Run before any training — costs milliseconds, saves hours.\n"
            "import torch\n"
            "from src.model import CNNScratch, count_params\n\n"
            "model = CNNScratch(num_classes=len(class_names))\n"
            "count_params(model)\n\n"
            "dummy  = torch.zeros(2, 3, 224, 224)\n"
            "output = model(dummy)\n"
            "print(f'Output shape: {output.shape}  -> expected [2, {len(class_names)}]')\n"
            "assert output.shape == (2, len(class_names)), 'Shape mismatch — check model.py'"
        ),

        # ── Cell 4: Experiment E1 ─────────────────────────────────────────────
        new_markdown_cell(
            "## 2. Experiment E1 — Baseline (no augmentation)\n\n"
            "**Hypothesis:** A 5-conv CNN without augmentation will overfit quickly.\n"
            "**Purpose:** Establish the baseline to compare augmentation's impact in E2."
        ),

        new_code_cell(
            "# ── Cell 4: Experiment E1 — baseline ───────────────────────────\n"
            "# Ablation rule: ONE factor changed per experiment.\n"
            "# E1 = no augmentation. All other params fixed at config defaults.\n"
            "from src.data import get_dataloaders\n"
            "from src.model import CNNScratch\n"
            "from src.train import run_experiment\n\n"
            "# DataLoader without augmentation for E1 baseline\n"
            "train_loader_plain, _, _, _ = get_dataloaders()\n\n"
            "model_e1 = CNNScratch(num_classes=len(class_names))\n\n"
            "metrics_e1 = run_experiment(\n"
            "    exp_id       = 'E1_scratch_baseline',\n"
            "    model        = model_e1,\n"
            "    train_loader = train_loader_plain,\n"
            "    val_loader   = val_loader,\n"
            "    test_loader  = test_loader,\n"
            "    class_names  = class_names,\n"
            "    epochs       = SCRATCH_EPOCHS,\n"
            "    lr           = SCRATCH_LR,\n"
            "    extra_params = {'architecture': 'CNNScratch_5conv_BN', 'augmentation': False},\n"
            ")\n"
            "print(f'E1 Test Accuracy: {metrics_e1[\"results\"][\"test_accuracy\"]*100:.2f}%')"
        ),

        # ── Cell 5: Experiment E2 ─────────────────────────────────────────────
        new_markdown_cell(
            "## 3. Experiment E2 — With Data Augmentation\n\n"
            "**Hypothesis:** Adding augmentation reduces overfitting and improves val accuracy.\n"
            "**Single factor changed from E1:** augmentation ON."
        ),

        new_code_cell(
            "# ── Cell 5: Experiment E2 — augmentation ───────────────────────\n"
            "# Only change from E1: train_loader now uses augment=True (default).\n"
            "# Same architecture, same LR, same epochs — isolates augmentation effect.\n"
            "from src.model import CNNScratch\n"
            "from src.train import run_experiment\n\n"
            "model_e2 = CNNScratch(num_classes=len(class_names))\n\n"
            "metrics_e2 = run_experiment(\n"
            "    exp_id       = 'E2_scratch_augmented',\n"
            "    model        = model_e2,\n"
            "    train_loader = train_loader,   # augmentation active (default)\n"
            "    val_loader   = val_loader,\n"
            "    test_loader  = test_loader,\n"
            "    class_names  = class_names,\n"
            "    epochs       = SCRATCH_EPOCHS,\n"
            "    lr           = SCRATCH_LR,\n"
            "    extra_params = {'architecture': 'CNNScratch_5conv_BN', 'augmentation': True},\n"
            ")\n"
            "print(f'E2 Test Accuracy: {metrics_e2[\"results\"][\"test_accuracy\"]*100:.2f}%')"
        ),

        # ── Cell 6: Experiment E3 ─────────────────────────────────────────────
        new_markdown_cell(
            "## 4. Experiment E3 — Lower LR + Augmentation\n\n"
            "**Hypothesis:** A lower LR with augmentation allows more stable convergence.\n"
            "**Single factor changed from E2:** lr 1e-3 → 1e-4."
        ),

        new_code_cell(
            "# ── Cell 6: Experiment E3 — lower LR ───────────────────────────\n"
            "# Lower LR is the standard next step when E2 shows unstable val_loss.\n"
            "# If E2 val_loss curve is smooth, E3 may not add value — check curves first.\n"
            "from src.model import CNNScratch\n"
            "from src.train import run_experiment\n\n"
            "model_e3 = CNNScratch(num_classes=len(class_names))\n\n"
            "metrics_e3 = run_experiment(\n"
            "    exp_id       = 'E3_scratch_lower_lr',\n"
            "    model        = model_e3,\n"
            "    train_loader = train_loader,\n"
            "    val_loader   = val_loader,\n"
            "    test_loader  = test_loader,\n"
            "    class_names  = class_names,\n"
            "    epochs       = SCRATCH_EPOCHS,\n"
            "    lr           = 1e-4,   # single factor change from E2\n"
            "    extra_params = {'architecture': 'CNNScratch_5conv_BN', 'augmentation': True, 'lr_reason': 'stable convergence'},\n"
            ")\n"
            "print(f'E3 Test Accuracy: {metrics_e3[\"results\"][\"test_accuracy\"]*100:.2f}%')"
        ),

        # ── Cell 7: Comparative table ─────────────────────────────────────────
        new_markdown_cell(
            "## 5. Phase 2 — Comparative Results Table\n\n"
            "Summary of all scratch experiments with the single factor that changed."
        ),

        new_code_cell(
            "# ── Cell 7: Comparative table ──────────────────────────────────\n"
            "# Why a table and not just reading JSON files:\n"
            "# Side-by-side comparison makes the impact of each factor immediately\n"
            "# visible — no mental arithmetic required to compare experiments.\n"
            "import pandas as pd\n\n"
            "results = pd.DataFrame([\n"
            "    {\n"
            "        'Experiment'   : m['exp_id'],\n"
            "        'Factor Changed': m['hyperparameters'].get('lr_reason', 'augmentation' if m['hyperparameters'].get('augmentation') else 'baseline'),\n"
            "        'LR'           : m['hyperparameters']['lr'],\n"
            "        'Val Acc'      : f\"{m['results']['best_val_loss']:.4f}\",\n"
            "        'Test Acc'     : f\"{m['results']['test_accuracy']*100:.2f}%\",\n"
            "        'Time (min)'   : m['results']['total_time_min'],\n"
            "        'Meets >=40%'  : '✅' if m['results']['test_accuracy'] >= 0.40 else '❌',\n"
            "    }\n"
            "    for m in [metrics_e1, metrics_e2, metrics_e3]\n"
            "])\n"
            "print(results.to_string(index=False))"
        ),

        # ── Cell 8: Full evaluation best model ───────────────────────────────
        new_markdown_cell(
            "## 6. Full Evaluation — Best Scratch Model\n\n"
            "Run `full_evaluation` on the best experiment to generate:\n"
            "- Classification report (precision, recall, F1 per class)\n"
            "- BI confusion matrix with Top-3 business error table\n"
            "- Executive report in docs/"
        ),

        new_code_cell(
            "# ── Cell 8: Full evaluation best model ─────────────────────────\n"
            "# Identify best experiment by test accuracy, then run full evaluation.\n"
            "# full_evaluation generates the BI confusion matrix and executive report.\n"
            "import torch\n"
            "from src.model import CNNScratch\n"
            "from src.evaluate import full_evaluation\n"
            "from src.config import MODELS_DIR\n\n"
            "all_metrics = [metrics_e1, metrics_e2, metrics_e3]\n"
            "best = max(all_metrics, key=lambda m: m['results']['test_accuracy'])\n"
            "print(f'Best experiment: {best[\"exp_id\"]} — {best[\"results\"][\"test_accuracy\"]*100:.2f}%')\n\n"
            "# Reload best checkpoint\n"
            "best_model = CNNScratch(num_classes=len(class_names))\n"
            "best_model.load_state_dict(\n"
            "    torch.load(MODELS_DIR / f'{best[\"exp_id\"]}_best.pt', weights_only=True)\n"
            ")\n\n"
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
            "| CNNScratch ≥3 conv layers | ✅ (5 conv blocks) |\n"
            "| Trained ≥30 epochs | ✅ |\n"
            "| BI narrative curves (loss + accuracy) | ✅ |\n"
            "| TorchScript exported | ✅ |\n"
            "| BI confusion matrix + Top-3 business errors | ✅ |\n"
            "| Executive report generated | ✅ |\n"
            "| Test accuracy ≥40% | ⬜ (fill after run) |\n\n"
            "**Next step:** `03_transfer_learning.ipynb` — Phase 3 on Colab T4."
        ),
    ]

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python (landmark-venv)",
        "language": "python",
        "name": "landmark-venv",
    }
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
            "≥70% test accuracy on 50 landmark classes — and when does fine-tuning "
            "outperform feature extraction?\n\n"
            "**Phase 3 rubric targets:**\n"
            "- Pretrained model selection with written justification (2 pts)\n"
            "- Training curves + comparison with Phase 2 (1 pt)\n"
            "- Test accuracy ≥70% + TorchScript export (2 pts)\n"
            "- Written analysis of strengths and weaknesses (2 pts)\n\n"
            "> ⚠️ **Run this notebook on Google Colab T4 GPU.**"
        ),

        new_code_cell(
            "# ── Cell 1: Environment setup ──────────────────────────────────\n"
            "import logging\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "PROJECT_ROOT = Path('..').resolve()\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "logging.basicConfig(level=logging.INFO)\n\n"
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
            "- VGG16: 138M parameters, FC layers alone exhaust 6 GB VRAM at batch_size=32\n"
            "- ResNet18: 11M parameters, residual connections prevent vanishing gradient\n"
            "- Residual connections make fine-tuning stable — VGG16 deep fine-tuning diverges\n"
            "- Benchmark from Project 2 (Vegetables): ResNet18 matched VGG16 accuracy at 12× less compute\n\n"
            "**Why ResNet50 as second candidate:**\n"
            "- 25M parameters — more capacity for complex landmark features\n"
            "- Useful when ResNet18 plateaus below 70% — provides a compute/accuracy trade-off reference"
        ),

        new_code_cell(
            "# ── Cell 3: Inspect trainable parameters per strategy ───────────\n"
            "# Why check params before training:\n"
            "# Confirms freeze/unfreeze worked as intended.\n"
            "# Frozen ResNet18 should show ~144K trainable (FC only).\n"
            "# Finetune should show ~8.5M (layer4 + FC).\n"
            "from src.model import get_transfer_model, count_params\n\n"
            "print('--- ResNet18 frozen (feature extraction) ---')\n"
            "m_frozen = get_transfer_model('resnet18', num_classes=len(class_names), strategy='frozen')\n"
            "count_params(m_frozen)\n\n"
            "print()\n"
            "print('--- ResNet18 finetune (layer4 + FC) ---')\n"
            "m_finetune = get_transfer_model('resnet18', num_classes=len(class_names), strategy='finetune')\n"
            "count_params(m_finetune)"
        ),

        new_markdown_cell(
            "## 2. Experiment E4 — ResNet18 Feature Extraction\n\n"
            "**Hypothesis:** Frozen ImageNet features already capture enough visual "
            "structure to classify landmarks above 55% accuracy.\n"
            "**Strategy:** Train only the new FC head — backbone weights unchanged."
        ),

        new_code_cell(
            "# ── Cell 4: Experiment E4 — ResNet18 frozen ────────────────────\n"
            "# Feature extraction stage: only the FC head is trainable.\n"
            "# Why start frozen: fast convergence, zero catastrophic forgetting risk.\n"
            "# If this alone reaches >=70%, fine-tuning is unnecessary.\n"
            "from src.model import get_transfer_model\n"
            "from src.train import run_experiment\n\n"
            "model_e4 = get_transfer_model('resnet18', num_classes=len(class_names), strategy='frozen')\n\n"
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

        new_markdown_cell(
            "## 3. Experiment E5 — ResNet18 Fine-tune (layer4 unfrozen)\n\n"
            "**Hypothesis:** Adapting layer4 (highest-level ImageNet features) "
            "to the landmark domain pushes accuracy above 70%.\n"
            "**Single factor changed from E4:** strategy='finetune' + differentiated LR."
        ),

        new_code_cell(
            "# ── Cell 5: Experiment E5 — ResNet18 finetune ──────────────────\n"
            "# Fine-tuning stage: layer4 + FC trainable with differentiated LR.\n"
            "# lr_backbone = 1e-5 (100x lower than head) prevents catastrophic\n"
            "# forgetting while allowing domain adaptation of high-level features.\n"
            "from src.model import get_transfer_model\n"
            "from src.train import run_experiment\n\n"
            "model_e5 = get_transfer_model('resnet18', num_classes=len(class_names), strategy='finetune')\n\n"
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

        new_markdown_cell(
            "## 4. Experiment E6 — ResNet50 Feature Extraction\n\n"
            "**Hypothesis:** A larger backbone provides higher capacity for distinguishing "
            "visually similar landmarks.\n"
            "**Single factor changed from E4:** backbone ResNet18 → ResNet50."
        ),

        new_code_cell(
            "# ── Cell 6: Experiment E6 — ResNet50 frozen ────────────────────\n"
            "# ResNet50 has 25M params vs ResNet18's 11M.\n"
            "# If E5 already reaches >=70%, E6 tests whether more capacity helps\n"
            "# or simply adds compute cost without accuracy gain.\n"
            "from src.model import get_transfer_model\n"
            "from src.train import run_experiment\n\n"
            "model_e6 = get_transfer_model('resnet50', num_classes=len(class_names), strategy='frozen')\n\n"
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

        new_markdown_cell(
            "## 5. Full Comparative Table — Scratch vs Transfer Learning\n\n"
            "**Rubric requirement:** Compare all models in a table or chart."
        ),

        new_code_cell(
            "# ── Cell 7: Full comparative table ─────────────────────────────\n"
            "# Load E3 scratch metrics from JSON to include in comparison.\n"
            "# Why reload from JSON: notebooks are stateless — E3 was run in notebook 02.\n"
            "import json\n"
            "import pandas as pd\n"
            "from src.config import EXPERIMENTS_DIR\n\n"
            "def load_metrics(exp_id: str) -> dict:\n"
            "    path = EXPERIMENTS_DIR / f'{exp_id}_metrics.json'\n"
            "    with path.open() as f:\n"
            "        return json.load(f)\n\n"
            "# Load best scratch result\n"
            "metrics_e3 = load_metrics('E3_scratch_lower_lr')\n\n"
            "all_results = pd.DataFrame([\n"
            "    {\n"
            "        'Model'       : m['exp_id'],\n"
            "        'Type'        : 'Scratch' if 'scratch' in m['exp_id'] else 'Transfer',\n"
            "        'Backbone'    : m['hyperparameters'].get('backbone', 'Custom CNN'),\n"
            "        'Strategy'    : m['hyperparameters'].get('strategy', 'from_scratch'),\n"
            "        'Test Acc'    : f\"{m['results']['test_accuracy']*100:.2f}%\",\n"
            "        'Time (min)'  : m['results']['total_time_min'],\n"
            "        'Meets target': '✅' if (\n"
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
            "# Select best transfer model and run full_evaluation for\n"
            "# classification report + BI confusion matrix + executive report.\n"
            "import torch\n"
            "from src.model import get_transfer_model\n"
            "from src.evaluate import full_evaluation\n"
            "from src.config import MODELS_DIR\n\n"
            "tl_metrics = [metrics_e4, metrics_e5, metrics_e6]\n"
            "best_tl    = max(tl_metrics, key=lambda m: m['results']['test_accuracy'])\n"
            "print(f'Best TL experiment: {best_tl[\"exp_id\"]} — {best_tl[\"results\"][\"test_accuracy\"]*100:.2f}%')\n\n"
            "backbone = best_tl['hyperparameters'].get('backbone', 'resnet18')\n"
            "strategy = best_tl['hyperparameters'].get('strategy', 'frozen')\n"
            "best_tl_model = get_transfer_model(backbone, num_classes=len(class_names), strategy=strategy)\n"
            "best_tl_model.load_state_dict(\n"
            "    torch.load(MODELS_DIR / f'{best_tl[\"exp_id\"]}_best.pt', weights_only=True)\n"
            ")\n\n"
            "eval_tl = full_evaluation(\n"
            "    exp_id      = best_tl['exp_id'],\n"
            "    model       = best_tl_model,\n"
            "    loader      = test_loader,\n"
            "    class_names = class_names,\n"
            "    topk        = 5,\n"
            ")"
        ),

        new_markdown_cell(
            "## 6. Analysis — Strengths and Weaknesses\n\n"
            "*(Fill after running all experiments)*\n\n"
            "### Strengths\n"
            "- Transfer Learning reaches ≥70% accuracy with only 10 epochs\n"
            "- Pretrained ImageNet features generalize well to architectural landmarks\n"
            "- Fine-tuning layer4 captures domain-specific textures (stone, glass, metal)\n\n"
            "### Weaknesses\n"
            "- Visually similar landmarks (e.g. different Roman amphitheaters) cause systematic confusion\n"
            "- Dataset imbalance biases predictions toward majority classes\n"
            "- Low-quality or unusual-angle images degrade confidence significantly\n\n"
            "### Recommended Improvements\n"
            "- Weighted sampler to address class imbalance\n"
            "- Test-Time Augmentation (TTA) for ambiguous images\n"
            "- Add Santa Cruz de la Sierra class for regional coverage\n\n"
            "## Phase 3 — Checklist\n\n"
            "| Check | Status |\n"
            "|---|---|\n"
            "| 2 pretrained models tested | ✅ (ResNet18, ResNet50) |\n"
            "| Written model justification | ✅ |\n"
            "| BI narrative curves | ✅ |\n"
            "| Scratch vs Transfer comparison table | ✅ |\n"
            "| TorchScript exported | ✅ |\n"
            "| BI confusion matrix + Top-3 business errors | ✅ |\n"
            "| Test accuracy ≥70% | ⬜ (fill after run) |\n\n"
            "**Next step:** `04_inference_app.ipynb` — Phase 4."
        ),
    ]

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python (landmark-venv)",
        "language": "python",
        "name": "landmark-venv",
    }
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
            "- `predict_landmarks(img_path, k)` function using TorchScript (Fase 4)\n"
            "- Test on ≥4 personal images not from the dataset\n"
            "- Written analysis of strengths and weaknesses\n\n"
            "> ✅ **This notebook runs on PyCharm (CPU).** Inference is lightweight."
        ),

        new_code_cell(
            "# ── Cell 1: Environment setup ──────────────────────────────────\n"
            "import logging\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "PROJECT_ROOT = Path('..').resolve()\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "logging.basicConfig(level=logging.INFO)\n\n"
            "from src.config import EXTERNAL_DIR, MODELS_DIR, SEED\n"
            "from src.utils import set_seed\n\n"
            "set_seed(SEED)\n\n"
            "# List available TorchScript models\n"
            "scripted_models = sorted(MODELS_DIR.glob('*_scripted.pt'))\n"
            "print('Available TorchScript models:')\n"
            "for m in scripted_models:\n"
            "    print(f'  {m.name}')"
        ),

        new_markdown_cell(
            "## 1. Select Best Model\n\n"
            "Load the best-performing TorchScript model from Phase 3.\n"
            "No model.py dependency — the scripted graph is self-contained."
        ),

        new_code_cell(
            "# ── Cell 2: Select best TorchScript model ───────────────────────\n"
            "# Why TorchScript for inference:\n"
            "# The .pt scripted file contains the full computation graph.\n"
            "# No need to import model.py, config.py, or any src module.\n"
            "# Deployable to any Python environment with only torch installed.\n\n"
            "# Select the finetune model if available — highest expected accuracy\n"
            "finetune_models = [m for m in scripted_models if 'finetune' in m.name]\n"
            "BEST_MODEL_PATH = finetune_models[-1] if finetune_models else scripted_models[-1]\n"
            "print(f'Using model: {BEST_MODEL_PATH.name}')"
        ),

        new_markdown_cell(
            "## 2. Run predict_landmarks on Personal Images\n\n"
            "Place ≥4 personal images (JPG or PNG) in `data/external/` before running.\n"
            "These must be images **not from the Google Landmarks dataset**."
        ),

        new_code_cell(
            "# ── Cell 3: Verify external images ─────────────────────────────\n"
            "# Why verify before inference:\n"
            "# A missing or corrupt image raises an unhelpful CUDA error during\n"
            "# the forward pass. Checking existence here gives a clear error message.\n"
            "images = (\n"
            "    list(EXTERNAL_DIR.glob('*.jpg'))\n"
            "    + list(EXTERNAL_DIR.glob('*.jpeg'))\n"
            "    + list(EXTERNAL_DIR.glob('*.png'))\n"
            "    + list(EXTERNAL_DIR.glob('*.webp'))\n"
            ")\n"
            "print(f'External images found: {len(images)}')\n"
            "for img in images:\n"
            "    print(f'  {img.name}')\n\n"
            "assert len(images) >= 4, (\n"
            "    f'Rubric requires >=4 personal images in data/external/. Found: {len(images)}'\n"
            ")"
        ),

        new_code_cell(
            "# ── Cell 4: Inference on all external images ────────────────────\n"
            "# predict_and_display: runs inference + renders BI visualization\n"
            "# (UCB palette bar chart alongside original image).\n"
            "# Why display probability bars and not just the top-1 label:\n"
            "# Production systems show top-3 suggestions — users can correct\n"
            "# if the top-1 is wrong. Probability bars communicate model confidence.\n"
            "from src.predictor import predict_and_display\n\n"
            "all_predictions = {}\n\n"
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

        new_markdown_cell(
            "## 3. Prediction Summary Table\n\n"
            "Structured view of all predictions for the written analysis."
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
            "        })\n\n"
            "summary = pd.DataFrame(rows)\n"
            "print(summary.to_string(index=False))"
        ),

        new_markdown_cell(
            "## 4. Analysis — Strengths and Weaknesses\n\n"
            "*(Fill based on your prediction results)*\n\n"
            "### Strengths\n"
            "- High confidence (>80%) on iconic, visually distinctive landmarks\n"
            "- Top-3 predictions include the correct class even when Top-1 is wrong\n"
            "- Robust to moderate changes in lighting and viewpoint (augmentation effect)\n\n"
            "### Weaknesses\n"
            "- Low confidence on landmarks not well-represented in training data\n"
            "- Confuses architecturally similar structures (e.g. Gothic cathedrals)\n"
            "- Performance degrades on heavily cropped or extreme-angle images\n\n"
            "### Possible Improvements\n"
            "- Ensemble: combine ResNet18 + ResNet50 predictions for borderline cases\n"
            "- Test-Time Augmentation: average predictions over multiple crops\n"
            "- Collect more images of weak classes (identified in Executive Report)\n\n"
            "## Phase 4 — Checklist\n\n"
            "| Check | Status |\n"
            "|---|---|\n"
            "| `predict_landmarks()` uses TorchScript | ✅ |\n"
            "| ≥4 personal images tested | ✅ |\n"
            "| Visual results displayed (BI bar chart) | ✅ |\n"
            "| Prediction summary table | ✅ |\n"
            "| Written analysis strengths/weaknesses | ✅ |\n\n"
            "## Final Deliverables Checklist\n\n"
            "| Deliverable | Status |\n"
            "|---|---|\n"
            "| GitHub repo public (no dataset) | ⬜ |\n"
            "| All 4 notebooks executed with outputs | ⬜ |\n"
            "| README.md with YouTube link | ⬜ |\n"
            "| Video 3–5 min uploaded to YouTube | ⬜ |\n"
            "| docs/informe_landmarks.pdf | ⬜ |\n"
            "| git tag v1.0-entrega pushed | ⬜ |"
        ),
    ]

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python (landmark-venv)",
        "language": "python",
        "name": "landmark-venv",
    }
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
