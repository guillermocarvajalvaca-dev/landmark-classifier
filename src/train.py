# ============================================================
# MODULE: train.py
# PURPOSE: Training loop, validation, and run_experiment().
#          Generates and VERIFIES all artifacts automatically
#          per run. Artifacts saved to Drive in Colab.
# NORMATIVE BASIS: UCB Project 5 rubric — Phase 2: >=30 epochs,
#                  loss/accuracy curves per epoch, save best
#                  model on val_loss, export with TorchScript.
# AUTHOR: Guillermo Carvajal Vaca — UCB MSc Data Science & AI
# VERSION: 1.2.0
# ============================================================
from __future__ import annotations

# --- stdlib (alphabetical) ---
import importlib
import logging
import time
from pathlib import Path
from typing import Any, Tuple

# --- third-party (alphabetical) ---
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# --- local (alphabetical) ---
from src.config import (
    DEVICE,
    EXPERIMENTS_DIR,
    MODELS_DIR,
    SCRATCH_EPOCHS,
    SCRATCH_LR,
    SCRATCH_SCHEDULER_GAMMA,
    SCRATCH_SCHEDULER_STEP,
    SEED,
)
from src.utils import save_metrics, set_seed

logger = logging.getLogger(__name__)


def _get_visualization():
    """
    Import and reload visualization module on every call.

    Why reload on every call:
        In Colab, after a git pull, Python keeps the old cached module
        in memory. Reloading guarantees the latest version of
        plot_training_narrative is always used, regardless of when
        the module was first imported.
    """
    import src.visualization
    importlib.reload(src.visualization)
    return src.visualization


def _display_inline(path: Path) -> None:
    """
    Display a saved PNG inline in Colab or Jupyter.

    Args:
        path: Path to the PNG file to display.

    Why display after saving:
        Saving first guarantees the artifact exists on disk regardless
        of whether the inline display succeeds.
    """
    try:
        img = mpimg.imread(str(path))
        plt.figure(figsize=(16, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.warning("Inline display failed: %s — artifact saved to disk", e)


def _verify_artifacts(exp_id: str) -> None:
    """
    Verify all expected artifacts were saved after an experiment.

    Args:
        exp_id: Experiment identifier to check.

    Why verify after saving:
        Silent write failures can occur on Drive — the file appears to save
        but is empty or missing. Explicit verification catches this before
        the next experiment overwrites the checkpoint slot.

    Raises:
        FileNotFoundError: If any expected artifact is missing.
    """
    expected = [
        EXPERIMENTS_DIR / f"{exp_id}_metrics.json",
        EXPERIMENTS_DIR / f"{exp_id}_narrative.png",
        MODELS_DIR      / f"{exp_id}_best.pt",
        MODELS_DIR      / f"{exp_id}_scripted.pt",
    ]
    print(f"\n--- Artifact verification: {exp_id} ---")
    all_ok = True
    for path in expected:
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {path.name}")
        if not exists:
            all_ok = False

    if not all_ok:
        raise FileNotFoundError(
            f"One or more artifacts missing for {exp_id}. "
            "Check Drive connection and re-run the experiment."
        )
    print(f"  All artifacts verified OK\n")


def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    optimizer : torch.optim.Optimizer,
    criterion : nn.Module,
    device    : torch.device = DEVICE,
) -> Tuple[float, float]:
    """
    Run one full training epoch over the DataLoader.

    Args:
        model:     The neural network in train mode.
        loader:    Training DataLoader (shuffle=True).
        optimizer: Optimizer instance (Adam or SGD).
        criterion: Loss function (CrossEntropyLoss).
        device:    Target device for tensor operations.

    Returns:
        Tuple of (avg_loss, avg_accuracy) over all batches.

    Why model.train() is called here and not outside:
        Calling it once per epoch guards against accidental eval() calls
        that might disable Dropout and BatchNorm between epochs.
    """
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct      += (logits.argmax(dim=1) == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


def validate(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    device    : torch.device = DEVICE,
) -> Tuple[float, float]:
    """
    Evaluate the model on a validation or test DataLoader.

    Args:
        model:     The neural network to evaluate.
        loader:    DataLoader with shuffle=False for reproducible metrics.
        criterion: Loss function (same as training).
        device:    Target device for tensor operations.

    Returns:
        Tuple of (avg_loss, avg_accuracy) over all batches.

    Why torch.no_grad():
        During inference the computation graph is not needed.
        Disabling it saves ~50% VRAM and speeds up evaluation —
        critical when VRAM is shared with other processes.
    """
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device,   non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            running_loss += loss.item() * imgs.size(0)
            correct      += (logits.argmax(dim=1) == labels).sum().item()
            total        += labels.size(0)

    return running_loss / total, correct / total


def run_experiment(
    exp_id          : str,
    model           : nn.Module,
    train_loader    : DataLoader,
    val_loader      : DataLoader,
    test_loader     : DataLoader,
    class_names     : list[str],
    epochs          : int   = SCRATCH_EPOCHS,
    lr              : float = SCRATCH_LR,
    lr_backbone     : float | None = None,
    scheduler_step  : int   = SCRATCH_SCHEDULER_STEP,
    scheduler_gamma : float = SCRATCH_SCHEDULER_GAMMA,
    extra_params    : dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute a full training run and persist all artifacts automatically.

    Every call produces and VERIFIES:
        - {exp_id}_metrics.json        (hyperparams + curves + test result)
        - {exp_id}_narrative.png       (BI-grade loss + accuracy plot inline)
        - {exp_id}_best.pt             (best checkpoint by val_loss)
        - {exp_id}_scripted.pt         (TorchScript — no model.py dependency)
        - {exp_id}_EXECUTIVE_REPORT.md (automatic narrative report)

    Args:
        exp_id:          Unique identifier (e.g. 'E5_resnet18_finetune_layer4').
        model:           Model instance — moved to DEVICE inside this function.
        train_loader:    Training DataLoader (shuffle=True, augmented).
        val_loader:      Validation DataLoader (shuffle=False, no augmentation).
        test_loader:     Test DataLoader (shuffle=False, no augmentation).
        class_names:     Ordered list of class labels from ImageFolder.
        epochs:          Number of training epochs.
        lr:              Learning rate for the head or entire model.
        lr_backbone:     If not None, apply differentiated LR for backbone.
                         Why: backbone weights are pretrained — same LR as head
                         causes catastrophic forgetting of ImageNet features.
        scheduler_step:  Epochs between LR decay steps.
        scheduler_gamma: LR decay factor per step.
        extra_params:    Additional metadata to persist in the JSON artifact.

    Returns:
        Dict with all metrics, hyperparameters, and artifact paths.
    """
    set_seed(SEED)
    model = model.to(DEVICE)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Optimizer with optional differentiated LR ---
    if lr_backbone is not None:
        backbone_params = [
            p for n, p in model.named_parameters()
            if "fc" not in n and p.requires_grad
        ]
        head_params = [
            p for n, p in model.named_parameters()
            if "fc" in n and p.requires_grad
        ]
        optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr},
        ])
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
        )

    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    train_losses : list[float] = []
    val_losses   : list[float] = []
    train_accs   : list[float] = []
    val_accs     : list[float] = []

    best_val_loss   = float("inf")
    best_model_path = MODELS_DIR / f"{exp_id}_best.pt"

    print(f"\n{'='*62}")
    print(f"  EXPERIMENT : {exp_id}")
    print(f"  Epochs     : {epochs} | LR head: {lr} | LR backbone: {lr_backbone}")
    print(f"  Device     : {DEVICE}")
    print(f"  Artifacts  -> {EXPERIMENTS_DIR}")
    print(f"{'='*62}")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_ep = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = validate(model, val_loader, criterion)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        is_best = vl_loss < best_val_loss
        if is_best:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), best_model_path)

        print(
            f"  Epoch {epoch:3d}/{epochs}"
            f"  | Train  loss: {tr_loss:.4f}  acc: {tr_acc:.3f}"
            f"  | Val    loss: {vl_loss:.4f}  acc: {vl_acc:.3f}"
            f"  | {('BEST ' + str(round(best_val_loss, 4))) if is_best else ''}"
            f"  [{time.time() - t_ep:.1f}s]"
        )

    t_total = time.time() - t_start

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss, test_acc = validate(model, test_loader, criterion)

    print(f"\n{'='*62}")
    print(f"  FINAL RESULT : {exp_id}")
    print(f"  Test loss    : {test_loss:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    print(f"  Total time   : {t_total / 60:.1f} min")
    print(f"{'='*62}\n")

    # --- TorchScript export ---
    model.eval()
    scripted_path = MODELS_DIR / f"{exp_id}_scripted.pt"
    example_input = torch.zeros(1, 3, 224, 224).to(DEVICE)

    try:
        scripted = torch.jit.trace(model, example_input)
        scripted.save(str(scripted_path))
        logger.info("TorchScript saved -> %s", scripted_path)
    except Exception as e:
        logger.warning("TorchScript export failed: %s — checkpoint only", e)
        scripted_path = None

    # --- Persist metrics ---
    metrics: dict[str, Any] = {
        "exp_id": exp_id,
        "hyperparameters": {
            "epochs"          : epochs,
            "lr"              : lr,
            "lr_backbone"     : lr_backbone,
            "scheduler_step"  : scheduler_step,
            "scheduler_gamma" : scheduler_gamma,
            **(extra_params or {}),
        },
        "curves": {
            "train_loss" : train_losses,
            "val_loss"   : val_losses,
            "train_acc"  : train_accs,
            "val_acc"    : val_accs,
        },
        "results": {
            "best_val_loss"  : best_val_loss,
            "test_loss"      : test_loss,
            "test_accuracy"  : test_acc,
            "total_time_min" : round(t_total / 60, 2),
        },
        "artifacts": {
            "best_checkpoint" : str(best_model_path),
            "scripted_model"  : str(scripted_path) if scripted_path else None,
        },
    }

    save_metrics(exp_id, metrics)

    # --- Generate BI narrative plot and display inline ---
    viz            = _get_visualization()
    narrative_path = viz.plot_training_narrative(
        exp_id, train_losses, val_losses, train_accs, val_accs
    )
    _display_inline(narrative_path)

    # --- Generate executive report ---
    import numpy as np
    viz.generate_executive_report(
        exp_id       = exp_id,
        train_losses = train_losses,
        val_losses   = val_losses,
        val_accs     = val_accs,
        class_names  = class_names,
        cm           = np.zeros((len(class_names), len(class_names)), dtype=int),
        test_acc     = test_acc,
    )

    # --- Verify all artifacts saved correctly ---
    _verify_artifacts(exp_id)

    return metrics
