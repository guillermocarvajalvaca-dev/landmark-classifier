# ============================================================
# MODULE: train.py
# PURPOSE: Training loop, validation, and run_experiment() —
#          the central traceability function that persists all
#          artifacts automatically per experimental run.
# NORMATIVE BASIS: UCB Project 5 rubric — Phase 2: >=30 epochs,
#                  loss/accuracy curves per epoch, save best
#                  model on val_loss, export with TorchScript.
# AUTHOR: Guillermo Carvajal Vaca — UCB MSc Data Science & AI
# VERSION: 1.0.0
# ============================================================
from __future__ import annotations

# --- stdlib (alphabetical) ---
import logging
import time
from typing import Any, Tuple

# --- third-party (alphabetical) ---
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# --- local (alphabetical) ---
from src.config import (
    DEVICE,
    MODELS_DIR,
    SCRATCH_EPOCHS,
    SCRATCH_LR,
    SCRATCH_SCHEDULER_GAMMA,
    SCRATCH_SCHEDULER_STEP,
    SEED,
)
from src.utils import save_metrics, set_seed
from src.visualization import generate_executive_report, plot_training_narrative

logger = logging.getLogger(__name__)


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
        in callbacks or custom loops that might disable Dropout/BatchNorm.
    """
    model.train()   # activates Dropout and BatchNorm training behavior
    running_loss = 0.0
    correct      = 0
    total        = 0

    for imgs, labels in loader:
        # non_blocking=True overlaps data transfer with GPU compute
        imgs   = imgs.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()              # clear gradients from previous step
        logits = model(imgs)               # forward pass
        loss   = criterion(logits, labels) # CrossEntropyLoss includes LogSoftmax
        loss.backward()                    # compute dL/dW for all trainable params
        optimizer.step()                   # apply gradient update

        running_loss += loss.item() * imgs.size(0)   # accumulate unnormalized loss
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
        Disabling it saves ~50% VRAM and speeds up evaluation significantly —
        critical on Colab T4 where VRAM is shared with the OS.
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

    This is the central traceability function. Every call produces:
        - {exp_id}_metrics.json      (hyperparams + curves + test result)
        - {exp_id}_narrative.png     (BI-grade loss + accuracy plot)
        - {exp_id}_best.pt           (best checkpoint by val_loss)
        - {exp_id}_scripted.pt       (TorchScript — no model.py dependency)
        - {exp_id}_EXECUTIVE_REPORT.md (automatic narrative report)

    Args:
        exp_id:          Unique identifier (e.g. 'E5_resnet18_finetune_layer4').
        model:           Model instance — moved to DEVICE inside this function.
        train_loader:    Training DataLoader (shuffle=True, augmented).
        val_loader:      Validation DataLoader (shuffle=False, no augmentation).
        test_loader:     Test DataLoader (shuffle=False, no augmentation).
        class_names:     Ordered list of class labels from ImageFolder.
        epochs:          Number of training epochs.
        lr:              Learning rate for the head (or entire model if no backbone LR).
        lr_backbone:     If not None, apply differentiated LR for backbone params.
                         Why: backbone weights are pretrained — using the same LR
                         as the head causes catastrophic forgetting of ImageNet features.
        scheduler_step:  Epochs between LR decay steps.
        scheduler_gamma: LR decay factor per step.
        extra_params:    Additional metadata to persist in the JSON artifact.

    Returns:
        Dict with all metrics, hyperparameters, and artifact paths.

    Raises:
        RuntimeError: If TorchScript export fails (logged as WARNING, not fatal).
    """
    set_seed(SEED)
    model = model.to(DEVICE)

    # --- Optimizer with optional differentiated LR ---
    if lr_backbone is not None:
        # Separate backbone params (low LR) from head params (normal LR)
        # Why: the backbone already learned general features — we only nudge it
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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / f"{exp_id}_best.pt"

    print(f"\n{'='*62}")
    print(f"  EXPERIMENT : {exp_id}")
    print(f"  Epochs     : {epochs} | LR head: {lr} | LR backbone: {lr_backbone}")
    print(f"  Device     : {DEVICE}")
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

        # Save checkpoint only when val_loss improves
        # Why val_loss and not val_acc: loss is smoother and less sensitive
        # to class imbalance than accuracy — more reliable early stopping signal
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

    # --- Load best checkpoint for final test evaluation ---
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss, test_acc = validate(model, test_loader, criterion)

    print(f"\n{'='*62}")
    print(f"  FINAL RESULT : {exp_id}")
    print(f"  Test loss    : {test_loss:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    print(f"  Total time   : {t_total / 60:.1f} min")
    print(f"{'='*62}\n")

    # --- TorchScript export ---
    # Why torch.jit.trace and not script: trace follows one execution path
    # through the model — sufficient for inference on fixed-shape inputs.
    # script requires all control flow to be TorchScript-compatible,
    # which some custom layers may violate unnecessarily.
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

    # --- Persist all artifacts ---
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
    plot_training_narrative(exp_id, train_losses, val_losses, train_accs, val_accs)

    # Confusion matrix requires full_evaluation — called from notebooks
    # Executive report generated here with available data only
    generate_executive_report(
        exp_id       = exp_id,
        train_losses = train_losses,
        val_losses   = val_losses,
        val_accs     = val_accs,
        class_names  = class_names,
        cm           = __import__("numpy").zeros((len(class_names), len(class_names)), dtype=int),
        test_acc     = test_acc,
    )

    return metrics
