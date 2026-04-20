# ============================================================
# MODULE: evaluate.py
# PURPOSE: Post-training evaluation — top-k accuracy, full
#          classification report, BI confusion matrix with
#          inline display. Decoupled from train.py for reuse
#          in notebooks and inference workflows.
# NORMATIVE BASIS: UCB Project 5 rubric — Phase 3: compare
#                  scratch vs transfer in table or chart.
#                  Phase 4: analysis of strengths and weaknesses.
# AUTHOR: Guillermo Carvajal Vaca — UCB MSc Data Science & AI
# VERSION: 1.2.0
# ============================================================
from __future__ import annotations

# --- stdlib (alphabetical) ---
import importlib
import logging
from typing import Tuple

# --- third-party (alphabetical) ---
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

# --- local (alphabetical) ---
from src.config import DEVICE

logger = logging.getLogger(__name__)


def _get_visualization():
    """
    Import and reload visualization module on every call.

    Why reload on every call:
        In Colab, after a git pull, Python keeps the old cached module
        in memory. Reloading guarantees the latest version of
        plot_confusion_matrix_bi is always used.
    """
    import src.visualization
    importlib.reload(src.visualization)
    return src.visualization


def top_k_accuracy(
    model  : nn.Module,
    loader : DataLoader,
    k      : int = 5,
    device : torch.device = DEVICE,
) -> Tuple[float, float]:
    """
    Compute top-1 and top-k accuracy over a DataLoader.

    Args:
        model:  Trained model in eval mode, already on device.
        loader: DataLoader with shuffle=False for reproducible results.
        k:      Number of top predictions to consider for top-k metric.
        device: Target device for tensor operations.

    Returns:
        Tuple of (top1_accuracy, topk_accuracy) as floats in [0, 1].

    Why top-k matters for 50-class classification:
        With 50 visually similar landmark classes, top-1 accuracy alone
        can be misleadingly low. Top-5 reveals whether the model has the
        correct class in its shortlist — critical for production systems
        where showing 3-5 suggestions is acceptable UX.
    """
    model.eval()
    top1_correct = 0
    topk_correct = 0
    total        = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device,   non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)

            top1_correct += (logits.argmax(dim=1) == labels).sum().item()
            topk_idx      = logits.topk(k, dim=1).indices
            topk_correct += (topk_idx == labels.unsqueeze(1)).any(dim=1).sum().item()
            total        += labels.size(0)

    top1_acc = top1_correct / total
    topk_acc = topk_correct / total

    logger.info("Top-1: %.4f | Top-%d: %.4f", top1_acc, k, topk_acc)
    return top1_acc, topk_acc


def full_evaluation(
    exp_id      : str,
    model       : nn.Module,
    loader      : DataLoader,
    class_names : list[str],
    device      : torch.device = DEVICE,
    topk        : int = 5,
) -> dict[str, object]:
    """
    Run complete post-training evaluation with all diagnostic artifacts.

    Computes top-1 accuracy, top-k accuracy, sklearn classification report,
    and generates a BI-grade confusion matrix heatmap with business error
    table. Displays all visuals inline in Colab/Jupyter.

    Args:
        exp_id:       Experiment identifier — artifact filename prefix.
        model:        Trained model. Moved to device internally.
        loader:       Test DataLoader (shuffle=False).
        class_names:  Ordered list of class labels matching ImageFolder.
        device:       Target device for tensor operations.
        topk:         k for top-k accuracy metric.

    Returns:
        Dict with keys: top1_accuracy, top{k}_accuracy,
        confusion_matrix_path, classification_report.

    Why move model to device inside full_evaluation:
        Callers may pass a model loaded from checkpoint on CPU.
        Centralizing the .to(device) call here prevents the
        'Input type and weight type should be the same' RuntimeError
        that occurs when tensors are on GPU but model weights are on CPU.

    Why collect all predictions before computing metrics:
        sklearn metrics require the full prediction array, not per-batch
        aggregates. Collecting in lists then converting once is more
        memory efficient than concatenating tensors on GPU.
    """
    # Why move to device here: prevents RuntimeError when model loaded from
    # CPU checkpoint and DataLoader sends tensors to GPU.
    model = model.to(device)
    model.eval()

    all_preds  : list[int] = []
    all_labels : list[int] = []
    topk_correct = 0
    total        = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device,   non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)

            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            topk_idx      = logits.topk(topk, dim=1).indices
            topk_correct += (topk_idx == labels.unsqueeze(1)).any(dim=1).sum().item()
            total        += labels.size(0)

    all_preds_np  = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    top1_acc = float((all_preds_np == all_labels_np).mean())
    topk_acc = float(topk_correct / total)

    report = classification_report(
        all_labels_np,
        all_preds_np,
        target_names  = class_names,
        zero_division = 0,
    )
    print(f"\n--- {exp_id} — Classification Report ---")
    print(report)
    print(f"  Top-1 Accuracy  : {top1_acc:.4f}  ({top1_acc * 100:.2f}%)")
    print(f"  Top-{topk} Accuracy : {topk_acc:.4f}  ({topk_acc * 100:.2f}%)")

    # --- BI confusion matrix with inline display ---
    cm      = confusion_matrix(all_labels_np, all_preds_np)
    viz     = _get_visualization()
    cm_path = viz.plot_confusion_matrix_bi(exp_id, cm, class_names)

    # --- Update executive report with real confusion matrix data ---
    viz.generate_executive_report(
        exp_id       = exp_id,
        train_losses = [],
        val_losses   = [],
        val_accs     = [],
        class_names  = class_names,
        cm           = cm,
        test_acc     = top1_acc,
    )

    logger.info(
        "[%s] Evaluation complete — Top-1: %.4f | Top-%d: %.4f",
        exp_id, top1_acc, topk, topk_acc,
    )

    return {
        "top1_accuracy"         : top1_acc,
        f"top{topk}_accuracy"   : topk_acc,
        "confusion_matrix_path" : str(cm_path),
        "classification_report" : report,
    }
