# ============================================================
# MODULE: evaluate.py
# PURPOSE: Post-training evaluation — top-k accuracy, full
#          classification report, and BI-grade confusion matrix.
#          Decoupled from train.py to enable reuse in notebooks
#          and inference workflows without re-running training.
# NORMATIVE BASIS: UCB Project 5 rubric — Phase 3: compare
#                  scratch vs transfer in table or chart.
#                  Phase 4: analysis of strengths and weaknesses.
# AUTHOR: Guillermo Carvajal Vaca — UCB MSc Data Science & AI
# VERSION: 1.0.0
# ============================================================
from __future__ import annotations

# --- stdlib (alphabetical) ---
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
from src.visualization import generate_executive_report, plot_confusion_matrix_bi

logger = logging.getLogger(__name__)


def top_k_accuracy(
    model  : nn.Module,
    loader : DataLoader,
    k      : int = 5,
    device : torch.device = DEVICE,
) -> Tuple[float, float]:
    """
    Compute top-1 and top-k accuracy over a DataLoader.

    Args:
        model:  Trained model in eval mode.
        loader: DataLoader with shuffle=False for reproducible results.
        k:      Number of top predictions to consider for top-k metric.
        device: Target device for tensor operations.

    Returns:
        Tuple of (top1_accuracy, topk_accuracy) as floats in [0, 1].

    Why top-k matters for 50-class classification:
        With 50 visually similar landmark classes, top-1 accuracy alone
        can be misleadingly low. Top-5 accuracy reveals whether the model
        has the correct class in its shortlist — critical for a production
        recommendation system where showing 3-5 suggestions is acceptable.
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

            # Top-1: single best prediction must match the true label
            top1_correct += (logits.argmax(dim=1) == labels).sum().item()

            # Top-k: true label must appear among the k highest logits
            # unsqueeze(1) broadcasts labels from [B] to [B,1] for comparison
            topk_idx      = logits.topk(k, dim=1).indices   # [B, k]
            topk_correct += (topk_idx == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

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
    and generates a BI-grade confusion matrix heatmap with business error table.
    Also updates the executive report with the full confusion matrix.

    Args:
        exp_id:       Experiment identifier — used as artifact filename prefix.
        model:        Trained model. Will be set to eval mode internally.
        loader:       Test DataLoader (shuffle=False).
        class_names:  Ordered list of class labels matching ImageFolder order.
        device:       Target device for tensor operations.
        topk:         k for top-k accuracy metric.

    Returns:
        Dict with keys:
            top1_accuracy, top{k}_accuracy,
            confusion_matrix_path, classification_report.

    Why collect all predictions before computing metrics:
        sklearn metrics require the full prediction array, not per-batch
        aggregates. Collecting in lists and converting once is more memory
        efficient than concatenating tensors incrementally on GPU.
    """
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

    # sklearn report: precision, recall, F1 per class + macro/weighted averages
    report = classification_report(
        all_labels_np,
        all_preds_np,
        target_names = class_names,
        zero_division = 0,   # suppress warnings for classes with no predictions
    )
    print(f"\n--- {exp_id} — Classification Report ---")
    print(report)
    print(f"  Top-1 Accuracy  : {top1_acc:.4f}  ({top1_acc * 100:.2f}%)")
    print(f"  Top-{topk} Accuracy : {topk_acc:.4f}  ({topk_acc * 100:.2f}%)")

    # Confusion matrix — raw counts for BI visualization
    cm      = confusion_matrix(all_labels_np, all_preds_np)
    cm_path = plot_confusion_matrix_bi(exp_id, cm, class_names)

    # Update executive report with real confusion matrix data
    generate_executive_report(
        exp_id       = exp_id,
        train_losses = [],    # not available at evaluation time — pass empty
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
        "top1_accuracy"             : top1_acc,
        f"top{topk}_accuracy"       : topk_acc,
        "confusion_matrix_path"     : str(cm_path),
        "classification_report"     : report,
    }
