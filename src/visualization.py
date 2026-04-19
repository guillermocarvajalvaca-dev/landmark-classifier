# ============================================================
# MODULE: visualization.py
# PURPOSE: BI-grade training visualizations with narrative
#          storytelling. Answers: "Is the model robust enough
#          for production?" Replaces basic curve plotting with
#          Grammar of Graphics (plotnine) + executive reports.
# NORMATIVE BASIS: UCB Project 5 rubric — Phase 2/3 curve
#                  requirements. UCB corporate palette applied.
# AUTHOR: Guillermo Carvajal Vaca — UCB MSc Data Science & AI
# VERSION: 1.0.0
# ============================================================
from __future__ import annotations

# --- stdlib (alphabetical) ---
import logging
import textwrap
from pathlib import Path

# --- third-party (alphabetical) ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotnine import (
    aes,
    annotate,
    element_blank,
    element_line,
    element_rect,
    element_text,
    geom_hline,
    geom_line,
    geom_point,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_minimal,
)

# --- local (alphabetical) ---
from src.config import DOCS_DIR, EXPERIMENTS_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UCB CORPORATE PALETTE
# Why these exact hex codes: institutional brand consistency across all
# deliverables submitted to UCB.
# #003262 = UCB Dark Blue  (primary)
# #FDB515 = UCB Gold       (highlight / optimal point)
# #C4820E = UCB Dark Gold  (warning / overfitting)
# ---------------------------------------------------------------------------
UCB_BLUE      : str = "#003262"
UCB_GOLD      : str = "#FDB515"
UCB_DARK_GOLD : str = "#C4820E"

PRODUCTION_THRESHOLD : float = 0.85   # rubric bonus threshold


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------
def _detect_overfitting_epoch(val_losses: list[float], patience: int = 3) -> int | None:
    """
    Detect the first epoch where overfitting begins.

    Args:
        val_losses: Per-epoch validation loss sequence.
        patience:   Consecutive epochs of increasing val_loss to confirm onset.

    Returns:
        1-indexed epoch number where overfitting starts, or None if not detected.

    Why patience=3:
        A single spike in val_loss can be noise. Three consecutive increases
        signal a structural trend — the model has stopped generalizing.
    """
    min_idx = int(np.argmin(val_losses))
    count   = 0

    for i in range(min_idx + 1, len(val_losses)):
        if val_losses[i] > val_losses[i - 1]:
            count += 1
            if count >= patience:
                return (i - patience + 1) + 1
        else:
            count = 0

    return None


def _build_curve_dataframe(
    train_losses : list[float],
    val_losses   : list[float],
    train_accs   : list[float],
    val_accs     : list[float],
) -> pd.DataFrame:
    """
    Reshape per-epoch lists into tidy long-format DataFrame for plotnine.

    Args:
        train_losses: Per-epoch training loss.
        val_losses:   Per-epoch validation loss.
        train_accs:   Per-epoch training accuracy.
        val_accs:     Per-epoch validation accuracy.

    Returns:
        Tidy DataFrame with columns [epoch, value, metric, split].

    Why long format:
        plotnine maps aesthetics to columns. Long format lets us map
        split to color with a single aes() call — no duplicated geom layers.
    """
    epochs  = list(range(1, len(train_losses) + 1))
    records = []

    for ep, tl, vl, ta, va in zip(epochs, train_losses, val_losses, train_accs, val_accs):
        records.extend([
            {"epoch": ep, "value": tl, "metric": "Loss",     "split": "Train"},
            {"epoch": ep, "value": vl, "metric": "Loss",     "split": "Validation"},
            {"epoch": ep, "value": ta, "metric": "Accuracy", "split": "Train"},
            {"epoch": ep, "value": va, "metric": "Accuracy", "split": "Validation"},
        ])

    return pd.DataFrame(records)


def _ucb_theme() -> theme:
    """
    Return UCB-branded plotnine theme.

    Why a dedicated theme function:
        Centralizing theme config avoids repetition across plots and ensures
        all visuals share the same typographic and color hierarchy.
    """
    return (
        theme_minimal()
        + theme(
            plot_title       = element_text(size=13, weight="bold", color=UCB_BLUE),
            plot_subtitle    = element_text(size=9,  color="#555555"),
            axis_title       = element_text(size=9,  color=UCB_BLUE),
            axis_text        = element_text(size=8),
            legend_title     = element_blank(),
            legend_text      = element_text(size=8),
            panel_grid_major = element_line(color="#EEEEEE"),
            panel_grid_minor = element_blank(),
            panel_background = element_rect(fill="white"),
        )
    )


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------
def plot_training_narrative(
    exp_id       : str,
    train_losses : list[float],
    val_losses   : list[float],
    train_accs   : list[float],
    val_accs     : list[float],
) -> Path:
    """
    Generate a BI-grade training narrative plot.

    Business question answered: "Is the model robust enough for production?"

    Storytelling elements:
        - Optimal model point (min val_loss) marked with UCB Gold line.
        - Overfitting onset annotated with a warning label.
        - Production threshold reference line at 85% accuracy.

    Args:
        exp_id:       Experiment identifier used as filename prefix and title.
        train_losses: Per-epoch training CrossEntropyLoss.
        val_losses:   Per-epoch validation CrossEntropyLoss.
        train_accs:   Per-epoch training accuracy (0-1).
        val_accs:     Per-epoch validation accuracy (0-1).

    Returns:
        Path to the saved PNG artifact.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    optimal_epoch   : int        = int(np.argmin(val_losses)) + 1
    optimal_val_acc : float      = val_accs[optimal_epoch - 1]
    overfit_epoch   : int | None = _detect_overfitting_epoch(val_losses)
    total_epochs    : int        = len(train_losses)
    x_breaks        : list[int]  = list(range(1, total_epochs + 1, max(1, total_epochs // 10)))

    logger.info(
        "[%s] Optimal epoch: %d | Val acc: %.3f | Overfit onset: %s",
        exp_id, optimal_epoch, optimal_val_acc,
        str(overfit_epoch) if overfit_epoch else "not detected",
    )

    df       = _build_curve_dataframe(train_losses, val_losses, train_accs, val_accs)
    df_loss  = df[df["metric"] == "Loss"].copy()
    df_acc   = df[df["metric"] == "Accuracy"].copy()
    color_map = {"Train": UCB_BLUE, "Validation": UCB_DARK_GOLD}

    # --- Loss panel ---
    p_loss = (
        ggplot(df_loss, aes(x="epoch", y="value", color="split"))
        + geom_line(size=1.1)
        + geom_point(size=1.5, alpha=0.7)
        + geom_vline(xintercept=optimal_epoch, color=UCB_GOLD, linetype="dashed", size=0.9)
        + annotate(
            "text", x=optimal_epoch + 0.4, y=max(val_losses) * 0.95,
            label="Optimal\nEpoch " + str(optimal_epoch),
            color=UCB_DARK_GOLD, size=7, ha="left",
        )
        + scale_color_manual(values=color_map)
        + scale_x_continuous(breaks=x_breaks)
        + labs(
            title    = exp_id + " — Training Diagnostic",
            subtitle = "Validation loss minimum marks the optimal checkpoint",
            x        = "Epoch",
            y        = "CrossEntropyLoss",
        )
        + _ucb_theme()
    )

    if overfit_epoch:
        p_loss = p_loss + annotate(
            "text",
            x=overfit_epoch, y=val_losses[overfit_epoch - 1] * 1.05,
            label="Overfit onset\nEpoch " + str(overfit_epoch),
            color="#D32F2F", size=7, ha="center",
        )

    # --- Accuracy panel ---
    p_acc = (
        ggplot(df_acc, aes(x="epoch", y="value", color="split"))
        + geom_line(size=1.1)
        + geom_point(size=1.5, alpha=0.7)
        + geom_hline(yintercept=PRODUCTION_THRESHOLD, color="#D32F2F", linetype="dotted", size=0.8)
        + geom_vline(xintercept=optimal_epoch, color=UCB_GOLD, linetype="dashed", size=0.9)
        + annotate(
            "text", x=total_epochs * 0.02, y=PRODUCTION_THRESHOLD + 0.02,
            label="Production threshold (" + str(int(PRODUCTION_THRESHOLD * 100)) + "%)",
            color="#D32F2F", size=7, ha="left",
        )
        + annotate(
            "point", x=optimal_epoch, y=optimal_val_acc,
            color=UCB_GOLD, size=4,
        )
        + annotate(
            "text", x=optimal_epoch + 0.4, y=optimal_val_acc - 0.03,
            label="Best: " + str(round(optimal_val_acc * 100, 1)) + "%",
            color=UCB_DARK_GOLD, size=7, ha="left",
        )
        + scale_color_manual(values=color_map)
        + scale_x_continuous(breaks=x_breaks)
        + scale_y_continuous(limits=[0, 1])
        + labs(
            title    = exp_id + " — Accuracy",
            subtitle = "Optimal val accuracy at epoch " + str(optimal_epoch),
            x        = "Epoch",
            y        = "Accuracy",
        )
        + _ucb_theme()
    )

    # --- Render both panels side by side ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")

    for p, ax in [(p_loss, ax1), (p_acc, ax2)]:
        p_fig = p.draw()
        p_fig.canvas.draw()
        img = np.frombuffer(p_fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(p_fig.canvas.get_width_height()[::-1] + (3,))
        ax.imshow(img)
        ax.axis("off")
        plt.close(p_fig)

    plt.tight_layout(pad=1.5)
    out_path = EXPERIMENTS_DIR / (exp_id + "_narrative.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info("Narrative plot saved -> %s", out_path)
    return out_path


def plot_confusion_matrix_bi(
    exp_id      : str,
    cm          : np.ndarray,
    class_names : list[str],
) -> Path:
    """
    Generate a BI-grade confusion matrix with Top-3 business error table.

    Storytelling elements:
        - Row-normalized heatmap in UCB Blue palette.
        - Top-3 most confused class pairs as an embedded executive alert table.

    Args:
        exp_id:       Experiment identifier.
        cm:           Raw confusion matrix from sklearn (shape [N, N]).
        class_names:  Ordered class labels matching ImageFolder order.

    Returns:
        Path to the saved PNG artifact.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    n        = len(class_names)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = cm.astype(float) / (row_sums + 1e-8)

    # Off-diagonal only: diagonal = correct predictions, not errors
    error_matrix = cm_norm.copy()
    np.fill_diagonal(error_matrix, 0)

    flat_indices = np.argsort(error_matrix.ravel())[::-1][:3]
    top3_errors  = [
        {
            "True Class"     : class_names[idx // n],
            "Predicted As"   : class_names[idx  % n],
            "Confusion Rate" : str(round(error_matrix.ravel()[idx] * 100, 1)) + "%",
        }
        for idx in flat_indices
    ]
    top3_df = pd.DataFrame(top3_errors)

    fig_size = max(14, n // 2)
    fig      = plt.figure(figsize=(fig_size + 4, fig_size * 0.9), facecolor="white")
    ax_hm    = fig.add_axes([0.0,  0.15, 0.82, 0.80])
    ax_table = fig.add_axes([0.84, 0.50, 0.15, 0.35])

    sns.heatmap(
        cm_norm,
        ax          = ax_hm,
        cmap        = sns.light_palette(UCB_BLUE, as_cmap=True),
        xticklabels = class_names,
        yticklabels = class_names,
        vmin=0, vmax=1,
        annot       = (n <= 25),
        fmt         = ".2f",
        linewidths  = 0.3,
        cbar_kws    = {"shrink": 0.6, "label": "Confusion Rate"},
    )
    ax_hm.set_title(
        exp_id + "\nConfusion Matrix — Row Normalized",
        fontsize=12, fontweight="bold", color=UCB_BLUE, pad=12,
    )
    ax_hm.set_xlabel("Predicted Label", fontsize=9, color=UCB_BLUE)
    ax_hm.set_ylabel("True Label",      fontsize=9, color=UCB_BLUE)

    tick_size = max(5, 10 - n // 10)
    ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=45, ha="right", fontsize=tick_size)
    ax_hm.set_yticklabels(ax_hm.get_yticklabels(), rotation=0,  fontsize=tick_size)

    ax_table.axis("off")
    ax_table.set_title(
        "Top-3 Business Errors\n(Data Collection Targets)",
        fontsize=8, fontweight="bold", color="#D32F2F", pad=6,
    )
    tbl = ax_table.table(
        cellText  = top3_df.values,
        colLabels = top3_df.columns,
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.6)

    for col in range(len(top3_df.columns)):
        tbl[0, col].set_facecolor(UCB_DARK_GOLD)
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    out_path = EXPERIMENTS_DIR / (exp_id + "_confusion_bi.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info("BI confusion matrix saved -> %s", out_path)
    return out_path


def generate_executive_report(
    exp_id       : str,
    train_losses : list[float],
    val_losses   : list[float],
    val_accs     : list[float],
    class_names  : list[str],
    cm           : np.ndarray,
    test_acc     : float,
) -> Path:
    """
    Generate an automatic executive Markdown report from training metrics.

    All insights and recommendations are derived programmatically —
    no manual input required.

    Args:
        exp_id:       Experiment identifier.
        train_losses: Per-epoch training loss.
        val_losses:   Per-epoch validation loss.
        val_accs:     Per-epoch validation accuracy.
        class_names:  Ordered class labels.
        cm:           Raw confusion matrix.
        test_acc:     Final test set accuracy (0-1).

    Returns:
        Path to the generated Markdown report.
    """
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    optimal_epoch   = int(np.argmin(val_losses)) + 1
    optimal_val_acc = val_accs[optimal_epoch - 1]
    overfit_epoch   = _detect_overfitting_epoch(val_losses)
    total_epochs    = len(val_losses)
    gap             = val_losses[-1] - min(val_losses)

    row_sums    = cm.sum(axis=1)
    per_class   = np.diag(cm) / (row_sums + 1e-8)
    weak_idx    = np.argsort(per_class)[:3]
    weak_classes = [class_names[i] for i in weak_idx]

    production_ready = test_acc >= PRODUCTION_THRESHOLD
    status_label     = "READY" if production_ready else "NOT READY"

    overfit_str = (
        "epoch " + str(overfit_epoch)
        if overfit_epoch
        else "not detected"
    )
    overfit_insight = (
        "Continuing training beyond that point degrades generalization "
        "— overfitting onset detected at epoch " + str(overfit_epoch) + "."
        if overfit_epoch
        else "No sustained overfitting was detected."
    )
    next_step = (
        "Proceed to Transfer Learning (Phase 3)."
        if test_acc >= 0.40
        else "Phase 2 threshold not met — review architecture and augmentation."
    )

    report = textwrap.dedent(
        "# Executive Report — " + exp_id + "\n"
        "> Auto-generated by visualization.py | UCB MSc Data Science & AI\n\n"
        "## Production Readiness Assessment\n"
        "| Metric | Value | Threshold | Status |\n"
        "|---|---|---|---|\n"
        "| Test Accuracy | " + str(round(test_acc * 100, 1)) + "% | "
        + str(int(PRODUCTION_THRESHOLD * 100)) + "% | " + status_label + " |\n"
        "| Optimal Val Accuracy | " + str(round(optimal_val_acc * 100, 1)) + "% | — | — |\n"
        "| Generalization Gap | " + str(round(gap, 4)) + " | — | "
        + ("High" if gap > 0.1 else "Acceptable") + " |\n\n"
        "## Automatic Insight\n"
        "The model reaches a performance plateau at **epoch " + str(optimal_epoch) + "** "
        "(val_loss = " + str(round(min(val_losses), 4)) + "). " + overfit_insight + "\n\n"
        "## Actionable Recommendations\n"
        "1. **Early Stopping**: Apply early stopping at epoch **"
        + str(overfit_epoch or optimal_epoch) + "**.\n"
        "2. **Data Collection Targets** (lowest per-class accuracy):\n"
        "   - **" + weak_classes[0] + "**\n"
        "   - **" + weak_classes[1] + "**\n"
        "   - **" + weak_classes[2] + "**\n"
        "3. **Next Experiment**: " + next_step + "\n\n"
        "## Training Summary\n"
        "- Total epochs: " + str(total_epochs) + "\n"
        "- Optimal checkpoint: epoch " + str(optimal_epoch) + "\n"
        "- Overfitting onset: " + overfit_str + "\n"
        "- Final test accuracy: **" + str(round(test_acc * 100, 1)) + "%**\n"
    )

    out_path = DOCS_DIR / (exp_id + "_EXECUTIVE_REPORT.md")
    out_path.write_text(report, encoding="utf-8")

    logger.info("Executive report saved -> %s", out_path)
    return out_path
