# ============================================================
# MODULE: utils.py
# PURPOSE: Cross-cutting infrastructure — seed fixing and
#          experiment artifact persistence (JSON only).
#          Visualization delegated to visualization.py.
# NORMATIVE BASIS: UCB Project 5 rubric — traceability
#                  requirement: one JSON per experiment run.
# AUTHOR: Guillermo Carvajal Vaca — UCB MSc Data Science & AI
# VERSION: 1.0.0
# ============================================================
from __future__ import annotations

# --- stdlib (alphabetical) ---
import json
import logging
import random
from functools import wraps
from pathlib import Path
from typing import Any

# --- third-party (alphabetical) ---
import numpy as np
import torch

# --- local (alphabetical) ---
from src.config import EXPERIMENTS_DIR, SEED

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ERROR HANDLING — production-grade decorator
# Why: bare exceptions lose context and expose internals.
# Structured logging enables post-hoc debugging of failed runs.
# ---------------------------------------------------------------------------
def with_error_context(func):
    """
    Decorator that captures full error context without exposing internals.

    Logs validation errors at ERROR level and unexpected failures at
    CRITICAL, then re-raises so the caller decides whether to abort.

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function with structured error logging.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.error("[VALIDATION_ERROR] %s: %s", func.__name__, e)
            raise
        except Exception as e:
            logger.critical(
                "[CRITICAL] %s -> %s: %s", func.__name__, type(e).__name__, e
            )
            raise RuntimeError(f"Unexpected failure in {func.__name__}") from e
    return wrapper


# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """
    Fix all sources of randomness for full reproducibility across runs.

    Args:
        seed: Integer seed applied to Python, NumPy, and PyTorch RNGs.

    Why deterministic=True + benchmark=False:
        benchmark=True lets cuDNN pick the fastest conv algorithm per input
        shape, but that choice varies between runs. Disabling it trades a
        small speed cost for exact reproducibility — mandatory for ablation
        studies where a single changed factor must explain any delta.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    logger.debug("Seed fixed to %d", seed)


# ---------------------------------------------------------------------------
# ARTIFACT PERSISTENCE
# ---------------------------------------------------------------------------
@with_error_context
def save_metrics(exp_id: str, metrics: dict[str, Any]) -> Path:
    """
    Persist experiment metrics to a JSON artifact.

    Args:
        exp_id:  Unique experiment identifier (e.g. 'E3_scratch_5conv_bn').
        metrics: Dict containing hyperparameters, per-epoch curves, and
                 final test results. Must be JSON-serializable.

    Returns:
        Path to the generated JSON file.

    Raises:
        RuntimeError: If the file cannot be written (wrapped by decorator).

    Why JSON and not CSV:
        Metrics include nested structures (curves as lists, hyperparams as
        dicts). JSON preserves hierarchy without flattening — readable by
        both humans and downstream analysis scripts.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPERIMENTS_DIR / f"{exp_id}_metrics.json"

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    logger.info("Metrics saved -> %s", out_path)
    return out_path
