"""Utility functions shared across ThriftNAS modules.

Public API
----------
set_seed, AverageMeter, accuracy, save_checkpoint, load_checkpoint,
CSVLogger, peak_memory_mb, get_device
"""

from __future__ import annotations

import csv
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from thriftnas.utils.metrics import (
    LatencyComparison,
    compare_latency,
    count_macs,
    count_params,
    cpu_latency,
    model_size_mb,
    predicted_latency,
)
from thriftnas.utils.plots import (
    pareto_front,
    plot_alpha_evolution,
    plot_e_alpha_over_epochs,
    plot_lambda_ablation,
    plot_pareto_front,
    plot_skip_count_vs_lambda,
)
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility.

    Sets Python ``random``, NumPy, and PyTorch (CPU + all GPUs) seeds, and
    enables cuDNN deterministic mode.

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops; slightly slower but reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Running statistics
# ---------------------------------------------------------------------------

class AverageMeter:
    """Tracks a running average of a scalar metric.

    Example
    -------
    >>> m = AverageMeter()
    >>> m.update(2, n=1)
    >>> m.update(4, n=3)
    >>> m.avg   # 14 / 4 = 3.5
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self.val: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
        self.avg: float = 0.0

    def update(self, val: float, n: int = 1) -> None:
        """Update meter with *val* observed *n* times.

        Parameters
        ----------
        val:
            The scalar value to record.
        n:
            Weight (number of samples this value represents).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy (%).

    Parameters
    ----------
    logits:
        Float tensor of shape ``(N, C)`` — raw model outputs.
    labels:
        Long tensor of shape ``(N,)`` — ground-truth class indices.

    Returns
    -------
    float
        Percentage of correctly classified samples in [0, 100].
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = preds.eq(labels).sum().item()
        return 100.0 * correct / labels.size(0)


# ---------------------------------------------------------------------------
# Checkpointing (atomic)
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict[str, Any], path: str) -> None:
    """Atomically save *state* to *path*.

    Writes to a temporary file first, then uses ``os.replace`` (atomic on
    the same filesystem/drive) to swap it into place.  If ``torch.save``
    raises for any reason, the previous checkpoint at *path* is untouched.

    Parameters
    ----------
    state:
        Dictionary containing the data to checkpoint (model weights,
        optimiser state, epoch, scaler state, etc.).
    path:
        Destination file path (e.g. ``"checkpoints/epoch_5.pt"``).
    """
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic on the same drive (POSIX rename semantics)


def load_checkpoint(path: str) -> dict[str, Any]:
    """Load a checkpoint from *path*.

    Parameters
    ----------
    path:
        File path written by :func:`save_checkpoint`.

    Returns
    -------
    dict
        The state dictionary stored in the checkpoint.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class CSVLogger:
    """Append-only CSV logger that writes a header exactly once.

    Opening a :class:`CSVLogger` on an existing file **appends** data
    without repeating the header.  A new file gets a header on the first
    :meth:`log` call.

    Parameters
    ----------
    path:
        Destination CSV file path.

    Example
    -------
    >>> logger = CSVLogger("results/run.csv")
    >>> logger.log({"epoch": 1, "loss": 0.5})
    >>> logger.log({"epoch": 2, "loss": 0.3})
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = Path(path)
        # Track whether the file already exists and is non-empty so we know
        # whether to write a header.
        self._header_written: bool = (
            self._path.exists() and self._path.stat().st_size > 0
        )

    def log(self, row: dict[str, Any]) -> None:
        """Append *row* to the CSV file.

        Parameters
        ----------
        row:
            Mapping from column names to values.  All rows must share the
            same keys (standard CSV restriction).
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def peak_memory_mb() -> float:
    """Return peak GPU memory allocated since the last reset (MB).

    Returns 0.0 when CUDA is unavailable (CPU-only runs).

    Returns
    -------
    float
        Peak memory in megabytes.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def get_device() -> torch.device:
    """Return the best available :class:`torch.device`.

    Prefers CUDA if available, otherwise falls back to CPU.

    Returns
    -------
    torch.device
        ``torch.device("cuda")`` or ``torch.device("cpu")``.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
