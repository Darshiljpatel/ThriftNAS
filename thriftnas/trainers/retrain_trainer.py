"""Retraining trainer for ThriftNAS (PROJECT_SPEC §5, §7, §8).

Trains a discrete :class:`~thriftnas.models.retrain_net.NetworkCIFAR` on the
full 50 k CIFAR-10 training set.  The test set is evaluated **exactly once**
after the final epoch; no test metrics appear in the per-epoch CSV.

CSV columns (exact, no "test" column)
--------------------------------------
epoch, lr, train_loss, train_acc, drop_path, epoch_time_s, peak_gpu_mb

Checkpointing
-------------
- ``checkpoint.pt``  – full training state (resume checkpoint, bigger).
- ``model_weights.pt`` – weights-only state dict (for evaluation, smaller).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thriftnas.utils import AverageMeter, CSVLogger, accuracy, get_device

# ---------------------------------------------------------------------------
# Exact per-epoch CSV columns  (spec §8: no "test" column)
# ---------------------------------------------------------------------------

CSV_COLUMNS: tuple[str, ...] = (
    "epoch", "lr", "train_loss", "train_acc",
    "drop_path", "epoch_time_s", "peak_gpu_mb",
)


# ---------------------------------------------------------------------------
# RetrainTrainer
# ---------------------------------------------------------------------------

class RetrainTrainer:
    """Retrain a discrete DARTS genotype on the full CIFAR-10 training set.

    Parameters
    ----------
    model:
        A :class:`~thriftnas.models.retrain_net.NetworkCIFAR` instance.
    epochs:
        Total training epochs.  Used by CosineAnnealingLR (T_max=epochs).
    peak_drop_path:
        Peak drop-path probability (0 → ``peak_drop_path`` linearly).
    amp:
        Enable CUDA AMP (float16).  Silently ignored on CPU.
    out_dir:
        Directory for checkpoint, weights file, and CSV log.
    """

    def __init__(
        self,
        model: nn.Module,
        epochs: int = 50,
        peak_drop_path: float = 0.2,
        amp: bool = False,
        out_dir: str | Path = "results/retrain",
    ) -> None:
        self.model          = model
        self.epochs         = epochs
        self.peak_drop_path = peak_drop_path
        self.amp            = amp
        self.out_dir        = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._epoch         = 0   # next epoch to run

        # ---- Optimiser (SGD + cosine LR, spec §5) --------------------------
        self.opt = torch.optim.SGD(
            model.parameters(),
            lr=0.025,
            momentum=0.9,
            weight_decay=3e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=epochs,
        )

        # ---- GradScaler (no-op when enabled=False) -------------------------
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp)

        # ---- CSV logger (spec: no test column) ----------------------------
        self._csv = CSVLogger(self.out_dir / "retrain_log.csv")

    # -----------------------------------------------------------------------
    # One training step
    # -----------------------------------------------------------------------

    def _train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[float, float]:
        """Forward + backward + optimizer step for one mini-batch.

        Returns
        -------
        (loss_item, top1_acc)
        """
        self.opt.zero_grad(set_to_none=True)

        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                logits = self.model(x)
                loss   = F.cross_entropy(logits, y)
        except torch.cuda.OutOfMemoryError:
            print("Lower --batch_size (try 64)")
            sys.exit(1)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.scaler.step(self.opt)
        self.scaler.update()

        acc = accuracy(logits.float().detach(), y)
        return float(loss.float().detach()), acc

    # -----------------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------------

    def train_epoch(self, train_loader, epoch: int) -> dict[str, Any]:
        """Run one full training epoch.

        Drop-path probability ramps linearly:
            ``drop_p = peak_drop_path * epoch / (epochs - 1)``
        (epoch 0 → 0; final epoch → peak_drop_path).

        Parameters
        ----------
        train_loader:
            DataLoader over the full 50 k training set.
        epoch:
            Zero-based epoch index.

        Returns
        -------
        dict
            Row logged to the CSV (keys match :data:`CSV_COLUMNS`).
        """
        model  = self.model
        device = next(model.parameters()).device

        # Linear drop-path ramp
        drop_p = (
            self.peak_drop_path * epoch / max(self.epochs - 1, 1)
        )
        model.drop_path_prob = drop_p
        model.train()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        loss_m  = AverageMeter()
        train_m = AverageMeter()
        t0      = time.perf_counter()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            n    = x.size(0)
            loss, acc = self._train_step(x, y)
            loss_m.update(loss, n)
            train_m.update(acc, n)

        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        epoch_time_s = time.perf_counter() - t0
        peak_gpu_mb  = (
            torch.cuda.max_memory_allocated(device) / 1024 ** 2
            if device.type == "cuda" else 0.0
        )

        row: dict[str, Any] = {
            "epoch":        epoch,
            "lr":           round(lr, 8),
            "train_loss":   round(loss_m.avg, 6),
            "train_acc":    round(train_m.avg, 4),
            "drop_path":    round(drop_p, 6),
            "epoch_time_s": round(epoch_time_s, 2),
            "peak_gpu_mb":  round(peak_gpu_mb, 1),
        }

        self._csv.log(row)
        self._save_checkpoint(self.out_dir / "checkpoint.pt", epoch)
        self._save_weights(self.out_dir / "model_weights.pt")
        self._epoch = epoch + 1
        return row

    # -----------------------------------------------------------------------
    # Final test evaluation  (called ONCE after the last epoch, spec §8)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_test(self, test_loader) -> dict[str, float]:
        """Evaluate on the test set exactly once after training completes.

        Parameters
        ----------
        test_loader:
            DataLoader constructed with ``get_test_loader(final_report=True)``.

        Returns
        -------
        dict with keys ``test_loss`` and ``test_acc``.
        """
        model  = self.model
        device = next(model.parameters()).device
        model.eval()

        loss_m = AverageMeter()
        acc_m  = AverageMeter()

        for x, y in test_loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = F.cross_entropy(logits, y)
            acc    = accuracy(logits.float(), y)
            n      = x.size(0)
            loss_m.update(float(loss), n)
            acc_m.update(acc, n)

        metrics = {
            "test_loss": round(loss_m.avg, 6),
            "test_acc":  round(acc_m.avg, 4),
        }

        # Write final_metrics.json atomically
        path = self.out_dir / "final_metrics.json"
        tmp  = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        os.replace(tmp, str(path))

        return metrics

    # -----------------------------------------------------------------------
    # Checkpointing (atomic)
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        """Atomically save full training state (resume checkpoint)."""
        state = {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "opt_state":       self.opt.state_dict(),
            "scaler_state":    self.scaler.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "rng": {
                "torch":  torch.get_rng_state(),
                "numpy":  np.random.get_state(),
                "python": random.getstate(),
                "cuda":   (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
            },
        }
        tmp = str(path) + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, str(path))   # atomic

    def _save_weights(self, path: Path) -> None:
        """Atomically save a weights-only state dict (spec §7 metrics file)."""
        tmp = str(path) + ".tmp"
        torch.save(self.model.state_dict(), tmp)
        os.replace(tmp, str(path))   # atomic

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore full training state from *path*.

        After loading, :attr:`start_epoch` returns the next epoch to run.
        """
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.opt.load_state_dict(state["opt_state"])
        self.scaler.load_state_dict(state["scaler_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])

        rng = state["rng"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])

        self._epoch = state["epoch"] + 1   # resume AFTER saved epoch

    # -----------------------------------------------------------------------
    # Property
    # -----------------------------------------------------------------------

    @property
    def start_epoch(self) -> int:
        """Next epoch index to run (0 at construction; updated after each epoch)."""
        return self._epoch
