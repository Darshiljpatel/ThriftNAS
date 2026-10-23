"""First-order DARTS search trainer.

Implements one step of the bilevel optimisation loop (DARTS §2, addendum §10):
  1. Alpha step  – update arch parameters on the alpha data split.
  2. Weight step – update network weights on the weight data split.

AMP is via torch.cuda.amp with one GradScaler per optimiser.
The expected cost E(alpha) is ALWAYS computed and logged (detached) even when
lam == 0, so the column is meaningful in all ablation runs.

CSV columns (exact, matches PROJECT_SPEC section 3)
----------------------------------------------------
epoch, lr, ce_w, ce_a, eff, lam_eff, train_acc, val_acc,
skip_count, epoch_time_s, peak_gpu_mb
"""

from __future__ import annotations

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
# Column specification (must match the spec exactly)
# ---------------------------------------------------------------------------

CSV_COLUMNS: tuple[str, ...] = (
    "epoch", "lr", "ce_w", "ce_a", "eff", "lam_eff",
    "train_acc", "val_acc", "skip_count", "epoch_time_s", "peak_gpu_mb",
)


# ---------------------------------------------------------------------------
# SearchTrainer
# ---------------------------------------------------------------------------

class SearchTrainer:
    """First-order DARTS trainer.

    Parameters
    ----------
    model:
        A :class:`~thriftnas.models.search_net.SearchNetwork` instance.
    c1:
        Stride-1 op costs tensor, shape ``(8,)``, ``PRIMITIVES`` order.
    c2:
        Stride-2 op costs tensor, shape ``(8,)``, ``PRIMITIVES`` order.
    lam:
        Cost regularisation strength.  When ``lam == 0`` the alpha step uses
        CE only; ``E(alpha)`` is still computed and logged (detached).
    epochs:
        Total training epochs.  Used by the CosineAnnealingLR scheduler.
    amp:
        Enable CUDA automatic mixed precision (float16).  Ignored silently
        when the model is on CPU.
    out_dir:
        Directory for checkpoints, alpha snapshots, and the CSV log.
    """

    def __init__(
        self,
        model: nn.Module,
        c1: torch.Tensor,
        c2: torch.Tensor,
        lam: float = 0.0,
        epochs: int = 25,
        amp: bool = False,
        out_dir: str | Path = "results/search",
    ) -> None:
        self.model   = model
        self.c1      = c1
        self.c2      = c2
        self.lam     = lam
        self.epochs  = epochs
        self.amp     = amp
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._epoch  = 0   # next epoch to run (updated after each epoch)

        # ---- Alpha optimiser (Adam, spec §3) --------------------------------
        self.a_opt = torch.optim.Adam(
            model.arch_parameters(),
            lr=3e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-3,
        )

        # ---- Weight optimiser (SGD + cosine LR, spec §3) -------------------
        self.w_opt = torch.optim.SGD(
            model.weights(),
            lr=0.025,
            momentum=0.9,
            weight_decay=3e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.w_opt, T_max=epochs, eta_min=0.001,
        )

        # ---- GradScalers (no-ops when enabled=False) -----------------------
        self.a_scaler = torch.amp.GradScaler("cuda", enabled=amp)
        self.w_scaler = torch.amp.GradScaler("cuda", enabled=amp)

        # ---- CSV logger ----------------------------------------------------
        self._csv = CSVLogger(self.out_dir / "search_log.csv")

    # -----------------------------------------------------------------------
    # Core mini-batch steps (public so tests can call them directly)
    # -----------------------------------------------------------------------

    def alpha_step(
        self,
        xa: torch.Tensor,
        ya: torch.Tensor,
    ) -> tuple[float, float, float]:
        """Perform one alpha optimisation step.

        1. Zero alpha grads.
        2. Forward pass under autocast → CE loss (float16/32).
        3. Compute E(alpha) in float32 outside autocast.
        4. Combine: loss_a = ce_a.float() [+ lam * eff if lam > 0].
        5. Backward + scaler step.

        Parameters
        ----------
        xa, ya:  Alpha-data batch (inputs, labels).

        Returns
        -------
        (ce_a_item, eff_item, lam_eff_item)
            All float scalars detached from the computation graph.

        Raises
        ------
        SystemExit(1):
            On ``torch.cuda.OutOfMemoryError`` – prints guidance and exits.
        """
        model = self.model
        self.a_opt.zero_grad(set_to_none=True)

        # Forward on alpha data (inside autocast when AMP enabled)
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                ce_a = F.cross_entropy(model(xa), ya)
        except torch.cuda.OutOfMemoryError:
            print("Lower --batch_size (try 32)")
            sys.exit(1)

        # Expected cost – float32, OUTSIDE autocast, always computed
        eff: torch.Tensor = model.expected_cost(self.c1, self.c2)

        # Build alpha loss (spec reference solution)
        loss_a = ce_a.float() + self.lam * eff if self.lam > 0 else ce_a.float()

        self.a_scaler.scale(loss_a).backward()
        self.a_scaler.step(self.a_opt)
        self.a_scaler.update()

        log_eff     = float(eff.detach())
        log_lam_eff = float(self.lam * eff.detach())
        return float(ce_a.float().detach()), log_eff, log_lam_eff

    def weight_step(
        self,
        xw: torch.Tensor,
        yw: torch.Tensor,
    ) -> tuple[float, float]:
        """Perform one weight optimisation step.

        1. Zero weight grads (clears any alpha-backward residuals).
        2. Forward under autocast → CE loss.
        3. Backward via w_scaler.
        4. unscale_ → clip_grad_norm_(5) → w_scaler.step → update.

        Parameters
        ----------
        xw, yw:  Weight-data batch (inputs, labels).

        Returns
        -------
        (ce_w_item, top1_acc)
            CE loss and top-1 training accuracy (%).

        Raises
        ------
        SystemExit(1):
            On ``torch.cuda.OutOfMemoryError``.
        """
        model = self.model
        self.w_opt.zero_grad(set_to_none=True)

        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                logits_w = model(xw)
                ce_w     = F.cross_entropy(logits_w, yw)
        except torch.cuda.OutOfMemoryError:
            print("Lower --batch_size (try 32)")
            sys.exit(1)

        self.w_scaler.scale(ce_w).backward()
        self.w_scaler.unscale_(self.w_opt)                     # order: 1st
        nn.utils.clip_grad_norm_(model.weights(), 5.0)         # order: 2nd
        self.w_scaler.step(self.w_opt)                         # order: 3rd
        self.w_scaler.update()

        acc = accuracy(logits_w.float().detach(), yw)
        return float(ce_w.float().detach()), acc

    # -----------------------------------------------------------------------
    # Validation (alpha subset only – spec §3, "never touch the test set")
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, alpha_loader) -> float:
        """Top-1 accuracy on the alpha data split (used as validation set)."""
        model  = self.model
        device = next(model.parameters()).device
        model.eval()
        meter  = AverageMeter()
        for x, y in alpha_loader:
            x, y = x.to(device), y.to(device)
            meter.update(accuracy(model(x), y), n=x.size(0))
        model.train()
        return meter.avg

    # -----------------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------------

    def train_epoch(
        self,
        weight_loader,
        alpha_loader,
        epoch: int,
    ) -> dict[str, Any]:
        """Run one full training epoch.

        Parameters
        ----------
        weight_loader:  DataLoader for the weight update split (50 k subset).
        alpha_loader:   DataLoader for the alpha update + validation split.
        epoch:          Zero-based epoch index.

        Returns
        -------
        dict
            The row logged to the CSV (column names are :data:`CSV_COLUMNS`).
        """
        model  = self.model
        device = next(model.parameters()).device
        model.train()

        # Reset peak GPU memory counter at the start of each epoch
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        ce_w_m    = AverageMeter()
        ce_a_m    = AverageMeter()
        eff_m     = AverageMeter()
        train_m   = AverageMeter()

        t0 = time.perf_counter()

        for (xw, yw), (xa, ya) in zip(weight_loader, alpha_loader):
            xw, yw = xw.to(device), yw.to(device)
            xa, ya = xa.to(device), ya.to(device)
            n = xw.size(0)

            # --- alpha step ---
            ce_a, eff, _lam_eff = self.alpha_step(xa, ya)
            ce_a_m.update(ce_a, n)
            eff_m.update(eff, n)

            # --- weight step ---
            ce_w, acc = self.weight_step(xw, yw)
            ce_w_m.update(ce_w, n)
            train_m.update(acc, n)

        # Validate on alpha subset (spec: "never touch the test set")
        val_acc = self._validate(alpha_loader)

        # Cosine LR scheduler steps once per epoch
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]

        epoch_time_s = time.perf_counter() - t0
        peak_gpu_mb  = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        )

        # skip_connect count in current discrete genotype (normal + reduce)
        geno       = model.genotype()
        skip_count = sum(
            1 for op, _ in geno["normal"] + geno["reduce"]
            if op == "skip_connect"
        )

        row: dict[str, Any] = {
            "epoch":        epoch,
            "lr":           round(lr, 8),
            "ce_w":         round(ce_w_m.avg, 6),
            "ce_a":         round(ce_a_m.avg, 6),
            "eff":          round(eff_m.avg, 6),
            "lam_eff":      round(self.lam * eff_m.avg, 6),
            "train_acc":    round(train_m.avg, 4),
            "val_acc":      round(val_acc, 4),
            "skip_count":   skip_count,
            "epoch_time_s": round(epoch_time_s, 2),
            "peak_gpu_mb":  round(peak_gpu_mb, 1),
        }

        self._csv.log(row)
        self._save_checkpoint(self.out_dir / "checkpoint.pt", epoch)
        self._save_alpha_snapshot(epoch)
        self._epoch = epoch + 1
        return row

    # -----------------------------------------------------------------------
    # Checkpointing (atomic)
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        """Atomically write full training state to *path*."""
        state = {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "a_opt_state":     self.a_opt.state_dict(),
            "w_opt_state":     self.w_opt.state_dict(),
            "a_scaler_state":  self.a_scaler.state_dict(),
            "w_scaler_state":  self.w_scaler.state_dict(),
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

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore full training state from *path*.

        After loading, :attr:`start_epoch` returns the next epoch to run.
        """
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.a_opt.load_state_dict(state["a_opt_state"])
        self.w_opt.load_state_dict(state["w_opt_state"])
        self.a_scaler.load_state_dict(state["a_scaler_state"])
        self.w_scaler.load_state_dict(state["w_scaler_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])

        rng = state["rng"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])

        self._epoch = state["epoch"] + 1  # resume AFTER the saved epoch

    def _save_alpha_snapshot(self, epoch: int) -> None:
        """Save current alpha tensors to a named snapshot file."""
        torch.save(
            {
                "alpha_normal": self.model.alpha_normal.detach().cpu(),
                "alpha_reduce": self.model.alpha_reduce.detach().cpu(),
            },
            self.out_dir / f"alpha_ep{epoch:03d}.pt",
        )

    # -----------------------------------------------------------------------
    # Property
    # -----------------------------------------------------------------------

    @property
    def start_epoch(self) -> int:
        """Next epoch index to run (0 at construction; updated after each epoch)."""
        return self._epoch
