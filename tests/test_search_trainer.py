"""Tests for thriftnas/trainers/search_trainer.py (Step 6 – T1 through T10).

Run with:  pytest tests/test_search_trainer.py -v
T11 is MANUAL (real GPU, 3 lambda values).
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from thriftnas.models.search_net import SearchNetwork
from thriftnas.ops.primitives import PRIMITIVES
from thriftnas.trainers.search_trainer import CSV_COLUMNS, SearchTrainer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_C_TINY   = 4    # tiny channel multiplier → fast CPU forward pass
_N_LAYERS = 5    # keep standard architecture


def _make_model(seed: int = 0) -> SearchNetwork:
    torch.manual_seed(seed)
    return SearchNetwork(C=_C_TINY, num_classes=10, layers=_N_LAYERS)


def _make_costs(seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Return small but non-zero cost tensors in PRIMITIVES order."""
    torch.manual_seed(seed)
    c1 = torch.rand(len(PRIMITIVES)) * 0.5 + 0.1   # in [0.1, 0.6]
    c2 = torch.rand(len(PRIMITIVES)) * 0.5 + 0.1
    return c1, c2


def _make_batch(n: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, 10, (n,))
    return x, y


def _make_trainer(
    model: SearchNetwork,
    c1: torch.Tensor,
    c2: torch.Tensor,
    lam: float = 0.0,
    tmp_dir: str | Path | None = None,
) -> SearchTrainer:
    out = tmp_dir or tempfile.mkdtemp()
    return SearchTrainer(model, c1, c2, lam=lam, epochs=5, amp=False, out_dir=out)


def _simple_loader(n_batches: int = 2, batch: int = 8) -> DataLoader:
    """Return a DataLoader yielding *n_batches* fixed batches."""
    torch.manual_seed(7)
    x = torch.randn(n_batches * batch, 3, 32, 32)
    y = torch.randint(0, 10, (n_batches * batch,))
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)


# ---------------------------------------------------------------------------
# T1 – one full step: losses finite, alpha changes, weights stable per step
# ---------------------------------------------------------------------------

class TestOneStep:
    """T1: one alpha+weight step has the correct update pattern."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.model = _make_model(seed=1)
        c1, c2     = _make_costs()
        self.trainer = _make_trainer(self.model, c1, c2, lam=0.1,
                                     tmp_dir=str(tmp_path))
        self.xa, self.ya = _make_batch(n=8, seed=10)
        self.xw, self.yw = _make_batch(n=8, seed=20)

    def _clone_weights(self):
        return [p.detach().clone() for p in self.model.weights()]

    def _clone_alpha(self):
        return (self.model.alpha_normal.detach().clone(),
                self.model.alpha_reduce.detach().clone())

    def test_losses_finite(self):
        ce_a, eff, _  = self.trainer.alpha_step(self.xa, self.ya)
        ce_w, acc     = self.trainer.weight_step(self.xw, self.yw)
        assert torch.isfinite(torch.tensor(ce_a)), "ce_a not finite"
        assert torch.isfinite(torch.tensor(eff)),  "eff not finite"
        assert torch.isfinite(torch.tensor(ce_w)), "ce_w not finite"
        assert eff > 0, "eff should be > 0 with non-zero costs"

    def test_alpha_changes_after_alpha_step(self):
        an_before, ar_before = self._clone_alpha()
        self.trainer.alpha_step(self.xa, self.ya)
        an_after, ar_after = self._clone_alpha()
        assert not torch.allclose(an_before, an_after), \
            "alpha_normal unchanged after alpha_step"

    def test_weights_unchanged_during_alpha_step(self):
        w_before = self._clone_weights()
        self.trainer.alpha_step(self.xa, self.ya)
        w_after = self._clone_weights()
        for wb, wa in zip(w_before, w_after):
            assert torch.allclose(wb, wa, atol=0), \
                "A weight tensor changed during alpha_step"

    def test_alpha_unchanged_during_weight_step(self):
        # Run alpha step first so alpha is in a non-initial state
        self.trainer.alpha_step(self.xa, self.ya)
        an_mid, ar_mid = self._clone_alpha()
        self.trainer.weight_step(self.xw, self.yw)
        an_end, ar_end = self._clone_alpha()
        assert torch.allclose(an_mid, an_end, atol=0), \
            "alpha_normal changed during weight_step"
        assert torch.allclose(ar_mid, ar_end, atol=0), \
            "alpha_reduce changed during weight_step"


# ---------------------------------------------------------------------------
# T2 – lam == 0: alpha gradient == gradient of CE alone
# ---------------------------------------------------------------------------

def test_t2_lam0_gradient_equals_ce(tmp_path):
    """T2: with lam=0, loss_a = ce only; alpha.grad must equal CE gradient."""
    torch.manual_seed(0)
    model = _make_model(seed=0)
    c1, c2 = _make_costs()
    xa, ya = _make_batch(n=8, seed=0)

    # Gradient from trainer (lam=0)
    model.zero_grad()
    with torch.autocast("cuda", enabled=False):
        ce = F.cross_entropy(model(xa), ya)
    eff = model.expected_cost(c1, c2)
    loss_a = ce.float()   # lam=0 → no eff term
    loss_a.backward()
    grad_n_trainer = model.alpha_normal.grad.clone()
    grad_r_trainer = model.alpha_reduce.grad.clone()

    # Reference gradient from CE alone
    model.zero_grad()
    with torch.autocast("cuda", enabled=False):
        ce2 = F.cross_entropy(model(xa), ya)
    ce2.float().backward()
    grad_n_ref = model.alpha_normal.grad.clone()
    grad_r_ref = model.alpha_reduce.grad.clone()

    assert torch.allclose(grad_n_trainer, grad_n_ref, atol=1e-8), \
        "alpha_normal gradient differs from CE gradient at lam=0"
    assert torch.allclose(grad_r_trainer, grad_r_ref, atol=1e-8), \
        "alpha_reduce gradient differs from CE gradient at lam=0"


# ---------------------------------------------------------------------------
# T3 – lam > 0: alpha gradient == grad(CE) + lam * grad(E)
# ---------------------------------------------------------------------------

def test_t3_lam_pos_gradient_decomposition(tmp_path):
    """T3: with lam>0, alpha.grad = grad_CE + lam * grad_E (atol 1e-6)."""
    LAM = 0.5
    torch.manual_seed(0)
    model = _make_model(seed=0)
    c1, c2 = _make_costs()
    xa, ya = _make_batch(n=8, seed=0)

    # Gradient from combined loss (trainer path)
    model.zero_grad()
    ce = F.cross_entropy(model(xa), ya)
    eff = model.expected_cost(c1, c2)
    loss_a = ce.float() + LAM * eff
    loss_a.backward()
    grad_n_combined = model.alpha_normal.grad.clone()
    grad_r_combined = model.alpha_reduce.grad.clone()

    # Gradient from CE alone
    model.zero_grad()
    ce2 = F.cross_entropy(model(xa), ya)
    ce2.float().backward()
    grad_n_ce = model.alpha_normal.grad.clone()
    grad_r_ce = model.alpha_reduce.grad.clone()

    # Gradient from E alone
    model.zero_grad()
    eff2 = model.expected_cost(c1, c2)
    eff2.backward()
    grad_n_e = model.alpha_normal.grad.clone()
    grad_r_e = model.alpha_reduce.grad.clone()

    expected_n = grad_n_ce + LAM * grad_n_e
    expected_r = grad_r_ce + LAM * grad_r_e

    assert torch.allclose(grad_n_combined, expected_n, atol=1e-6), \
        "alpha_normal: grad(CE + lam*E) != grad_CE + lam*grad_E"
    assert torch.allclose(grad_r_combined, expected_r, atol=1e-6), \
        "alpha_reduce: grad(CE + lam*E) != grad_CE + lam*grad_E"


# ---------------------------------------------------------------------------
# T4 – E logged at lam==0 equals E logged at lam==1 at step 0 (same seed)
# ---------------------------------------------------------------------------

def test_t4_eff_independent_of_lam(tmp_path):
    """T4: E(alpha) is independent of lam; > 0; same at step 0 for both."""
    c1, c2 = _make_costs()
    xa, ya = _make_batch(n=8, seed=5)

    # lam = 0
    torch.manual_seed(0)
    model0 = _make_model(seed=0)
    _, eff0, _ = _make_trainer(model0, c1, c2, lam=0.0,
                               tmp_dir=str(tmp_path / "t4a")).alpha_step(xa, ya)

    # lam = 1  (same seed → same alpha init → same E at step 0)
    torch.manual_seed(0)
    model1 = _make_model(seed=0)
    _, eff1, _ = _make_trainer(model1, c1, c2, lam=1.0,
                               tmp_dir=str(tmp_path / "t4b")).alpha_step(xa, ya)

    assert eff0 > 0, "eff at lam=0 should be > 0"
    assert abs(eff0 - eff1) < 1e-5, \
        f"eff at step 0 differs between lam=0 ({eff0:.8f}) and lam=1 ({eff1:.8f})"


# ---------------------------------------------------------------------------
# T5 – CE removed (loss = E only): 50 Adam steps reduce expensive-op weight
# ---------------------------------------------------------------------------

def test_t5_minimize_eff_only(tmp_path):
    """T5: 50 Adam steps on E alone reduce the most-expensive op's softmax weight."""
    torch.manual_seed(0)
    model = _make_model(seed=0)

    # Make op k artificially the most expensive
    k  = 1   # sep_conv_5x5
    c1 = torch.full((len(PRIMITIVES),), 0.1)
    c2 = torch.full((len(PRIMITIVES),), 0.1)
    c1[k] = 10.0
    c2[k] = 10.0

    model.alpha_normal.data.zero_()
    model.alpha_reduce.data.zero_()
    initial_w = float(F.softmax(model.alpha_normal[0].detach().float(), dim=-1)[k])

    opt = torch.optim.Adam(model.arch_parameters(), lr=0.1)
    for _ in range(50):
        opt.zero_grad()
        model.expected_cost(c1, c2).backward()
        opt.step()

    final_w = float(F.softmax(model.alpha_normal[0].detach().float(), dim=-1)[k])
    assert final_w < initial_w, \
        f"Expensive op weight did not decrease: {initial_w:.4f} -> {final_w:.4f}"


# ---------------------------------------------------------------------------
# T6 – call-order: unscale_ BEFORE clip_grad_norm_ BEFORE w_scaler.step
# ---------------------------------------------------------------------------

def test_t6_weight_step_call_order(tmp_path):
    """T6: in weight_step, unscale_ -> clip_grad_norm_ -> w_scaler.step order verified."""
    model = _make_model(seed=0)
    c1, c2 = _make_costs()
    trainer = _make_trainer(model, c1, c2, lam=0.0, tmp_dir=str(tmp_path))
    xw, yw = _make_batch(n=8, seed=0)

    call_log: list[str] = []

    orig_unscale = type(trainer.w_scaler).unscale_
    orig_step    = type(trainer.w_scaler).step
    orig_clip    = torch.nn.utils.clip_grad_norm_

    def patched_unscale_(self_scaler, optimizer):
        call_log.append("unscale_")
        return orig_unscale(self_scaler, optimizer)

    def patched_clip(params, max_norm, **kw):
        call_log.append("clip")
        return orig_clip(params, max_norm, **kw)

    def patched_step(self_scaler, optimizer, *args, **kw):
        call_log.append("step")
        return orig_step(self_scaler, optimizer, *args, **kw)

    # Run alpha step first so its backward grads don't interfere
    trainer.alpha_step(*_make_batch(n=8, seed=1))

    with patch.object(type(trainer.w_scaler), "unscale_", patched_unscale_), \
         patch.object(type(trainer.w_scaler), "step",     patched_step), \
         patch.object(torch.nn.utils, "clip_grad_norm_",  patched_clip):
        trainer.weight_step(xw, yw)

    assert "unscale_" in call_log, "w_scaler.unscale_ was never called"
    assert "clip"     in call_log, "clip_grad_norm_ was never called"
    assert "step"     in call_log, "w_scaler.step was never called"
    assert call_log.index("unscale_") < call_log.index("clip"), \
        f"unscale_ not called before clip_grad_norm_: {call_log}"
    assert call_log.index("clip") < call_log.index("step"), \
        f"clip_grad_norm_ not called before w_scaler.step: {call_log}"


# ---------------------------------------------------------------------------
# T7 – CSV header exact; 1-epoch run writes exactly 1 data row
# ---------------------------------------------------------------------------

def test_t7_csv_header_and_row(tmp_path):
    """T7: CSV header matches CSV_COLUMNS exactly; one epoch → one data row."""
    model = _make_model(seed=0)
    c1, c2 = _make_costs()
    trainer = _make_trainer(model, c1, c2, lam=0.1, tmp_dir=str(tmp_path))
    loader  = _simple_loader(n_batches=2, batch=8)

    trainer.train_epoch(loader, loader, epoch=0)

    csv_path = tmp_path / "search_log.csv"
    assert csv_path.exists(), "search_log.csv not created"

    with open(csv_path, newline="") as fh:
        reader = list(csv.reader(fh))

    assert len(reader) == 2, f"Expected 2 lines (header + 1 row), got {len(reader)}"

    header = reader[0]
    assert header == list(CSV_COLUMNS), \
        f"Header mismatch:\n  got:      {header}\n  expected: {list(CSV_COLUMNS)}"


# ---------------------------------------------------------------------------
# T8 – resume: 2 epochs straight == 1 epoch + save + load + 1 epoch (atol 1e-6)
# ---------------------------------------------------------------------------

def test_t8_resume(tmp_path):
    """T8: resuming from checkpoint produces identical alpha to running straight."""
    c1, c2  = _make_costs()
    loader  = _simple_loader(n_batches=2, batch=8)

    # --- Run A: 2 epochs straight ---
    dirA = tmp_path / "runA"
    torch.manual_seed(0)
    modelA = _make_model(seed=0)
    trainerA = _make_trainer(modelA, c1, c2, lam=0.1, tmp_dir=str(dirA))
    trainerA.train_epoch(loader, loader, epoch=0)
    trainerA.train_epoch(loader, loader, epoch=1)
    an_A = modelA.alpha_normal.detach().clone()
    ar_A = modelA.alpha_reduce.detach().clone()

    # --- Run B: 1 epoch, save, load, 1 more epoch ---
    dirB = tmp_path / "runB"
    torch.manual_seed(0)
    modelB = _make_model(seed=0)
    trainerB = _make_trainer(modelB, c1, c2, lam=0.1, tmp_dir=str(dirB))
    trainerB.train_epoch(loader, loader, epoch=0)
    ckpt_path = dirB / "checkpoint.pt"
    assert ckpt_path.exists(), "Checkpoint not saved after epoch 0"

    # Fresh trainer with fresh model, then load checkpoint
    torch.manual_seed(999)   # different seed — should be overwritten by RNG restore
    modelB2 = _make_model(seed=999)
    trainerB2 = _make_trainer(modelB2, c1, c2, lam=0.1, tmp_dir=str(dirB))
    trainerB2.load_checkpoint(ckpt_path)
    assert trainerB2.start_epoch == 1, \
        f"start_epoch after resume should be 1, got {trainerB2.start_epoch}"
    trainerB2.train_epoch(loader, loader, epoch=1)
    an_B = modelB2.alpha_normal.detach().clone()
    ar_B = modelB2.alpha_reduce.detach().clone()

    assert torch.allclose(an_A, an_B, atol=1e-6), \
        "alpha_normal diverged after resume (max diff {:.2e})".format(
            (an_A - an_B).abs().max().item())
    assert torch.allclose(ar_A, ar_B, atol=1e-6), \
        "alpha_reduce diverged after resume (max diff {:.2e})".format(
            (ar_A - ar_B).abs().max().item())


# ---------------------------------------------------------------------------
# T9 – OOM: monkeypatch forward → OutOfMemoryError → print + exit(1)
# ---------------------------------------------------------------------------

def test_t9_oom_exits(tmp_path, monkeypatch, capsys):
    """T9: OOM in forward → 'Lower --batch_size (try 32)' + SystemExit(1)."""
    model = _make_model(seed=0)
    c1, c2 = _make_costs()
    trainer = _make_trainer(model, c1, c2, lam=0.0, tmp_dir=str(tmp_path))
    xa, ya  = _make_batch(n=8, seed=0)

    def _oom_forward(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")

    monkeypatch.setattr(model, "forward", _oom_forward)

    with pytest.raises(SystemExit) as exc_info:
        trainer.alpha_step(xa, ya)

    assert exc_info.value.code == 1, \
        f"Expected exit code 1, got {exc_info.value.code}"
    captured = capsys.readouterr()
    assert "Lower --batch_size" in captured.out, \
        f"OOM message not printed. stdout: {captured.out!r}"


# ---------------------------------------------------------------------------
# T10 – grep: search.py and search_trainer.py never import get_test_loader
# ---------------------------------------------------------------------------

def test_t10_no_test_loader_import():
    """T10: neither search.py nor search_trainer.py reference get_test_loader."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    files_to_check = [
        root / "search.py",
        root / "thriftnas" / "trainers" / "search_trainer.py",
    ]
    for fpath in files_to_check:
        assert fpath.exists(), f"File not found: {fpath}"
        text = fpath.read_text(encoding="utf-8")
        assert "get_test_loader" not in text, \
            f"get_test_loader found in {fpath.name}"
