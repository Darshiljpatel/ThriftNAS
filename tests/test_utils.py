"""Tests for thriftnas.utils (Step 1 – T1 through T7).

Run with:  pytest tests/test_utils.py -v
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from thriftnas.utils import (
    AverageMeter,
    CSVLogger,
    accuracy,
    get_device,
    load_checkpoint,
    peak_memory_mb,
    save_checkpoint,
    set_seed,
)


# ---------------------------------------------------------------------------
# T1 – set_seed reproducibility
# ---------------------------------------------------------------------------

class TestSetSeed:
    """T1: set_seed(0) called twice produces identical draws from all RNGs."""

    def _draw(self) -> tuple[torch.Tensor, np.ndarray, float]:
        t = torch.rand(3)
        n = np.random.rand(3)
        p = random.random()
        return t, n, p

    def test_torch_identical(self) -> None:
        set_seed(0)
        t1, _, _ = self._draw()
        set_seed(0)
        t2, _, _ = self._draw()
        assert torch.equal(t1, t2), "torch.rand(3) must be identical after set_seed(0) twice"

    def test_numpy_identical(self) -> None:
        set_seed(0)
        _, n1, _ = self._draw()
        set_seed(0)
        _, n2, _ = self._draw()
        assert np.array_equal(n1, n2), "numpy rand(3) must be identical after set_seed(0) twice"

    def test_python_random_identical(self) -> None:
        set_seed(0)
        _, _, p1 = self._draw()
        set_seed(0)
        _, _, p2 = self._draw()
        assert p1 == p2, "python random() must be identical after set_seed(0) twice"


# ---------------------------------------------------------------------------
# T2 – AverageMeter
# ---------------------------------------------------------------------------

class TestAverageMeter:
    """T2: update(2, n=1) then update(4, n=3) → avg == 3.5."""

    def test_weighted_average(self) -> None:
        m = AverageMeter()
        m.update(2, n=1)   # sum=2, count=1
        m.update(4, n=3)   # sum=2+12=14, count=4  → avg=3.5
        assert m.avg == pytest.approx(3.5), f"Expected avg=3.5, got {m.avg}"

    def test_count_and_sum(self) -> None:
        m = AverageMeter()
        m.update(2, n=1)
        m.update(4, n=3)
        assert m.count == 4
        assert m.sum == pytest.approx(14.0)

    def test_reset_clears(self) -> None:
        m = AverageMeter()
        m.update(10, n=5)
        m.reset()
        assert m.count == 0 and m.sum == 0.0 and m.avg == 0.0


# ---------------------------------------------------------------------------
# T3 – accuracy
# ---------------------------------------------------------------------------

class TestAccuracy:
    """T3: 3 of 4 samples correct → 75.0 %."""

    def test_three_of_four(self) -> None:
        # Labels: [0, 1, 2, 3]; logits argmax: [0, 1, 2, 0] → 3 correct
        logits = torch.tensor([
            [10.0,  0.0,  0.0,  0.0],   # pred 0, label 0 ✓
            [ 0.0, 10.0,  0.0,  0.0],   # pred 1, label 1 ✓
            [ 0.0,  0.0, 10.0,  0.0],   # pred 2, label 2 ✓
            [10.0,  0.0,  0.0,  0.0],   # pred 0, label 3 ✗
        ])
        labels = torch.tensor([0, 1, 2, 3])
        result = accuracy(logits, labels)
        assert result == pytest.approx(75.0), f"Expected 75.0, got {result}"

    def test_all_correct(self) -> None:
        logits = torch.eye(4) * 10
        labels = torch.arange(4)
        assert accuracy(logits, labels) == pytest.approx(100.0)

    def test_none_correct(self) -> None:
        # All predictions point to class 1, all labels are 0
        logits = torch.zeros(4, 2)
        logits[:, 1] = 10.0
        labels = torch.zeros(4, dtype=torch.long)
        assert accuracy(logits, labels) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# T4 – save / load checkpoint (round-trip, no .tmp residue)
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestCheckpoint:
    """T4: save_checkpoint / load_checkpoint restores all state exactly."""

    def test_round_trip(self, tmp_path: Path) -> None:
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cpu")

        # Dirty the optimiser state with a fake step.
        dummy = model(torch.randn(2, 4))
        dummy.sum().backward()
        scaler.scale(dummy.sum())   # just to initialise scaler internals
        optimizer.step()
        optimizer.zero_grad()

        state = {
            "epoch": 7,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }
        ckpt_path = str(tmp_path / "epoch_7.pt")
        save_checkpoint(state, ckpt_path)

        loaded = load_checkpoint(ckpt_path)

        # Epoch
        assert loaded["epoch"] == 7

        # Model weights
        for key in state["model"]:
            assert torch.equal(loaded["model"][key], state["model"][key]), \
                f"Model tensor mismatch at key '{key}'"

        # Optimiser state (compare param-group LR + state tensors)
        assert loaded["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.01)

        # GradScaler state
        assert loaded["scaler"] == scaler.state_dict()

    def test_no_tmp_residue(self, tmp_path: Path) -> None:
        """T4 (residue check): no .tmp file remains after a successful save."""
        ckpt_path = str(tmp_path / "ck.pt")
        save_checkpoint({"epoch": 1}, ckpt_path)
        tmp_file = ckpt_path + ".tmp"
        assert not os.path.exists(tmp_file), ".tmp file must be removed after save_checkpoint"


# ---------------------------------------------------------------------------
# T5 – atomic write: previous checkpoint survives a failed save
# ---------------------------------------------------------------------------

class TestAtomicCheckpoint:
    """T5: if torch.save raises, the existing checkpoint is untouched."""

    def test_previous_checkpoint_intact(self, tmp_path: Path) -> None:
        ckpt_path = str(tmp_path / "ck.pt")

        # Write a valid first checkpoint.
        save_checkpoint({"epoch": 1}, ckpt_path)

        # Simulate a failure during the second save.
        with patch("torch.save", side_effect=RuntimeError("disk full")):
            with pytest.raises(RuntimeError, match="disk full"):
                save_checkpoint({"epoch": 2}, ckpt_path)

        # The original checkpoint must still be loadable.
        loaded = load_checkpoint(ckpt_path)
        assert loaded["epoch"] == 1, \
            "Previous checkpoint must survive a failed save attempt"

        # No .tmp file should remain (torch.save raised before os.replace).
        assert not os.path.exists(ckpt_path + ".tmp"), \
            ".tmp must not exist after a failed torch.save"


# ---------------------------------------------------------------------------
# T6 – CSVLogger
# ---------------------------------------------------------------------------

class TestCSVLogger:
    """T6: 3 rows → 1 header + 3 data lines; re-opening appends without header."""

    def test_header_once_three_rows(self, tmp_path: Path) -> None:
        csv_path = str(tmp_path / "log.csv")
        logger = CSVLogger(csv_path)
        for i in range(3):
            logger.log({"epoch": i, "loss": 0.1 * i})

        lines = Path(csv_path).read_text().splitlines()
        assert lines[0] == "epoch,loss", f"Expected header 'epoch,loss', got '{lines[0]}'"
        assert len(lines) == 4, f"Expected 1 header + 3 data lines (4 total), got {len(lines)}"

    def test_reopen_appends_no_second_header(self, tmp_path: Path) -> None:
        csv_path = str(tmp_path / "log.csv")

        # First session: 3 rows
        logger1 = CSVLogger(csv_path)
        for i in range(3):
            logger1.log({"epoch": i, "loss": float(i)})

        # Second session on the SAME file: should append, not add a header
        logger2 = CSVLogger(csv_path)
        logger2.log({"epoch": 3, "loss": 3.0})

        lines = Path(csv_path).read_text().splitlines()
        # Still only 1 header
        header_count = sum(1 for ln in lines if ln.startswith("epoch"))
        assert header_count == 1, \
            f"Header must appear exactly once; found {header_count} times"
        # 4 data rows total
        assert len(lines) == 5, \
            f"Expected 1 header + 4 data lines (5 total), got {len(lines)}"


# ---------------------------------------------------------------------------
# T7 – package importability (exercised at collection time, no extra test body)
# ---------------------------------------------------------------------------

def test_import_thriftnas() -> None:
    """T7: 'import thriftnas' must succeed (validates pip install -e .)."""
    import thriftnas  # noqa: F401 – import is the test
    assert hasattr(thriftnas, "__version__")
