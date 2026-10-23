"""Tests for thriftnas/data/cifar10.py (Step 2 – T1 through T9, T8 manual).

Run with:  pytest tests/test_cifar10.py -v

NOTE: Tests that load images require CIFAR-10 on disk (data/cifar-10-batches-py/).
      They are skipped automatically when the dataset is absent.
      T1-T3, T7a never touch the disk and always run.
      T8 (grep) is performed manually / in the verify script.
"""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from thriftnas.data.cifar10 import (
    split_indices,
    get_search_loaders,
    get_retrain_loader,
    get_test_loader,
    _train_transform,
    _eval_transform,
    _DATA_ROOT,
    _N_TRAIN,
    _TRAIN_SPLIT_SIZE,
)

# Skip marker applied to every test that actually reads image files.
_cifar_present = (Path(_DATA_ROOT) / "cifar-10-batches-py").exists()
needs_cifar = pytest.mark.skipif(
    not _cifar_present,
    reason="CIFAR-10 not downloaded yet (run: python -c \"import torchvision; "
           "torchvision.datasets.CIFAR10('data', train=True, download=True)\")",
)

# ---------------------------------------------------------------------------
# T1 – subset sizes
# ---------------------------------------------------------------------------

def test_t1_subset_sizes() -> None:
    """T1: weight subset == 10 000 and alpha subset == 10 000."""
    alpha_idx, weight_idx = split_indices()
    assert len(weight_idx) == _TRAIN_SPLIT_SIZE, \
        f"weight_idx length {len(weight_idx)} != {_TRAIN_SPLIT_SIZE}"
    assert len(alpha_idx) == _TRAIN_SPLIT_SIZE, \
        f"alpha_idx length {len(alpha_idx)} != {_TRAIN_SPLIT_SIZE}"


# ---------------------------------------------------------------------------
# T2 – disjoint
# ---------------------------------------------------------------------------

def test_t2_disjoint() -> None:
    """T2: the intersection of weight_idx and alpha_idx is empty."""
    alpha_idx, weight_idx = split_indices()
    inter = set(weight_idx) & set(alpha_idx)
    assert len(inter) == 0, f"Found {len(inter)} overlapping indices"


# ---------------------------------------------------------------------------
# T3 – seed stability
# ---------------------------------------------------------------------------

def test_t3_same_seed_identical() -> None:
    """T3a: two calls with the same seed give the same index lists."""
    a1, w1 = split_indices(seed=0)
    a2, w2 = split_indices(seed=0)
    assert a1 == a2 and w1 == w2, "Same seed must produce identical splits"


def test_t3_different_seeds_differ() -> None:
    """T3b: seed 0 and seed 1 give different index lists."""
    a0, w0 = split_indices(seed=0)
    a1, w1 = split_indices(seed=1)
    assert a0 != a1 or w0 != w1, "Different seeds must produce different splits"


# ---------------------------------------------------------------------------
# T4 – batch shape, dtype, label range
# ---------------------------------------------------------------------------

@needs_cifar
def test_t4_batch_shape(tmp_path: pytest.FixtureLookupError) -> None:
    """T4: one weight batch has shape (48, 3, 32, 32), dtype float32, labels in [0,9]."""
    weight_loader, _ = get_search_loaders(batch_size=48, num_workers=0)
    imgs, labels = next(iter(weight_loader))
    assert imgs.shape == (48, 3, 32, 32), f"Unexpected shape {imgs.shape}"
    assert imgs.dtype == torch.float32, f"Unexpected dtype {imgs.dtype}"
    assert labels.min().item() >= 0 and labels.max().item() <= 9, \
        f"Labels out of [0,9]: min={labels.min()}, max={labels.max()}"


# ---------------------------------------------------------------------------
# T5 – normalisation statistics
# ---------------------------------------------------------------------------

@needs_cifar
def test_t5_normalisation_stats() -> None:
    """T5: over 5000 eval-transformed training images mean ≈0 (±0.05), std ≈1 (±0.1)."""
    import torchvision
    from thriftnas.data.cifar10 import _DATA_ROOT
    from torch.utils.data import DataLoader, Subset

    tf = _eval_transform()
    ds = torchvision.datasets.CIFAR10(_DATA_ROOT, train=True, download=True, transform=tf)
    subset = Subset(ds, list(range(5000)))
    loader = DataLoader(subset, batch_size=500, num_workers=0, shuffle=False)

    all_imgs = torch.cat([imgs for imgs, _ in loader], dim=0)  # (5000,3,32,32)
    # Compute per-channel mean and std over N×H×W
    mean = all_imgs.mean(dim=(0, 2, 3))
    std  = all_imgs.std(dim=(0, 2, 3))

    for c in range(3):
        assert abs(mean[c].item()) < 0.05, \
            f"Channel {c} mean {mean[c]:.4f} not within ±0.05 of 0"
        assert abs(std[c].item() - 1.0) < 0.1, \
            f"Channel {c} std {std[c]:.4f} not within ±0.1 of 1"


# ---------------------------------------------------------------------------
# T6 – randomness of train transform
# ---------------------------------------------------------------------------

@needs_cifar
def test_t6_train_transform_random_eval_deterministic() -> None:
    """T6: train transform varies across calls; eval transform does not."""
    import torchvision
    from thriftnas.data.cifar10 import _DATA_ROOT

    # Load the raw PIL dataset (no transform)
    ds_raw = torchvision.datasets.CIFAR10(_DATA_ROOT, train=True, download=True, transform=None)
    pil_img, _ = ds_raw[0]

    train_tf = _train_transform()
    eval_tf  = _eval_transform()

    t1 = train_tf(pil_img)
    t2 = train_tf(pil_img)
    e1 = eval_tf(pil_img)
    e2 = eval_tf(pil_img)

    # Train transform must differ at least sometimes (virtually guaranteed with
    # RandomCrop + RandomHorizontalFlip on a non-trivially constant image).
    assert not torch.equal(t1, t2), \
        "Train transform produced identical tensors on two calls — randomness not working"
    assert torch.equal(e1, e2), \
        "Eval transform produced different tensors on two calls — should be deterministic"


# ---------------------------------------------------------------------------
# T7 – get_test_loader guard
# ---------------------------------------------------------------------------

def test_t7_test_loader_guard_raises() -> None:
    """T7a: get_test_loader without final_report=True raises ValueError."""
    with pytest.raises(ValueError, match="final_report"):
        get_test_loader(96)


@needs_cifar
def test_t7_test_loader_length() -> None:
    """T7b: with final_report=True the dataset has 10 000 samples."""
    loader = get_test_loader(100, final_report=True, num_workers=0)
    assert len(loader.dataset) == 10_000, \
        f"Test dataset length {len(loader.dataset)} != 10000"


# ---------------------------------------------------------------------------
# T9 – retrain loader size and cutout default off
# ---------------------------------------------------------------------------

@needs_cifar
def test_t9_retrain_loader_size() -> None:
    """T9: retrain loader has 50 000 samples and cutout is off by default."""
    loader = get_retrain_loader(batch_size=96, num_workers=0)
    assert len(loader.dataset) == _N_TRAIN, \
        f"Retrain dataset length {len(loader.dataset)} != {_N_TRAIN}"


@needs_cifar
def test_t9_cutout_default_off() -> None:
    """T9: get_retrain_loader(cutout=False) succeeds without error."""
    # If cutout defaulted to True it would raise NotImplementedError.
    loader = get_retrain_loader(batch_size=96, num_workers=0, cutout=False)
    assert loader is not None
