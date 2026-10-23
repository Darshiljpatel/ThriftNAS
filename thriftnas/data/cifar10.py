"""CIFAR-10 data loading for ThriftNAS.

Public API
----------
split_indices       – reproducible disjoint index split for search
get_search_loaders  – (weight_loader, alpha_loader) from two disjoint 10k subsets
get_retrain_loader  – full 50k training set loader
get_test_loader     – 10k test loader, gated by final_report=True
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_ROOT = str(Path(__file__).resolve().parents[2] / "data")

# CIFAR-10 channel statistics (computed over the 50k training set).
_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2470, 0.2435, 0.2616)

_TRAIN_SPLIT_SIZE = 10_000   # images per search subset (weight / alpha)
_N_TRAIN          = 50_000   # total CIFAR-10 training images


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _train_transform(cutout: bool = False) -> T.Compose:
    """Return the training augmentation pipeline.

    Parameters
    ----------
    cutout:
        If True, Cutout regularisation would be added (not yet implemented;
        flag kept for the spec-mandated interface, default False).
    """
    ops: list = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ]
    # Cutout is behind a flag (default off) as specified in §5.
    # TODO (step 5): insert Cutout(n_holes=1, length=16) here when flag=True.
    if cutout:
        raise NotImplementedError("Cutout is not yet implemented (see step 5).")
    return T.Compose(ops)


def _eval_transform() -> T.Compose:
    """Return the evaluation transform (ToTensor + Normalize, no augmentation)."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])


# ---------------------------------------------------------------------------
# Index splitting
# ---------------------------------------------------------------------------

def split_indices(
    n: int = _N_TRAIN,
    size: int = _TRAIN_SPLIT_SIZE,
    seed: int = 0,
) -> Tuple[list, list]:
    """Return two disjoint index lists of length *size* from [0, n).

    Uses a deterministic ``torch.Generator`` so results are seed-stable and
    independent of the global RNG state.

    Parameters
    ----------
    n:
        Total pool size (50 000 for CIFAR-10 training set).
    size:
        Number of indices in each split (10 000).
    seed:
        Integer seed for the permutation generator.

    Returns
    -------
    (alpha_idx, weight_idx)
        Two disjoint lists of length *size*; 30 000 indices are left unused.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    alpha_idx  = perm[:size]
    weight_idx = perm[size : 2 * size]
    return alpha_idx, weight_idx


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _common_kw(batch_size: int, num_workers: int) -> dict:
    """Shared DataLoader keyword arguments."""
    return dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        # persistent_workers requires num_workers > 0; avoids per-batch
        # worker re-spawn overhead on Windows (§10).
        persistent_workers=num_workers > 0,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_search_loaders(
    batch_size: int = 48,
    seed: int = 0,
    num_workers: int = 2,
    cutout: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Return (weight_loader, alpha_loader) from two disjoint 10k subsets.

    Both subsets are drawn from the 50k CIFAR-10 training set via a
    fixed-seed permutation; 30k images are left unused during search.

    Parameters
    ----------
    batch_size:
        Mini-batch size (spec default 48; try 32 if OOM).
    seed:
        Permutation seed; change to get different but reproducible splits.
    num_workers:
        DataLoader worker processes (spec default 2 on Windows, §10).
    cutout:
        Passed to the train transform (default False per §5).

    Returns
    -------
    (weight_loader, alpha_loader)
        Both shuffle, drop_last=True, pin_memory=True.
    """
    tf   = _train_transform(cutout=cutout)
    ds   = torchvision.datasets.CIFAR10(_DATA_ROOT, train=True, download=True, transform=tf)
    kw   = _common_kw(batch_size, num_workers)

    alpha_idx, weight_idx = split_indices(seed=seed)

    weight_loader = DataLoader(Subset(ds, weight_idx), shuffle=True, **kw)
    alpha_loader  = DataLoader(Subset(ds, alpha_idx),  shuffle=True, **kw)
    return weight_loader, alpha_loader


def get_retrain_loader(
    batch_size: int = 96,
    num_workers: int = 2,
    cutout: bool = False,
) -> DataLoader:
    """Return a DataLoader over the full 50k CIFAR-10 training set.

    Parameters
    ----------
    batch_size:
        Mini-batch size (spec default 96; try 64 if OOM).
    num_workers:
        DataLoader worker processes.
    cutout:
        Cutout flag (default False per §5).
    """
    tf = _train_transform(cutout=cutout)
    ds = torchvision.datasets.CIFAR10(_DATA_ROOT, train=True, download=True, transform=tf)
    return DataLoader(ds, shuffle=True, **_common_kw(batch_size, num_workers))


def get_test_loader(
    batch_size: int,
    *,
    final_report: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    """Return a DataLoader over the 10k CIFAR-10 test set.

    .. warning::
        The test set must only be touched **once**, after the final training
        epoch (spec §8).  This function enforces that contract by requiring
        ``final_report=True``.

    Parameters
    ----------
    batch_size:
        Mini-batch size.
    final_report:
        Must be explicitly set to ``True``; raises ``ValueError`` otherwise.
    num_workers:
        DataLoader worker processes.

    Raises
    ------
    ValueError
        If *final_report* is not ``True``.
    """
    if not final_report:
        raise ValueError(
            "get_test_loader() requires final_report=True.  "
            "The test set must only be evaluated once, after the final epoch (spec §8)."
        )
    tf = _eval_transform()
    ds = torchvision.datasets.CIFAR10(_DATA_ROOT, train=False, download=True, transform=tf)
    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        shuffle=False,   # deterministic order for the single final evaluation
    )
    return DataLoader(ds, **kw)
