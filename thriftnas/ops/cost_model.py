"""Cost model for ThriftNAS: CPU latency profiling and MAC counting.

Implements the lookup-table approach from spec §4 and addendum §10.
Profile the ops (never call from search/retrain code); import from
scripts/profile_ops.py only.

Public API
----------
time_op              – CPU timing under inference_mode (single thread)
count_macs           – fvcore FlopCountAnalysis wrapper
build_cost_table     – profile all PRIMITIVES, compute c(o)
build_flops_only_table – MACs-only cost table for the ablation
make_metadata        – build the metadata dict stored in every JSON
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from thriftnas.ops.primitives import OPS, PRIMITIVES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_C       = 32
_DEFAULT_SIZE    = 16
_DEFAULT_WARMUP  = 20
_DEFAULT_RUNS    = 200
_DEFAULT_THREADS = 1

#: Approximation disclaimer stored in every saved JSON (spec §4).
APPROX_NOTE: str = (
    "Lookup-table latency is an approximation of true end-to-end latency. "
    "The prediction excludes the stem, global average pooling, and classifier "
    "layers (those costs are captured in end-to-end measurements)."
)


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------

def time_op(
    op: nn.Module,
    x: torch.Tensor,
    warmup: int = _DEFAULT_WARMUP,
    runs: int = _DEFAULT_RUNS,
) -> tuple[float, float]:
    """Time *op* on CPU (single thread, eval mode, inference_mode).

    Parameters
    ----------
    op:
        Module to profile.  Set to eval mode internally.
    x:
        CPU input tensor (batch size 1 recommended).
    warmup:
        Passes before timing starts (20 per spec §4).
    runs:
        Timed passes (≥200 per spec §4).

    Returns
    -------
    (median_ms, std_ms)
        Median and standard deviation of per-call wall-clock times in ms.
        Flag the op if std > 10 % of median (spec §4).
    """
    torch.set_num_threads(_DEFAULT_THREADS)
    op.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            op(x)
        times: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            op(x)
            times.append((time.perf_counter() - t0) * 1e3)
    arr = np.asarray(times, dtype=np.float64)
    return float(np.median(arr)), float(arr.std())


def count_macs(op: nn.Module, x: torch.Tensor) -> int:
    """Count multiply-accumulate operations (MACs) with fvcore.

    Convention (spec §4, addendum §10):
    - Conv/linear: each multiply-add counts as 1 MAC.
    - Pooling, BN, ReLU: 0 MACs (fvcore does not register these by default).

    Parameters
    ----------
    op:
        Module to analyse (set to eval mode internally).
    x:
        Input tensor with batch size 1.

    Returns
    -------
    int
        Total MAC count.
    """
    op.eval()
    with torch.no_grad():
        fa = FlopCountAnalysis(op, x)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        return int(fa.total())


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

def make_metadata(
    C: int = _DEFAULT_C,
    size: int = _DEFAULT_SIZE,
    warmup: int = _DEFAULT_WARMUP,
    runs: int = _DEFAULT_RUNS,
) -> dict[str, Any]:
    """Return the metadata dict that is stored in every JSON output."""
    return {
        "C":          C,
        "input_size": [1, C, size, size],
        "threads":    _DEFAULT_THREADS,
        "warmup":     warmup,
        "runs":       runs,
        "note":       APPROX_NOTE,
    }


def build_cost_table(
    C: int = _DEFAULT_C,
    size: int = _DEFAULT_SIZE,
    strides: tuple[int, ...] = (1, 2),
    warmup: int = _DEFAULT_WARMUP,
    runs: int = _DEFAULT_RUNS,
) -> tuple[list[dict[str, Any]], float, float]:
    """Profile all PRIMITIVES at two strides and compute c(o).

    .. code-block:: text

        c(o) = 0.5 * lat_median / lat_max  +  0.5 * MACs / MACs_max

    Maxima are taken over ALL (op, stride) entries so stride-1 and stride-2
    costs are comparable (addendum §10).

    Parameters
    ----------
    C:
        Channel count (profiling convention: 32, affine=True).
    size:
        Spatial resolution H = W = *size* (convention: 16).
    strides:
        Which strides to profile.
    warmup, runs:
        Passed to :func:`time_op`.

    Returns
    -------
    (entries, lat_max, mac_max)
        *entries* – list of dicts with keys:
            op, stride, lat_median_ms, lat_std_ms, macs, c, flagged.
        *lat_max*, *mac_max* – normalisation denominators.
    """
    x = torch.zeros(1, C, size, size)
    entries: list[dict[str, Any]] = []

    for stride in strides:
        for name in PRIMITIVES:
            op = OPS[name](C, stride, affine=True)
            lat_med, lat_std = time_op(op, x, warmup=warmup, runs=runs)
            macs = count_macs(op, x)
            entries.append({
                "op":            name,
                "stride":        stride,
                "lat_median_ms": lat_med,
                "lat_std_ms":    lat_std,
                "macs":          macs,
                "c":             None,   # computed below
                # Flag if timing is noisy (spec §4)
                "flagged":       bool(lat_std > 0.1 * lat_med),
            })

    # Normalise over ALL entries (both strides)
    lat_max = max(e["lat_median_ms"] for e in entries)
    mac_max = max(e["macs"] for e in entries)

    for e in entries:
        lat_term = (e["lat_median_ms"] / lat_max) if lat_max > 0 else 0.0
        mac_term = (e["macs"] / mac_max)          if mac_max > 0 else 0.0
        e["c"] = 0.5 * lat_term + 0.5 * mac_term

    return entries, lat_max, mac_max


def build_flops_only_table(
    C: int = _DEFAULT_C,
    size: int = _DEFAULT_SIZE,
    strides: tuple[int, ...] = (1, 2),
) -> tuple[list[dict[str, Any]], float]:
    """Build the FLOPs-only ablation table: c(o) = MACs / MACs_max.

    No timing is performed.

    Returns
    -------
    (entries, mac_max)
        *entries* – list of dicts with keys: op, stride, macs, c.
        *mac_max* – normalisation denominator.
    """
    x = torch.zeros(1, C, size, size)
    entries: list[dict[str, Any]] = []

    for stride in strides:
        for name in PRIMITIVES:
            op = OPS[name](C, stride, affine=True)
            macs = count_macs(op, x)
            entries.append({"op": name, "stride": stride, "macs": macs, "c": None})

    mac_max = max(e["macs"] for e in entries)
    for e in entries:
        e["c"] = (e["macs"] / mac_max) if mac_max > 0 else 0.0

    return entries, mac_max
