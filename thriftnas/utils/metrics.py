"""Evaluation metrics and latency estimation for ThriftNAS (PROJECT_SPEC §7, Step 10).

Provides:
- count_params: parameter count in millions
- count_macs: multiply-accumulates in millions (fvcore)
- model_size_mb: on-disk size in MB of the weights-only file
- cpu_latency: CPU inference time (batch 1, 1 thread, 20 warmup, 200 timed)
- predicted_latency: lookup-table sum over cell ops for a genotype
- compare_latency: prediction error percentage against measured latency
"""

from __future__ import annotations

import os
import time
from collections import namedtuple
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from thriftnas.models.genotypes import Genotype

# ---------------------------------------------------------------------------
# Namedtuple for latency comparison
# ---------------------------------------------------------------------------

LatencyComparison = namedtuple(
    "LatencyComparison", ["predicted", "measured", "error_pct"]
)


# ---------------------------------------------------------------------------
# 1. Parameter and complexity metrics
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> float:
    """Return model parameters in millions (M).

    Parameters
    ----------
    model:
        PyTorch module.

    Returns
    -------
    float
        Number of parameters divided by 1,000,000.
    """
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def count_macs(
    model: nn.Module,
    input_size: tuple[int, ...] = (1, 3, 32, 32),
) -> float:
    """Return multiply-accumulate operations (MACs) in millions (M) using fvcore.

    Parameters
    ----------
    model:
        PyTorch module.
    input_size:
        Input tensor shape (default: 1x3x32x32 for CIFAR-10).

    Returns
    -------
    float
        Total MAC count divided by 1,000,000.
    """
    device = next(model.parameters()).device
    x = torch.randn(*input_size, device=device)
    model.eval()
    with torch.no_grad():
        fa = FlopCountAnalysis(model, x)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        total = int(fa.total())
    return total / 1e6


def model_size_mb(path: str | Path) -> float:
    """Return on-disk size in Megabytes (MB) of the weights-only state dict file.

    Parameters
    ----------
    path:
        File path to the saved state dict.

    Returns
    -------
    float
        Size on disk in MB (bytes / 2^20).
    """
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# 2. CPU inference latency measurement
# ---------------------------------------------------------------------------

def cpu_latency(
    model: nn.Module,
    input_size: tuple[int, ...] = (1, 3, 32, 32),
    warmup: int = 20,
    runs: int = 200,
) -> tuple[float, float]:
    """Measure CPU latency of a model in milliseconds (mean +/- std).

    Timing conditions (spec §7, §4):
    - Batch size 1
    - torch.set_num_threads(1)
    - model.eval()
    - torch.inference_mode()
    - 20 warm-up runs, 200 timed runs

    Returns
    -------
    (mean_ms, std_ms)
    """
    torch.set_num_threads(1)
    cpu_model = model.to("cpu")
    cpu_model.eval()

    x = torch.randn(*input_size, device="cpu")

    with torch.inference_mode():
        for _ in range(warmup):
            cpu_model(x)

        times: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            cpu_model(x)
            times.append((time.perf_counter() - t0) * 1e3)

    arr = np.asarray(times, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# 3. Lookup-table predicted latency for genotypes
# ---------------------------------------------------------------------------

def _build_latency_map(cost_table: Any) -> dict[tuple[str, int], float]:
    """Parse various cost table formats into an (op, stride) -> latency_ms dict."""
    mapping: dict[tuple[str, int], float] = {}

    # Case 1: Dict with "entries" list (from cost_table.json)
    if isinstance(cost_table, dict) and "entries" in cost_table:
        for entry in cost_table["entries"]:
            mapping[(entry["op"], int(entry["stride"]))] = float(entry["lat_median_ms"])
        return mapping

    # Case 2: List of entry dicts
    if isinstance(cost_table, list):
        for entry in cost_table:
            mapping[(entry["op"], int(entry["stride"]))] = float(entry["lat_median_ms"])
        return mapping

    # Case 3: Nested dict: table[op][stride] = lat_ms
    if isinstance(cost_table, dict):
        for k, v in cost_table.items():
            if isinstance(k, tuple):
                mapping[(k[0], int(k[1]))] = float(v)
            elif isinstance(v, dict):
                for stride, lat in v.items():
                    mapping[(k, int(stride))] = float(lat)
            else:
                # Flat map with stride 1 assumed or custom
                mapping[(k, 1)] = float(v)
        return mapping

    raise ValueError(f"Unsupported cost_table format: {type(cost_table)}")


def predicted_latency(
    genotype: Genotype,
    cost_table: Any,
    layers: int | None = None,
) -> float:
    """Predict network latency from the lookup table as the sum over cell ops.

    Notes
    -----
    - Normal cells have all stride-1 edges.
    - Reduction cells have stride-2 edges for inputs < 2 (from previous cells)
      and stride-1 edges for intermediate connections (inputs >= 2).
    - If `layers` is None, returns the sum of the genotype cells (1 normal + 1 reduction).
    - If `layers` is specified, scales to the stacked network (2 reduction cells,
      `layers - 2` normal cells).

    Parameters
    ----------
    genotype:
        Genotype instance with `normal` and `reduce` gene lists.
    cost_table:
        Lookup table from `scripts/profile_ops.py` or dict.
    layers:
        Total cells in the unrolled network (optional).

    Returns
    -------
    float
        Summed lookup-table latency in milliseconds.
    """
    lat_map = _build_latency_map(cost_table)

    # Normal cell: 8 edges, all stride 1
    normal_lat = 0.0
    for op_name, _ in genotype.normal:
        normal_lat += lat_map.get((op_name, 1), 0.0)

    # Reduction cell: 8 edges (input_index < 2 -> stride 2; input_index >= 2 -> stride 1)
    reduce_lat = 0.0
    for op_name, in_idx in genotype.reduce:
        stride = 2 if in_idx < 2 else 1
        reduce_lat += lat_map.get((op_name, stride), 0.0)

    if layers is None:
        return float(normal_lat + reduce_lat)

    num_reduce = 2
    num_normal = max(0, layers - num_reduce)
    total = (num_normal * normal_lat) + (num_reduce * reduce_lat)
    return float(total)


def compare_latency(predicted: float, measured: float) -> LatencyComparison:
    """Compare predicted vs measured latency and return percentage error.

    error_pct = abs(pred - meas) / meas * 100

    Returns
    -------
    LatencyComparison(predicted, measured, error_pct)
    """
    if measured <= 0:
        err = 0.0
    else:
        err = abs(predicted - measured) / measured * 100.0
    return LatencyComparison(predicted=predicted, measured=measured, error_pct=err)
