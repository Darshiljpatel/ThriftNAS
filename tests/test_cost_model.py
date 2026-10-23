"""Tests for thriftnas/ops/cost_model.py (Step 4 – T1 through T6).

Run with:  pytest tests/test_cost_model.py -v

T7, T8, T9 are MANUAL — run profile_ops.py twice from the terminal
and check the console output.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn as nn

from thriftnas.ops.cost_model import (
    APPROX_NOTE,
    _DEFAULT_THREADS,
    _DEFAULT_RUNS,
    build_cost_table,
    build_flops_only_table,
    count_macs,
    make_metadata,
    time_op,
)
from thriftnas.ops.primitives import OPS, PRIMITIVES

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_C    = 32
_SIZE = 16
_X    = torch.zeros(1, _C, _SIZE, _SIZE)

# Use minimal warmup/runs so tests finish in a few seconds.
_WARMUP_FAST = 2
_RUNS_FAST   = 10


# ---------------------------------------------------------------------------
# T1 – table has 16 entries, all c in [0,1], max(lat/lat_max) == 1
# ---------------------------------------------------------------------------

class TestTableStructure:
    """T1: 16 entries, c ∈ [0,1], max(lat/lat_max) == 1."""

    @pytest.fixture(scope="class")
    def table(self):
        entries, lat_max, mac_max = build_cost_table(
            C=_C, size=_SIZE, warmup=_WARMUP_FAST, runs=_RUNS_FAST
        )
        return entries, lat_max, mac_max

    def test_entry_count(self, table):
        entries, _, _ = table
        assert len(entries) == 16, \
            f"Expected 16 entries (8 ops × 2 strides), got {len(entries)}"

    def test_c_in_unit_interval(self, table):
        entries, _, _ = table
        for e in entries:
            assert 0.0 <= e["c"] <= 1.0, \
                f"{e['op']} stride={e['stride']}: c={e['c']:.4f} not in [0,1]"

    def test_max_lat_normalised_to_one(self, table):
        entries, lat_max, _ = table
        max_ratio = max(e["lat_median_ms"] / lat_max for e in entries)
        assert math.isclose(max_ratio, 1.0, rel_tol=1e-6), \
            f"max(lat/lat_max) = {max_ratio:.6f}, expected 1.0"


# ---------------------------------------------------------------------------
# T2 – MACs of zero, skip_connect(s=1), pooling are 0 (or < 1% of sep_conv_3x3)
# ---------------------------------------------------------------------------

class TestZeroMACs:
    """T2: zero, identity skip, pooling have 0 MACs from fvcore."""

    @pytest.fixture(scope="class")
    def sep_macs(self):
        return count_macs(OPS["sep_conv_3x3"](_C, 1, True), _X)

    def test_zero_macs(self):
        macs = count_macs(OPS["zero"](_C, 1, True), _X)
        assert macs == 0, f"zero op MACs should be 0, got {macs}"

    def test_skip_connect_stride1_macs(self):
        macs = count_macs(OPS["skip_connect"](_C, 1, True), _X)
        assert macs == 0, f"skip_connect stride=1 MACs should be 0, got {macs}"

    def test_max_pool_macs(self, sep_macs):
        macs = count_macs(OPS["max_pool_3x3"](_C, 1, True), _X)
        # fvcore returns 0 for unsupported ops (pool); accept < 1% of sep_conv MACs
        assert macs < sep_macs * 0.01, \
            f"max_pool_3x3 MACs={macs} >= 1% of sep_conv_3x3 MACs={sep_macs}"

    def test_avg_pool_macs(self, sep_macs):
        macs = count_macs(OPS["avg_pool_3x3"](_C, 1, True), _X)
        assert macs < sep_macs * 0.01, \
            f"avg_pool_3x3 MACs={macs} >= 1% of sep_conv_3x3 MACs={sep_macs}"


# ---------------------------------------------------------------------------
# T3 – MACs ranges at C=32, 16x16, stride=1
# ---------------------------------------------------------------------------

class TestMACsRanges:
    """T3: MACs for three ops fall within spec-mandated ranges."""

    def test_sep_conv_3x3(self):
        macs = count_macs(OPS["sep_conv_3x3"](_C, 1, True), _X)
        # 671,744 = conv-only; 720,000 accounts for fvcore BN-affine counting
        assert 671_744 <= macs <= 720_000, \
            f"sep_conv_3x3 MACs={macs:,} not in [671744, 720000]"

    def test_sep_conv_5x5(self):
        macs = count_macs(OPS["sep_conv_5x5"](_C, 1, True), _X)
        assert 933_888 <= macs <= 1_000_000, \
            f"sep_conv_5x5 MACs={macs:,} not in [933888, 1000000]"

    def test_dil_conv_3x3(self):
        macs = count_macs(OPS["dil_conv_3x3"](_C, 1, True), _X)
        assert 335_872 <= macs <= 360_000, \
            f"dil_conv_3x3 MACs={macs:,} not in [335872, 360000]"


# ---------------------------------------------------------------------------
# T4 – ratio sep_conv_5x5 / sep_conv_3x3 in [1.3, 1.5]
# ---------------------------------------------------------------------------

def test_t4_mac_ratio():
    """T4: MACs(sep_conv_5x5) / MACs(sep_conv_3x3) ∈ [1.3, 1.5]."""
    m3 = count_macs(OPS["sep_conv_3x3"](_C, 1, True), _X)
    m5 = count_macs(OPS["sep_conv_5x5"](_C, 1, True), _X)
    ratio = m5 / m3
    assert 1.3 <= ratio <= 1.5, \
        f"MACs ratio 5x5/3x3 = {ratio:.4f}, expected [1.3, 1.5]"


# ---------------------------------------------------------------------------
# T5 – FLOPs-only table correctness and metadata contents
# ---------------------------------------------------------------------------

class TestFlopsOnlyTable:
    """T5: c = MACs/MACs_max; zero c = 0; metadata has note, threads, runs."""

    @pytest.fixture(scope="class")
    def fo_table(self):
        entries, mac_max = build_flops_only_table(C=_C, size=_SIZE)
        return entries, mac_max

    def test_c_formula(self, fo_table):
        entries, mac_max = fo_table
        for e in entries:
            expected = e["macs"] / mac_max if mac_max > 0 else 0.0
            assert math.isclose(e["c"], expected, rel_tol=1e-9), \
                f"{e['op']} stride={e['stride']}: c={e['c']:.6f} != MACs/MACs_max={expected:.6f}"

    def test_zero_c_is_zero(self, fo_table):
        entries, _ = fo_table
        for e in entries:
            if e["op"] == "zero":
                assert e["c"] == 0.0, \
                    f"zero op c should be 0.0, got {e['c']}"

    def test_metadata_note(self):
        meta = make_metadata()
        assert "approximation" in meta["note"].lower(), \
            "Metadata note must contain the approximation disclaimer"
        assert APPROX_NOTE in meta["note"]

    def test_metadata_threads(self):
        meta = make_metadata()
        assert meta["threads"] == 1, \
            f"metadata threads must be 1, got {meta['threads']}"

    def test_metadata_runs(self):
        meta = make_metadata(runs=_DEFAULT_RUNS)
        assert meta["runs"] == _DEFAULT_RUNS, \
            f"metadata runs must be {_DEFAULT_RUNS}, got {meta['runs']}"


# ---------------------------------------------------------------------------
# T6 – timing function calls op exactly warmup+runs times; uses inference_mode
# ---------------------------------------------------------------------------

class _CountingModule(nn.Module):
    """Records call count and inference-mode state on each forward."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count: int = 0
        self.inference_states: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        self.inference_states.append(torch.is_inference_mode_enabled())
        return x


class TestTimingMechanism:
    """T6: call count == warmup + runs; all calls under inference_mode."""

    @pytest.mark.parametrize("warmup,runs", [(3, 7), (1, 5)])
    def test_call_count(self, warmup: int, runs: int) -> None:
        counter = _CountingModule()
        x = torch.zeros(1, 1, 4, 4)
        time_op(counter, x, warmup=warmup, runs=runs)
        expected = warmup + runs
        assert counter.call_count == expected, \
            f"Expected {expected} calls (warmup={warmup}+runs={runs}), got {counter.call_count}"

    @pytest.mark.parametrize("warmup,runs", [(2, 4)])
    def test_inference_mode_enabled(self, warmup: int, runs: int) -> None:
        counter = _CountingModule()
        x = torch.zeros(1, 1, 4, 4)
        time_op(counter, x, warmup=warmup, runs=runs)
        assert all(counter.inference_states), \
            "All calls to op inside time_op must be under torch.inference_mode()"
        assert len(counter.inference_states) == warmup + runs
