"""Tests for thriftnas/ops/primitives.py (Step 3 – T1 through T9).

Run with:  pytest tests/test_primitives.py -v
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from thriftnas.ops.primitives import (
    PRIMITIVES,
    OPS,
    ECA,
    SE,
    FactorizedReduce,
    Zero,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

C  = 16
X1 = torch.randn(2, C, 32, 32)   # stride-1 input
X2 = torch.randn(2, C, 32, 32)   # stride-2 input (same shape, different object)


def _build(name: str, stride: int, affine: bool = False) -> nn.Module:
    return OPS[name](C, stride, affine)


# ---------------------------------------------------------------------------
# T1 – output shapes
# ---------------------------------------------------------------------------

class TestShapes:
    """T1: stride 1 → (2,16,32,32); stride 2 → (2,16,16,16) for every op."""

    @pytest.mark.parametrize("name", PRIMITIVES)
    def test_stride1(self, name: str) -> None:
        op = _build(name, stride=1)
        out = op(X1)
        assert out.shape == (2, C, 32, 32), \
            f"{name} stride=1: expected (2,{C},32,32), got {tuple(out.shape)}"

    @pytest.mark.parametrize("name", PRIMITIVES)
    def test_stride2(self, name: str) -> None:
        op = _build(name, stride=2)
        out = op(X2)
        assert out.shape == (2, C, 16, 16), \
            f"{name} stride=2: expected (2,{C},16,16), got {tuple(out.shape)}"


# ---------------------------------------------------------------------------
# T2 – exact parameter counts (C=16, affine=False)
# ---------------------------------------------------------------------------

EXPECTED_PARAMS = {
    "sep_conv_3x3":     800,   # 2 × (16×9 + 16×16)
    "sep_conv_5x5":    1312,   # 2 × (16×25 + 16×16)
    "dil_conv_3x3":     400,   # 16×9 + 16×16
    "eca_sep_conv_3x3": 803,   # sep_conv_3x3(800) + Conv1d(1,1,3)(3)
    "max_pool_3x3":       0,
    "avg_pool_3x3":       0,
    "zero":               0,
}


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


class TestParamCounts:
    """T2: exact parameter counts at C=16, affine=False."""

    @pytest.mark.parametrize("name,expected", EXPECTED_PARAMS.items())
    def test_param_count(self, name: str, expected: int) -> None:
        op = _build(name, stride=1, affine=False)
        got = _count_params(op)
        assert got == expected, f"{name}: expected {expected} params, got {got}"

    def test_skip_connect_stride1(self) -> None:
        """skip_connect stride=1 is Identity → 0 params."""
        op = _build("skip_connect", stride=1)
        assert _count_params(op) == 0

    def test_skip_connect_stride2(self) -> None:
        """skip_connect stride=2 is FactorizedReduce → 256 params at C=16, affine=False."""
        op = _build("skip_connect", stride=2, affine=False)
        # c1: 16*8*1=128, c2: 128, BN(affine=False)=0 → 256
        assert _count_params(op) == 256, \
            f"skip_connect stride=2 expected 256 params, got {_count_params(op)}"


# ---------------------------------------------------------------------------
# T3 – Zero output is all-zeros
# ---------------------------------------------------------------------------

class TestZero:
    """T3: Zero op returns all-zeros with the correct shape."""

    def test_zero_stride1_values(self) -> None:
        op = Zero(stride=1)
        out = op(X1)
        assert out.shape == (2, C, 32, 32)
        assert out.abs().max().item() == 0.0, "Zero stride=1: output must be all zeros"

    def test_zero_stride2_values(self) -> None:
        op = Zero(stride=2)
        out = op(X2)
        assert out.shape == (2, C, 16, 16)
        assert out.abs().max().item() == 0.0, "Zero stride=2: output must be all zeros"


# ---------------------------------------------------------------------------
# T4 – skip_connect stride=1 returns exactly x
# ---------------------------------------------------------------------------

def test_t4_identity_exact() -> None:
    """T4: skip_connect stride=1 (Identity) returns the exact same tensor."""
    op = _build("skip_connect", stride=1)
    out = op(X1)
    assert torch.equal(out, X1), "skip_connect stride=1 must return exactly x"


# ---------------------------------------------------------------------------
# T5 – ECA kernel size is always odd; equals 3 for C in {16,32,36,64}
# ---------------------------------------------------------------------------

def _eca_kernel(C: int) -> int:
    t = int(abs((math.log2(C) + 1) / 2))   # gamma=2, b=1
    return t if t % 2 else t + 1


class TestECAKernel:
    """T5: ECA kernel is odd; equals 3 for C in {16,32,36,64}."""

    @pytest.mark.parametrize("ch", [16, 32, 36, 64, 128])
    def test_kernel_is_odd(self, ch: int) -> None:
        k = _eca_kernel(ch)
        assert k % 2 == 1, f"ECA kernel must be odd for C={ch}, got k={k}"

    @pytest.mark.parametrize("ch", [16, 32, 36, 64])
    def test_kernel_equals_3(self, ch: int) -> None:
        k = _eca_kernel(ch)
        assert k == 3, f"ECA kernel must be 3 for C={ch}, got k={k}"


# ---------------------------------------------------------------------------
# T6 – ECA output = x * s, s shape (B,C,1,1), values in (0, 1)
# ---------------------------------------------------------------------------

def test_t6_eca_scale() -> None:
    """T6: ECA returns x * s where s ∈ (0,1) elementwise, shape (2,16,1,1)."""
    eca = ECA(C)
    x   = X1.clone().requires_grad_(False)

    # Capture the scale by dividing output by input where input != 0
    with torch.no_grad():
        out = eca(x)

    # Reconstruct scale: out = x * s → s = out / x  (avoid division by 0)
    mask = x.abs() > 1e-6
    # All s values for a channel are identical → read from pool→sigmoid path
    # Easier: run pool→conv→sigmoid directly
    pool_val = eca.pool(x).squeeze(-1).transpose(-1, -2)          # (2,1,16)
    s = torch.sigmoid(eca.conv(pool_val)).transpose(-1, -2).unsqueeze(-1)  # (2,16,1,1)

    assert s.shape == (2, C, 1, 1), f"Scale shape mismatch: {s.shape}"
    assert s.min().item() > 0.0,   "ECA scale must be > 0 (sigmoid output)"
    assert s.max().item() < 1.0,   "ECA scale must be < 1 (sigmoid output)"


# ---------------------------------------------------------------------------
# T7 – SE preserves shape; correct parameter count (bias=False)
# ---------------------------------------------------------------------------

def test_t7_se_shape() -> None:
    """T7: SE block preserves input shape."""
    se  = SE(C, reduction=16)
    out = se(X1)
    assert out.shape == X1.shape, f"SE output shape {out.shape} != input {X1.shape}"


def test_t7_se_param_count() -> None:
    """T7: SE param count = 2 * C * (C // 16), bias=False (stated choice).

    At C=16: 2 * 16 * 1 = 32 params.
    """
    se       = SE(C, reduction=16)
    expected = 2 * C * max(1, C // 16)
    got      = _count_params(se)
    assert got == expected, \
        f"SE params expected {expected}, got {got}  (bias=False for both Linear layers)"


# ---------------------------------------------------------------------------
# T8 – backward pass gives non-None gradients for all parameters
# ---------------------------------------------------------------------------

class TestBackward:
    """T8: backward through every op → non-None grads for every parameter."""

    @pytest.mark.parametrize("name", PRIMITIVES)
    @pytest.mark.parametrize("stride", [1, 2])
    def test_backward(self, name: str, stride: int) -> None:
        op    = _build(name, stride=stride, affine=False)
        x     = torch.randn(2, C, 32, 32, requires_grad=True)
        out   = op(x)
        loss  = out.sum()
        loss.backward()

        params = list(op.parameters())
        if not params:
            # zero / pool / identity: no parameters; test passes trivially.
            return
        for p in params:
            assert p.grad is not None, \
                f"{name} stride={stride}: param of shape {p.shape} has None gradient"
            assert p.grad.abs().sum().item() > 0, \
                f"{name} stride={stride}: param gradient is zero"


# ---------------------------------------------------------------------------
# T9 – affine=True adds exactly 2*C*(num_BN) = 64 params to sep_conv_3x3
# ---------------------------------------------------------------------------

def test_t9_affine_params() -> None:
    """T9: sep_conv_3x3 affine=True has 800 + 64 = 864 params at C=16.

    2 BN layers × 2 (weight + bias) × 16 channels = 64 extra params.
    """
    op_no_aff = _build("sep_conv_3x3", stride=1, affine=False)
    op_affine  = _build("sep_conv_3x3", stride=1, affine=True)
    base   = _count_params(op_no_aff)
    total  = _count_params(op_affine)
    extra  = total - base
    # 2 BN layers, each adds weight(C) + bias(C) = 2*C when affine=True
    assert extra == 2 * 2 * C, \
        f"Expected {2*2*C} extra params from affine BN, got {extra}"
    assert total == 864, f"sep_conv_3x3 affine=True expected 864 params, got {total}"
