"""Tests for thriftnas/models/search_net.py (Step 5 – T1 through T10).

Run with:  pytest tests/test_search_net.py -v
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from thriftnas.models.search_net import (
    SearchNetwork,
    _NUM_EDGES,
    _NUM_OPS,
    _NUM_NODES,
)
from thriftnas.ops.primitives import PRIMITIVES

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def net() -> SearchNetwork:
    """Default SearchNetwork(C=16, num_classes=10, layers=5) on CPU."""
    torch.manual_seed(0)
    return SearchNetwork(C=16, num_classes=10, layers=5)


# ---------------------------------------------------------------------------
# T1 – alpha shapes and edge count
# ---------------------------------------------------------------------------

def test_t1_alpha_shapes(net: SearchNetwork) -> None:
    """T1: alpha_normal and alpha_reduce are both (14, 8)."""
    assert net.alpha_normal.shape == (_NUM_EDGES, _NUM_OPS), \
        f"alpha_normal shape {net.alpha_normal.shape} != ({_NUM_EDGES}, {_NUM_OPS})"
    assert net.alpha_reduce.shape == (_NUM_EDGES, _NUM_OPS), \
        f"alpha_reduce shape {net.alpha_reduce.shape} != ({_NUM_EDGES}, {_NUM_OPS})"

def test_t1_edge_count() -> None:
    """T1: total edges == 2+3+4+5 == 14."""
    edges = sum(2 + i for i in range(_NUM_NODES))
    assert edges == 14, f"Edge count {edges} != 14"


# ---------------------------------------------------------------------------
# T2 – reduction cell indices for layers=5
# ---------------------------------------------------------------------------

def test_t2_reduction_indices() -> None:
    """T2: for layers=5, reduction cells are at indices 1 and 3."""
    net = SearchNetwork(layers=5)
    assert net._reduction_indices == {1, 3}, \
        f"Reduction indices {net._reduction_indices} != {{1, 3}}"
    # verify the formula directly
    layers = 5
    assert layers // 3 == 1
    assert 2 * layers // 3 == 3


# ---------------------------------------------------------------------------
# T3 – forward pass shape and finiteness
# ---------------------------------------------------------------------------

def test_t3_forward_shape(net: SearchNetwork) -> None:
    """T3: forward on (2, 3, 32, 32) returns (2, 10) with finite values."""
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (2, 10), f"Output shape {out.shape} != (2, 10)"
    assert torch.isfinite(out).all(), "Output contains non-finite values"


# ---------------------------------------------------------------------------
# T4 – stride2_mask has exactly 8 True and 6 False
# ---------------------------------------------------------------------------

def test_t4_stride2_mask(net: SearchNetwork) -> None:
    """T4: stride2_mask has 8 True (stride-2 edges) and 6 False."""
    m = net.stride2_mask
    assert int(m.sum()) == 8,      f"stride2_mask has {int(m.sum())} True, expected 8"
    assert int((~m).sum()) == 6,   f"stride2_mask has {int((~m).sum())} False, expected 6"
    assert m.shape == (14,),        f"stride2_mask shape {m.shape} != (14,)"


# ---------------------------------------------------------------------------
# T5 – uniform alpha (all zeros): E == 20 * mean(c1) + 8 * mean(c2)
# ---------------------------------------------------------------------------

def test_t5_uniform_alpha_cost() -> None:
    """T5: with zero alpha (uniform softmax) E == 20*mean(c1) + 8*mean(c2)."""
    net = SearchNetwork()
    net.alpha_normal.data.zero_()
    net.alpha_reduce.data.zero_()

    torch.manual_seed(123)
    c1 = torch.rand(8)
    c2 = torch.rand(8)

    E        = net.expected_cost(c1, c2)
    expected = 20.0 * c1.mean() + 8.0 * c2.mean()

    assert abs(E.item() - expected.item()) < 1e-5, \
        f"E={E.item():.8f}, expected={expected.item():.8f}, diff={abs(E.item()-expected.item()):.2e}"


# ---------------------------------------------------------------------------
# T6 – one-hot alpha (logit 50 on op k): E ~= 20*c1[k] + 8*c2[k]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [0, 3, 7])
def test_t6_onehot_alpha_cost(k: int) -> None:
    """T6: with one-hot alpha (logit 50 on op k) E ~= 20*c1[k] + 8*c2[k]."""
    net = SearchNetwork()
    net.alpha_normal.data.fill_(-50.0)
    net.alpha_normal.data[:, k] = 50.0
    net.alpha_reduce.data.fill_(-50.0)
    net.alpha_reduce.data[:, k] = 50.0

    torch.manual_seed(k + 42)
    c1 = torch.rand(8)
    c2 = torch.rand(8)

    E        = net.expected_cost(c1, c2)
    expected = 20.0 * c1[k] + 8.0 * c2[k]

    assert abs(E.item() - expected.item()) < 1e-3, \
        f"k={k}: E={E.item():.6f}, expected={expected.item():.6f}"


# ---------------------------------------------------------------------------
# T7 – zero-op dominant: E close to 20*c1[zero] + 8*c2[zero] (== 0 here)
# ---------------------------------------------------------------------------

def test_t7_zero_op_dominant() -> None:
    """T7: with 'zero' op dominant and zero costs, E < 1e-3."""
    net = SearchNetwork()
    zero_idx = PRIMITIVES.index("zero")

    net.alpha_normal.data.fill_(-50.0)
    net.alpha_normal.data[:, zero_idx] = 50.0
    net.alpha_reduce.data.fill_(-50.0)
    net.alpha_reduce.data[:, zero_idx] = 50.0

    # zero op has 0 cost; other ops cost 1.0
    c1 = torch.ones(8)
    c2 = torch.ones(8)
    c1[zero_idx] = 0.0
    c2[zero_idx] = 0.0

    E = net.expected_cost(c1, c2)
    target = 20.0 * c1[zero_idx] + 8.0 * c2[zero_idx]   # == 0
    assert abs(E.item() - target) < 1e-3, \
        f"Zero-dominant E={E.item():.6f}, target={target:.6f}"


# ---------------------------------------------------------------------------
# T8 – backward populates only alpha grads; weight grads remain None
# ---------------------------------------------------------------------------

def test_t8_backward_alpha_only() -> None:
    """T8: E.backward() → alpha grads non-None, network weight grads None."""
    net = SearchNetwork()
    # Ensure no stale grads
    net.zero_grad()

    c1 = torch.rand(8)
    c2 = torch.rand(8)
    E  = net.expected_cost(c1, c2)
    E.backward()

    # Alpha grads must exist and be finite
    for name, alpha in [("alpha_normal", net.alpha_normal),
                        ("alpha_reduce", net.alpha_reduce)]:
        assert alpha.grad is not None, f"{name}.grad is None after backward"
        assert torch.isfinite(alpha.grad).all(), \
            f"{name}.grad contains non-finite values"

    # Network weight grads must still be None
    for p in net.weights():
        assert p.grad is None, \
            f"A weight param of shape {tuple(p.shape)} has non-None grad after expected_cost.backward()"


# ---------------------------------------------------------------------------
# T9 – 50 Adam steps on E decrease the weight of the most expensive op
# ---------------------------------------------------------------------------

def test_t9_adam_decreases_expensive_op() -> None:
    """T9: minimising E with Adam reduces the softmax weight of the costliest op."""
    net = SearchNetwork()
    net.alpha_normal.data.zero_()
    net.alpha_reduce.data.zero_()

    # Make op k = 1 (sep_conv_5x5) much more expensive than the rest
    k  = 1
    c1 = torch.full((8,), 0.1)
    c1[k] = 10.0
    c2 = c1.clone()

    opt     = torch.optim.Adam(net.arch_parameters(), lr=0.1)
    initial = float(F.softmax(net.alpha_normal[0].detach().float(), dim=-1)[k])

    for _ in range(50):
        opt.zero_grad()
        net.expected_cost(c1, c2).backward()
        opt.step()

    final = float(F.softmax(net.alpha_normal[0].float(), dim=-1)[k])
    assert final < initial, \
        f"Expensive-op weight did not decrease: {initial:.4f} -> {final:.4f}"


# ---------------------------------------------------------------------------
# T10 – weights() and arch_parameters() are disjoint and cover all params
# ---------------------------------------------------------------------------

def test_t10_parameter_partition(net: SearchNetwork) -> None:
    """T10: weights() ∩ arch_parameters() = ∅ and their union = all params."""
    all_ids    = {id(p) for p in net.parameters()}
    weight_ids = {id(p) for p in net.weights()}
    arch_ids   = {id(p) for p in net.arch_parameters()}

    assert weight_ids.isdisjoint(arch_ids), \
        "weights() and arch_parameters() share parameters!"
    assert weight_ids | arch_ids == all_ids, \
        (f"weights() + arch_parameters() do not cover all params. "
         f"Missing: {len(all_ids - (weight_ids | arch_ids))} params.")
