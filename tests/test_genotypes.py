"""Tests for thriftnas/models/genotypes.py (Step 7 – T1 through T7).

Run with:  pytest tests/test_genotypes.py -v
"""

from __future__ import annotations

import torch
import pytest
from pathlib import Path

from thriftnas.ops.primitives import PRIMITIVES
from thriftnas.models.genotypes import (
    Genotype,
    _CONCAT,
    _NUM_NODES,
    _ZERO_IDX,
    derive_genotype,
    count_ops,
    load_genotype,
    save_genotype,
    plot_genotype,
    plot_side_by_side,
)
from thriftnas.models.search_net import SearchNetwork

# ---------------------------------------------------------------------------
# Helper: make alpha
# ---------------------------------------------------------------------------

def _random_alpha(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(14, len(PRIMITIVES))


def _derive_normal_reduce(seed: int = 0):
    alpha_n = _random_alpha(seed)
    alpha_r = _random_alpha(seed + 1)
    return derive_genotype(alpha_n, alpha_r)


# ---------------------------------------------------------------------------
# T1 – structural invariants
# ---------------------------------------------------------------------------

class TestStructuralInvariants:
    """T1: 8 entries per cell; no 'zero' op; valid input indices."""

    @pytest.fixture(autouse=True)
    def geno(self):
        self.geno = _derive_normal_reduce(seed=0)

    def test_normal_has_8_entries(self):
        assert len(self.geno.normal) == 8, \
            f"normal has {len(self.geno.normal)} entries, expected 8"

    def test_reduce_has_8_entries(self):
        assert len(self.geno.reduce) == 8, \
            f"reduce has {len(self.geno.reduce)} entries, expected 8"

    def test_no_zero_op_in_normal(self):
        ops = [op for op, _ in self.geno.normal]
        assert "zero" not in ops, f"'zero' found in normal gene: {ops}"

    def test_no_zero_op_in_reduce(self):
        ops = [op for op, _ in self.geno.reduce]
        assert "zero" not in ops, f"'zero' found in reduce gene: {ops}"

    def test_input_indices_in_range(self):
        """For intermediate node i (0-indexed), input_index ∈ [0, i+1]."""
        for cell_name, gene in [("normal", self.geno.normal),
                                 ("reduce", self.geno.reduce)]:
            for i in range(_NUM_NODES):  # 4 intermediate nodes
                max_idx = i + 1          # spec: [0, 2+i-1] = [0, i+1]
                for op, src in gene[2 * i: 2 * i + 2]:
                    assert 0 <= src <= max_idx, (
                        f"{cell_name} node {i}: input_index {src} not in "
                        f"[0, {max_idx}] for op '{op}'"
                    )


# ---------------------------------------------------------------------------
# T2 – handcrafted alpha: node-1 entries must match spec expectation
# ---------------------------------------------------------------------------

def test_t2_handcrafted_alpha_node1():
    """T2: edge 3 has zero-dominant alpha → dropped; node 1 → (sep5,0),(skip,2)."""
    # PRIMITIVES: sep3=0, sep5=1, dil3=2, maxpool=3, avgpool=4, skip=5, eca=6, zero=7
    # Node 1 uses global alpha rows 2,3,4  (j=0,1,2 within the node)
    alpha = torch.zeros(14, len(PRIMITIVES))

    # edge j=0 (global row 2): sep_conv_5x5 wins
    alpha[2, 1] = 5.0   # sep_conv_5x5

    # edge j=1 (global row 3): zero dominates, max_pool is best non-zero but small
    alpha[3, 3] = 1.0   # max_pool_3x3
    alpha[3, 7] = 8.0   # zero — MUST be excluded and this edge must be dropped

    # edge j=2 (global row 4): skip_connect wins
    alpha[4, 5] = 3.0   # skip_connect

    geno = derive_genotype(alpha, alpha)
    node1_entries = geno.normal[2:4]   # entries for intermediate node 1

    assert node1_entries == [("sep_conv_5x5", 0), ("skip_connect", 2)], (
        f"node-1 entries wrong: {node1_entries}\n"
        "Expected: [('sep_conv_5x5', 0), ('skip_connect', 2)]"
    )


# ---------------------------------------------------------------------------
# T3 – node 0 always keeps both edges (it has exactly 2 incoming edges)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 7, 42])
def test_t3_node0_keeps_both_edges(seed: int):
    """T3: node 0 always retains both of its 2 incoming edges."""
    geno = _derive_normal_reduce(seed=seed)
    for cell_name, gene in [("normal", geno.normal), ("reduce", geno.reduce)]:
        n0 = gene[0:2]
        srcs = [src for _, src in n0]
        assert sorted(srcs) == [0, 1], (
            f"{cell_name} node 0: expected input_indices [0,1], got {srcs}"
        )


# ---------------------------------------------------------------------------
# T4 – save → load gives identical Genotype
# ---------------------------------------------------------------------------

def test_t4_save_load_roundtrip(tmp_path: Path):
    """T4: JSON round-trip preserves genotype exactly."""
    geno_orig = _derive_normal_reduce(seed=3)
    p = tmp_path / "geno.json"
    save_genotype(geno_orig, p)

    assert p.exists(),              "JSON file not created"
    assert p.stat().st_size > 0,   "JSON file is empty"

    geno_loaded = load_genotype(p)
    assert geno_loaded.normal        == geno_orig.normal
    assert geno_loaded.normal_concat == geno_orig.normal_concat
    assert geno_loaded.reduce        == geno_orig.reduce
    assert geno_loaded.reduce_concat == geno_orig.reduce_concat


# ---------------------------------------------------------------------------
# T5 – count_ops values sum to 16
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 99])
def test_t5_count_ops_sum_is_16(seed: int):
    """T5: count_ops sums to 16 (8 normal + 8 reduce)."""
    geno = _derive_normal_reduce(seed=seed)
    counts = count_ops(geno)
    total = sum(counts.values())
    assert total == 16, \
        f"count_ops values sum to {total}, expected 16. counts: {counts}"
    # All returned ops must be valid PRIMITIVES (minus "zero")
    for op in counts:
        assert op in PRIMITIVES, f"Unknown op '{op}' in count_ops"
        assert op != "zero",    f"'zero' appears in count_ops"


# ---------------------------------------------------------------------------
# T6 – PNG files are created and non-empty
# ---------------------------------------------------------------------------

def test_t6_plot_genotype_creates_pngs(tmp_path: Path):
    """T6: plot_genotype writes cell_normal.png and cell_reduce.png."""
    geno = _derive_normal_reduce(seed=0)
    plot_genotype(geno, tmp_path)

    for name in ("cell_normal.png", "cell_reduce.png"):
        p = tmp_path / name
        assert p.exists(),            f"{name} not created"
        assert p.stat().st_size > 0,  f"{name} is empty"


def test_t6_plot_side_by_side_creates_png(tmp_path: Path):
    """T6: plot_side_by_side writes a non-empty PNG."""
    geno_a = _derive_normal_reduce(seed=0)
    geno_b = _derive_normal_reduce(seed=5)
    out = tmp_path / "compare.png"
    plot_side_by_side(geno_a, geno_b, ["seed-0", "seed-5"], out)
    assert out.exists(),            "compare.png not created"
    assert out.stat().st_size > 0,  "compare.png is empty"


# ---------------------------------------------------------------------------
# T7 – skip_count from derive_genotype equals skip_count from SearchNetwork.genotype()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 2, 11])
def test_t7_skip_count_matches_search_net(seed: int):
    """T7: skip_count via derive_genotype == skip_count via SearchNetwork.genotype()."""
    torch.manual_seed(seed)
    model = SearchNetwork(C=4, num_classes=10, layers=5)

    # ---- derive_genotype path ----
    geno = derive_genotype(
        model.alpha_normal.detach(),
        model.alpha_reduce.detach(),
    )
    skip_derived = sum(1 for op, _ in geno.normal + geno.reduce
                       if op == "skip_connect")

    # ---- SearchNetwork.genotype() path (as used by search_trainer) ----
    net_geno = model.genotype()   # returns {"normal": [...], "reduce": [...]}
    skip_net = sum(1 for op, _ in net_geno["normal"] + net_geno["reduce"]
                   if op == "skip_connect")

    assert skip_derived == skip_net, (
        f"seed={seed}: skip_count mismatch: "
        f"derive_genotype={skip_derived}, model.genotype()={skip_net}"
    )
