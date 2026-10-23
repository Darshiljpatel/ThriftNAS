"""DARTS differentiable architecture search network.

Public API
----------
MixedOp        – softmax-weighted mixture of the 8 primitive operations.
SearchCell     – one DARTS cell (normal or reduction), 14 edges.
SearchNetwork  – full supernet with shared alpha_normal / alpha_reduce.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from thriftnas.ops.primitives import PRIMITIVES, OPS, FactorizedReduce

# ---------------------------------------------------------------------------
# Constants (derived from PRIMITIVES so they stay in sync)
# ---------------------------------------------------------------------------

_NUM_NODES: int = 4
_NUM_EDGES: int = sum(2 + i for i in range(_NUM_NODES))   # 2+3+4+5 = 14
_NUM_OPS:   int = len(PRIMITIVES)                          # 8


# ---------------------------------------------------------------------------
# Helper: ReLU → Conv2d → BN  (used in cell preprocessing)
# ---------------------------------------------------------------------------

class ReLUConvBN(nn.Module):
    """ReLU → Conv2d(k×k, bias=False) → BatchNorm."""

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        affine: bool = False,
    ) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# ---------------------------------------------------------------------------
# MixedOp
# ---------------------------------------------------------------------------

class MixedOp(nn.Module):
    """Softmax-weighted sum of the 8 primitive operations.

    Pooling ops are followed by BatchNorm(affine=False) in the supernet
    (spec §2) so they produce differentiable, normalised outputs.
    """

    def __init__(self, C: int, stride: int) -> None:
        super().__init__()
        self._ops = nn.ModuleList()
        for name in PRIMITIVES:
            op = OPS[name](C, stride, affine=False)
            if "pool" in name:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input feature map.
            w: per-op softmax weights, shape (8,).
        """
        return sum(wi * op(x) for wi, op in zip(w, self._ops))


# ---------------------------------------------------------------------------
# SearchCell
# ---------------------------------------------------------------------------

class SearchCell(nn.Module):
    """One DARTS cell: 2 input nodes + 4 intermediate nodes, 14 edges total.

    Output is the channel-wise concatenation of the 4 intermediate nodes,
    giving 4 * C_curr output channels.

    Args:
        C_prev_prev:    Channel count of x_{t-2}.
        C_prev:         Channel count of x_{t-1}.
        C_curr:         Internal channel count for this cell.
        reduction:      True  → this is a reduction cell (strides 2 on edges
                                from input nodes).
        reduction_prev: True  → x_{t-2} came from a reduction cell and its
                                spatial size must be halved by preprocess0.
    """

    def __init__(
        self,
        C_prev_prev: int,
        C_prev: int,
        C_curr: int,
        reduction: bool,
        reduction_prev: bool,
    ) -> None:
        super().__init__()
        self._num_nodes = _NUM_NODES
        self._reduction = reduction

        # Preprocessing: project both inputs to C_curr channels.
        # If the cell two steps back was a reduction cell, its spatial dims are
        # larger than the current ones → apply FactorizedReduce to match.
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C_curr, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C_curr, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C_curr, 1, 1, 0, affine=False)

        # One MixedOp per edge.
        # In a reduction cell, edges from the two input nodes (j < 2) use stride 2.
        self._ops = nn.ModuleList()
        for i in range(self._num_nodes):     # intermediate nodes 2 … 5
            for j in range(2 + i):           # 2+i edges into node i+2
                stride = 2 if (reduction and j < 2) else 1
                self._ops.append(MixedOp(C_curr, stride))

    def forward(
        self,
        x_prev_prev: torch.Tensor,
        x_prev: torch.Tensor,
        weights: torch.Tensor,              # (14, 8) – softmax of alpha
    ) -> torch.Tensor:
        s0 = self.preprocess0(x_prev_prev)
        s1 = self.preprocess1(x_prev)
        states = [s0, s1]
        offset = 0
        for i in range(self._num_nodes):
            node_out = sum(
                self._ops[offset + j](states[j], weights[offset + j])
                for j in range(2 + i)
            )
            states.append(node_out)
            offset += 2 + i
        # Concat 4 intermediate node outputs along channel dim
        return torch.cat(states[2:], dim=1)


# ---------------------------------------------------------------------------
# SearchNetwork
# ---------------------------------------------------------------------------

class SearchNetwork(nn.Module):
    """Full DARTS supernet.

    Architecture
    ------------
    stem  →  cells[0..layers-1]  →  GlobalAvgPool  →  Linear(C_final, num_classes)

    Reduction cells (at indices layers//3 and 2*layers//3) double the channel
    width and halve the spatial resolution.

    Alpha parameters
    ----------------
    alpha_normal : nn.Parameter  shape (14, 8)  – shared across normal cells
    alpha_reduce : nn.Parameter  shape (14, 8)  – shared across reduction cells
    Both are initialised with 1e-3 * randn.

    stride2_mask : registered buffer  shape (14,)  bool
        True  → edge uses stride-2 cost in :meth:`expected_cost`.
        False → edge uses stride-1 cost.
        Pattern: [j < 2 for i in range(4) for j in range(2+i)].

    Args:
        C:           Channel multiplier.  Stem outputs 3*C channels.
        num_classes: Output classes.
        layers:      Total number of cells.
    """

    def __init__(
        self,
        C: int = 16,
        num_classes: int = 10,
        layers: int = 5,
    ) -> None:
        super().__init__()
        self._C           = C
        self._num_classes = num_classes
        self._layers      = layers

        # Stem: 3 → 3*C channels (32×32 → 32×32 with padding=1)
        stem_out = 3 * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_out),
        )

        # Reduction cell indices
        self._reduction_indices: set[int] = {layers // 3, 2 * layers // 3}

        # Build cells, tracking channel widths
        C_curr                     = C
        C_prev_prev, C_prev        = stem_out, stem_out
        reduction_prev             = False
        self.cells                 = nn.ModuleList()

        for i in range(layers):
            reduction = (i in self._reduction_indices)
            if reduction:
                C_curr *= 2          # channel doubling at every reduction cell
            cell = SearchCell(
                C_prev_prev = C_prev_prev,
                C_prev      = C_prev,
                C_curr      = C_curr,
                reduction   = reduction,
                reduction_prev = reduction_prev,
            )
            self.cells.append(cell)
            C_prev_prev    = C_prev
            C_prev         = _NUM_NODES * C_curr   # 4 * C_curr (concat output)
            reduction_prev = reduction

        # Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Linear(C_prev, num_classes)

        # Architecture parameters (initialised small so softmax ≈ uniform)
        self.alpha_normal = nn.Parameter(
            1e-3 * torch.randn(_NUM_EDGES, _NUM_OPS))
        self.alpha_reduce = nn.Parameter(
            1e-3 * torch.randn(_NUM_EDGES, _NUM_OPS))

        # stride-2 mask: True where the edge comes from an input node (j < 2)
        mask = [j < 2 for i in range(_NUM_NODES) for j in range(2 + i)]
        self.register_buffer("stride2_mask", torch.tensor(mask))

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = s1 = self.stem(x)

        wn = F.softmax(self.alpha_normal, dim=-1)   # (14, 8)
        wr = F.softmax(self.alpha_reduce,  dim=-1)   # (14, 8)

        for i, cell in enumerate(self.cells):
            w    = wr if (i in self._reduction_indices) else wn
            s0, s1 = s1, cell(s0, s1, w)

        out = self.global_pool(s1).view(s1.size(0), -1)
        return self.classifier(out)

    # -----------------------------------------------------------------------
    # Parameter access helpers
    # -----------------------------------------------------------------------

    def arch_parameters(self) -> list[nn.Parameter]:
        """Return [alpha_normal, alpha_reduce]."""
        return [self.alpha_normal, self.alpha_reduce]

    def weights(self) -> list[nn.Parameter]:
        """Return all parameters *except* alpha (stem, cells, classifier)."""
        arch_ids = {id(p) for p in self.arch_parameters()}
        return [p for p in self.parameters() if id(p) not in arch_ids]

    # -----------------------------------------------------------------------
    # Differentiable expected cost
    # -----------------------------------------------------------------------

    def expected_cost(
        self,
        c1: torch.Tensor,    # (8,) stride-1 costs, PRIMITIVES order
        c2: torch.Tensor,    # (8,) stride-2 costs, PRIMITIVES order
    ) -> torch.Tensor:
        """Differentiable architecture cost proxy (spec §4, addendum §10).

        .. code-block:: text

            E = Σ_{14 normal edges} softmax(alpha_normal) · c1
              + Σ_{8 stride-2 reduce edges} softmax(alpha_reduce) · c2
              + Σ_{6 stride-1 reduce edges} softmax(alpha_reduce) · c1

        Computed in float32 regardless of alpha dtype.  Differentiable
        w.r.t. alpha_normal and alpha_reduce; does **not** involve any
        network weights.
        """
        wn = F.softmax(self.alpha_normal.float(), dim=-1)   # (14, 8)
        wr = F.softmax(self.alpha_reduce.float(),  dim=-1)   # (14, 8)
        m  = self.stride2_mask                                # (14,) bool
        return (wn * c1).sum() + (wr[m] * c2).sum() + (wr[~m] * c1).sum()

    # -----------------------------------------------------------------------
    # Genotype (top-2 edges per intermediate node, excluding 'zero')
    # -----------------------------------------------------------------------

    def genotype(self) -> dict:
        """Return the current best discrete genotype (argmax per edge)."""

        def _parse(weights: torch.Tensor) -> list[tuple[str, int]]:
            zero_idx = PRIMITIVES.index("zero") if "zero" in PRIMITIVES else -1
            gene, offset = [], 0
            for i in range(_NUM_NODES):
                n_in = 2 + i
                W    = weights[offset: offset + n_in].detach()
                # For each incoming edge, find the best non-zero op
                edges = []
                for j in range(n_in):
                    row = W[j].clone()
                    if zero_idx >= 0:
                        row[zero_idx] = -float("inf")
                    best_op = int(row.argmax())
                    edges.append((float(W[j, best_op]), PRIMITIVES[best_op], j))
                top2 = sorted(edges, key=lambda e: -e[0])[:2]
                gene.extend((op, inp) for _, op, inp in top2)
                offset += n_in
            return gene

        with torch.no_grad():
            wn = F.softmax(self.alpha_normal, dim=-1)
            wr = F.softmax(self.alpha_reduce,  dim=-1)
        return {"normal": _parse(wn), "reduce": _parse(wr)}
