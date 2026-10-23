"""DARTS-style retraining network for ThriftNAS (PROJECT_SPEC §5, §7).

Public API
----------
NetworkCIFAR(C, num_classes, layers, genotype, drop_path_prob)
    DARTS cell-based classifier with optional drop-path regularisation.

Notes
-----
- NO auxiliary head (flag kept, default off).
- NO Cutout (handled in the data loader, default off).
- Drop-path ramps from 0 to ``drop_path_prob`` linearly during training;
  it is applied in train mode only and never to identity (skip_connect) edges.
- Channels double at each reduction cell (reduction at layers//3 and 2*layers//3).
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from thriftnas.models.genotypes import Genotype
from thriftnas.ops.primitives import FactorizedReduce, OPS

# ---------------------------------------------------------------------------
# Parameter-count warning threshold for ThriftNAS models
# ---------------------------------------------------------------------------

_THRIFTNAS_PARAM_WARN = 1_500_000   # 1.5 M


# ---------------------------------------------------------------------------
# Drop-path  (reference solution verbatim, PROJECT_SPEC §8)
# ---------------------------------------------------------------------------

def drop_path(x: torch.Tensor, p: float) -> torch.Tensor:
    """Randomly drop entire samples from a feature map during training.

    Parameters
    ----------
    x:
        Feature map of shape ``(B, C, H, W)``.
    p:
        Drop probability in ``[0, 1)``.  When ``p == 0`` the input is
        returned unchanged.

    Returns
    -------
    torch.Tensor
        Feature map with the same shape as ``x``; scaled by ``1/(1-p)``
        so the expected value is preserved.
    """
    if p > 0.0:
        keep = 1.0 - p
        mask = torch.bernoulli(
            torch.full((x.size(0), 1, 1, 1), keep, device=x.device)
        ).to(x.dtype)
        x = x / keep * mask
    return x


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ReLUConvBN(nn.Module):
    """ReLU → Conv2d → BN.

    Used as the pre-processing module for cell inputs (1×1 with padding 0).

    Parameters
    ----------
    C_in, C_out:
        Input / output channel counts.
    k, stride, padding:
        Convolution geometry.
    affine:
        Whether BN has learnable affine parameters (``True`` in retrain).
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        k: int,
        stride: int,
        padding: int,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, k, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------

class Cell(nn.Module):
    """One DARTS cell (normal or reduction) built from a discrete genotype.

    Parameters
    ----------
    genotype:
        A :class:`~thriftnas.models.genotypes.Genotype` providing the list of
        ``(op_name, input_index)`` pairs for normal and reduction cells.
    C_pp, C_p, C:
        Channels of the two predecessor states and the current cell's width.
    reduction:
        If ``True``, this is a reduction cell (edges from input nodes use
        stride 2; others stride 1).
    reduction_prev:
        If ``True``, the *previous* cell was a reduction cell, so the
        ``s0`` pre-processor must use :class:`~thriftnas.ops.primitives.FactorizedReduce`
        to match spatial dimensions.
    """

    def __init__(
        self,
        genotype: Genotype,
        C_pp: int,
        C_p: int,
        C: int,
        reduction: bool,
        reduction_prev: bool,
    ) -> None:
        super().__init__()
        # Pre-processing: align s0 / s1 to C channels
        self.pre0: nn.Module = (
            FactorizedReduce(C_pp, C, affine=True)
            if reduction_prev
            else ReLUConvBN(C_pp, C, 1, 1, 0)
        )
        self.pre1 = ReLUConvBN(C_p, C, 1, 1, 0)

        # Unpack genotype for this cell type
        gene = genotype.reduce if reduction else genotype.normal
        names, idx = zip(*gene)          # two parallel tuples
        self._idx  = list(idx)           # input-node indices for each op

        # Build ops: edges from the two input nodes (indices 0 and 1) in a
        # reduction cell use stride 2; all other edges use stride 1.
        self.ops = nn.ModuleList(
            OPS[n](C, 2 if (reduction and i < 2) else 1, True)
            for n, i in zip(names, self._idx)
        )

    def forward(self, s0: torch.Tensor, s1: torch.Tensor, drop_p: float) -> torch.Tensor:
        """Forward pass with optional drop-path.

        Parameters
        ----------
        s0, s1:
            The two predecessor feature maps (``c_{k-2}`` and ``c_{k-1}``).
        drop_p:
            Drop-path probability.  Applied only in training mode and never
            to ``nn.Identity`` ops (skip_connect at stride 1).
        """
        st = [self.pre0(s0), self.pre1(s1)]   # state list: [s0', s1', node0, …]

        for i in range(4):                     # 4 intermediate nodes
            hs = []
            for k in (2 * i, 2 * i + 1):      # the 2 ops feeding this node
                h = self.ops[k](st[self._idx[k]])
                # Apply drop-path in train mode only, never to identity ops
                if (
                    self.training
                    and drop_p > 0.0
                    and not isinstance(self.ops[k], nn.Identity)
                ):
                    h = drop_path(h, drop_p)
                hs.append(h)
            st.append(hs[0] + hs[1])

        # Concatenate the 4 intermediate nodes (indices 2..5)
        return torch.cat(st[2:], dim=1)


# ---------------------------------------------------------------------------
# NetworkCIFAR
# ---------------------------------------------------------------------------

class NetworkCIFAR(nn.Module):
    """DARTS retrain network for CIFAR-10.

    Architecture
    ------------
    stem (3×3 Conv → BN, 3 → 3*C)
    → ``layers`` cells (normal / reduction per position)
    → global average pool
    → linear classifier

    Reduction cells are placed at positions ``layers // 3`` and
    ``2 * layers // 3``; channels double at each reduction.

    Parameters
    ----------
    C:
        Initial channel count (``C_init``).  Default 36 (PROJECT_SPEC §5).
    num_classes:
        Number of output classes (10 for CIFAR-10).
    layers:
        Total number of cells.  Default 8.
    genotype:
        Discrete architecture to unroll.
    drop_path_prob:
        Peak drop-path probability (ramps 0 → ``drop_path_prob`` linearly
        with epoch from the caller).  Default 0.0 (no drop-path).
    auxiliary:
        Enable auxiliary head (flag only, default ``False`` per spec §5).
    """

    def __init__(
        self,
        C: int = 36,
        num_classes: int = 10,
        layers: int = 8,
        genotype: Genotype | None = None,
        drop_path_prob: float = 0.0,
        auxiliary: bool = False,        # behind flag, default off
    ) -> None:
        super().__init__()
        if genotype is None:
            raise ValueError("genotype must be provided to NetworkCIFAR")

        self.drop_path_prob = drop_path_prob
        self._auxiliary     = auxiliary          # stored but unused by default

        # ---- Stem: 3×3 Conv (3 → 3*C) + BN  (no ReLU before first cell) ----
        C_stem = 3 * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_stem, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_stem),
        )

        # ---- Build cells ----------------------------------------------------
        reduction_positions = {layers // 3, 2 * layers // 3}

        C_pp, C_p, C_cur = C_stem, C_stem, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        for i in range(layers):
            reduction = (i in reduction_positions)
            if reduction:
                # Channel doubles at each reduction (DARTS convention)
                C_cur *= 2

            cell = Cell(
                genotype       = genotype,
                C_pp           = C_pp,
                C_p            = C_p,
                C              = C_cur,
                reduction      = reduction,
                reduction_prev = reduction_prev,
            )
            self.cells.append(cell)

            # Output of one cell: 4 nodes concatenated → 4 * C_cur channels
            C_pp           = C_p
            C_p            = 4 * C_cur
            reduction_prev = reduction

        # ---- Classifier head -----------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Linear(C_p, num_classes)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        out = self.global_pool(s1)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


# ---------------------------------------------------------------------------
# Factory + parameter-count check
# ---------------------------------------------------------------------------

def build_network(
    C: int,
    num_classes: int,
    layers: int,
    genotype: Genotype,
    drop_path_prob: float = 0.0,
    is_thriftnas: bool = True,
) -> NetworkCIFAR:
    """Construct a :class:`NetworkCIFAR` and optionally warn if too large.

    Parameters
    ----------
    C, num_classes, layers, genotype, drop_path_prob:
        Forwarded to :class:`NetworkCIFAR`.
    is_thriftnas:
        If ``True`` (default), emit a warning when the model exceeds
        1.5 M parameters.  Set ``False`` for baselines (ResNet-20, etc.)
        which are not expected to be small.

    Returns
    -------
    NetworkCIFAR
    """
    model = NetworkCIFAR(
        C              = C,
        num_classes    = num_classes,
        layers         = layers,
        genotype       = genotype,
        drop_path_prob = drop_path_prob,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NetworkCIFAR: {n_params:,} parameters")

    if is_thriftnas and n_params > _THRIFTNAS_PARAM_WARN:
        warnings.warn(
            f"ThriftNAS model has {n_params:,} parameters "
            f"(> {_THRIFTNAS_PARAM_WARN:,} threshold). "
            "Consider reducing C or layers.",
            stacklevel=2,
        )

    return model
