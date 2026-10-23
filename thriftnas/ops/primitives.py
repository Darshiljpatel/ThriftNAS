"""DARTS operation primitives for ThriftNAS.

Public API
----------
PRIMITIVES  – ordered list of 8 operation names (canonical search space order).
OPS         – dict mapping name -> factory lambda(C, stride, affine) -> nn.Module.

All convolutions use bias=False.  BN affine is controlled by the caller.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Canonical op order (used to index alpha vectors – never change this list).
# ---------------------------------------------------------------------------

PRIMITIVES: list[str] = [
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "eca_sep_conv_3x3",
    "zero",
]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SepConv(nn.Module):
    """Double separable convolution: two stacked depthwise+pointwise blocks.

    Structure (per DARTS convention):
        ReLU → DW(k,stride,p) → PW(1×1) → BN →
        ReLU → DW(k,1,p)      → PW(1×1) → BN
    """

    def __init__(self, C: int, k: int, stride: int, affine: bool) -> None:
        super().__init__()
        p = k // 2
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, k, stride=stride, padding=p, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, k, stride=1, padding=p, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class DilConv(nn.Module):
    """Single dilated separable convolution (dilation=2, padding=2).

    Structure: ReLU → DW(3×3,dil=2) → PW(1×1) → BN
    """

    def __init__(self, C: int, stride: int, affine: bool) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            # dilation=2, padding=2 keeps spatial size at stride=1
            nn.Conv2d(C, C, 3, stride=stride, padding=2, dilation=2, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al. 2020).

    Kernel size k derived from C: t = ⌈(log₂C + b) / γ⌉, k = t if odd else t+1.
    Default γ=2, b=1.  A single Conv1d(1,1,k) with no bias.
    """

    def __init__(self, C: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        t = int(abs((math.log2(C) + b) / gamma))
        k = t if t % 2 else t + 1          # ensure k is odd
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pool: (B,C,1,1) → squeeze → (B,C,1) → transpose → (B,1,C)
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        # conv + sigmoid → (B,1,C) → transpose → (B,C,1) → unsqueeze → (B,C,1,1)
        y = torch.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)
        return x * y


class SE(nn.Module):
    """Squeeze-and-Excitation block (Hu et al. 2018).

    Both linear layers use bias=False.
    Parameters: 2 × C × (C // reduction)  (no bias terms).
    """

    def __init__(self, C: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, C // reduction)        # guard against reduction > C
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(C, mid, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(mid, C, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        y = self.pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class EcaSepConv(nn.Module):
    """sep_conv_3x3 followed by an ECA block (the eca_sep_conv_3x3 op)."""

    def __init__(self, C: int, stride: int, affine: bool) -> None:
        super().__init__()
        self.sep = SepConv(C, 3, stride, affine)
        self.eca = ECA(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.eca(self.sep(x))


class FactorizedReduce(nn.Module):
    """Stride-2 spatial reduction that preserves channel count via two offset convolutions.

    Concatenates two 1×1 convolutions that sample at complementary offsets
    (standard DARTS trick) to avoid information loss at the boundary.
    """

    def __init__(self, C_in: int, C_out: int, affine: bool = True) -> None:
        super().__init__()
        assert C_out % 2 == 0, "C_out must be even for FactorizedReduce"
        self.relu = nn.ReLU(inplace=False)
        # Two convolutions sample shifted grids → concatenated to fill C_out channels.
        self.c1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.c2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        # c2 samples at a 1-pixel offset so the two conv outputs align perfectly.
        return self.bn(torch.cat([self.c1(x), self.c2(x[:, :, 1:, 1:])], dim=1))


class Zero(nn.Module):
    """Zero operation: returns a zero tensor with the correct spatial size.

    At stride 1 multiplies by 0 (preserves shape).
    At stride 2 slices then multiplies by 0 (halves H and W).
    Gradient w.r.t. input is zero, which effectively disconnects the edge.
    """

    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


# ---------------------------------------------------------------------------
# Op registry
# ---------------------------------------------------------------------------

OPS: dict[str, object] = {
    "sep_conv_3x3":
        lambda C, stride, affine: SepConv(C, 3, stride, affine),
    "sep_conv_5x5":
        lambda C, stride, affine: SepConv(C, 5, stride, affine),
    "dil_conv_3x3":
        lambda C, stride, affine: DilConv(C, stride, affine),
    "max_pool_3x3":
        lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    "avg_pool_3x3":
        lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1,
                                                count_include_pad=False),
    "skip_connect":
        lambda C, stride, affine: (
            nn.Identity() if stride == 1
            else FactorizedReduce(C, C, affine=affine)
        ),
    "eca_sep_conv_3x3":
        lambda C, stride, affine: EcaSepConv(C, stride, affine),
    "zero":
        lambda C, stride, affine: Zero(stride),
}
