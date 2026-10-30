"""Baseline architectures for ThriftNAS (PROJECT_SPEC §6, Step 9).

Provides CIFAR-adapted implementations of:
1. ResNet-20 (CIFAR version, He et al. 2016)
2. MobileNetV2 (width 1.0, Sandler et al. 2018 adapted for 32x32 inputs)

Both baselines use:
- The same CIFAR-10 data pipeline and standard augmentation
- The same optimiser (SGD + cosine schedule), batch size, and epoch count
- No Cutout and no auxiliary head (spec §5, §6)
- Evaluated on test set exactly once after the final epoch (spec §8)
"""

from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from thriftnas.ops.cost_model import count_macs

# ---------------------------------------------------------------------------
# MobileNetV2 changes versus ImageNet (spec Step 9 requirement)
# ---------------------------------------------------------------------------

MOBILENETV2_CHANGES_VS_IMAGENET: tuple[str, ...] = (
    "1. Input resolution: Designed for CIFAR-10 32x32x3 images instead of ImageNet 224x224x3.",
    "2. Stem Conv2d stride: Changed from stride=2 (224->112) to stride=1 (32->32) to prevent early loss of spatial resolution.",
    "3. Stage 2 stride: Inverted residual stage 2 (c=24, n=2) uses stride=1 instead of stride=2, keeping resolution at 32x32.",
    "4. Downsampling schedule: Subsequent downsamplings occur at stage 3 (c=32, stride=2 -> 16x16), stage 4 (c=64, stride=2 -> 8x8), and stage 6 (c=160, stride=2 -> 4x4). Final conv (1280 channels) operates on 4x4 features before global average pooling.",
    "5. Classifier head: Output linear layer is Linear(1280, 10) instead of Linear(1280, 1000).",
    "6. No auxiliary head & no Cutout: Single primary head without auxiliary branches; standard augmentation only."
)


# ===========================================================================
# 1. ResNet-20 (CIFAR version)
# ===========================================================================

class BasicBlock(nn.Module):
    """Standard ResNet basic block for CIFAR (He et al. 2016)."""

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = (
            nn.Sequential()
            if stride == 1 and cin == cout
            else nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(y + self.sc(x), inplace=True)


class ResNet20(nn.Module):
    """CIFAR-10 ResNet-20 (He et al. 2016).

    Architecture
    ------------
    - Stem: 3x3 Conv (3 -> 16 channels), stride 1, padding 1 + BN + ReLU
    - Stage 1: 3 BasicBlocks (16 channels, stride 1) -> 32x32
    - Stage 2: 3 BasicBlocks (32 channels, stride 2 on first block) -> 16x16
    - Stage 3: 3 BasicBlocks (64 channels, stride 2 on first block) -> 8x8
    - Global average pooling -> 1x1
    - Linear classifier (64 -> 10)

    Total layers: 1 + 2*3 + 2*3 + 2*3 + 1 = 20 weighted layers.
    Parameters: ~0.272M (fits T2: 0.26M - 0.29M).
    MACs: ~41.8M (fits T2: 38M - 43M).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.auxiliary = False          # No auxiliary head (spec §5, §6)
        self.drop_path_prob = 0.0       # Compatible with RetrainTrainer

        # Stem: 3x3 conv (3 -> 16)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # 3 stages of 3 basic blocks (16, 32, 64 channels)
        self.stage1 = nn.Sequential(
            BasicBlock(16, 16, stride=1),
            BasicBlock(16, 16, stride=1),
            BasicBlock(16, 16, stride=1),
        )
        self.stage2 = nn.Sequential(
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 32, stride=1),
            BasicBlock(32, 32, stride=1),
        )
        self.stage3 = nn.Sequential(
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


# ===========================================================================
# 2. MobileNetV2 (CIFAR-10 adapted, width 1.0)
# ===========================================================================

class ConvBNReLU(nn.Sequential):
    """Conv2d + BatchNorm2d + ReLU6 module."""

    def __init__(
        self,
        cin: int,
        cout: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                cin,
                cout,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(cout),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """Inverted Residual block (Sandler et al. 2018)."""

    def __init__(
        self,
        cin: int,
        cout: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and cin == cout

        hidden_dim = int(round(cin * expand_ratio))
        layers: list[nn.Module] = []

        if expand_ratio != 1:
            # 1x1 pointwise expansion
            layers.append(ConvBNReLU(cin, hidden_dim, kernel_size=1))

        # 3x3 depthwise convolution
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
            # 1x1 linear projection (no non-linearity)
            nn.Conv2d(hidden_dim, cout, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cout),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """CIFAR-adapted MobileNetV2 (width 1.0).

    Adapted for 32x32 inputs to avoid early downsampling collapse.
    Parameters: ~2.237M (fits T3: 2.2M - 2.4M).
    MACs: ~96.2M.
    """

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.auxiliary = False          # No auxiliary head (spec §5, §6)
        self.drop_path_prob = 0.0       # Compatible with RetrainTrainer

        input_channel = int(32 * width_mult)
        last_channel = int(1280 * max(1.0, width_mult))

        # Inverted residual settings: [t, c, n, s]
        # Stage 2 changed from s=2 to s=1 for 32x32 input adaptation.
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # stride 1 (adapted for CIFAR, was 2 in ImageNet)
            [6, 32, 3, 2],  # stride 2 (32x32 -> 16x16)
            [6, 64, 4, 2],  # stride 2 (16x16 -> 8x8)
            [6, 96, 3, 1],  # stride 1 (8x8)
            [6, 160, 3, 2], # stride 2 (8x8 -> 4x4)
            [6, 320, 1, 1], # stride 1 (4x4)
        ]

        # Stem: Conv3x3 with stride 1 (adapted for CIFAR, was stride 2 in ImageNet)
        features: list[nn.Module] = [
            ConvBNReLU(3, input_channel, kernel_size=3, stride=1)
        ]

        # Build InvertedResidual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        stride=stride,
                        expand_ratio=t,
                    )
                )
                input_channel = output_channel

        # Final 1x1 conv layer before pooling (1280 channels)
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        # Global average pool + Linear classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)


# ===========================================================================
# Model builder factory
# ===========================================================================

def build_baseline(name: str, num_classes: int = 10) -> nn.Module:
    """Build a baseline model by name.

    Parameters
    ----------
    name:
        Model name ('resnet20' or 'mobilenet_v2').
    num_classes:
        Number of target classes (default 10).
    """
    key = name.lower().replace("-", "_")
    if key in ("resnet20", "resnet_20"):
        return ResNet20(num_classes=num_classes)
    elif key in ("mobilenetv2", "mobilenet_v2"):
        return MobileNetV2(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown baseline '{name}'. Choose from 'resnet20', 'mobilenet_v2'.")


def count_parameters(model: nn.Module) -> int:
    """Return total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def count_model_macs(model: nn.Module, input_size: tuple[int, ...] = (1, 3, 32, 32)) -> int:
    """Return total MACs for a single input using fvcore."""
    device = next(model.parameters()).device
    x = torch.randn(*input_size, device=device)
    return count_macs(model, x)
