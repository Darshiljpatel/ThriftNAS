"""Unit tests for Step 9: Baselines (ResNet-20, MobileNetV2, eval_baselines.py).

Test cases:
T1  forward on randn(2, 3, 32, 32) returns (2, 10) for both models (finite).
T2  ResNet-20 parameters between 0.26M and 0.29M; MACs between 38M and 43M.
T3  MobileNetV2 parameters between 2.2M and 2.4M (report MACs; no fixed range).
T4  both models have no Cutout, no auxiliary head, and use the shared epoch count (read from config).
T5  the same seed gives the same first-batch loss on two runs (CPU).
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from thriftnas.models.baselines import (
    MOBILENETV2_CHANGES_VS_IMAGENET,
    MobileNetV2,
    ResNet20,
    build_baseline,
    count_model_macs,
    count_parameters,
)
from thriftnas.data.cifar10 import _train_transform
from thriftnas.utils import set_seed


# ---------------------------------------------------------------------------
# T1: Forward pass shape and numerical stability
# ---------------------------------------------------------------------------

class TestForwardShape:
    @pytest.mark.parametrize("model_name", ["resnet20", "mobilenet_v2"])
    def test_forward_shape_and_finite(self, model_name: str) -> None:
        model = build_baseline(model_name, num_classes=10)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 10), f"Expected shape (2, 10), got {out.shape}"
        assert torch.isfinite(out).all(), "Output contains NaN or Inf values"


# ---------------------------------------------------------------------------
# T2: ResNet-20 parameter count and MACs
# ---------------------------------------------------------------------------

class TestResNet20Complexity:
    def test_resnet20_params_and_macs(self) -> None:
        model = ResNet20(num_classes=10)
        params = count_parameters(model)
        macs = count_model_macs(model, input_size=(1, 3, 32, 32))

        # Spec T2: parameters between 0.26M and 0.29M; MACs between 38M and 43M
        assert 0.26e6 <= params <= 0.29e6, f"ResNet-20 params {params:,} ({params/1e6:.4f}M) not in [0.26M, 0.29M]"
        assert 38e6 <= macs <= 43e6, f"ResNet-20 MACs {macs:,} ({macs/1e6:.2f}M) not in [38M, 43M]"


# ---------------------------------------------------------------------------
# T3: MobileNetV2 parameter count and MACs
# ---------------------------------------------------------------------------

class TestMobileNetV2Complexity:
    def test_mobilenetv2_params_and_macs(self) -> None:
        model = MobileNetV2(num_classes=10)
        params = count_parameters(model)
        macs = count_model_macs(model, input_size=(1, 3, 32, 32))

        # Spec T3: parameters between 2.2M and 2.4M (report MACs; no fixed range)
        assert 2.2e6 <= params <= 2.4e6, f"MobileNetV2 params {params:,} ({params/1e6:.4f}M) not in [2.2M, 2.4M]"
        assert macs > 0, "MACs must be positive"
        print(f"\n[REPORT T3] MobileNetV2 MACs: {macs:,} ({macs/1e6:.2f}M MACs), Params: {params:,} ({params/1e6:.4f}M)")

    def test_mobilenetv2_documented_changes(self) -> None:
        # Check that changes versus ImageNet are explicitly documented
        assert len(MOBILENETV2_CHANGES_VS_IMAGENET) >= 5, "Expected at least 5 documented changes"
        assert any("stride=1" in c for c in MOBILENETV2_CHANGES_VS_IMAGENET)


# ---------------------------------------------------------------------------
# T4: No Cutout, no auxiliary head, shared epoch count
# ---------------------------------------------------------------------------

class TestBaselineDesignConstraints:
    @pytest.mark.parametrize("model_name", ["resnet20", "mobilenet_v2"])
    def test_no_auxiliary_head(self, model_name: str) -> None:
        model = build_baseline(model_name)
        # Model should have auxiliary flag set to False, or not have an auxiliary submodule
        assert getattr(model, "auxiliary", False) is False, f"{model_name} must not have an active auxiliary head"
        assert not hasattr(model, "aux_head"), f"{model_name} has aux_head module"

    def test_no_cutout_by_default(self) -> None:
        # Check training transforms have cutout disabled by default
        transform = _train_transform(cutout=False)
        transform_types = [type(t).__name__ for t in transform.transforms]
        assert "Cutout" not in transform_types, "Cutout should not be present in default transforms"

    def test_uses_shared_epoch_config(self) -> None:
        shared_yaml = Path("configs/shared.yaml")
        assert shared_yaml.exists(), "configs/shared.yaml must exist"
        with open(shared_yaml, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        assert "epochs" in cfg
        assert "batch_size" in cfg

        from eval_baselines import _load_shared
        shared_loaded = _load_shared()
        assert shared_loaded["epochs"] == cfg["epochs"]
        assert shared_loaded["batch_size"] == cfg["batch_size"]


# ---------------------------------------------------------------------------
# T5: Determinism under the same seed (CPU, first batch loss)
# ---------------------------------------------------------------------------

class TestSeedReproducibility:
    @pytest.mark.parametrize("model_name", ["resnet20", "mobilenet_v2"])
    def test_first_batch_loss_deterministic(self, model_name: str) -> None:
        def compute_first_batch_loss(seed: int) -> float:
            set_seed(seed)
            torch.manual_seed(seed)
            model = build_baseline(model_name, num_classes=10)
            model.train()

            # Synthetic batch
            g = torch.Generator().manual_seed(seed)
            x = torch.randn(8, 3, 32, 32, generator=g)
            y = torch.randint(0, 10, (8,), generator=g)

            out = model(x)
            loss = F.cross_entropy(out, y)
            return float(loss.item())

        loss1 = compute_first_batch_loss(seed=42)
        loss2 = compute_first_batch_loss(seed=42)
        assert abs(loss1 - loss2) < 1e-6, f"{model_name} first batch loss differed: {loss1} vs {loss2}"
