"""Tests for Step 12: Interactive Demo (app.py and inference.py) (PROJECT_SPEC §7).

T1  predict() returns exactly 3 classes, probabilities sorted in descending order, each in [0, 1], and sum <= 1.0001.
T2  grayscale (mode "L"), RGBA and palette images all run without error.
T3  a 4000x3000 image runs in under 2 seconds including resize.
T4  a missing checkpoint returns a clear message string and does not raise.
T5  switching the model changes the Params/MACs/Size displayed, and the values equal the rows of results.csv.
T6  grep app.py and inference.py: no literals that look like accuracies or latencies (for example "9[0-9]\\.[0-9]").
T7  no network calls: block sockets in the test and run predict(); it still works.
T8  timing: inference time reported is a positive float, measured with torch.inference_mode and set_num_threads(1).
"""

from __future__ import annotations

import csv
import re
import socket
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from thriftnas.models.baselines import build_baseline
from thriftnas.utils.inference import (
    CLASSES,
    get_model_metadata,
    load_model,
    measure_inference_latency,
    predict,
)


@pytest.fixture(scope="module")
def model() -> nn.Module:
    """Fixture providing a fast ResNet20 model on CPU."""
    m = build_baseline("resnet20")
    m.to("cpu")
    m.eval()
    return m


# ---------------------------------------------------------------------------
# T1: predict() returns 3 classes sorted descending in [0, 1]
# ---------------------------------------------------------------------------

class TestPredictOutput:
    def test_top3_properties(self, model: nn.Module) -> None:
        img = Image.new("RGB", (32, 32), color=(120, 150, 180))
        res = predict(model, img)

        assert len(res) == 3, f"Expected exactly 3 predictions, got {len(res)}"
        for cls_name, prob in res:
            assert cls_name in CLASSES, f"Invalid class name '{cls_name}'"
            assert 0.0 <= prob <= 1.0, f"Probability {prob} outside [0, 1]"

        # Check descending order
        assert res[0][1] >= res[1][1] >= res[2][1], "Probabilities not sorted in descending order"

        # Check sum of top-3 <= 1.0001
        total_p = sum(p for _, p in res)
        assert total_p <= 1.0001, f"Sum of top-3 probabilities {total_p} > 1.0001"


# ---------------------------------------------------------------------------
# T2: Diverse PIL image modes (L, RGBA, P)
# ---------------------------------------------------------------------------

class TestImageModes:
    @pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
    def test_image_modes(self, model: nn.Module, mode: str) -> None:
        if mode == "L":
            img = Image.new("L", (48, 48), color=128)
        elif mode == "RGBA":
            img = Image.new("RGBA", (48, 48), color=(255, 100, 50, 200))
        elif mode == "P":
            # Palette mode
            raw = Image.new("RGB", (48, 48), color=(50, 150, 250))
            img = raw.convert("P")
        else:
            raise ValueError(mode)

        res = predict(model, img)
        assert len(res) == 3
        assert sum(p for _, p in res) <= 1.0001


# ---------------------------------------------------------------------------
# T3: Large image (4000x3000) execution time < 2.0s
# ---------------------------------------------------------------------------

class TestLargeImageSpeed:
    def test_large_image_under_2_seconds(self, model: nn.Module) -> None:
        large_img = Image.new("RGB", (4000, 3000), color=(80, 120, 160))
        t0 = time.perf_counter()
        res = predict(model, large_img)
        duration = time.perf_counter() - t0

        assert len(res) == 3
        assert duration < 2.0, f"Inference took {duration:.3f}s (exceeded 2.0s limit)"


# ---------------------------------------------------------------------------
# T4: Missing checkpoint returns clear string without raising
# ---------------------------------------------------------------------------

class TestMissingCheckpoint:
    def test_missing_checkpoint_graceful(self) -> None:
        missing_path = Path("results/nonexistent_dir/weights.pt")
        loaded, err = load_model("ThriftNAS_Missing", checkpoint_path=missing_path)

        assert loaded is None, "Model should be None when checkpoint is missing"
        assert err is not None, "Error message must be returned"
        assert "not found" in err.lower() or "missing" in err.lower()


# ---------------------------------------------------------------------------
# T5: Dynamic metadata matches results/results.csv
# ---------------------------------------------------------------------------

class TestDynamicMetadata:
    def test_metadata_matches_csv_and_changes(self) -> None:
        csv_path = Path("results/results.csv")
        assert csv_path.exists(), "results/results.csv must exist"

        meta_r20 = get_model_metadata("ResNet-20", results_csv_path=csv_path)
        meta_mb = get_model_metadata("MobileNetV2", results_csv_path=csv_path)

        # Changing model changes displayed metadata
        assert meta_r20["Params (M)"] != meta_mb["Params (M)"]
        assert meta_r20["MACs (M)"] != meta_mb["MACs (M)"]

        # Matches rows in results.csv
        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = {r["Model"]: r for r in reader}

        assert "ResNet-20" in rows
        assert "MobileNetV2" in rows
        assert meta_r20["Params (M)"] == rows["ResNet-20"]["Params (M)"]
        assert meta_mb["Params (M)"] == rows["MobileNetV2"]["Params (M)"]


# ---------------------------------------------------------------------------
# T6: Grep app.py and inference.py for hardcoded accuracy/latency literals
# ---------------------------------------------------------------------------

class TestNoHardcodedLiterals:
    def test_grep_no_accuracy_latency_literals(self) -> None:
        files = [Path("app.py"), Path("thriftnas/utils/inference.py")]
        # Pattern matching floats like 90.5, 92.1, 93.8, etc.
        pattern = re.compile(r"\b9[0-9]\.[0-9]+\b")

        for f in files:
            assert f.exists()
            content = f.read_text(encoding="utf-8")
            matches = pattern.findall(content)
            assert not matches, (
                f"Found forbidden hardcoded literal(s) {matches} in {f.name}: "
                f"metrics must be read dynamically from results/results.csv."
            )


# ---------------------------------------------------------------------------
# T7: Zero network calls during predict()
# ---------------------------------------------------------------------------

class TestOfflinePredict:
    def test_predict_without_network(self, model: nn.Module, monkeypatch: pytest.MonkeyPatch) -> None:
        # Block network sockets
        def guard(*args, **kwargs):
            raise RuntimeError("Network disabled: all inference must run completely offline.")

        monkeypatch.setattr(socket, "socket", guard)

        img = Image.new("RGB", (32, 32), color=(50, 100, 150))
        res = predict(model, img)
        assert len(res) == 3


# ---------------------------------------------------------------------------
# T8: Latency timing returns positive float
# ---------------------------------------------------------------------------

class TestTimingMeasurement:
    def test_measured_latency_positive_float(self, model: nn.Module) -> None:
        img = Image.new("RGB", (32, 32), color=(20, 40, 60))
        lat_ms = measure_inference_latency(model, img, runs=10, warmup=2)

        assert isinstance(lat_ms, float), "Latency must be a float"
        assert lat_ms > 0.0, f"Expected positive latency, got {lat_ms}"
