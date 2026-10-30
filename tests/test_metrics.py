"""Tests for Step 10: Evaluation metrics and evaluate.py (PROJECT_SPEC §7).

T1  count_params of a known module (nn.Linear(10, 5)) == 55 (as 55 / 1e6 M).
T2  model_size_mb on a saved weights-only file is within 5% of params * 4 / 2**20.
T3  predicted_latency with a synthetic table (all ops 1.0 ms stride 1, 2.0 ms stride 2) and a genotype with 8 + 8
    edges, where the reduction cell has 4 stride-2 edges, equals the hand-computed sum.
T4  error_pct for predicted 8, measured 10 is 20.0.
T5  cpu_latency runs warmup + runs forward passes (mock) with threads == 1.
T6  evaluate.py with two fake final_metrics.json files writes a table with both rows and the exact column names.
T7  grep: only final-evaluation code calls get_test_loader(final_report=True).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from thriftnas.models.genotypes import Genotype
from thriftnas.utils.metrics import (
    compare_latency,
    count_macs,
    count_params,
    cpu_latency,
    model_size_mb,
    predicted_latency,
)
from evaluate import COLUMNS, _load_and_enrich_metrics, aggregate_runs, write_results_files


# ---------------------------------------------------------------------------
# T1: count_params on known module
# ---------------------------------------------------------------------------

class TestCountParams:
    def test_linear_10_5_params(self) -> None:
        # nn.Linear(10, 5): weight is 5x10 = 50, bias is 5 -> total 55 params
        m = nn.Linear(10, 5)
        p = count_params(m)
        expected = 55 / 1e6
        assert pytest.approx(p, rel=1e-6) == expected


# ---------------------------------------------------------------------------
# T2: model_size_mb within 5% of raw float32 parameter bytes
# ---------------------------------------------------------------------------

class TestModelSizeMB:
    def test_weights_file_size_within_5pct(self, tmp_path: Path) -> None:
        # Create a module with enough parameters so pickle header is negligible (<1%)
        # 500 x 500 = 250,000 weights + 500 bias = 250,500 params * 4 bytes ≈ 1,002,000 bytes
        m = nn.Linear(500, 500)
        p_count = sum(p.numel() for p in m.parameters())
        expected_mb = (p_count * 4) / (2 ** 20)

        weights_file = tmp_path / "model_weights.pt"
        torch.save(m.state_dict(), weights_file)

        actual_mb = model_size_mb(weights_file)
        diff_pct = abs(actual_mb - expected_mb) / expected_mb * 100.0

        assert diff_pct <= 5.0, (
            f"Expected {expected_mb:.4f} MB, got {actual_mb:.4f} MB "
            f"(diff: {diff_pct:.2f}% > 5%)"
        )


# ---------------------------------------------------------------------------
# T3: predicted_latency on synthetic table & genotype
# ---------------------------------------------------------------------------

class TestPredictedLatency:
    def test_predicted_latency_synthetic(self) -> None:
        # Synthetic table: all ops 1.0 ms at stride 1, 2.0 ms at stride 2
        synth_table = {
            "entries": [
                {"op": "sep_conv_3x3", "stride": 1, "lat_median_ms": 1.0},
                {"op": "sep_conv_3x3", "stride": 2, "lat_median_ms": 2.0},
                {"op": "skip_connect", "stride": 1, "lat_median_ms": 1.0},
                {"op": "skip_connect", "stride": 2, "lat_median_ms": 2.0},
            ]
        }

        # Genotype with 8 normal edges (all stride 1) and 8 reduce edges
        # In reduction cell: input_index < 2 has stride 2; input_index >= 2 has stride 1
        # Exactly 4 stride-2 edges (input_index < 2) and 4 stride-1 edges (input_index >= 2)
        normal_gene = [("sep_conv_3x3", 0)] * 8
        reduce_gene = (
            [("sep_conv_3x3", 0), ("skip_connect", 1)] * 2  # 4 edges from inputs 0 and 1 (stride 2)
            + [("sep_conv_3x3", 2), ("skip_connect", 3)] * 2  # 4 edges from inputs 2 and 3 (stride 1)
        )

        g = Genotype(
            normal=normal_gene,
            normal_concat=[2, 3, 4, 5],
            reduce=reduce_gene,
            reduce_concat=[2, 3, 4, 5],
        )

        # Expected sum:
        # Normal cell: 8 edges * 1.0 ms = 8.0 ms
        # Reduction cell: 4 edges * 2.0 ms + 4 edges * 1.0 ms = 8.0 + 4.0 = 12.0 ms
        # Total: 8.0 + 12.0 = 20.0 ms
        pred = predicted_latency(g, synth_table)
        assert pred == 20.0, f"Expected 20.0 ms, got {pred} ms"


# ---------------------------------------------------------------------------
# T4: compare_latency percentage error
# ---------------------------------------------------------------------------

class TestCompareLatency:
    def test_compare_latency_error_pct(self) -> None:
        # predicted 8, measured 10 -> abs(8 - 10) / 10 * 100 = 20.0%
        res = compare_latency(predicted=8.0, measured=10.0)
        assert res.predicted == 8.0
        assert res.measured == 10.0
        assert pytest.approx(res.error_pct, rel=1e-6) == 20.0


# ---------------------------------------------------------------------------
# T5: cpu_latency execution and single thread validation
# ---------------------------------------------------------------------------

class TestCPULatencyMock:
    def test_warmup_and_runs_called_with_single_thread(self) -> None:
        call_count = 0

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                nonlocal call_count
                call_count += 1
                # Must be running with 1 thread
                assert torch.get_num_threads() == 1
                return x

        m = DummyModel()
        mean_ms, std_ms = cpu_latency(m, input_size=(1, 1, 4, 4), warmup=5, runs=15)

        # Total passes must equal warmup + runs (5 + 15 = 20)
        assert call_count == 20, f"Expected 20 calls, got {call_count}"
        assert mean_ms > 0, "Latency mean must be positive"
        assert std_ms >= 0, "Latency std must be non-negative"


# ---------------------------------------------------------------------------
# T6: evaluate.py table generation with two fake final_metrics.json
# ---------------------------------------------------------------------------

class TestEvaluateOutput:
    def test_evaluate_writes_csv_and_md_with_exact_columns(self, tmp_path: Path) -> None:
        dir1 = tmp_path / "model1"
        dir2 = tmp_path / "model2"
        dir1.mkdir()
        dir2.mkdir()

        data1 = {
            "model": "ResNet-20",
            "test_acc": 92.1,
            "params_m": 0.2725,
            "macs_m": 41.22,
            "size_mb": 1.08,
            "cpu_lat_mean": 12.5,
            "cpu_lat_std": 0.4,
            "predicted_latency": None,
            "prediction_error": None,
            "search_gpu_hours": None,
        }
        data2 = {
            "model": "ThriftNAS (lam=0.1)",
            "test_acc": 93.4,
            "params_m": 1.45,
            "macs_m": 180.5,
            "size_mb": 5.8,
            "cpu_lat_mean": 28.2,
            "cpu_lat_std": 0.8,
            "predicted_latency": 25.0,
            "prediction_error": 11.3,
            "search_gpu_hours": 1.85,
        }

        with open(dir1 / "final_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(data1, fh)
        with open(dir2 / "final_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(data2, fh)

        # Load and aggregate
        run1 = _load_and_enrich_metrics(dir1 / "final_metrics.json", measure_cpu=False)
        run2 = _load_and_enrich_metrics(dir2 / "final_metrics.json", measure_cpu=False)

        rows = aggregate_runs([run1, run2])
        csv_file, md_file = write_results_files(rows, out_dir=tmp_path)

        assert csv_file.exists()
        assert md_file.exists()

        # Check CSV columns
        with open(csv_file, encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            assert header == COLUMNS, f"Header mismatch: {header} vs {COLUMNS}"
            data_rows = list(reader)
            assert len(data_rows) == 2, f"Expected 2 data rows, got {len(data_rows)}"
            model_names = [r[0] for r in data_rows]
            assert "ResNet-20" in model_names
            assert "ThriftNAS (lam=0.1)" in model_names

        # Check Markdown content
        with open(md_file, encoding="utf-8") as fh:
            content = fh.read()
            for col in COLUMNS:
                assert col in content, f"Column '{col}' missing from markdown table"
            assert "ResNet-20" in content
            assert "ThriftNAS (lam=0.1)" in content


# ---------------------------------------------------------------------------
# T7: Only final-evaluation code calls get_test_loader(final_report=True)
# ---------------------------------------------------------------------------

class TestGrepTestLoaderFinalReport:
    def test_only_final_evaluation_calls_final_report_true(self) -> None:
        root = Path(__file__).resolve().parents[1]
        py_files = list(root.glob("*.py")) + list((root / "thriftnas").rglob("*.py"))

        allowed_files = {
            "retrain.py",
            "eval_baselines.py",
            "retrain_trainer.py",
            "test_cifar10.py",
            "test_retrain.py",
            "test_baselines.py",
            "test_metrics.py",
            "cifar10.py",
        }

        pattern = re.compile(r"get_test_loader\s*\([^)]*final_report\s*=\s*True", re.MULTILINE)

        for f in py_files:
            if f.name in allowed_files:
                continue
            text = f.read_text(encoding="utf-8")
            match = pattern.search(text)
            assert match is None, (
                f"Forbidden call to get_test_loader(final_report=True) found in {f.name}: "
                f"only final-evaluation scripts may evaluate the test set."
            )
