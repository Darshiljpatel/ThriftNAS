"""Tests for Step 11: Plotting and Visualization utilities (PROJECT_SPEC §7).

T1  pareto_front([(100, 90), (200, 92), (300, 91)]) returns [(100, 90), (200, 92)].
T2  pareto_front([(100, 90), (100, 91)]) keeps only (100, 91).
T3  every saved PNG exists, is non-empty, and its DPI metadata is 300 (PIL Image.info["dpi"]).
T4  every axis has a non-empty label (check via the returned Figure object).
T5  with only 1 seed per model no error bars are drawn and no division by zero occurs.
T6  each Pareto plot has one text label per data point.
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image
import pytest
import matplotlib.pyplot as plt

from thriftnas.utils.plots import (
    pareto_front,
    plot_alpha_evolution,
    plot_e_alpha_over_epochs,
    plot_lambda_ablation,
    plot_pareto_front,
    plot_skip_count_vs_lambda,
)
from scripts.make_plots import main as make_plots_main


# ---------------------------------------------------------------------------
# T1 & T2: Pareto front algorithm
# ---------------------------------------------------------------------------

class TestParetoFront:
    def test_t1_dominated_points_filtered(self) -> None:
        # Lower cost and higher accuracy are better
        # (300, 91) is dominated by (200, 92) (higher cost AND lower accuracy)
        points = [(100, 90), (200, 92), (300, 91)]
        res = pareto_front(points)
        assert res == [(100, 90), (200, 92)]

    def test_t2_same_cost_highest_accuracy_kept(self) -> None:
        # Same cost: 100, keep 91
        points = [(100, 90), (100, 91)]
        res = pareto_front(points)
        assert res == [(100, 91)]


# ---------------------------------------------------------------------------
# T3: DPI metadata of saved PNGs is 300
# ---------------------------------------------------------------------------

class TestSavedFiguresDPI:
    def test_saved_png_300_dpi(self, tmp_path: Path) -> None:
        fig_path = tmp_path / "test_fig.png"
        pts = [
            {"model": "A", "cost": 10.0, "acc": 90.0},
            {"model": "B", "cost": 20.0, "acc": 92.0},
        ]
        fig = plot_pareto_front(
            data_points=pts,
            cost_key="cost",
            acc_key="acc",
            label_key="model",
            output_path=fig_path,
        )
        plt.close(fig)

        assert fig_path.exists(), "PNG file was not created"
        assert fig_path.stat().st_size > 0, "PNG file is empty"

        with Image.open(fig_path) as img:
            dpi = img.info.get("dpi")
            assert dpi is not None, "DPI metadata missing in PNG"
            assert round(dpi[0]) == 300, f"Expected 300 DPI, got {dpi[0]}"
            assert round(dpi[1]) == 300, f"Expected 300 DPI, got {dpi[1]}"


# ---------------------------------------------------------------------------
# T4: Every axis has a non-empty label
# ---------------------------------------------------------------------------

class TestAxisLabels:
    def test_pareto_axis_labels_non_empty(self) -> None:
        pts = [{"model": "A", "cost": 10.0, "acc": 90.0}]
        fig = plot_pareto_front(pts, cost_key="cost", acc_key="acc", label_key="model")
        for ax in fig.axes:
            assert ax.get_xlabel() != "", f"Empty xlabel on axis {ax}"
            assert ax.get_ylabel() != "", f"Empty ylabel on axis {ax}"
        plt.close(fig)

    def test_lambda_ablation_axis_labels_non_empty(self) -> None:
        entries = [
            {"lambda": "0.0", "acc": 93.0, "macs_m": 200.0, "lat_ms": 30.0},
            {"lambda": "0.1", "acc": 92.0, "macs_m": 120.0, "lat_ms": 15.0},
        ]
        fig = plot_lambda_ablation(entries)
        for ax in fig.axes:
            assert ax.get_ylabel() != "", f"Empty ylabel on axis {ax}"
        assert fig.axes[0].get_xlabel() != "", "Empty xlabel on primary axis"
        plt.close(fig)

    def test_skip_count_axis_labels_non_empty(self) -> None:
        entries = [{"lambda": 0.0, "skip_count": 0}, {"lambda": 0.1, "skip_count": 2}]
        fig = plot_skip_count_vs_lambda(entries)
        for ax in fig.axes:
            assert ax.get_xlabel() != "", f"Empty xlabel on axis {ax}"
            assert ax.get_ylabel() != "", f"Empty ylabel on axis {ax}"
        plt.close(fig)

    def test_e_alpha_axis_labels_non_empty(self) -> None:
        runs = {"0.0": [8.0, 8.5, 9.0], "0.1": [8.0, 7.5, 7.0]}
        fig = plot_e_alpha_over_epochs(runs)
        for ax in fig.axes:
            assert ax.get_xlabel() != "", f"Empty xlabel on axis {ax}"
            assert ax.get_ylabel() != "", f"Empty ylabel on axis {ax}"
        plt.close(fig)

    def test_alpha_evolution_axis_labels_non_empty(self) -> None:
        epochs = [0, 1, 2]
        op_data = {"op1": [0.5, 0.6, 0.7], "op2": [0.5, 0.4, 0.3]}
        fig = plot_alpha_evolution(epochs, op_data)
        for ax in fig.axes:
            assert ax.get_xlabel() != "", f"Empty xlabel on axis {ax}"
            assert ax.get_ylabel() != "", f"Empty ylabel on axis {ax}"
        plt.close(fig)


# ---------------------------------------------------------------------------
# T5: With only 1 seed per model no error bars are drawn
# ---------------------------------------------------------------------------

class TestSingleSeedNoErrorBars:
    def test_no_error_bars_when_single_seed(self) -> None:
        # Single seed: std is 0.0 or None
        pts = [
            {"model": "A", "cost": 10.0, "acc": 90.0, "cost_std": 0.0, "acc_std": 0.0},
            {"model": "B", "cost": 20.0, "acc": 92.0, "cost_std": None, "acc_std": None},
        ]
        fig = plot_pareto_front(pts, cost_key="cost", acc_key="acc", label_key="model")
        ax = fig.axes[0]
        # Inspect errorbar container
        errorbar_container = ax.containers[0]
        assert not errorbar_container.has_xerr, "Error bars on X should not be drawn for single seed"
        assert not errorbar_container.has_yerr, "Error bars on Y should not be drawn for single seed"
        plt.close(fig)


# ---------------------------------------------------------------------------
# T6: Each Pareto plot has one text label per data point
# ---------------------------------------------------------------------------

class TestParetoTextLabels:
    def test_one_label_per_data_point(self) -> None:
        pts = [
            {"model": "ResNet-20", "cost": 41.2, "acc": 92.1},
            {"model": "MobileNetV2", "cost": 91.3, "acc": 92.8},
            {"model": "ThriftNAS", "cost": 125.0, "acc": 93.5},
        ]
        fig = plot_pareto_front(pts, cost_key="cost", acc_key="acc", label_key="model")
        ax = fig.axes[0]

        # Extract text annotations
        label_texts = [t.get_text() for t in ax.texts]
        for p in pts:
            assert p["model"] in label_texts, f"Label {p['model']} missing from plot"
        assert len(ax.texts) == len(pts), f"Expected {len(pts)} labels, got {len(ax.texts)}"
        plt.close(fig)
