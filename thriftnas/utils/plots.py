"""Plotting and visualization utilities for ThriftNAS (PROJECT_SPEC §7, Step 11).

Generates 300-dpi publication-quality figures:
1. Pareto front plots (Accuracy vs. MACs, Accuracy vs. Latency)
2. Lambda ablation (Accuracy, MACs, Latency across trade-off lambda)
3. Skip-connection count vs. lambda
4. Alpha parameter evolution across search epochs
5. Expected cell cost E(alpha) across search epochs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. Pareto frontier computation (spec §11 reference solution)
# ---------------------------------------------------------------------------

def pareto_front(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Compute the Pareto frontier for (cost, accuracy) points.

    Lower cost and higher accuracy are better.
    Ties in cost keep the highest accuracy.

    Parameters
    ----------
    points:
        Sequence of (cost, accuracy) tuples.

    Returns
    -------
    list[tuple[float, float]]
        Pareto-optimal (cost, accuracy) points sorted by cost ascending.
    """
    best, front = -1.0, []
    for c, a in sorted(points, key=lambda p: (p[0], -p[1])):
        if a > best:
            front.append((c, a))
            best = a
    return front


# ---------------------------------------------------------------------------
# 2. Pareto Front Plotting
# ---------------------------------------------------------------------------

def plot_pareto_front(
    data_points: Sequence[dict[str, Any]],
    cost_key: str,
    acc_key: str = "acc",
    label_key: str = "label",
    x_label: str = "Computational Cost (MACs [M])",
    y_label: str = "Top-1 Test Accuracy (%)",
    title: str = "Pareto Frontier: Accuracy vs. Efficiency",
    output_path: str | Path | None = None,
    is_dummy: bool = False,
) -> plt.Figure:
    """Plot accuracy vs cost with labeled points and the Pareto frontier line.

    Parameters
    ----------
    data_points:
        List of dicts with cost, accuracy, and label fields.
    cost_key:
        Dict key for cost on X-axis.
    acc_key:
        Dict key for accuracy on Y-axis.
    label_key:
        Dict key for model label.
    x_label, y_label, title:
        Labels and title with explicit units.
    output_path:
        File path to save the 300-dpi PNG.
    is_dummy:
        If True, marks the figure with a 'DUMMY DATA' watermark.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    pts = [(float(d[cost_key]), float(d[acc_key])) for d in data_points]
    costs = [p[0] for p in pts]
    accs = [p[1] for p in pts]

    # Check for error bars (only draw when multiple seeds / std > 0, spec T5)
    cost_errs = [float(d.get(f"{cost_key}_std", 0.0) or 0.0) for d in data_points]
    acc_errs = [float(d.get(f"{acc_key}_std", 0.0) or 0.0) for d in data_points]
    has_cost_err = any(e > 0.0 for e in cost_errs)
    has_acc_err = any(e > 0.0 for e in acc_errs)

    xerr = cost_errs if has_cost_err else None
    yerr = acc_errs if has_acc_err else None

    # Scatter points with error bars if present
    ax.errorbar(
        costs,
        accs,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        color="#2b5c8f",
        ecolor="#7ea6c9",
        elinewidth=1.5,
        capsize=3,
        markersize=8,
        label="Models",
        zorder=4,
    )

    # Compute & plot Pareto frontier
    front = pareto_front(pts)
    if front:
        front_x = [f[0] for f in front]
        front_y = [f[1] for f in front]
        ax.plot(
            front_x,
            front_y,
            linestyle="--",
            color="#d9534f",
            linewidth=2,
            label="Pareto Frontier",
            zorder=3,
        )

    # Label every point (spec T6: one text label per data point)
    for d, (x, y) in zip(data_points, pts):
        lbl = str(d.get(label_key, ""))
        ax.text(
            x,
            y + 0.3,
            lbl,
            fontsize=9,
            weight="bold",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.8),
            zorder=5,
        )

    ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6, zorder=1)
    ax.legend(loc="lower right", frameon=True)

    if is_dummy:
        fig.text(
            0.5,
            0.5,
            "DUMMY DATA",
            fontsize=36,
            color="red",
            alpha=0.25,
            ha="center",
            va="center",
            weight="bold",
            rotation=30,
        )

    fig.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300)

    return fig


# ---------------------------------------------------------------------------
# 3. Lambda Ablation Plotting (Grouped Bars)
# ---------------------------------------------------------------------------

def plot_lambda_ablation(
    lambda_entries: Sequence[dict[str, Any]],
    output_path: str | Path | None = None,
    is_dummy: bool = False,
) -> plt.Figure:
    """Grouped bar plot comparing Top-1 Accuracy, MACs, and Latency per lambda."""
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

    lambdas = [str(e["lambda"]) for e in lambda_entries]
    accs = [float(e["acc"]) for e in lambda_entries]
    macs = [float(e["macs_m"]) for e in lambda_entries]
    lats = [float(e["lat_ms"]) for e in lambda_entries]

    x = np.arange(len(lambdas))
    width = 0.25

    # Check for error bars
    acc_err = [float(e.get("acc_std", 0.0) or 0.0) for e in lambda_entries]
    mac_err = [float(e.get("macs_std", 0.0) or 0.0) for e in lambda_entries]
    lat_err = [float(e.get("lat_std", 0.0) or 0.0) for e in lambda_entries]

    yerr_acc = acc_err if any(err > 0 for err in acc_err) else None
    yerr_mac = mac_err if any(err > 0 for err in mac_err) else None
    yerr_lat = lat_err if any(err > 0 for err in lat_err) else None

    # Primary axis: Accuracy (%)
    rects1 = ax1.bar(
        x - width,
        accs,
        width,
        yerr=yerr_acc,
        capsize=3,
        label="Top-1 Accuracy (%)",
        color="#2b5c8f",
    )
    ax1.set_xlabel("Trade-off Parameter λ", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Top-1 Accuracy (%)", fontsize=11, color="#2b5c8f", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#2b5c8f")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"λ = {lam}" for lam in lambdas])

    # Secondary axis: MACs (M) & Latency (ms)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(
        x,
        macs,
        width,
        yerr=yerr_mac,
        capsize=3,
        label="MACs (M)",
        color="#28a745",
    )
    rects3 = ax2.bar(
        x + width,
        lats,
        width,
        yerr=yerr_lat,
        capsize=3,
        label="CPU Latency (ms)",
        color="#e67e22",
    )
    ax2.set_ylabel("MACs (M) / CPU Latency (ms)", fontsize=11, color="#333333", fontweight="bold")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Lambda Trade-Off Ablation (Accuracy vs. Resource Cost)", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")

    if is_dummy:
        fig.text(
            0.5,
            0.5,
            "DUMMY DATA",
            fontsize=36,
            color="red",
            alpha=0.25,
            ha="center",
            va="center",
            weight="bold",
            rotation=30,
        )

    fig.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300)

    return fig


# ---------------------------------------------------------------------------
# 4. Skip Count vs. Lambda Plotting
# ---------------------------------------------------------------------------

def plot_skip_count_vs_lambda(
    lambda_skip_entries: Sequence[dict[str, Any]],
    output_path: str | Path | None = None,
    is_dummy: bool = False,
) -> plt.Figure:
    """Plot final derived cell skip count vs. trade-off lambda."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    lambdas = [float(e["lambda"]) for e in lambda_skip_entries]
    skips = [int(e["skip_count"]) for e in lambda_skip_entries]

    ax.plot(
        lambdas,
        skips,
        marker="s",
        markersize=8,
        linewidth=2,
        color="#8e44ad",
        label="Derived Skip Connections",
    )

    ax.set_xlabel("Trade-off Parameter λ", fontsize=11, fontweight="bold")
    ax.set_ylabel("Skip Connection Count (per cell)", fontsize=11, fontweight="bold")
    ax.set_title("Skip-Connection Frequency vs. Penalty λ", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    if is_dummy:
        fig.text(
            0.5,
            0.5,
            "DUMMY DATA",
            fontsize=36,
            color="red",
            alpha=0.25,
            ha="center",
            va="center",
            weight="bold",
            rotation=30,
        )

    fig.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300)

    return fig


# ---------------------------------------------------------------------------
# 5. Expected Cost E(alpha) Over Epochs
# ---------------------------------------------------------------------------

def plot_e_alpha_over_epochs(
    runs_data: dict[str, Sequence[float]],
    output_path: str | Path | None = None,
    is_dummy: bool = False,
) -> plt.Figure:
    """Plot expected architecture cost E(alpha) per search epoch across lambda values."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    colors = ["#c0392b", "#2980b9", "#27ae60", "#d35400", "#8e44ad"]
    for i, (lam_label, e_vals) in enumerate(runs_data.items()):
        epochs = list(range(len(e_vals)))
        color = colors[i % len(colors)]
        ax.plot(
            epochs,
            e_vals,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"λ = {lam_label}",
            color=color,
        )

    ax.set_xlabel("Search Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Expected Cell Cost E(α) [Normalized]", fontsize=11, fontweight="bold")
    ax.set_title("Evolution of Expected Cost E(α) During Search", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    if is_dummy:
        fig.text(
            0.5,
            0.5,
            "DUMMY DATA",
            fontsize=36,
            color="red",
            alpha=0.25,
            ha="center",
            va="center",
            weight="bold",
            rotation=30,
        )

    fig.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300)

    return fig


# ---------------------------------------------------------------------------
# 6. Alpha Evolution Over Epochs
# ---------------------------------------------------------------------------

def plot_alpha_evolution(
    epochs: Sequence[int],
    op_weights: dict[str, Sequence[float]],
    lam_name: str = "0.0",
    output_path: str | Path | None = None,
    is_dummy: bool = False,
) -> plt.Figure:
    """Plot softmax(alpha) per operation over epochs for a search run."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    for op_name, weights in op_weights.items():
        ax.plot(epochs, weights, label=op_name, linewidth=1.8)

    ax.set_xlabel("Search Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Softmax Op Weight", fontsize=11, fontweight="bold")
    ax.set_title(f"Architecture Parameters α Evolution (λ = {lam_name})", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    if is_dummy:
        fig.text(
            0.5,
            0.5,
            "DUMMY DATA",
            fontsize=36,
            color="red",
            alpha=0.25,
            ha="center",
            va="center",
            weight="bold",
            rotation=30,
        )

    fig.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300)

    return fig
