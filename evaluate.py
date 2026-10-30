"""Evaluate and aggregate experiment results for ThriftNAS (PROJECT_SPEC §7, Step 10).

Discovers all `final_metrics.json` files in the results directory, computes any
missing metrics (size, CPU latency, predicted latency), aggregates across seeds,
and writes:
- `results/results.csv`
- `results/results.md`

Required table columns (exact):
Model | Top-1 (%) | Params (M) | MACs (M) | Size (MB) | CPU latency (ms, mean +/- std) | Predicted latency (ms) | Prediction error (%) | Search GPU-hours
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from thriftnas.models.baselines import build_baseline
from thriftnas.models.genotypes import load_genotype
from thriftnas.models.retrain_net import build_network
from thriftnas.utils.metrics import (
    compare_latency,
    count_macs,
    count_params,
    cpu_latency,
    model_size_mb,
    predicted_latency,
)

# ---------------------------------------------------------------------------
# Exact column specification (PROJECT_SPEC §7, Step 10)
# ---------------------------------------------------------------------------

COLUMNS: list[str] = [
    "Model",
    "Top-1 (%)",
    "Params (M)",
    "MACs (M)",
    "Size (MB)",
    "CPU latency (ms, mean +/- std)",
    "Predicted latency (ms)",
    "Prediction error (%)",
    "Search GPU-hours",
]


# ---------------------------------------------------------------------------
# Model discovery & metric enrichment
# ---------------------------------------------------------------------------

def _find_final_metrics(results_dir: Path) -> list[Path]:
    """Find all final_metrics.json files in results_dir."""
    return sorted(list(results_dir.rglob("final_metrics.json")))


def _normalize_model_name(name: str) -> str:
    """Normalize model naming for reports."""
    name_lower = name.lower().replace("-", "_")
    if "resnet20" in name_lower:
        return "ResNet-20"
    if "mobilenet" in name_lower:
        return "MobileNetV2"
    if "retrain_t10" in name_lower:
        return "ThriftNAS (lam=0, 1ep)"
    if "lam0.1" in name_lower:
        return "ThriftNAS (lam=0.1)"
    if "lam1" in name_lower:
        return "ThriftNAS (lam=1.0)"
    if "lam0" in name_lower:
        return "ThriftNAS (lam=0.0)"
    return name


def _load_and_enrich_metrics(
    json_path: Path,
    cost_table: Any | None = None,
    measure_cpu: bool = True,
) -> dict[str, Any]:
    """Load final_metrics.json and compute any missing metrics from disk."""
    with open(json_path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    folder = json_path.parent
    model_name = data.get("model") or folder.name
    model_display = _normalize_model_name(model_name)

    weights_file = folder / "model_weights.pt"
    genotype_file = folder / "genotype.json"
    if not genotype_file.exists():
        # Check parent search folder
        if (folder.parent / "genotype.json").exists():
            genotype_file = folder.parent / "genotype.json"
        elif Path("results/search_lam0/genotype.json").exists() and "thriftnas" in model_display.lower():
            genotype_file = Path("results/search_lam0/genotype.json")

    # 1. Size (MB)
    size_mb = data.get("size_mb")
    if size_mb is None and weights_file.exists():
        size_mb = model_size_mb(weights_file)
    data["size_mb"] = size_mb

    # 2. Model reconstruction for params/MACs/latency if needed
    model_obj = None
    if "resnet" in model_name.lower():
        model_obj = build_baseline("resnet20")
    elif "mobilenet" in model_name.lower():
        model_obj = build_baseline("mobilenet_v2")
    elif genotype_file.exists():
        try:
            g = load_genotype(genotype_file)
            model_obj = build_network(
                C=data.get("C", 36),
                num_classes=10,
                layers=data.get("layers", 8),
                genotype=g,
                is_thriftnas=False,
            )
            data["genotype"] = g
        except Exception:
            pass

    # 3. Parameters (M)
    params_m = data.get("params_m")
    if params_m is None and model_obj is not None:
        params_m = count_params(model_obj)
    data["params_m"] = params_m

    # 4. MACs (M)
    macs_m = data.get("macs_m")
    if macs_m is None and model_obj is not None:
        macs_m = count_macs(model_obj)
    data["macs_m"] = macs_m

    # 5. CPU Latency (ms)
    cpu_mean = data.get("cpu_lat_mean")
    cpu_std = data.get("cpu_lat_std")
    if cpu_mean is None and model_obj is not None and measure_cpu:
        try:
            # Measure with single thread, 20 warmup, 50 runs for speed during eval
            cpu_mean, cpu_std = cpu_latency(model_obj, warmup=10, runs=50)
        except Exception:
            pass
    data["cpu_lat_mean"] = cpu_mean
    data["cpu_lat_std"] = cpu_std

    # 6. Predicted latency & prediction error
    pred_lat = data.get("predicted_latency")
    if pred_lat is None and "genotype" in data and cost_table is not None:
        pred_lat = predicted_latency(data["genotype"], cost_table, layers=data.get("layers", 8))
    data["predicted_latency"] = pred_lat

    pred_error = data.get("prediction_error")
    if pred_error is None and pred_lat is not None and cpu_mean is not None:
        pred_error = compare_latency(pred_lat, cpu_mean).error_pct
    data["prediction_error"] = pred_error

    # 7. Search GPU-hours
    gpu_hours = data.get("search_gpu_hours")
    if gpu_hours is None:
        # Check if search_log.csv has epoch_time_s
        search_log = folder / "search_log.csv"
        if not search_log.exists() and (folder.parent / "search_log.csv").exists():
            search_log = folder.parent / "search_log.csv"
        if search_log.exists():
            try:
                with open(search_log, encoding="utf-8") as sfh:
                    reader = csv.DictReader(sfh)
                    total_sec = sum(float(r["epoch_time_s"]) for r in reader if "epoch_time_s" in r)
                    gpu_hours = total_sec / 3600.0
            except Exception:
                gpu_hours = None
    data["search_gpu_hours"] = gpu_hours

    data["model_display"] = model_display
    return data


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------

def aggregate_runs(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group runs by model name, compute mean +/- std across seeds, warn on <3 seeds."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        m = r["model_display"]
        grouped.setdefault(m, []).append(r)

    rows: list[dict[str, Any]] = []

    for model_name, run_list in grouped.items():
        n_seeds = len(run_list)
        if n_seeds < 3:
            warnings.warn(
                f"Model '{model_name}' has fewer than 3 seeds ({n_seeds} seed(s)).",
                UserWarning,
            )

        # Accuracy
        accs = [r["test_acc"] for r in run_list if r.get("test_acc") is not None]
        if not accs:
            acc_str = "-"
        elif n_seeds == 1:
            acc_str = f"{accs[0]:.2f} (1 seed)"
        else:
            acc_str = f"{np.mean(accs):.2f} +/- {np.std(accs):.2f}"

        # Params (M)
        p_list = [r["params_m"] for r in run_list if r.get("params_m") is not None]
        p_str = f"{np.mean(p_list):.4f}" if p_list else "-"

        # MACs (M)
        m_list = [r["macs_m"] for r in run_list if r.get("macs_m") is not None]
        m_str = f"{np.mean(m_list):.2f}" if m_list else "-"

        # Size (MB)
        s_list = [r["size_mb"] for r in run_list if r.get("size_mb") is not None]
        s_str = f"{np.mean(s_list):.2f}" if s_list else "-"

        # CPU latency (ms, mean +/- std)
        lat_means = [r["cpu_lat_mean"] for r in run_list if r.get("cpu_lat_mean") is not None]
        lat_stds = [r["cpu_lat_std"] for r in run_list if r.get("cpu_lat_std") is not None]
        if not lat_means:
            cpu_str = "-"
        elif n_seeds == 1:
            std_val = lat_stds[0] if lat_stds else 0.0
            cpu_str = f"{lat_means[0]:.2f} +/- {std_val:.2f} (1 seed)"
        else:
            cpu_str = f"{np.mean(lat_means):.2f} +/- {np.std(lat_means):.2f}"

        # Predicted latency (ms)
        pred_lats = [r["predicted_latency"] for r in run_list if r.get("predicted_latency") is not None]
        pred_str = f"{np.mean(pred_lats):.2f}" if pred_lats else "-"

        # Prediction error (%)
        err_list = [r["prediction_error"] for r in run_list if r.get("prediction_error") is not None]
        err_str = f"{np.mean(err_list):.1f}" if err_list else "-"

        # Search GPU-hours
        gpu_list = [r["search_gpu_hours"] for r in run_list if r.get("search_gpu_hours") is not None]
        gpu_str = f"{np.mean(gpu_list):.3f}" if gpu_list else "-"

        row = {
            "Model": model_name,
            "Top-1 (%)": acc_str,
            "Params (M)": p_str,
            "MACs (M)": m_str,
            "Size (MB)": s_str,
            "CPU latency (ms, mean +/- std)": cpu_str,
            "Predicted latency (ms)": pred_str,
            "Prediction error (%)": err_str,
            "Search GPU-hours": gpu_str,
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Output formatting (CSV and Markdown)
# ---------------------------------------------------------------------------

def generate_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Format rows as a GitHub-flavored Markdown table with exact column names."""
    header = "| " + " | ".join(COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(COLUMNS)) + " |"
    body_lines = []
    for r in rows:
        line = "| " + " | ".join(str(r.get(c, "-")) for c in COLUMNS) + " |"
        body_lines.append(line)
    return "\n".join([header, separator] + body_lines)


def write_results_files(
    rows: list[dict[str, Any]],
    out_dir: Path = Path("results"),
) -> tuple[Path, Path]:
    """Write results/results.csv and results/results.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    md_path = out_dir / "results.md"

    # 1. Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # 2. Write Markdown
    md_table = generate_markdown_table(rows)
    md_content = (
        "# ThriftNAS: Evaluation & Comparison Results\n\n"
        "> Evaluation metrics comparing ThriftNAS models against baselines (PROJECT_SPEC §7).\n\n"
        f"{md_table}\n"
    )
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_content)

    return csv_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="ThriftNAS: aggregate final metrics into results.csv and results.md."
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Root directory to search for final_metrics.json (default: results)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="results",
        help="Directory to save results.csv and results.md (default: results)"
    )
    parser.add_argument(
        "--cost_table", type=str, default="results/cost_table.json",
        help="Path to cost_table.json (default: results/cost_table.json)"
    )
    parser.add_argument(
        "--no_cpu_eval", action="store_true",
        help="Skip live CPU latency measurement for unbenchmarked models"
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    cost_table_path = Path(args.cost_table)
    cost_table = None
    if cost_table_path.exists():
        with open(cost_table_path, encoding="utf-8") as fh:
            cost_table = json.load(fh)

    metric_files = _find_final_metrics(results_dir)
    print(f"Found {len(metric_files)} final_metrics.json file(s) in {results_dir}")

    raw_runs = []
    for mf in metric_files:
        run_data = _load_and_enrich_metrics(
            mf,
            cost_table=cost_table,
            measure_cpu=not args.no_cpu_eval,
        )
        raw_runs.append(run_data)

    aggregated_rows = aggregate_runs(raw_runs)
    csv_p, md_p = write_results_files(aggregated_rows, out_dir=Path(args.out_dir))

    print(f"\nSaved CSV: {csv_p}")
    print(f"Saved Markdown: {md_p}\n")

    # Display results.md
    with open(md_p, encoding="utf-8") as fh:
        print(fh.read())


if __name__ == "__main__":
    main()
