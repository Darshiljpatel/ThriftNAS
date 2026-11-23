"""Stage-aware latency predictor (SALP) for ThriftNAS.

Problem
-------
The original lookup table (`results/cost_table.json`) profiles every op once, at a single
geometry (C=32, 16x16), and predicts network latency as the plain sum of the 16 cell ops.
That ignores two things:

1. **Geometry.** The retrain network runs ops at three different stages
   (C=36 @ 32x32, C=72 @ 16x16, C=144 @ 8x8). An op's cost changes a lot between stages.
2. **The skeleton.** The stem, the two 1x1 preprocessing convs in every cell, the
   FactorizedReduce after each reduction cell, node additions/concat, global pooling and
   the classifier are never counted. Once the cells get cheap (high lambda), the skeleton
   dominates, which is why the old error grows from ~70 % to ~95 %.

Method
------
- **Stage table:** profile every primitive at every (C, H, stride) geometry that actually
  occurs in the 8-cell retrain network (5 geometries x 8 primitives = 40 entries).
- **Skeleton cost:** profile the genotype-independent parts once (stem, pre0/pre1 for each
  of the 8 cells, node add+concat glue, global pool + classifier).
- **Prediction for any genotype** = skeleton + sum over all 8 cells of the stage-table
  lookups for that cell's ops. No per-genotype profiling and no training needed, so it can
  rank architectures *before* they are retrained.

Validation
----------
Every genotype found on disk (full searches + pilots, including the degenerate lambda=1.0
cell) is built with random weights and timed end-to-end. Latency does not depend on the
weight values, so this is a fair check. The predictor never sees these measurements, so
all architectures are out-of-sample.

Usage
-----
    python scripts/stage_aware_latency.py            # writes results/latency_predictor/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thriftnas.models.genotypes import load_genotype  # noqa: E402
from thriftnas.models.retrain_net import NetworkCIFAR  # noqa: E402
from thriftnas.ops.primitives import OPS, PRIMITIVES  # noqa: E402
from thriftnas.utils.metrics import predicted_latency  # noqa: E402

C_INIT = 36
LAYERS = 8
IMG = 32


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_fn(fn, warmup: int, runs: int) -> float:
    """Median wall-clock time of fn() in ms (single thread, inference mode)."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        t = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            t.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(t))


def time_module(m: nn.Module, shape: tuple[int, ...], warmup: int, runs: int) -> float:
    m.eval()
    x = torch.randn(*shape)
    return time_fn(lambda: m(x), warmup, runs)


# ---------------------------------------------------------------------------
# Network geometry (mirrors NetworkCIFAR exactly)
# ---------------------------------------------------------------------------

def cell_geometry() -> list[dict]:
    """Per-cell shapes for the C=36, 8-layer retrain network."""
    red_pos = {LAYERS // 3, 2 * LAYERS // 3}
    C_stem = 3 * C_INIT
    C_pp, C_p, C = C_stem, C_stem, C_INIT
    H_pp, H_p = IMG, IMG
    reduction_prev = False
    cells = []
    for i in range(LAYERS):
        reduction = i in red_pos
        if reduction:
            C *= 2
        H_in = H_p                       # spatial size after preprocessing
        H_out = H_in // 2 if reduction else H_in
        cells.append(dict(i=i, reduction=reduction, reduction_prev=reduction_prev,
                          C=C, C_pp=C_pp, C_p=C_p, H_pp=H_pp, H_p=H_p,
                          H_in=H_in, H_out=H_out))
        C_pp, C_p = C_p, 4 * C
        H_pp, H_p = H_p, H_out
        reduction_prev = reduction
    return cells


def op_geometry(cell: dict, input_idx: int) -> tuple[int, int, int]:
    """(C, H_in, stride) for one edge of a cell."""
    if cell["reduction"] and input_idx < 2:
        return cell["C"], cell["H_in"], 2
    return cell["C"], cell["H_out"], 1


# ---------------------------------------------------------------------------
# Build the predictor
# ---------------------------------------------------------------------------

def build_predictor(warmup: int, runs: int) -> dict:
    cells = cell_geometry()

    # 1) Stage table: every primitive at every geometry that occurs.
    geoms = sorted({op_geometry(c, k) for c in cells for k in (0, 2)})
    table = {}
    for C, H, s in geoms:
        for name in PRIMITIVES:
            op = OPS[name](C, s, True)
            table[f"{name}|C{C}|H{H}|s{s}"] = time_module(op, (1, C, H, H), warmup, runs)
        print(f"  stage table: C={C:<3} H={H:<2} stride={s} done")

    # 2) Skeleton: genotype-independent parts, profiled with the real modules.
    dummy = load_genotype(ROOT / "results/search_lam0.0/genotype.json")
    net = NetworkCIFAR(C=C_INIT, num_classes=10, layers=LAYERS, genotype=dummy).eval()
    skel = {"stem": time_module(net.stem, (1, 3, IMG, IMG), warmup, runs)}
    pre = 0.0
    glue = 0.0
    for c, cell in zip(cells, net.cells):
        pre += time_module(cell.pre0, (1, c["C_pp"], c["H_pp"], c["H_pp"]), warmup, runs)
        pre += time_module(cell.pre1, (1, c["C_p"], c["H_p"], c["H_p"]), warmup, runs)
        a = [torch.randn(1, c["C"], c["H_out"], c["H_out"]) for _ in range(8)]
        glue += time_fn(lambda: torch.cat([a[0] + a[1], a[2] + a[3],
                                           a[4] + a[5], a[6] + a[7]], 1), warmup, runs)
    skel["preprocess"] = pre
    skel["add_concat"] = glue
    C_last = 4 * cells[-1]["C"]
    head = nn.Sequential(net.global_pool, nn.Flatten(), net.classifier)
    skel["head"] = time_module(head, (1, C_last, cells[-1]["H_out"], cells[-1]["H_out"]),
                               warmup, runs)
    skel["total"] = sum(skel.values())
    print(f"  skeleton: {skel['total']:.2f} ms")

    return {"table": table, "skeleton": skel, "cells": cells,
            "config": {"C": C_INIT, "layers": LAYERS, "threads": 1,
                       "warmup": warmup, "runs": runs}}


def predict(pred: dict, genotype) -> dict:
    """Predicted latency of a genotype, split into skeleton and cell-op parts."""
    ops_ms = 0.0
    for c in pred["cells"]:
        gene = genotype.reduce if c["reduction"] else genotype.normal
        for name, idx in gene:
            C, H, s = op_geometry(c, idx)
            ops_ms += pred["table"][f"{name}|C{C}|H{H}|s{s}"]
    sk = pred["skeleton"]["total"]
    return {"skeleton_ms": sk, "ops_ms": ops_ms, "total_ms": sk + ops_ms}


def measure_end_to_end(genotype, warmup: int, runs: int) -> float:
    net = NetworkCIFAR(C=C_INIT, num_classes=10, layers=LAYERS, genotype=genotype).eval()
    return time_module(net, (1, 3, IMG, IMG), warmup, runs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def spearman(a: list[float], b: list[float]) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--out_dir", default="results/latency_predictor")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(0)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Building stage-aware predictor ...")
    pred = build_predictor(args.warmup, args.runs)
    with open(out / "stage_table.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in pred.items() if k != "cells"}, fh, indent=2)

    with open(ROOT / "results/cost_table.json", encoding="utf-8") as fh:
        old_table = json.load(fh)

    genos = sorted(p.parent.name for p in (ROOT / "results").glob("search_*/genotype.json"))
    # search_lam0 is a 1-epoch smoke run kept for history; skip it.
    genos = [g for g in genos if g != "search_lam0"]

    rows = []
    print("Validating on every genotype on disk ...")
    for g in genos:
        geno = load_genotype(ROOT / "results" / g / "genotype.json")
        measured = measure_end_to_end(geno, args.warmup, args.runs)
        old = predicted_latency(geno, old_table, layers=LAYERS)
        new = predict(pred, geno)
        row = {
            "architecture": g.replace("search_", ""),
            "measured_ms": round(measured, 3),
            "old_pred_ms": round(old, 3),
            "old_error_pct": round(abs(old - measured) / measured * 100, 1),
            "new_pred_ms": round(new["total_ms"], 3),
            "new_error_pct": round(abs(new["total_ms"] - measured) / measured * 100, 1),
            "skeleton_share_pct": round(new["skeleton_ms"] / new["total_ms"] * 100, 1),
        }
        rows.append(row)
        print(f"  {row['architecture']:<24} measured {measured:7.2f} ms | "
              f"old {old:6.2f} ({row['old_error_pct']:5.1f}%) | "
              f"new {new['total_ms']:6.2f} ({row['new_error_pct']:5.1f}%)")

    # Calibration: isolated-module timing misses per-call framework overhead, so the raw sum
    # is biased low by a roughly constant factor. Fit one scalar k = mean(measured / raw) and
    # evaluate it leave-one-out, so no architecture is scored with a k that saw it.
    ratios = [r["measured_ms"] / r["new_pred_ms"] for r in rows]
    for i, r in enumerate(rows):
        k = float(np.mean(ratios[:i] + ratios[i + 1:]))
        r["calibrated_pred_ms"] = round(r["new_pred_ms"] * k, 3)
        r["calibrated_error_pct"] = round(
            abs(r["calibrated_pred_ms"] - r["measured_ms"]) / r["measured_ms"] * 100, 1)
    k_all = float(np.mean(ratios))
    pred["calibration_k"] = k_all
    with open(out / "stage_table.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in pred.items() if k != "cells"}, fh, indent=2)

    meas = [r["measured_ms"] for r in rows]
    summary = {
        "n_architectures": len(rows),
        "old_mean_error_pct": round(float(np.mean([r["old_error_pct"] for r in rows])), 1),
        "new_mean_error_pct": round(float(np.mean([r["new_error_pct"] for r in rows])), 1),
        "calibrated_mean_error_pct_loo": round(
            float(np.mean([r["calibrated_error_pct"] for r in rows])), 1),
        "calibrated_max_error_pct_loo": round(
            float(np.max([r["calibrated_error_pct"] for r in rows])), 1),
        "calibration_k": round(k_all, 3),
        "old_spearman": round(spearman([r["old_pred_ms"] for r in rows], meas), 3),
        "new_spearman": round(spearman([r["new_pred_ms"] for r in rows], meas), 3),
        "skeleton_ms": round(pred["skeleton"]["total"], 3),
    }
    for r in rows:
        print(f"  {r['architecture']:<24} calibrated (LOO) {r['calibrated_pred_ms']:7.2f} ms "
              f"({r['calibrated_error_pct']:4.1f}%)")

    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # Figure: predicted vs measured, old vs new.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    lim = max(meas + [r["calibrated_pred_ms"] for r in rows]) * 1.1
    ax.plot([0, lim], [0, lim], color="#888", lw=1, ls="--", label="perfect prediction")
    ax.scatter(meas, [r["old_pred_ms"] for r in rows], s=46, color="#c0504d",
               label=f"Original lookup table (mean error {summary['old_mean_error_pct']}%)")
    ax.scatter(meas, [r["new_pred_ms"] for r in rows], s=46, color="#9dc3e6",
               label=f"Stage-aware, raw (mean error {summary['new_mean_error_pct']}%)")
    ax.scatter(meas, [r["calibrated_pred_ms"] for r in rows], s=46, color="#1f4e79",
               label=f"Stage-aware, calibrated LOO "
                     f"(mean error {summary['calibrated_mean_error_pct_loo']}%)")
    for r in rows:
        ax.annotate(r["architecture"], (r["measured_ms"], r["calibrated_pred_ms"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points", color="#444")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Measured end-to-end CPU latency (ms)")
    ax.set_ylabel("Predicted latency (ms)")
    ax.set_title(f"Latency prediction on {len(rows)} unseen architectures")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/latency_predictor.png", dpi=160)

    print("\nSummary:", json.dumps(summary, indent=2))
    print(f"Wrote {out} and results/figures/latency_predictor.png")


if __name__ == "__main__":
    main()
