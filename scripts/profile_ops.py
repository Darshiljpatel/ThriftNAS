"""Profile all DARTS ops and save latency + MACs cost tables.

Usage
-----
  python scripts/profile_ops.py                       # 16x16, C=32
  python scripts/profile_ops.py --multi_res           # also 32x32 and 8x8 + Spearman
  python scripts/profile_ops.py --warmup 5 --runs 50  # fast test run

Outputs
-------
  results/cost_table.json              combined lat+MACs cost table
  results/cost_table_flops_only.json   MACs-only ablation table
  results/figures/op_costs.png         bar chart: latency vs MACs per op

NOTE: Profile with other apps closed, charger plugged in, Windows set to
      High Performance.  Do not run while a search is in progress.
      Run at least twice and compare; medians must agree within a few percent
      for ops with median >= 0.2 ms (spec 4, T7).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless; no display required
import matplotlib.pyplot as plt
import numpy as np

# Allow running as `python scripts/profile_ops.py` without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thriftnas.ops.cost_model import (
    APPROX_NOTE,
    _DEFAULT_C,
    _DEFAULT_RUNS,
    _DEFAULT_SIZE,
    _DEFAULT_WARMUP,
    build_cost_table,
    build_flops_only_table,
    make_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(data: dict, path: Path) -> None:
    """Atomically write *data* as JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, str(path))


def _print_table(entries: list[dict], title: str = "") -> None:
    """Print a cost table ranked by c(o), ascending."""
    if title:
        print("\n" + "=" * 72)
        print("  " + title)
        print("=" * 72)
    hdr = "{:<22} {:>6} {:>10} {:>8} {:>10} {:>6} {}".format(
        "op", "stride", "median_ms", "std_ms", "MACs", "c", "flag"
    )
    print(hdr)
    print("-" * 72)
    for e in sorted(entries, key=lambda x: x["c"]):
        flag = "[NOISY]" if e.get("flagged") else ""
        print("{:<22} {:>6} {:>10.4f} {:>8.4f} {:>10d} {:>6.4f}  {}".format(
            e["op"], e["stride"],
            e["lat_median_ms"], e["lat_std_ms"],
            e["macs"], e["c"], flag,
        ))


def _spearman_rho(a: list[float], b: list[float]) -> tuple[float, float]:
    """Return (rho, pvalue) via scipy.stats.spearmanr."""
    from scipy.stats import spearmanr
    result = spearmanr(a, b)
    # scipy >= 1.7 uses .statistic; older uses .correlation
    rho = getattr(result, "statistic", getattr(result, "correlation", float("nan")))
    return float(rho), float(result.pvalue)


def _bar_chart(entries: list[dict], out_path: Path) -> None:
    """Save a grouped bar chart: normalised latency vs normalised MACs per op."""
    labels   = ["{}\ns={}".format(e["op"], e["stride"]) for e in entries]
    lat_max  = max(e["lat_median_ms"] for e in entries)
    mac_max  = max(e["macs"]          for e in entries)
    lat_norm = [e["lat_median_ms"] / lat_max for e in entries]
    mac_norm = [(e["macs"] / mac_max) if mac_max > 0 else 0.0 for e in entries]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w / 2, lat_norm, w, label="Latency / lat_max", color="#4C72B0")
    ax.bar(x + w / 2, mac_norm, w, label="MACs / MACs_max",   color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Normalised cost (0-1)")
    ax.set_title(
        "Op costs -- C={}, {}x{} input  (Latency is approximate; see JSON note)".format(
            _DEFAULT_C, _DEFAULT_SIZE, _DEFAULT_SIZE
        )
    )
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("Bar chart saved -> {}".format(out_path))


# ---------------------------------------------------------------------------
# Multi-resolution Spearman analysis (--multi_res)
# ---------------------------------------------------------------------------

def _profile_multi_res(C: int, sizes: list[int], warmup: int, runs: int) -> None:
    """Profile at multiple spatial sizes and print Spearman rank correlations."""
    print("\n>>> Multi-resolution profiling (Spearman rank correlation) ...")
    size_entries: dict[int, list[dict]] = {}
    for sz in sizes:
        print("  Profiling at {}x{} ...".format(sz, sz), flush=True)
        entries, _, _ = build_cost_table(C=C, size=sz, warmup=warmup, runs=runs)
        size_entries[sz] = entries

    # One latency vector per spatial size (same (op, stride) order)
    lat_vecs: dict[int, list[float]] = {
        sz: [e["lat_median_ms"] for e in ents]
        for sz, ents in size_entries.items()
    }

    print("\nSpearman rank correlations of op latency across spatial sizes:")
    print("  {:<16} {:>8} {:>12}".format("Sizes", "rho", "p-value"))
    print("  " + "-" * 38)
    for s1, s2 in combinations(sizes, 2):
        rho, pval = _spearman_rho(lat_vecs[s1], lat_vecs[s2])
        print("  {}x{} vs {}x{}:  rho = {:+.4f},  p = {:.4e}".format(
            s1, s1, s2, s2, rho, pval))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile ThriftNAS DARTS ops and build cost tables."
    )
    parser.add_argument("--warmup",    type=int, default=_DEFAULT_WARMUP,
                        help="Warm-up runs before timing (default: {})".format(_DEFAULT_WARMUP))
    parser.add_argument("--runs",      type=int, default=_DEFAULT_RUNS,
                        help="Timed runs per op (default: {})".format(_DEFAULT_RUNS))
    parser.add_argument("--C",         type=int, default=_DEFAULT_C,
                        help="Channel count (default: {})".format(_DEFAULT_C))
    parser.add_argument("--size",      type=int, default=_DEFAULT_SIZE,
                        help="Spatial size H=W (default: {})".format(_DEFAULT_SIZE))
    parser.add_argument("--multi_res", action="store_true",
                        help="Also profile at 32x32 and 8x8 and print Spearman rho")
    args = parser.parse_args()

    print("\nThriftNAS op profiler -- C={}, {}x{}, warmup={}, runs={}".format(
        args.C, args.size, args.size, args.warmup, args.runs))
    print("NOTE: {}\n".format(APPROX_NOTE))

    # -----------------------------------------------------------------------
    # Main table: all 8 ops x stride {1, 2} at the requested spatial size
    # -----------------------------------------------------------------------
    print("Profiling 8 ops x 2 strides ...", flush=True)
    entries, lat_max, mac_max = build_cost_table(
        C=args.C, size=args.size,
        warmup=args.warmup, runs=args.runs,
    )
    _print_table(entries, title="Ranked cost table -- {}x{}".format(
        args.size, args.size))

    # Report noisy ops (std > 10 % of median)
    noisy = [e for e in entries if e.get("flagged")]
    if noisy:
        print("\n[!] {} noisy op(s) (std > 10% of median):".format(len(noisy)))
        for e in noisy:
            print("    {:<22} stride={}  median={:.4f} ms  std={:.4f} ms".format(
                e["op"], e["stride"], e["lat_median_ms"], e["lat_std_ms"]))
    else:
        print("\n[OK] No ops flagged as noisy (std <= 10% of median for all).")

    # -----------------------------------------------------------------------
    # Save results/cost_table.json
    # -----------------------------------------------------------------------
    meta = make_metadata(C=args.C, size=args.size,
                         warmup=args.warmup, runs=args.runs)
    cost_table_path = Path("results/cost_table.json")
    _save_json(
        {"metadata": meta, "lat_max": lat_max, "mac_max": mac_max,
         "entries": entries},
        cost_table_path,
    )
    print("\nSaved -> {}".format(cost_table_path))

    # -----------------------------------------------------------------------
    # Save results/cost_table_flops_only.json
    # -----------------------------------------------------------------------
    fo_entries, fo_mac_max = build_flops_only_table(C=args.C, size=args.size)
    fo_meta = make_metadata(C=args.C, size=args.size,
                            warmup=args.warmup, runs=args.runs)
    flops_only_path = Path("results/cost_table_flops_only.json")
    _save_json(
        {"metadata": fo_meta, "mac_max": fo_mac_max, "entries": fo_entries},
        flops_only_path,
    )
    print("Saved -> {}".format(flops_only_path))

    # -----------------------------------------------------------------------
    # Bar chart
    # -----------------------------------------------------------------------
    _bar_chart(entries, Path("results/figures/op_costs.png"))

    # -----------------------------------------------------------------------
    # Optional multi-resolution Spearman analysis
    # -----------------------------------------------------------------------
    if args.multi_res:
        _profile_multi_res(
            C=args.C,
            sizes=[8, 16, 32],
            warmup=args.warmup,
            runs=args.runs,
        )


if __name__ == "__main__":
    main()
