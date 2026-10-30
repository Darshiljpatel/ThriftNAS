"""Step 13: Analysis of 5-epoch pilot search across lambdas 0, 0.1, 1.0."""

from __future__ import annotations

import csv
import json
from pathlib import Path


LAMBDAS = [0.0, 0.1, 1.0]
HEAVY_OPS = {"sep_conv_5x5", "dil_conv_3x3"}
CHEAP_OPS = {"max_pool_3x3", "avg_pool_3x3", "skip_connect", "zero"}


def analyze_pilot() -> None:
    print("=" * 80)
    print("STEP 13: 5-EPOCH PILOT SEARCH VERIFICATION & ANALYSIS")
    print("=" * 80)

    total_epoch_time = 0.0
    total_epoch_count = 0

    for lam in LAMBDAS:
        out_dir = Path(f"results/search_pilot_lam{lam}")
        csv_path = out_dir / "search_log.csv"
        geno_path = out_dir / "genotype.json"

        print(f"\n" + "-" * 80)
        print(f"PILOT RUN: Lambda = {lam}")
        print("-" * 80)

        if not csv_path.exists():
            print(f"Log not found at {csv_path}. (Has this run completed?)")
            continue

        with open(csv_path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        header = f"{'Ep':>3} | {'ce_w':>7} | {'ce_a':>7} | {'eff':>7} | {'lam_eff':>8} | {'train%':>6} | {'val%':>6} | {'skip':>4} | {'time(s)':>7} | {'gpu(MB)':>7}"
        print(header)
        print("-" * len(header))

        for r in rows:
            ep = int(r["epoch"])
            ce_w = float(r["ce_w"])
            ce_a = float(r["ce_a"])
            eff = float(r["eff"])
            lam_eff = float(r["lam_eff"])
            tr_acc = float(r["train_acc"])
            val_acc = float(r["val_acc"])
            skip = int(r["skip_count"])
            t_s = float(r["epoch_time_s"])
            gpu = float(r["peak_gpu_mb"])

            total_epoch_time += t_s
            total_epoch_count += 1

            print(f"{ep:3d} | {ce_w:7.4f} | {ce_a:7.4f} | {eff:7.4f} | {lam_eff:8.4f} | {tr_acc:6.1f} | {val_acc:6.1f} | {skip:4d} | {t_s:7.1f} | {gpu:7.0f}")

        # T3: r ratio at last pilot epoch
        if rows and lam > 0:
            last = rows[-1]
            ce_a_last = float(last["ce_a"])
            lam_eff_last = float(last["lam_eff"])
            r = lam_eff_last / ce_a_last if ce_a_last > 0 else 0.0

            if r < 0.05:
                verdict = "WEAK (< 0.05: cost penalty has little effect)"
            elif r <= 1.0:
                verdict = "USEFUL (0.05 - 1.0: balanced search trade-off)"
            elif r <= 2.0:
                verdict = "STRONG (1.0 - 2.0: cost penalty active)"
            else:
                verdict = "COST DOMINATES (> 2.0: cost penalty overpowers accuracy)"

            print(f"\n[T3 Addendum Rule Check] r = lam_eff / ce_a = {lam_eff_last:.4f} / {ce_a_last:.4f} = {r:.4f} -> {verdict}")

        # Genotype & op counts
        if geno_path.exists():
            with open(geno_path, encoding="utf-8") as fh:
                g_data = json.load(fh)
            all_ops = [op for op, _ in g_data.get("normal", []) + g_data.get("reduce", [])]
            heavy = sum(1 for op in all_ops if op in HEAVY_OPS)
            cheap = sum(1 for op in all_ops if op in CHEAP_OPS)
            print(f"\n[T4 Genotype Op Counts] Total ops: {len(all_ops)} | Heavy (sep5x5, dil3x3): {heavy} | Cheap (pool, skip, zero): {cheap}")
            print(f"Genotype:\n{json.dumps(g_data, indent=2)}")

    # T5: Average epoch time & Projected GPU-hours
    if total_epoch_count > 0:
        avg_t = total_epoch_time / total_epoch_count
        proj_hours = (25 * 3 * avg_t) / 3600.0
        print("\n" + "=" * 80)
        print(f"[T5 Runtime Projections]")
        print(f"Average Epoch Time: {avg_t:.2f} s ({avg_t / 60:.2f} min)")
        print(f"Projected Total GPU-Hours for Full 25 epochs x 3 lambdas: {proj_hours:.2f} GPU-hours")
        print("=" * 80)


if __name__ == "__main__":
    analyze_pilot()
