"""Automated runner for Darshil's track (Track 1: ThriftNAS lambda=0.1 and lambda=0.01).

Runs sequentially:
1. Search lambda=0.1 (25 epochs) -> results/search_lam0.1
2. Retrain lambda=0.1 genotype (50 epochs) -> results/retrain_lam0.1
3. Search lambda=0.01 (25 epochs) -> results/search_lam0.01
4. Retrain lambda=0.01 genotype (50 epochs) -> results/retrain_lam0.01

Includes automatic checkpoint detection to resume if interrupted.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path


SEARCH_RUNS = [
    {"lam": 0.1, "epochs": 25, "out_dir": Path("results/search_lam0.1")},
    {"lam": 0.01, "epochs": 25, "out_dir": Path("results/search_lam0.01")},
]


def run_command(cmd: list[str]) -> None:
    print(f"\n[EXEC] {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[DONE] Elapsed time: {elapsed / 60:.2f} minutes")


def main() -> None:
    print("=" * 70)
    print("THRIFTNAS - TRACK 1 (DARSHIL)")
    print("Sequential Search & Retrain for lambda=0.1 and lambda=0.01")
    print("=" * 70)

    for run_cfg in SEARCH_RUNS:
        lam = run_cfg["lam"]
        epochs = run_cfg["epochs"]
        s_out = run_cfg["out_dir"]
        s_out.mkdir(parents=True, exist_ok=True)

        geno_path = s_out / "genotype.json"
        s_ckpt = s_out / "checkpoint.pt"
        s_csv = s_out / "search_log.csv"

        # -------------------------------------------------------------
        # Step A: Architecture Search
        # -------------------------------------------------------------
        s_done = False
        if geno_path.exists() and s_csv.exists():
            with open(s_csv, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if len(rows) >= epochs:
                print(f"\n[SKIP] Search for lambda={lam} already completed ({len(rows)}/{epochs} epochs in {s_csv}).")
                s_done = True

        if not s_done:
            print(f"\n>>> [1/2] Starting Architecture Search: lambda={lam} ({epochs} epochs)...")
            cmd = [
                sys.executable,
                "-u",
                "search.py",
                "--lam", str(lam),
                "--epochs", str(epochs),
                "--batch_size", "48",
                "--out_dir", str(s_out),
            ]
            if s_ckpt.exists():
                print(f"[RESUME] Found existing checkpoint at {s_ckpt}. Resuming search...")
                cmd.extend(["--resume", str(s_ckpt)])

            run_command(cmd)

        # -------------------------------------------------------------
        # Step B: Retraining Discovered Architecture
        # -------------------------------------------------------------
        r_out = Path(f"results/retrain_lam{lam}")
        r_out.mkdir(parents=True, exist_ok=True)
        r_metrics = r_out / "final_metrics.json"
        r_ckpt = r_out / "checkpoint.pt"

        if r_metrics.exists():
            print(f"\n[SKIP] Retrain for lambda={lam} already completed ({r_metrics} found).")
            continue

        print(f"\n>>> [2/2] Starting Retraining for lambda={lam} (50 epochs on full CIFAR-10)...")
        r_cmd = [
            sys.executable,
            "-u",
            "retrain.py",
            "--genotype", str(geno_path),
            "--out_dir", str(r_out),
        ]
        if r_ckpt.exists():
            print(f"[RESUME] Found existing retrain checkpoint at {r_ckpt}. Resuming retrain...")
            r_cmd.extend(["--resume", str(r_ckpt)])

        run_command(r_cmd)

    print("\n" + "=" * 70)
    print("ALL SEARCH & RETRAIN RUNS FOR TRACK 1 COMPLETED SUCCESSFULLY!")
    print("Now you can commit and push your branch to GitHub.")
    print("=" * 70)


if __name__ == "__main__":
    main()

