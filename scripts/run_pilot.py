"""Step 13: 5-epoch pilot search across lambdas 0, 0.1, 1.0 on CIFAR-10."""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path


LAMBDAS = [0.0, 0.1, 1.0]
EPOCHS = 5
BATCH_SIZE = 48


def run_pilot_search() -> None:
    print("=" * 70)
    print("ThriftNAS Step 13: 5-Epoch Pilot Search (Lambda = 0, 0.1, 1.0)")
    print(f"Batch size: {BATCH_SIZE} | Epochs per run: {EPOCHS}")
    print("=" * 70)

    for lam in LAMBDAS:
        out_dir = Path(f"results/search_pilot_lam{lam}")
        out_dir.mkdir(parents=True, exist_ok=True)

        geno_path = out_dir / "genotype.json"
        ckpt_path = out_dir / "checkpoint.pt"
        csv_path = out_dir / "search_log.csv"

        # Check if already completed
        if geno_path.exists() and csv_path.exists():
            with open(csv_path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if len(rows) >= EPOCHS:
                print(f"\n[SKIP] Run for lambda={lam} already complete ({len(rows)} epochs found in {csv_path}).")
                continue

        cmd = [
            sys.executable,
            "-u",
            "search.py",
            "--lam", str(lam),
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--out_dir", str(out_dir),
        ]

        if ckpt_path.exists():
            print(f"\n[RESUME] Found existing checkpoint for lambda={lam} at {ckpt_path}. Resuming...")
            cmd.extend(["--resume", str(ckpt_path)])
        else:
            print(f"\n>>> Running pilot search for lambda = {lam}...")

        print(f"Command: {' '.join(cmd)}")
        t0 = time.time()
        res = subprocess.run(cmd)
        elapsed = time.time() - t0

        if res.returncode != 0:
            print(f"Error: search for lambda={lam} exited with code {res.returncode}")
            sys.exit(res.returncode)

    print("\n" + "=" * 70)
    print("ALL 3 PILOT SEARCHES COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_pilot_search()
