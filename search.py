"""Architecture search CLI for ThriftNAS (DARTS §3, addendum §10).

Usage
-----
  python search.py --lam 0.1 --epochs 25 --cost_table results/cost_table.json
  python search.py --lam 0   --no_amp --out_dir results/search_lam0
  python search.py --resume results/search/checkpoint.pt --lam 0.1

The test loader is NEVER accessed here.  All evaluation uses the alpha
(architecture) data split as a proxy validation set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from thriftnas.data.cifar10 import get_search_loaders
from thriftnas.models.genotypes import derive_genotype, save_genotype
from thriftnas.models.search_net import SearchNetwork
from thriftnas.ops.primitives import PRIMITIVES
from thriftnas.trainers.search_trainer import SearchTrainer
from thriftnas.utils import get_device, set_seed


# ---------------------------------------------------------------------------
# Cost table helpers
# ---------------------------------------------------------------------------

def _load_costs(cost_table_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (c1, c2) tensors in PRIMITIVES order from a cost_table.json.

    c1  – stride-1 combined cost  c(o)
    c2  – stride-2 combined cost  c(o)
    """
    with open(cost_table_path, encoding="utf-8") as fh:
        ct = json.load(fh)

    op_to_idx = {name: i for i, name in enumerate(PRIMITIVES)}
    c1 = torch.zeros(len(PRIMITIVES))
    c2 = torch.zeros(len(PRIMITIVES))

    for entry in ct["entries"]:
        idx = op_to_idx.get(entry["op"])
        if idx is None:
            continue
        if entry["stride"] == 1:
            c1[idx] = float(entry["c"])
        elif entry["stride"] == 2:
            c2[idx] = float(entry["c"])

    return c1, c2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="ThriftNAS: efficiency-aware DARTS architecture search."
    )
    parser.add_argument("--lam",         type=float, default=0.0,
                        help="Cost regularisation weight (default: 0)")
    parser.add_argument("--epochs",      type=int,   default=25,
                        help="Number of search epochs (default: 25)")
    parser.add_argument("--batch_size",  type=int,   default=48,
                        help="Mini-batch size per data split (default: 48)")
    parser.add_argument("--cost_table",  type=str,
                        default="results/cost_table.json",
                        help="Path to cost_table.json (default: results/cost_table.json)")
    parser.add_argument("--seed",        type=int,   default=0,
                        help="Global random seed (default: 0)")
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--num_workers", type=int,   default=2,
                        help="DataLoader worker processes (default: 2)")
    parser.add_argument("--out_dir",     type=str,
                        default="results/search",
                        help="Output directory (default: results/search)")
    parser.add_argument("--no_amp",      action="store_true",
                        help="Disable automatic mixed precision")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device()
    amp    = (not args.no_amp) and (device.type == "cuda")

    print("ThriftNAS search -- device={}, amp={}, lam={}, epochs={}".format(
        device, amp, args.lam, args.epochs))

    # ---- Cost tensors -------------------------------------------------------
    c1, c2 = _load_costs(args.cost_table)
    c1 = c1.to(device)
    c2 = c2.to(device)
    print("Cost table loaded from {}".format(args.cost_table))

    # ---- Model --------------------------------------------------------------
    model = SearchNetwork(C=16, num_classes=10, layers=5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("SearchNetwork: {:,} total parameters ({:,} arch)".format(
        n_params, sum(p.numel() for p in model.arch_parameters())))

    # ---- Data loaders -------------------------------------------------------
    w_loader, a_loader = get_search_loaders(
        batch_size  = args.batch_size,
        seed        = args.seed,
        num_workers = args.num_workers,
    )
    print("Data: weight_loader={} batches, alpha_loader={} batches".format(
        len(w_loader), len(a_loader)))

    # ---- Trainer ------------------------------------------------------------
    trainer = SearchTrainer(
        model   = model,
        c1      = c1,
        c2      = c2,
        lam     = args.lam,
        epochs  = args.epochs,
        amp     = amp,
        out_dir = args.out_dir,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print("Resumed from {} (next epoch: {})".format(
            args.resume, trainer.start_epoch))

    # ---- Training loop ------------------------------------------------------
    for epoch in range(trainer.start_epoch, args.epochs):
        row = trainer.train_epoch(w_loader, a_loader, epoch)
        print(
            "Ep {epoch:3d} | lr={lr:.6f} | ce_w={ce_w:.4f} ce_a={ce_a:.4f} "
            "| eff={eff:.4f} lam_eff={lam_eff:.4f} "
            "| acc_tr={train_acc:.1f}% acc_val={val_acc:.1f}% "
            "| skip={skip_count} | {epoch_time_s:.1f}s | {peak_gpu_mb:.0f}MB".format(
                **row)
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Derive and save final genotype -------------------------------------
    geno = derive_genotype(model.alpha_normal, model.alpha_reduce)
    geno_path = Path(args.out_dir) / "genotype.json"
    save_genotype(geno, geno_path)
    print("Genotype saved to {}".format(geno_path))
    print("Final Derived Genotype:\n{}".format(geno))
    print("Search complete. Results in: {}".format(args.out_dir))


if __name__ == "__main__":
    main()
