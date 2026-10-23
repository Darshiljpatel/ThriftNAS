"""Retrain CLI for ThriftNAS (PROJECT_SPEC §5, §7, §8, addendum §10).

Usage
-----
  python retrain.py --genotype results/search_lam0/genotype.json
  python retrain.py --genotype results/search_lam0/genotype.json --resume results/retrain/checkpoint.pt
  python retrain.py --genotype results/search_lam0/genotype.json --no_amp --batch_size 64

Epochs and default batch size are read from configs/shared.yaml.  Do NOT
hard-code them here -- this script deliberately avoids any literal epoch
count so that T9 (grep for hard-coded epochs) passes.

The test set is evaluated EXACTLY ONCE after the final epoch (spec §8).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

from thriftnas.data.cifar10 import get_retrain_loader, get_test_loader
from thriftnas.models.genotypes import load_genotype
from thriftnas.models.retrain_net import build_network
from thriftnas.trainers.retrain_trainer import RetrainTrainer
from thriftnas.utils import get_device, set_seed

# ---------------------------------------------------------------------------
# shared.yaml loader  (single source of truth for epochs + batch_size)
# ---------------------------------------------------------------------------

_SHARED_YAML = Path(__file__).resolve().parent / "configs" / "shared.yaml"


def _load_shared() -> dict:
    with open(_SHARED_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    shared = _load_shared()

    parser = argparse.ArgumentParser(
        description="ThriftNAS: retrain a derived genotype on CIFAR-10."
    )
    parser.add_argument(
        "--genotype", type=str, required=True,
        help="Path to genotype.json produced by search.py"
    )
    parser.add_argument(
        "--epochs", type=int, default=shared["epochs"],
        help=f"Training epochs (default from shared.yaml: {shared['epochs']})"
    )
    parser.add_argument(
        "--batch_size", type=int, default=shared["batch_size"],
        help=f"Mini-batch size (default from shared.yaml: {shared['batch_size']})"
    )
    parser.add_argument(
        "--C", type=int, default=36,
        help="Initial channel width (default: 36)"
    )
    parser.add_argument(
        "--layers", type=int, default=8,
        help="Number of cells (default: 8)"
    )
    parser.add_argument(
        "--drop_path_prob", type=float, default=0.2,
        help="Peak drop-path probability (default: 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Global random seed (default: 0)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint path to resume from"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="DataLoader workers (default: 2)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="results/retrain",
        help="Output directory (default: results/retrain)"
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable automatic mixed precision"
    )
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device()
    amp    = (not args.no_amp) and (device.type == "cuda")

    print("ThriftNAS retrain -- device={}, amp={}, epochs={}, batch={}".format(
        device, amp, args.epochs, args.batch_size))

    # ---- Genotype -----------------------------------------------------------
    genotype = load_genotype(args.genotype)
    print("Genotype loaded from {}".format(args.genotype))

    # ---- Model --------------------------------------------------------------
    model = build_network(
        C              = args.C,
        num_classes    = 10,
        layers         = args.layers,
        genotype       = genotype,
        drop_path_prob = 0.0,   # trainer sets this each epoch
        is_thriftnas   = True,
    ).to(device)

    # ---- Data loaders -------------------------------------------------------
    train_loader = get_retrain_loader(
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )
    print("Train loader: {} batches/epoch".format(len(train_loader)))

    # ---- Trainer ------------------------------------------------------------
    trainer = RetrainTrainer(
        model          = model,
        epochs         = args.epochs,
        peak_drop_path = args.drop_path_prob,
        amp            = amp,
        out_dir        = args.out_dir,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print("Resumed from {} (next epoch: {})".format(
            args.resume, trainer.start_epoch))

    # ---- Training loop (no test set per epoch) ------------------------------
    for epoch in range(trainer.start_epoch, args.epochs):
        row = trainer.train_epoch(train_loader, epoch)
        print(
            "Ep {epoch:3d} | lr={lr:.6f} | loss={train_loss:.4f} "
            "| acc={train_acc:.1f}% | dp={drop_path:.3f} "
            "| {epoch_time_s:.1f}s | {peak_gpu_mb:.0f}MB".format(**row)
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Final test evaluation (spec §8: exactly once) ----------------------
    print("Training complete. Evaluating on test set (final report only)...")
    test_loader = get_test_loader(
        batch_size  = args.batch_size,
        final_report = True,
        num_workers  = args.num_workers,
    )
    metrics = trainer.evaluate_test(test_loader)
    print("Final test -- loss={test_loss:.4f}  acc={test_acc:.2f}%".format(**metrics))
    print("Saved final_metrics.json and model_weights.pt to {}".format(args.out_dir))


if __name__ == "__main__":
    main()
