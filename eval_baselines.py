"""Train and evaluate baseline models for ThriftNAS (PROJECT_SPEC §6, Step 9).

Supports:
- ResNet-20 (CIFAR-10)
- MobileNetV2 (CIFAR-10 adapted, width 1.0)

Vanilla DARTS is evaluated by retraining the lambda=0 genotype with `retrain.py`
(PROJECT_SPEC §6: "Vanilla DARTS = search with lambda=0, derive the genotype,
retrain with the same recipe").

All baselines use:
- The exact same CIFAR-10 training set (50k) and test set (10k)
- The same data augmentation (RandomCrop 32 pad 4 + RandomHorizontalFlip)
- The same optimizer: SGD (lr 0.025, cosine LR schedule to 0.001, momentum 0.9, wd 3e-4)
- Epoch count and batch size loaded from configs/shared.yaml (never hardcoded)
- No Cutout and no auxiliary head
- The test set is evaluated EXACTLY ONCE after the final epoch (spec §8)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from thriftnas.data.cifar10 import get_retrain_loader, get_test_loader
from thriftnas.models.baselines import (
    build_baseline,
    count_model_macs,
    count_parameters,
)
from thriftnas.trainers.retrain_trainer import RetrainTrainer
from thriftnas.utils import get_device, set_seed

# ---------------------------------------------------------------------------
# shared.yaml loader (single source of truth for epochs + batch_size)
# ---------------------------------------------------------------------------

_SHARED_YAML = Path(__file__).resolve().parent / "configs" / "shared.yaml"


def _load_shared() -> dict[str, Any]:
    with open(_SHARED_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------

def train_and_eval_baseline(
    model_name: str,
    epochs: int,
    batch_size: int,
    seed: int = 0,
    out_dir: str | Path = "results/baselines",
    resume: str | None = None,
    no_amp: bool = False,
    num_workers: int = 2,
    eval_only: bool = False,
) -> dict[str, Any]:
    """Train and evaluate a single baseline model."""
    set_seed(seed)
    device = get_device()
    amp = (not no_amp) and (device.type == "cuda")

    out_path = Path(out_dir) / model_name
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Baseline: {model_name.upper()} | device={device} | amp={amp} | epochs={epochs} | batch={batch_size}")
    print("=" * 70)

    # ---- Model --------------------------------------------------------------
    model = build_baseline(model_name, num_classes=10).to(device)
    params = count_parameters(model)
    macs = count_model_macs(model, input_size=(1, 3, 32, 32))
    print(f"Model {model_name}: {params:,} parameters ({params/1e6:.4f}M), {macs:,} MACs ({macs/1e6:.2f}M)")

    # ---- Trainer ------------------------------------------------------------
    trainer = RetrainTrainer(
        model=model,
        epochs=epochs,
        peak_drop_path=0.0,     # Baselines do not use drop-path
        amp=amp,
        out_dir=out_path,
    )

    if resume:
        trainer.load_checkpoint(resume)
        print(f"Resumed from {resume} (next epoch: {trainer.start_epoch})")

    # ---- Data loaders -------------------------------------------------------
    if not eval_only:
        train_loader = get_retrain_loader(batch_size=batch_size, num_workers=num_workers)
        print(f"Train loader: {len(train_loader)} batches/epoch")

        for epoch in range(trainer.start_epoch, epochs):
            row = trainer.train_epoch(train_loader, epoch)
            print(
                "Ep {epoch:3d} | lr={lr:.6f} | loss={train_loss:.4f} "
                "| acc={train_acc:.1f}% | {epoch_time_s:.1f}s | {peak_gpu_mb:.0f}MB".format(**row)
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Final test evaluation (spec §8: exactly once) ----------------------
    print(f"\nEvaluating {model_name} on CIFAR-10 test set (final report only)...")
    test_loader = get_test_loader(
        batch_size=batch_size,
        final_report=True,
        num_workers=num_workers,
    )
    test_metrics = trainer.evaluate_test(test_loader)
    print(f"Final test [{model_name}] -- loss={test_metrics['test_loss']:.4f}  acc={test_metrics['test_acc']:.2f}%")

    # Save summary metadata
    summary = {
        "model": model_name,
        "parameters": params,
        "params_m": round(params / 1e6, 4),
        "macs": macs,
        "macs_m": round(macs / 1e6, 2),
        "test_acc": test_metrics["test_acc"],
        "test_loss": test_metrics["test_loss"],
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
    }
    with open(out_path / "final_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    shared = _load_shared()

    parser = argparse.ArgumentParser(
        description="ThriftNAS: train and evaluate baseline models (ResNet-20, MobileNetV2)."
    )
    parser.add_argument(
        "--model", type=str, default="resnet20",
        choices=["resnet20", "mobilenet_v2", "all"],
        help="Baseline to run ('resnet20', 'mobilenet_v2', or 'all')"
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
        "--seed", type=int, default=0,
        help="Global random seed (default: 0)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="DataLoader workers (default: 2)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="results/baselines",
        help="Output directory (default: results/baselines)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint path to resume from"
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable automatic mixed precision"
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Only evaluate existing checkpoint on the test set without training"
    )
    args = parser.parse_args(argv)

    models_to_run = ["resnet20", "mobilenet_v2"] if args.model == "all" else [args.model]

    results = []
    for m in models_to_run:
        res = train_and_eval_baseline(
            model_name=m,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            out_dir=args.out_dir,
            resume=args.resume,
            no_amp=args.no_amp,
            num_workers=args.num_workers,
            eval_only=args.eval_only,
        )
        results.append(res)

    print("\n" + "=" * 70)
    print("BASELINE SUMMARY RESULTS")
    print("=" * 70)
    print(f"{'Model':<16} {'Params (M)':<12} {'MACs (M)':<12} {'Test Acc (%)':<14} {'Test Loss':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<16} {r['params_m']:<12.4f} {r['macs_m']:<12.2f} {r['test_acc']:<14.2f} {r['test_loss']:<10.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
