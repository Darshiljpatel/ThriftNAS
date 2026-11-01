"""Attention ablation for ThriftNAS (PROJECT_SPEC §2: "An SE block is the alternative
for the attention ablation").

Runs the same search three ways, changing ONLY the attention slot of the search space:

    --attention eca    slot 6 = eca_sep_conv_3x3   (sep_conv_3x3 + ECA)   [the default space]
    --attention se     slot 6 = se_sep_conv_3x3    (sep_conv_3x3 + SE)
    --attention none   slot 6 = sep_conv_3x3_dup   (plain sep_conv_3x3, no attention)

Why swap a slot instead of shortening the list: PRIMITIVES stays length 8, so the alpha
shape (14, 8), the stride-2 mask and `_ZERO_IDX == 7` are all unchanged.  Nothing in the
package or the test-suite sees a different search space size.

Note on `--attention none`: slot 6 becomes a *second* plain sep_conv_3x3.  The softmax then
splits probability between two identical operations, so for that variant the fair comparison
is `sep_conv_3x3 + sep_conv_3x3_dup` combined against `eca_sep_conv_3x3` alone.  This is
stated in the results rather than hidden.

The cost table is patched for the swapped operation: its latency and MACs are profiled here
(same convention as scripts/profile_ops.py -- C=32, 16x16, 1 thread, median of N runs) and
written into a variant-specific JSON so the penalty term stays correct.  Without this the
swapped op would have c = 0 and look free, which would invalidate the experiment.

Usage
-----
  python scripts/run_attention_ablation.py --attention se   --lam 0.1 --epochs 5
  python scripts/run_attention_ablation.py --attention none --lam 0.1 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thriftnas.ops import primitives as P
from thriftnas.ops.cost_model import count_macs, make_metadata, time_op

# ---------------------------------------------------------------------------
# The attention slot
# ---------------------------------------------------------------------------

_ATTN_SLOT = 6   # PRIMITIVES index of the attention op in the default search space


class SeSepConv(nn.Module):
    """sep_conv_3x3 followed by a Squeeze-and-Excitation block (the SE ablation arm)."""

    def __init__(self, C: int, stride: int, affine: bool) -> None:
        super().__init__()
        self.sep = P.SepConv(C, 3, stride, affine)
        self.se = P.SE(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.sep(x))


#: variant name -> (op name placed in slot 6, factory)
VARIANTS: dict[str, tuple[str, object]] = {
    "eca": (
        "eca_sep_conv_3x3",
        lambda C, stride, affine: P.EcaSepConv(C, stride, affine),
    ),
    "se": (
        "se_sep_conv_3x3",
        lambda C, stride, affine: SeSepConv(C, stride, affine),
    ),
    "none": (
        "sep_conv_3x3_dup",
        lambda C, stride, affine: P.SepConv(C, 3, stride, affine),
    ),
}


def apply_variant(variant: str) -> str:
    """Swap PRIMITIVES[6] and register its factory in OPS.  Returns the op name.

    Must be called BEFORE any SearchNetwork is constructed, because MixedOp reads
    PRIMITIVES at construction time.
    """
    name, factory = VARIANTS[variant]
    P.OPS[name] = factory
    P.PRIMITIVES[_ATTN_SLOT] = name
    assert len(P.PRIMITIVES) == 8, "search space must stay length 8"
    assert P.PRIMITIVES[7] == "zero", "'zero' must stay at index 7"
    return name


# ---------------------------------------------------------------------------
# Cost table patching
# ---------------------------------------------------------------------------

def build_variant_cost_table(
    base_table: Path,
    op_name: str,
    out_path: Path,
    C: int = 32,
    size: int = 16,
    warmup: int = 20,
    runs: int = 200,
) -> Path:
    """Copy *base_table* and replace the attention entries with *op_name* profiled here.

    The normalisation denominators (lat_max, MACs_max) are recomputed over all entries so
    the variant table stays on the same 0-1 scale as the original.
    """
    with open(base_table, encoding="utf-8") as fh:
        table = json.load(fh)

    # Drop whatever attention op the base table carried.
    known_attention = {n for n, _ in VARIANTS.values()}
    entries = [e for e in table["entries"] if e["op"] not in known_attention]

    x = torch.zeros(1, C, size, size)
    for stride in (1, 2):
        op = P.OPS[op_name](C, stride, affine=True)
        lat_med, lat_std = time_op(op, x, warmup=warmup, runs=runs)
        macs = count_macs(op, x)
        entries.append({
            "op": op_name,
            "stride": stride,
            "lat_median_ms": lat_med,
            "lat_std_ms": lat_std,
            "macs": macs,
            "c": None,
            "flagged": bool(lat_std > 0.1 * lat_med),
        })
        print(f"  profiled {op_name} stride={stride}: "
              f"{lat_med:.4f} ms (std {lat_std:.4f}), {macs:,} MACs")

    # MACs-only tables have no latency column; detect and handle both shapes.
    has_latency = all("lat_median_ms" in e for e in entries)
    mac_max = max(e["macs"] for e in entries)
    lat_max = max(e["lat_median_ms"] for e in entries) if has_latency else None

    for e in entries:
        mac_term = (e["macs"] / mac_max) if mac_max > 0 else 0.0
        if has_latency and lat_max:
            lat_term = e["lat_median_ms"] / lat_max
            e["c"] = 0.5 * lat_term + 0.5 * mac_term
        else:
            e["c"] = mac_term

    out = {
        "metadata": make_metadata(C=C, size=size, warmup=warmup, runs=runs),
        "mac_max": mac_max,
        "entries": entries,
    }
    if lat_max is not None:
        out["lat_max"] = lat_max
    out["metadata"]["attention_variant"] = op_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Attention ablation: run the search with ECA / SE / no attention."
    )
    parser.add_argument("--attention", choices=sorted(VARIANTS), required=True,
                        help="which operation occupies the attention slot")
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_cost_table", type=str,
                        default="results/cost_table.json")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="default: results/ablation_attn_<variant>_lam<lam>")
    parser.add_argument("--profile_runs", type=int, default=200)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args(argv)

    # ---- 1. Swap the search space BEFORE importing anything that builds a model ----
    op_name = apply_variant(args.attention)
    print(f"Attention variant '{args.attention}' -> slot {_ATTN_SLOT} = '{op_name}'")
    print(f"PRIMITIVES = {P.PRIMITIVES}")

    # Imported after the swap so these modules see the patched search space.
    from thriftnas.data.cifar10 import get_search_loaders
    from thriftnas.models.genotypes import derive_genotype, save_genotype
    from thriftnas.models.search_net import SearchNetwork
    from thriftnas.trainers.search_trainer import SearchTrainer
    from thriftnas.utils import get_device, set_seed

    out_dir = Path(args.out_dir or
                   f"results/ablation_attn_{args.attention}_lam{args.lam}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 2. Build a cost table that prices the swapped op correctly ----
    print(f"\nProfiling '{op_name}' for the variant cost table "
          f"(runs={args.profile_runs}) ...")
    table_path = build_variant_cost_table(
        base_table=Path(args.base_cost_table),
        op_name=op_name,
        out_path=out_dir / "cost_table.json",
        runs=args.profile_runs,
    )
    print(f"Variant cost table -> {table_path}")

    with open(table_path, encoding="utf-8") as fh:
        ct = json.load(fh)
    idx = {name: i for i, name in enumerate(P.PRIMITIVES)}
    c1 = torch.zeros(len(P.PRIMITIVES))
    c2 = torch.zeros(len(P.PRIMITIVES))
    for e in ct["entries"]:
        i = idx.get(e["op"])
        if i is None:
            continue
        (c1 if e["stride"] == 1 else c2)[i] = float(e["c"])
    print("c1 =", [round(float(v), 4) for v in c1])
    print("c2 =", [round(float(v), 4) for v in c2])

    # ---- 3. Run the search ----
    set_seed(args.seed)
    device = get_device()
    amp = (not args.no_amp) and (device.type == "cuda")
    print(f"\nSearch -- device={device}, amp={amp}, lam={args.lam}, epochs={args.epochs}")

    model = SearchNetwork(C=16, num_classes=10, layers=5).to(device)
    w_loader, a_loader = get_search_loaders(
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers)

    trainer = SearchTrainer(
        model=model, c1=c1.to(device), c2=c2.to(device),
        lam=args.lam, epochs=args.epochs, amp=amp, out_dir=out_dir,
    )

    for epoch in range(args.epochs):
        row = trainer.train_epoch(w_loader, a_loader, epoch)
        print("Ep {epoch:3d} | ce_w={ce_w:.4f} ce_a={ce_a:.4f} | eff={eff:.4f} "
              "lam_eff={lam_eff:.4f} | acc_val={val_acc:.1f}% | skip={skip_count} "
              "| {epoch_time_s:.1f}s".format(**row))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    geno = derive_genotype(model.alpha_normal, model.alpha_reduce)
    save_genotype(geno, out_dir / "genotype.json")

    n = [o for o, _ in geno.normal]
    attn_in_normal = n.count(op_name)
    print(f"\nGenotype saved -> {out_dir / 'genotype.json'}")
    print(f"normal cell ops: {n}")
    print(f"'{op_name}' selected on {attn_in_normal}/8 normal-cell edges")

    with open(out_dir / "ablation_summary.json", "w", encoding="utf-8") as fh:
        json.dump({
            "attention_variant": args.attention,
            "slot_op": op_name,
            "lam": args.lam,
            "epochs": args.epochs,
            "seed": args.seed,
            "normal_ops": n,
            "reduce_ops": [o for o, _ in geno.reduce],
            "attn_edges_in_normal": attn_in_normal,
        }, fh, indent=2)


if __name__ == "__main__":
    main()
