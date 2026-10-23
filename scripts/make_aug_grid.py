"""Generate a 4×8 grid of augmented CIFAR-10 training images.

Saves to results/figures/aug_grid.png.
Run with:  python scripts/make_aug_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision
import torchvision.utils as vutils
from thriftnas.data.cifar10 import _DATA_ROOT, _train_transform

if __name__ == "__main__":
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    tf = _train_transform()
    ds = torchvision.datasets.CIFAR10(_DATA_ROOT, train=True, download=True, transform=tf)

    # Fix seed for reproducibility of the grid
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(ds), generator=g)[:32].tolist()

    imgs = torch.stack([ds[i][0] for i in indices])   # (32,3,32,32), normalised

    # De-normalise for display: x_disp = x * std + mean
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    imgs_disp = (imgs * std + mean).clamp(0, 1)

    grid = vutils.make_grid(imgs_disp, nrow=8, padding=2)
    # Save via torchvision (no matplotlib dependency)
    from torchvision.utils import save_image
    save_image(grid, out_dir / "aug_grid.png")
    print(f"Saved grid to {out_dir / 'aug_grid.png'}")
