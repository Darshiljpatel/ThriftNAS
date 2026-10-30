"""Inference and model serving utilities for ThriftNAS (PROJECT_SPEC §7, Step 12).

Provides:
- Lazy CPU model loading
- Top-3 class prediction with probabilities
- 20-run median CPU inference latency measurement
- Dynamic metadata loading from results/results.csv (zero hardcoded values)
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from thriftnas.models.baselines import build_baseline
from thriftnas.models.genotypes import load_genotype
from thriftnas.models.retrain_net import build_network

# CIFAR-10 class labels
CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Standard CIFAR-10 inference preprocessing
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])

# Global in-memory cache for lazy CPU model instances
_MODEL_CACHE: dict[str, nn.Module] = {}


# ---------------------------------------------------------------------------
# Model discovery & loading
# ---------------------------------------------------------------------------

def get_available_model_names() -> list[str]:
    """Return list of supported model display names."""
    return [
        "ResNet-20",
        "MobileNetV2",
        "Vanilla DARTS (lam=0)",
        "ThriftNAS (lam=0.1)",
        "ThriftNAS (lam=1.0)",
    ]


def _resolve_checkpoint(model_name: str) -> Path | None:
    """Find the default weights path for a model."""
    name_lower = model_name.lower()
    if "resnet" in name_lower:
        candidates = [
            Path("results/baselines_t6/resnet20/model_weights.pt"),
            Path("results/baselines/resnet20/model_weights.pt"),
        ]
    elif "mobilenet" in name_lower:
        candidates = [
            Path("results/baselines_t6/mobilenet_v2/model_weights.pt"),
            Path("results/baselines/mobilenet_v2/model_weights.pt"),
        ]
    elif "lam=0.1" in name_lower or "lam0.1" in name_lower:
        candidates = [
            Path("results/retrain_lam0.1/model_weights.pt"),
            Path("results/search_lam0.1/checkpoint.pt"),
        ]
    elif "lam=1" in name_lower or "lam1" in name_lower:
        candidates = [
            Path("results/retrain_lam1.0/model_weights.pt"),
            Path("results/search_lam1.0/checkpoint.pt"),
        ]
    else:
        # Vanilla DARTS / lam=0
        candidates = [
            Path("results/retrain_t10/model_weights.pt"),
            Path("results/retrain_lam0/model_weights.pt"),
            Path("results/search_lam0/checkpoint.pt"),
        ]

    for c in candidates:
        if c.exists():
            return c
    return candidates[0] if candidates else None


def load_model(
    model_name: str,
    checkpoint_path: str | Path | None = None,
) -> tuple[nn.Module | None, str | None]:
    """Load a model onto CPU lazily.

    Parameters
    ----------
    model_name:
        Display name of the model.
    checkpoint_path:
        Optional path to weights file.

    Returns
    -------
    (model, error_message)
        If loading succeeds, returns (model, None).
        If checkpoint is missing or invalid, returns (None, error_msg) without raising.
    """
    ckpt = Path(checkpoint_path) if checkpoint_path else _resolve_checkpoint(model_name)

    cache_key = f"{model_name}:{ckpt}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], None

    if ckpt is None or not ckpt.exists():
        return None, f"Checkpoint not found at '{ckpt}'. Please train the model or provide weights."

    name_lower = model_name.lower()
    try:
        if "resnet" in name_lower:
            model = build_baseline("resnet20")
        elif "mobilenet" in name_lower:
            model = build_baseline("mobilenet_v2")
        else:
            # ThriftNAS / DARTS model: find genotype
            genotype_path = Path("results/search_lam0/genotype.json")
            if "0.1" in name_lower and Path("results/search_lam0.1/genotype.json").exists():
                genotype_path = Path("results/search_lam0.1/genotype.json")
            elif ("1.0" in name_lower or "lam=1" in name_lower) and Path("results/search_lam1.0/genotype.json").exists():
                genotype_path = Path("results/search_lam1.0/genotype.json")

            if not genotype_path.exists():
                return None, f"Genotype JSON not found at '{genotype_path}'."

            g = load_genotype(genotype_path)
            model = build_network(C=36, num_classes=10, layers=8, genotype=g, is_thriftnas=False)

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Filter unexpected keys if needed
        model.load_state_dict(state, strict=False)
        model.to("cpu")
        model.eval()

        _MODEL_CACHE[cache_key] = model
        return model, None
    except Exception as e:
        return None, f"Failed to load checkpoint '{ckpt}': {e}"


# ---------------------------------------------------------------------------
# Prediction & latency
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict(
    model: nn.Module,
    img: Image.Image,
) -> list[tuple[str, float]]:
    """Predict top-3 CIFAR-10 classes and probabilities for an input image.

    Preprocessing:
    - Converts any image mode (L, RGBA, P) to RGB
    - Resizes to 32x32
    - Normalizes with CIFAR-10 channel mean and std

    Parameters
    ----------
    model:
        Model in eval mode on CPU.
    img:
        PIL Image of arbitrary size and mode.

    Returns
    -------
    list of 3 (class_name, probability) tuples sorted descending by probability.
    """
    torch.set_num_threads(1)
    rgb_img = img.convert("RGB")
    x = INFERENCE_TRANSFORM(rgb_img).unsqueeze(0)

    logits = model(x)
    probs = F.softmax(logits, dim=-1)[0]
    top = torch.topk(probs, 3)

    return [(CLASSES[int(idx.item())], float(prob.item())) for prob, idx in zip(top.values, top.indices)]


@torch.inference_mode()
def measure_inference_latency(
    model: nn.Module,
    img: Image.Image,
    runs: int = 20,
    warmup: int = 3,
) -> float:
    """Measure median CPU inference time over `runs` iterations in milliseconds.

    Returns
    -------
    float
        Median latency in milliseconds.
    """
    torch.set_num_threads(1)
    rgb_img = img.convert("RGB")
    x = INFERENCE_TRANSFORM(rgb_img).unsqueeze(0)

    # Warmup passes
    for _ in range(warmup):
        model(x)

    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1e3)

    return float(np.median(times))


# ---------------------------------------------------------------------------
# Results metadata retrieval (Zero hardcoded values)
# ---------------------------------------------------------------------------

def get_model_metadata(
    model_name: str,
    results_csv_path: str | Path = "results/results.csv",
) -> dict[str, str]:
    """Read a model's published metrics from results/results.csv.

    Never hard-codes any metric (PROJECT_SPEC §12 rule).
    """
    p = Path(results_csv_path)
    if not p.exists():
        return {
            "Top-1 (%)": "N/A (run evaluate.py)",
            "Params (M)": "N/A",
            "MACs (M)": "N/A",
            "Size (MB)": "N/A",
            "CPU latency (ms, mean +/- std)": "N/A",
        }

    target = model_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    with open(p, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_model = row.get("Model", "").lower().replace("-", "").replace("_", "").replace(" ", "")
            if target in row_model or row_model in target:
                return {
                    "Top-1 (%)": row.get("Top-1 (%)", "N/A"),
                    "Params (M)": row.get("Params (M)", "N/A"),
                    "MACs (M)": row.get("MACs (M)", "N/A"),
                    "Size (MB)": row.get("Size (MB)", "N/A"),
                    "CPU latency (ms, mean +/- std)": row.get("CPU latency (ms, mean +/- std)", "N/A"),
                }

    return {
        "Top-1 (%)": "Pending run",
        "Params (M)": "Pending run",
        "MACs (M)": "Pending run",
        "Size (MB)": "Pending run",
        "CPU latency (ms, mean +/- std)": "Pending run",
    }


def compare_all_models(
    img: Image.Image,
    model_names: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Run an image through all available models and construct a comparison table."""
    if model_names is None:
        model_names = get_available_model_names()

    rows: list[dict[str, str]] = []
    for name in model_names:
        model, err = load_model(name)
        if err or model is None:
            rows.append({
                "Model": name,
                "Prediction": "Unavailable",
                "Probability": "-",
                "Latency (ms)": "-",
                "Status": err or "Missing",
            })
            continue

        preds = predict(model, img)
        top_cls, top_prob = preds[0]
        lat_ms = measure_inference_latency(model, img, runs=10)

        rows.append({
            "Model": name,
            "Prediction": top_cls,
            "Probability": f"{top_prob * 100:.1f}%",
            "Latency (ms)": f"{lat_ms:.2f} ms",
            "Status": "Ready",
        })

    return rows
