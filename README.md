# ThriftNAS: Efficiency-Aware Neural Architecture Search on CIFAR-10

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-203%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Course / Lab**: AI/ML Lab (Course: IS353IA)  
> **Target**: Efficiency-aware first-order Differentiable Architecture Search (DARTS) for resource-constrained edge vision on CIFAR-10.

---

## 📌 1. Project Overview

**ThriftNAS** is a lightweight, hardware-free Neural Architecture Search (NAS) framework based on first-order DARTS. While standard DARTS frequently discovers over-parameterized, computationally bloated cells, ThriftNAS co-optimizes **task classification accuracy** alongside **empirical computational complexity** using software-measurable proxies.

### Key Highlights
- **Hardware-Free Multi-Objective Search**: Does not require tethered edge microcontrollers or phones during search. Instead, it leverages an empirical, offline CPU latency and MAC lookup table normalized to $[0, 1]$.
- **Attention-Augmented Primitive Space**: Introduces an Efficient Channel Attention (ECA) 3×3 separable convolution alongside standard convolutions, pooling, and skip connections.
- **$r$-Ratio Diagnostic Rule**: A mathematical criterion ($r = \frac{\lambda \cdot \mathbb{E}[\alpha]}{\mathcal{L}_{\text{val}}}$) evaluated within 5 pilot epochs to identify stable vs. collapsing cost penalties before committing to long GPU runs.
- **Complete Offline Demo**: A local Gradio application with interactive CPU classification, side-by-side model comparison, Pareto front visualization, and offline search dynamic replays.

---

## 📐 2. Mathematical Formulation

### Bilevel Optimization with Efficiency Penalty
The architecture parameters $\alpha$ are updated on a held-out validation split $\mathcal{D}_{\alpha}$ using an efficiency-regularized objective:

$$\min_{\alpha} \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha) + \lambda \cdot \mathbb{E}[\alpha]$$

where $w^*(\alpha) \approx w - \xi \nabla_w \mathcal{L}_{\text{train}}(w, \alpha)$ (first-order approximation $\xi = 0$).

### Differentiable Operation Cost Model
For each candidate operation $o \in \mathcal{O}$ at stride $s \in \{1, 2\}$, the combined cost $c(o, s)$ is:

$$c(o, s) = 0.5 \cdot \frac{\text{lat}(o, s)}{\text{lat}_{\max}} + 0.5 \cdot \frac{\text{MACs}(o, s)}{\text{MACs}_{\max}}$$

- $\text{lat}(o, s)$: Median single-threaded CPU latency measured across 200 benchmark runs on $16 \times 16 \times 32$ feature maps.
- $\text{MACs}(o, s)$: Multiply-accumulate operations (fvcore convention: convolutions count FLOPs; pooling and ReLU count as 0 MACs).
- $\mathbb{E}[\alpha]$: Softmax-weighted expectation across all 14 normal cell edges and 14 reduction cell edges:

$$\mathbb{E}[\alpha] = \sum_{e \in \text{Normal}} \sum_{o \in \mathcal{O}} \text{softmax}(\alpha_e)[o] \cdot c(o, 1) + \sum_{e \in \text{Reduce}} \sum_{o \in \mathcal{O}} \text{softmax}(\alpha_e)[o] \cdot c(o, s_e)$$

---

## 📂 3. Project Structure

```text
ThriftNAS/
├── configs/
│   └── shared.yaml             # Single source of truth for retraining epochs & batch size
├── data/
│   ├── demo_examples/          # 10 sample CIFAR-10 test images for Gradio demo
│   └── cifar-10-batches-py/    # Raw CIFAR-10 dataset (git-ignored)
├── results/
│   ├── cost_table.json         # Measured empirical op latency & MAC table
│   ├── results.csv             # Markdown & CSV benchmark summary
│   ├── figures/                # 300-DPI Pareto curves, ablation plots, cell graphs
│   └── search_pilot_lam*/      # 5-epoch pilot search logs and derived genotypes
├── scripts/
│   ├── run_pilot.py            # Sequential runner for 5-epoch pilot searches with auto-resume
│   ├── analyze_pilot.py        # Automated test verification & r-ratio computation
│   ├── make_plots.py           # Publication-ready plotting generator
│   └── check_env.py            # Hardware & CUDA environment validator
├── tests/                      # Pytest suite (203 automated tests)
├── thriftnas/
│   ├── data/cifar10.py         # 10k/10k search splits & 50k retrain loaders
│   ├── ops/primitives.py       # 8 primitive operations + ECA channel attention
│   ├── models/
│   │   ├── search_net.py       # First-order DARTS continuous supernet
│   │   ├── retrain_net.py      # NetworkCIFAR discrete evaluation network
│   │   ├── genotypes.py        # Discrete genotype derivation & Graphviz visualization
│   │   └── baselines.py        # CIFAR-adapted ResNet-20 and MobileNetV2
│   ├── trainers/
│   │   ├── search_trainer.py   # Bilevel search loop with AMP & atomic checkpointing
│   │   └── retrain_trainer.py  # Drop-path retraining trainer
│   └── utils/
│       ├── metrics.py          # Params, MACs, disk size, and CPU latency profilers
│       ├── plots.py            # Pareto frontiers and search dynamic visualizers
│       └── inference.py        # Offline CPU inference engine
├── app.py                      # Interactive 4-tab Gradio demo
├── search.py                   # Supernet architecture search CLI
├── retrain.py                  # Discrete genotype retrain CLI
├── eval_baselines.py           # ResNet-20 and MobileNetV2 training & evaluation CLI
├── evaluate.py                 # Multi-model evaluation and markdown table builder
└── pyproject.toml              # Build specifications & pytest configuration
```

---

## ✅ 4. What Has Been Done (Completed Steps 1–13)

| Step | Component | Status | Description |
| :---: | :--- | :---: | :--- |
| **1** | **Environment & CUDA** | ✅ Done | Python 3.11, PyTorch 2.1+, NVIDIA RTX 3050 Laptop GPU (6 GB VRAM) verification. |
| **2** | **Data Loading** | ✅ Done | Disjoint 10k weight / 10k alpha search splits; 50k full retrain loader; test set gated by `final_report=True`. |
| **3** | **Primitive Ops & ECA** | ✅ Done | 8 candidate operations implemented, including `eca_sep_conv_3x3` (channel attention). |
| **4** | **Cost Table Profiler** | ✅ Done | Profiler executed at $C=32$, $16 \times 16$; verified agreement across multiple runs; generated `cost_table.json`. |
| **5** | **Search Supernet** | ✅ Done | 5-cell `SearchNetwork` with mixed ops and differentiable expected cost $\mathbb{E}[\alpha]$. |
| **6** | **Search Trainer** | ✅ Done | First-order bilevel optimization, AMP with dual `GradScaler`, atomic checkpoints, and `search.py` CLI. |
| **7** | **Genotype Derivation** | ✅ Done | Discrete cell extraction (`derive_genotype`), Graphviz diagram export (`cell_normal.png`, `cell_reduce.png`). |
| **8** | **Retrain Network** | ✅ Done | `NetworkCIFAR` discrete architecture with auxiliary drop-path, cosine LR schedule, and `retrain.py` CLI. |
| **9** | **Baselines** | ✅ Done | ResNet-20 (0.27M params, ~40.8M MACs) and MobileNetV2 adapted to 32×32 without early downsampling. |
| **10** | **Evaluation & Metrics** | ✅ Done | Standardized metrics reporting (Params, MACs, Size MB, single-threaded CPU latency, predicted vs. measured). |
| **11** | **Plotting Suite** | ✅ Done | Automated 300-DPI publication figures: Pareto frontiers, $\lambda$-ablation, $\alpha$-evolution, and skip connection counts. |
| **12** | **Offline Gradio Demo** | ✅ Done | 4-tab offline app (`app.py`): Classify, Compare All Models, Benchmark Results, and Search Replay. |
| **13** | **5-Epoch Pilot Search** | ✅ Done | Ran pilot searches on full CIFAR-10 splits across $\lambda \in \{0.0, 0.1, 1.0\}$; verified $r$-ratio diagnostic and op shifts. |

### Unit & Integration Test Coverage
- Total passing tests: **203 tests** across 12 test suites.
- Command to run:
  ```bash
  pytest
  ```

---

## 🔬 5. Key Findings from Step 13 Pilot Searches

The 5-epoch pilot runs on CIFAR-10 confirmed the project's central research hypothesis:

```text
PILOT RUN COMPARISON (Epoch 4 Metrics):
┌─────────────┬──────────┬──────────┬──────────┬────────────┬─────────────┬──────────────────────────────────────────┐
│   Lambda    │   ce_w   │   ce_a   │   eff    │  r-Ratio   │ Heavy Ops*  │ Diagnostic Verdict / Op Behavior         │
├─────────────┼──────────┼──────────┼──────────┼────────────┼─────────────┼──────────────────────────────────────────┤
│ λ = 0.0     │  0.9449  │  1.0183  │  9.0680  │   0.0000   │  11 ops     │ Unconstrained DARTS; picks heavy convs   │
│ λ = 0.1     │  0.9799  │  1.0425  │  7.8688  │   0.7548   │   1 op      │ USEFUL (0.05 ≤ r ≤ 1.0); balanced search │
│ λ = 1.0     │  1.0221  │  1.0832  │  7.1610  │   6.6109   │   0 ops     │ COST DOMINATES (r > 2.0); skips/pooling  │
└─────────────┴──────────┴──────────┴──────────┴────────────┴─────────────┴──────────────────────────────────────────┘
* Heavy ops: sep_conv_5x5, dil_conv_3x3.
```

- **Measured Search Speed**: ~4.5 minutes/epoch on NVIDIA RTX 3050 Laptop GPU (Peak VRAM: 2,157 MB).
- **Addendum Rule Validated**: $\lambda = 0.1$ operates in the sweet spot ($r \approx 0.75$), encouraging lightweight structures without collapsing feature extraction.

---

## 🚀 6. What Needs to Be Done (Next Steps)

> 💡 **2-Person Execution Plan**: To run the remaining training in parallel across two laptops in ~2.5 hours instead of 6+ hours, see **[TWO_PERSON_IMPLEMENTATION_PLAN.md](TWO_PERSON_IMPLEMENTATION_PLAN.md)** for exact command tracks and git merge instructions for Darshil and Priyansh.

1. **Step 14: Full Architecture Search (25 Epochs)**
   - Execute full 25-epoch search runs for $\lambda = 0.0$ (Vanilla DARTS baseline), $\lambda = 0.1$ (ThriftNAS), and optionally $\lambda = 0.01$.
   - Estimated search time: ~1.5 hours per run (~4.5 hours total).
2. **Step 15: Full Retraining (50 Epochs on full 50k CIFAR-10)**
   - Retrain the derived genotypes and baselines:
     - ResNet-20 (~15 min)
     - MobileNetV2 (~20 min)
     - Vanilla DARTS $\lambda = 0$ genotype (~25 min)
     - ThriftNAS $\lambda = 0.1$ genotype (~20 min)
   - Total retraining time: ~1.5 hours.
3. **Step 16: Final Evaluation & Report Generation**
   - Run `python evaluate.py` to evaluate the final checkpoints once on the 10k test set.
   - Populate final test accuracies, MACs, parameters, and CPU latencies in `results/results.csv` and `results/results.md`.
4. **Step 17: Final Pareto Plots**
   - Run `python scripts/make_plots.py` to produce final Pareto curves and search evolution figures.
5. **Step 18: Demo Walkthrough & Final Documentation**
   - Launch the Gradio app with final model weights for live demonstration.
   - Finalize the lab report and presentation slides.

---

## 💻 7. How to Run

### Setup Environment
```powershell
# Clone the repository
git clone https://github.com/Darshiljpatel/ThriftNAS.git
cd ThriftNAS

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install package in editable mode with dev dependencies
pip install -e .[dev]
```

### Launch the Interactive Gradio Demo
```powershell
python app.py
```
Open your browser at `http://127.0.0.1:7860/` to test single-image classification, multi-model side-by-side comparison, Pareto curves, and search dynamics replay.

### Run Architecture Search
```powershell
# Example: Search with efficiency penalty lambda = 0.1
python search.py --lam 0.1 --epochs 25 --batch_size 48 --out_dir results/search_lam0.1
```

### Retrain Discovered Architecture
```powershell
python retrain.py --genotype results/search_lam0.1/genotype.json --out_dir results/retrain_lam0.1
```

### Evaluate Baselines
```powershell
python eval_baselines.py --model resnet20
python eval_baselines.py --model mobilenet_v2
```

### Run Test Suite
```powershell
pytest
```

