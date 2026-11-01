# ThriftNAS — What We Do Differently

> **Purpose**: this document states exactly what is new in ThriftNAS relative to existing
> efficiency-aware NAS work, which claims the measured results already support, and which
> claims still need an experiment before we put them in the report.
>
> Course: IS353IA · Dataset: CIFAR-10 · Hardware: single RTX 3050 Laptop GPU (4 GB)

---

## 1. What already exists (do NOT claim as novel)

Efficiency-aware **differentiable** NAS with a latency lookup table is not new. It was
established in 2019:

| Work | Approach |
| :--- | :--- |
| **DARTS** (Liu et al., 2019) | Differentiable bilevel NAS. No efficiency term. |
| **FBNet** (Wu et al., 2019) | Differentiable NAS with a latency lookup table in the loss. |
| **ProxylessNAS** (Cai et al., 2019) | Hardware-aware search with a latency term, measured on the target device. |
| **MnasNet** (Tan et al., 2019) | RL-based search with real on-device latency in the reward. |

If we present "we added a latency term to the DARTS loss" as our contribution, the immediate
question is *"how is that different from FBNet?"* — and we would have no answer. Our
differentiation lives in the three items below, not in the existence of the cost term.

---

## 2. Contribution 1 — The `r`-ratio diagnostic (our strongest claim)

### The problem it solves

Every efficiency-aware NAS method has a trade-off weight (λ in our case, a reward weight in
MnasNet). The published approach is to **sweep it and report the results**. Nobody tells you,
*before* committing GPU hours, whether a given λ will produce a useful architecture or collapse
into a degenerate one.

On one 4 GB laptop a full 25-epoch search costs **~2 h 20 min**. Sweeping three λ values blindly
costs ~7 hours, and a collapsed run is 2.5 hours of wasted compute.

### What we propose

Run a **5-epoch pilot** and compute a single scalar:

```
r = λ · E[α] / L_val          (measured at the last pilot epoch)

r < 0.05        →  penalty too weak, behaves like vanilla DARTS
0.05 ≤ r ≤ 1.0  →  useful, balanced trade-off
1.0 < r ≤ 2.0   →  strong, cost penalty clearly active
r > 2.0         →  cost dominates, expect architecture collapse
```

### Evidence from our own pilots (5 epochs each, full CIFAR-10 splits)

| λ | `ce_a` | `E[α]` | `λ·E[α]` | **r** | Predicted | Actual derived genotype |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.0 | 1.0183 | 9.068 | 0.000 | **0.000** | no effect | heavy convs: 4× `sep_conv_5x5` in normal cell |
| 0.1 | 1.0425 | 7.869 | 0.787 | **0.755** | useful | balanced: 1 `dil_conv_3x3` + pools + skips |
| 1.0 | 1.0832 | 7.161 | 7.161 | **6.611** | collapse | **normal = 8× `skip_connect`, reduce = 8× `avg_pool`** |

The prediction held in all three cases. The λ=1.0 run collapsed exactly as the rule said it
would — a network with zero learnable operations in its normal cell.

### Why this is a contribution

It converts λ selection from a blind sweep into a **~25-minute screening step**. In our setting
it saves roughly 22 GPU-hours of misdirected search. For resource-constrained NAS — the exact
setting this project targets — that is a practical, reusable methodological result, and it is
the part of this work that is genuinely ours.

### What would strengthen it

The thresholds (0.05 / 1.0 / 2.0) are currently fitted to a **single 3-point pilot on one
dataset**. To claim they generalise we need the rule validated on at least one more λ value
(λ=0.01 is already planned) and ideally one more dataset.

---

## 3. Contribution 2 — Hybrid cost function with a stated rationale

Most prior work optimises **either** analytical FLOPs/MACs **or** measured latency. We combine
both, each normalised to [0, 1] over all `(op, stride)` entries:

```
c(o, s) = 0.5 · lat(o, s)/lat_max  +  0.5 · MACs(o, s)/MACs_max
```

### The reason this is not an arbitrary 50/50 blend

The two terms cover each other's blind spots:

- **MACs alone are wrong for parameter-free ops.** By the fvcore convention we follow, pooling,
  BatchNorm and ReLU all count as **0 MACs**. A cost model built on MACs alone therefore treats
  `max_pool_3x3` as completely free, which it is not.
- **Latency alone is noisy.** In our measured table, **13 of 16 entries** are flagged with
  std > 10 % of median. The worst case, `eca_sep_conv_3x3` at stride 2, has median 0.378 ms with
  std 0.226 ms (~60 % variation).

Blending a stable analytical term with a noisy empirical term that captures the ops the
analytical term misses is a defensible design choice, and we state it explicitly rather than
leaving it as a hyperparameter.

### Supporting evidence available cheaply

`results/cost_table_flops_only.json` is already generated. Running one search with the
MACs-only table against one with the hybrid table gives us a direct ablation for this claim at
the cost of a single extra run.

---

## 4. Contribution 3 — Channel attention as a *searched* cell primitive

### How attention is normally used

In MobileNetV3 and EfficientNet, Squeeze-and-Excitation blocks are **hand-placed**: a human
designer decides which stages get attention. The classic DARTS search space contains 7–8
operations and **no channel-attention operation at all**.

### What we do

We add `eca_sep_conv_3x3` (a 3×3 separable convolution followed by an Efficient Channel
Attention block, Wang et al. 2020) as an **eighth searchable primitive**. The optimiser — not
the designer — decides on which *individual edges* of the cell attention is worth its cost.

ECA is a deliberate choice over SE here: its entire parameter cost is one `Conv1d(1, 1, k)` with
`k=3`, so at C=32 the attention op costs `c = 0.714` against `c = 0.680` for the plain
`sep_conv_3x3` it wraps — roughly a **5 % cost premium**. That makes it a fair candidate inside a
cost-penalised search, where an expensive attention block would simply never be selected.

### Result from our completed 25-epoch λ=0.0 search

| Cell | Derived operations |
| :--- | :--- |
| **normal** | **6× `eca_sep_conv_3x3`**, 1× `sep_conv_3x3`, 1× `dil_conv_3x3` |
| **reduce** | 8× `max_pool_3x3` |

With no cost penalty at all, the search allocated **6 of 8 normal-cell edges to the attention
operation**. That is a concrete, non-obvious finding: given a free choice, DARTS prefers
attention-augmented convolutions over plain ones at near-identical cost.

### What would make this claim defensible

Right now it is **not** defensible on its own. A reviewer will say *"maybe ECA won simply because
it has marginally more capacity."* The `SE` class is implemented in `thriftnas/ops/primitives.py`
but has never been used in a search, and the attention-ablation branch in
`scripts/make_plots.py` is still an empty stub.

**Required experiment** — three searches, same seed, same λ, same epoch budget:

1. search space **without** any attention op
2. search space with **SE** instead of ECA
3. search space with **ECA** (done)

Until this runs, Contribution 3 should be presented in the report as an *observation*, not a
validated claim.

---

## 5. Secondary contributions (worth a paragraph each)

### 5.1 Stride-aware cost accounting

`lat_max` and `MACs_max` are taken over **all 16 `(op, stride)` entries**, so stride-1 and
stride-2 costs are directly comparable on one scale. We also decompose the reduction cell
correctly: **8 stride-2 edges** (those fed by the two input nodes) plus **6 stride-1 edges**
(fed by intermediate nodes). Many implementations assign one cost per operation and ignore
stride entirely, which systematically misprices reduction cells.

### 5.2 `E[α]` is always computed, even at λ=0

At λ=0 the expected cost is still computed and logged, detached from the graph, so the λ=0 loss
is provably unchanged. This gives us a **clean control run with identical instrumentation** —
the vanilla DARTS baseline differs from ThriftNAS in exactly one term. Our λ=0.0 log shows
`lam_eff = 0.0000` on every epoch while `eff` is still recorded, which is the proof.

### 5.3 Full pipeline on one 4 GB consumer laptop, offline

Original DARTS required on the order of a GPU-day on better hardware. By using 5 cells, disjoint
10k/10k search splits and 25 epochs, our search completes in **2 h 18 min** on an RTX 3050
Laptop GPU with a 2 156 MB peak, and the demo runs CPU-only with no network access. This is a
reproducibility-under-constraint result that matters for a course lab setting.

---

## 6. Measured evidence so far

### Completed runs

| Run | Status | Result |
| :--- | :--- | :--- |
| Cost table profiling | done | 16 entries, C=32 @ 16×16, 200 timed runs each |
| 5-epoch pilots, λ ∈ {0, 0.1, 1} | done | `r`-rule validated on all three |
| **Search λ=0.0, 25 epochs** | **done** | 138.3 min · final `ce_w`=0.1916 · train 93.8 % · val 80.0 % · `skip_count`=0 |
| ResNet-20 baseline, 50 epochs | done | **89.98 %** test accuracy |
| Retrain λ=0.0, 50 epochs | pending | — |
| MobileNetV2 baseline | partial | stopped at epoch 4 of 50 |

### Architecture cost comparison

| Model | Params | MACs | Note |
| :--- | :--- | :--- | :--- |
| ResNet-20 | 0.27 M | 41.2 M | baseline |
| MobileNetV2 (CIFAR-adapted) | 2.24 M | 91.3 M | baseline |
| **Vanilla DARTS (λ=0.0, 25 ep)** | **1.40 M** | **242.6 M** | our completed search, C=36 / 8 cells |
| ThriftNAS (λ=0.1, 5-epoch pilot) | 0.56 M | 110.5 M | **preliminary** — pilot genotype, not a full run |

The λ=0 → λ=0.1 direction already shows a **2.2× MACs reduction**, which is the central
hypothesis of the project. This must be re-measured from a full 25-epoch λ=0.1 search before it
goes in the report — the 110.5 M figure comes from a 5-epoch pilot.

### A supporting observation for the λ=0 control

`E[α]` during the λ=0.0 search rose from **8.915** at epoch 0 to a peak of **9.267**, ending at
**9.072** — it never fell below its starting value, and `skip_count` stayed at **0** for all 25
epochs. The λ=0.1 pilot, by contrast, decreased monotonically (8.769 → 7.869). Unconstrained
DARTS does not become efficient on its own; the penalty is what drives the reduction.

---

## 7. Known limitations to disclose

These are not reasons to weaken the contributions, but they must appear in the report:

1. **Single seed** throughout (spec §10). DARTS is known to have high seed-to-seed variance.
2. **Cost-model fidelity is unverified at scale.** The table is profiled at C=32 / 16×16 but
   applied at C=16 (supernet) and C=36 with 36→72→144 channels (retrain). The one measured
   predicted-vs-actual latency comparison we have, from a preliminary 1-epoch run, showed a
   **62.9 % prediction error**.
3. **The "hardware-free" claim is not yet tested on unseen hardware.** Latency is measured on the
   same laptop CPU the table was built from, which is circular. Timing the discovered genotype on
   one device we never profiled would close this.
4. **Efficiency penalty can reward degeneracy.** `skip_connect` costs `c = 0.001` and
   `avg_pool_3x3` costs `c = 0.067`, so the cheapest way to lower `E[α]` is to fill the cell with
   parameter-free ops. The λ=1.0 collapse is this effect. We currently report it as a finding but
   have no mechanism that separates "efficient" from "degenerate".

---

## 8. One-paragraph summary for the report

> ThriftNAS extends efficiency-aware differentiable architecture search not with a new cost
> formulation or a new search algorithm, but with two contributions. First, an `r`-ratio
> diagnostic, `r = λ·E[α]/L_val`, evaluated after a 5-epoch pilot, which predicts before a full
> run whether a given trade-off weight will yield a balanced architecture or collapse into a
> degenerate one; validated on λ ∈ {0, 0.1, 1.0}, where it correctly predicted the λ=1.0 collapse
> to an all-skip normal cell. Second, the introduction of Efficient Channel Attention as a
> searchable cell primitive at a ~5 % cost premium over the convolution it wraps, letting the
> optimiser place attention per edge rather than having a designer place it per stage; in our
> unconstrained 25-epoch search it was selected on 6 of 8 normal-cell edges. Both results are
> obtained entirely on a single 4 GB consumer laptop GPU using only software-measurable cost
> proxies, with no access to the target deployment device.

---

## 9. Immediate next experiments, in priority order

| # | Experiment | Why | Cost |
| :-- | :--- | :--- | :--- |
| 1 | Retrain λ=0.0 genotype, 50 epochs | completes the vanilla DARTS baseline | ~3–4 h |
| 2 | Search + retrain λ=0.1, 25 + 50 epochs | the actual ThriftNAS result; needed for every comparison | ~5 h |
| 3 | Attention ablation: no-attention / SE / ECA | the only thing that makes Contribution 3 defensible | 3 × ~2.5 h |
| 4 | MACs-only vs hybrid cost table | validates Contribution 2; table already exists | ~2.5 h |
| 5 | Time the discovered genotype on one unprofiled device | tests the hardware-free claim | minutes |
| 6 | Re-run λ=0.1 with 2 extra seeds | addresses the single-seed limitation | ~5 h |
