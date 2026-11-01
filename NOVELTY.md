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

### ⚠️ Status update — the rule failed its first out-of-sample test

**Read this before using the claim above.** When the full 25+50-epoch runs for every λ became
available on `main`, the rule's prediction for λ=0.1 turned out to be wrong:

| λ | pilot `r` | pilot `skip_count` | rule's prediction | **actual full-run result** | correct? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.0 | 0.000 | 0 | no effect, heavy convs | 93.39 % @ 242.6 M MACs | ✅ |
| 0.01 | **0.086** | **2** | "useful, balanced" | 93.38 % @ 223.6 M MACs — but only **8 % MACs saved** | ⚠️ |
| **0.1** | **0.755** | **11** | **"useful, balanced"** | **82.38 % @ 101.1 M MACs — 11 points lost** | ❌ |
| 1.0 | 6.611 | 8 (all-skip normal cell) | collapse | all-skip / all-avgpool cell | ✅ |

At λ=0.1 the rule said *useful*, but the architecture lost **11 accuracy points** against λ=0.0
for a 2.4× MACs reduction. That is not a balanced trade-off, so the `0.05 ≤ r ≤ 1.0` band is
drawn in the wrong place.

What the real data shows is that the usable band for λ sits **between 0.01 and 0.1**, and it is
narrow: λ=0.01 costs us only **0.01 points** of accuracy while λ=0.1 costs **11 points**.

### ✅ Repair experiment — completed

We ran the missing 5-epoch pilot at **λ=0.01** (28 min, `results/search_pilot_lam0.01/`) as an
out-of-sample test, because its full-run outcome (93.38 %) was already known. Per-epoch trace:

| epoch | `ce_a` | `E[α]` | `λ·E[α]` | `skip_count` | `val_acc` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 1.8187 | 8.8949 | 0.0889 | 3 | 40.7 % |
| 1 | 1.4659 | 8.8941 | 0.0889 | 1 | 46.5 % |
| 2 | 1.2669 | 8.8893 | 0.0889 | 0 | 56.2 % |
| 3 | 1.1356 | 8.8899 | 0.0889 | 0 | 60.7 % |
| **4** | **1.0389** | **8.8948** | **0.0889** | **2** | **64.5 % → r = 0.0856** |

### Three findings

**(a) `r` does separate the cases — the thresholds were simply drawn in the wrong place.**

```
r = 0.086  →  93.38 %  (good accuracy)
r = 0.755  →  82.38 %  (11 points lost)
r = 6.611  →  degenerate all-skip cell
```

The gap between the good case and the bad case is **~9×**, which is a wide and clean separation.
But the originally stated band, `0.05 ≤ r ≤ 1.0 = useful`, puts **both 0.086 and 0.755 inside
it** — which is exactly why it mis-predicted λ=0.1. The band is far too wide.

**(b) λ=0.01 is not actually a good trade-off — it is almost no trade-off.**

Its derived pilot genotype is essentially the same *character* as λ=0.0's: heavy convolutions,
dominated by `sep_conv_5x5`, with no skips in the normal cell.

| λ | pilot normal cell | pilot reduce cell |
| :--- | :--- | :--- |
| 0.0 | 4× `sep_conv_5x5`, 2× `dil_conv_3x3`, 2× `eca_sep_conv_3x3` | 4× `sep_conv_5x5`, 2× `max_pool`, … |
| **0.01** | **4× `sep_conv_5x5`, 2× `sep_conv_3x3`, 1× `dil_conv_3x3`, 1× `max_pool`** | **6× `max_pool`, 2× `skip_connect`** |

This matches the full-run numbers exactly: 93.39 % → 93.38 % accuracy and only 242.6 → 223.6 M
MACs. At `r = 0.086` the penalty is **too weak to change the architecture**, so the correct
verdict for λ=0.01 is *"too weak"*, not *"useful"*. That moves the lower boundary up: the
original `r < 0.05 = too weak` rule should read roughly **`r ≲ 0.1 = too weak`**.

**(c) Pilot `skip_count` is a sharper discriminator than `r` itself.**

```
λ=0.0   skip=0   → 93.39 %
λ=0.01  skip=2   → 93.38 %
λ=0.1   skip=11  → 82.38 %     ← jumps 2 → 11 exactly where accuracy collapses
λ=1.0   skip=8   → degenerate (all 8 normal-cell edges are skips)
```

`skip_count` in the 5-epoch derived genotype moves from **2 to 11** precisely across the
good/bad boundary, and it needs no threshold calibration at all — it directly counts how many
learnable operations the penalty has already removed. This is the "second signal" the diagnostic
was missing.

### Revised diagnostic (evidence-based)

> Run a 5-epoch pilot. Read **both** `r = λ·E[α]/L_val` **and** `skip_count` of the derived
> genotype.
>
> | Condition | Verdict |
> | :--- | :--- |
> | `r ≲ 0.1` **and** `skip_count ≤ 2` | **too weak** — architecture will barely differ from λ=0 |
> | `skip_count ≥ 8` | **collapsing** — expect severe accuracy loss, do not commit GPU hours |
> | `0.1 < r < 0.5` **and** `3 ≤ skip_count ≤ 6` | **candidate useful band — UNVERIFIED, no λ sampled here yet** |
> | `r > 2` | **degenerate** — cells become parameter-free |

### What this means for the project

The diagnostic survives, with measured boundaries instead of guessed ones, plus a second signal
that is strictly more informative than the original scalar. **But the band it now points at
(`0.1 < r < 0.5`) has never been sampled.** Translating back to λ, that is roughly
**λ ∈ [0.02, 0.05]** — which is precisely the sweep listed as experiment #2 below, and the reason
that experiment is the project's highest priority.

Framing for the report: *"we proposed a cheap screening diagnostic, tested it out-of-sample
against a known outcome, found its stated thresholds wrong, re-derived them from measurement,
and identified a strictly better second signal."* That is a stronger and more honest
contribution than a rule validated only on the three points it was fitted to.

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

### Ablation run 1 — λ=0.01: inconclusive, and that is itself informative

We ran a 5-epoch pilot with `cost_table_flops_only.json` at λ=0.01, holding λ, seed, epochs and
batch size identical to the hybrid pilot — only the cost table differed
(`results/search_pilot_lam0.01_flopsonly/`).

**Prediction tested**: the MACs-only table prices pooling at exactly 0, so it should select
*more* pooling/skip operations than the hybrid table.

| | Hybrid (lat + MACs) | MACs-only |
| :--- | :--- | :--- |
| normal cell | 4× `sep_conv_5x5`, 2× `sep_conv_3x3`, 1× `dil_conv_3x3`, 1× `max_pool` | 3× `sep_conv_5x5`, **5× `dil_conv_3x3`** |
| reduce cell | 6× `max_pool`, 2× `skip_connect` | 6× `max_pool`, 1× `sep_conv_3x3`, 1× `dil_conv_3x3` |
| pooling ops | 7/16 | 6/16 |
| skip ops | 2/16 | 0/16 |
| **cheap ops total** | **9/16** | **6/16** |
| MACs @ C=36, 8 cells | 244.99 M | 225.55 M |

**The prediction did not hold** — MACs-only selected *fewer* cheap operations, not more.

**Why**: at λ=0.01 the penalty is simply too weak for the cost table to matter.
`λ·E[α] = 0.079` against `ce_a = 1.029`, i.e. the penalty is **7.7 %** of the loss, so the alpha
gradient is dominated almost entirely by cross-entropy. The difference between the two genotypes
is search noise, not a cost-model effect.

This **independently confirms the §2 finding** that λ=0.01 deserves the verdict *"too weak"*: at
that setting even the choice of cost model is irrelevant. Two contributions arriving at the same
conclusion from different directions is a useful consistency check.

**Conclusion**: the ablation has to be run at a λ where the penalty actually bites. We re-ran it
at λ=0.1 (`r = 0.755`), where a hybrid pilot already exists for comparison.

### Ablation run 2 — λ=0.1: completed, and the result is decisive

`results/search_pilot_lam0.1_flopsonly/` against the existing `results/search_pilot_lam0.1/`.
Same λ, seed, epochs, batch size — only the cost table differs.

**Final derived genotypes:**

| | Hybrid (lat + MACs) | MACs-only |
| :--- | :--- | :--- |
| normal cell | 6× `skip_connect`, 1× `max_pool`, 1× `dil_conv_3x3` | 5× `skip_connect`, 3× `max_pool` |
| reduce cell | 5× `skip_connect`, 3× `max_pool` | 5× `skip_connect`, 3× `max_pool` |
| skip ops | 11/16 | 10/16 |
| pooling ops | 4/16 | 6/16 |
| **cheap ops total** | **15/16** | **16/16** |
| **learnable ops** | **1/16** | **0/16** |
| params | 0.5552 M | 0.4952 M |
| MACs | 110.47 M | 101.09 M |

### ⚠️ Correction to an intermediate reading

Mid-run, the per-epoch `skip_count` trace (hybrid 12/12/11 vs MACs-only 6/8/9) suggested that the
hybrid table caused *more* degeneracy. **The final genotypes show that was misleading.** MACs-only
did not avoid degeneracy — it simply moved it from `skip_connect` into `max_pool`, ending with
**16/16 cheap operations and zero learnable operations**, which is worse than hybrid's 15/16.

### Finding 1 — the cost table composition is not the lever; λ is

At λ=0.01 both tables leave the architecture essentially untouched (§ above). At λ=0.1 **both
tables collapse the cell almost completely** — one learnable operation in hybrid, none at all in
MACs-only. The choice between them moves `cheap_total` by exactly one operation out of sixteen.

**This does not support Contribution 2 as originally stated.** The claim was that including
latency matters because MACs prices pooling at zero. Measured across the two λ values we have,
the composition of the cost table makes almost no difference to the discovered architecture; λ
dominates entirely. Contribution 2 should therefore be reported as **a reasoned design choice
that our ablation could not show to be consequential**, not as a validated contribution.

### Finding 2 — `skip_count` is an incomplete degeneracy metric

This is a direct refinement of §2's Finding (c). `skip_count` only counts identity edges, so it
**undercounts** a collapse that routes into pooling instead:

| | `skip_count` | **learnable ops** |
| :--- | :--- | :--- |
| Hybrid λ=0.1 | 11/16 | **1/16** |
| MACs-only λ=0.1 | 10/16 (looks better!) | **0/16** (actually worse) |

The correct signal is **the number of learnable operations in the derived genotype** — i.e.
`16 − (pool + skip + zero)`. It is threshold-free, counts every form of degeneracy, and in this
comparison it ranks the two runs correctly where `skip_count` ranks them backwards.

**Revised diagnostic, final form:**

> | Condition | Verdict |
> | :--- | :--- |
> | `r ≲ 0.1` **and** `learnable ≥ 14/16` | **too weak** — architecture ≈ λ=0 |
> | `learnable ≤ 4/16` | **collapsing** — expect severe accuracy loss |
> | `0.1 < r < 0.5` **and** `8 ≤ learnable ≤ 13` | **candidate band — still UNSAMPLED** |
> | `learnable = 0` | **fully degenerate** — network cannot learn |

### Finding 3 — this explains the 82.38 % result mechanically

The λ=0.1 model that scored 82.38 % on `main` has 0.4952 M params and 101.09 M MACs — **exactly
matching our MACs-only pilot genotype**. Independent of cost table, λ=0.1 converges to a cell with
no learnable operations. Its 82.38 % therefore comes almost entirely from the stem, the per-cell
1×1 preprocessing convolutions and the classifier, not from the searched cell at all.

That is the real explanation for why it loses to ResNet-20: **there is no searched network left to
speak of.**

### What the project should actually fix

Not the cost table — the **cost floor**. `skip_connect` at `c = 0.001` and pooling at `c = 0`
mean the cheapest route to a low `E[α]` is always to empty the cell. Two concrete options:

1. Exclude parameter-free operations from the cost reward entirely (price them at the mean cost of
   the learnable ops), or
2. Add a hard constraint: reject genotypes with fewer than *k* learnable operations.

Either makes the penalty push toward *cheap learnable* operations such as `dil_conv_3x3`
(`c = 0.311`, 352 K MACs) instead of toward nothing at all.

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

### Full results, all models (from `origin/main`, all 50-epoch retrains)

| Model | Test Acc | Params | MACs | Size | CPU latency | Pred. latency error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ResNet-20 | 89.98 % | 0.27 M | 41.2 M | — | 2.60 ms | — |
| MobileNetV2 (CIFAR-adapted) | 92.01 % | 2.24 M | 91.3 M | 8.77 MB | 7.47 ms | — |
| Vanilla DARTS (λ=0.0) | **93.39 %** | 1.40 M | 242.6 M | 5.65 MB | 40.96 ms | 69.6 % |
| ThriftNAS (λ=0.01) | **93.38 %** | 1.28 M | 223.6 M | 5.15 MB | 43.10 ms | 72.7 % |
| ThriftNAS (λ=0.1) | **82.38 %** | 0.50 M | 101.1 M | 1.94 MB | 10.58 ms | 95.3 % |

### What this table actually says — read carefully

This is **not** the clean efficiency/accuracy trade-off the project set out to demonstrate:

- **λ=0.01 is almost free.** It costs **0.01 accuracy points** (93.39 → 93.38) and saves only
  **8 % of MACs** (242.6 → 223.6 M). Nearly no movement in either direction.
- **λ=0.1 is too expensive.** It saves **2.4× MACs** (242.6 → 101.1 M) but costs **11 accuracy
  points** (93.39 → 82.38 %). At 82.38 % it is **beaten by plain ResNet-20 (89.98 %)**, which
  uses **2.5× fewer MACs** still — so the λ=0.1 architecture is not on the Pareto front at all.
- **The usable band for λ lies between 0.01 and 0.1 and is narrow**, and we have not sampled it.
  No λ value we have tried yet produces a point that dominates the baselines.

**The single most valuable missing experiment is therefore a λ sweep inside (0.01, 0.1)** — e.g.
λ ∈ {0.02, 0.03, 0.05} — to find the knee of the curve. Without it, the project's central claim
has no supporting data point.

### Seed stability (an accidental but useful result)

λ=0.0 was retrained twice, independently, with the same genotype and recipe:

| Run | Test Acc |
| :--- | :--- |
| `origin/main` | 93.39 % |
| our re-run (`feat/baselines-darts`) | 93.13 % |

**Spread: 0.26 points.** This is a real, if small, measurement of seed variance and it partially
answers limitation #1 below. Note it in the report rather than discarding it as duplicated work.

### Our own run details (λ=0.0, for reference)

- Search, 25 epochs: **138.3 min** · final `ce_w`=0.1916 · train 93.8 % · val 80.0 % ·
  `skip_count`=0 on every epoch
- Retrain, 50 epochs: **83.8 min** · final train acc 94.0 % vs test 93.13 % (**0.9 % gap**, no
  overfitting) · LR annealed to exactly 0.0 · drop-path ramped 0 → 0.2 as designed
- Hardware: RTX 3050 Laptop (4 GB), peak 2 156 MB during search and 1 866 MB during retrain

### A supporting observation for the λ=0 control

`E[α]` during the λ=0.0 search rose from **8.915** at epoch 0 to a peak of **9.267**, ending at
**9.072** — it never fell below its starting value, and `skip_count` stayed at **0** for all 25
epochs. The λ=0.1 pilot, by contrast, decreased monotonically (8.769 → 7.869). Unconstrained
DARTS does not become efficient on its own; the penalty is what drives the reduction.

---

## 7. Known limitations to disclose

These are not reasons to weaken the contributions, but they must appear in the report:

1. **Single seed** throughout (spec §10). DARTS is known to have high seed-to-seed variance.
2. **The cost model's latency prediction is badly wrong, on every single model.** The table is
   profiled at C=32 / 16×16 but applied at C=16 (supernet) and C=36 with 36→72→144 channels
   (retrain). Measured prediction errors across all three completed models:

   | Model | Predicted | Measured | Error |
   | :--- | :--- | :--- | :--- |
   | λ=0.0 | 12.46 ms | 40.96 ms | **69.6 %** |
   | λ=0.01 | 11.75 ms | 43.10 ms | **72.7 %** |
   | λ=0.1 | 0.49 ms | 10.58 ms | **95.3 %** |

   The error is not a one-off and it grows as the architecture gets cheaper — at λ=0.1 the table
   predicts 0.49 ms against a measured 10.58 ms, a **21× underestimate**. The reason is that the
   additive sum over cell ops omits the stem, preprocessing 1×1 convolutions, global pooling and
   classifier, and those fixed costs dominate once the cell itself becomes cheap. **Any claim
   that we optimise latency must be qualified by this**, because the quantity being minimised is
   not a faithful proxy for the quantity we actually measure.
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
> formulation or a new search algorithm, but with two contributions, both evaluated entirely on a
> single 4 GB consumer laptop GPU using only software-measurable cost proxies and no access to a
> target deployment device. First, a cheap screening diagnostic for the efficiency trade-off
> weight: `r = λ·E[α]/L_val`, computed after a 5-epoch pilot rather than a full 75-epoch
> search-and-retrain cycle. It correctly anticipates the degenerate all-skip collapse at λ=1.0,
> but we also show where it fails — at λ=0.1 it reports a balanced trade-off while the full run
> loses 11 accuracy points — and we characterise the additional signal needed to separate those
> cases. Second, the introduction of Efficient Channel Attention as a searchable cell primitive
> at a ~5 % cost premium over the convolution it wraps, letting the optimiser place attention per
> edge rather than having a designer place it per stage; in our unconstrained 25-epoch search it
> was selected on 6 of 8 normal-cell edges. We additionally report that the additive lookup-table
> latency proxy underestimates measured end-to-end latency by 70–95 % across all discovered
> models, and attribute this to fixed stem and classifier costs that the per-op sum omits.

---

## 9. Immediate next experiments, in priority order

| # | Experiment | Why | Cost | Status |
| :-- | :--- | :--- | :--- | :--- |
| — | Retrain λ=0.0, 50 epochs | vanilla DARTS baseline | ~1.5 h | ✅ done (93.39 % / 93.13 %) |
| — | Search + retrain λ=0.1 and λ=0.01 | standard pipeline | ~10 h | ✅ done (Darshil) |
| **1** | **5-epoch pilot at λ=0.01** | out-of-sample test of the `r`-rule against a known-good outcome | ~28 min | ✅ **done** — `r`=0.086, thresholds re-derived, `skip_count` found to be a better signal (see §2) |
| **2** | **λ sweep in (0.01, 0.1): λ ∈ {0.02, 0.03, 0.05}** | **the project has no Pareto-winning point yet — this is the highest-value experiment.** The revised diagnostic now points directly at this band | 3 × ~3.7 h | ⬜ not started |
| 3 | Attention ablation: no-attention / SE / ECA | the only thing that makes Contribution 3 defensible | 3 × ~2.5 h | ⬜ not started — needs a `--search_space` flag added to `search.py` |
| **4** | **MACs-only vs hybrid cost table, λ=0.01, 5-epoch pilot** | validates Contribution 2; controlled ablation — same λ, seed and epochs, only the cost table differs | ~28 min | 🔄 **running now** |
| 5 | Time a discovered genotype on one unprofiled device | tests the hardware-free claim; also probes the 95 % latency error | minutes | ⬜ not started |
| 6 | Add the stem/head fixed cost to `predicted_latency()` | would likely fix most of the 69–95 % prediction error | ~1 h coding | ⬜ not started |

### Note for Darshil

Experiment **2** is the one that matters most for the report. Right now our best efficient model
(λ=0.1, 82.38 %) is **worse than plain ResNet-20 (89.98 %) at 2.5× the MACs**, so we cannot yet
claim ThriftNAS finds a better accuracy/efficiency trade-off than a hand-designed baseline. A λ
somewhere around 0.02–0.05 is where that point should exist, and nobody has run it. If you have
GPU time free, that sweep is the best use of it.
