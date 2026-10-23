# ThriftNAS: Project Spec (source of truth, v2)

Read this file before every task and follow it. If any code or request conflicts with it, follow this
file and tell me about the conflict. Do not invent results. Before writing code, restate in two lines
what you will build.

## 1. What the project is
ThriftNAS is a PyTorch project for a college AI/ML lab (course IS353IA) on CIFAR-10: an
efficiency-aware, first-order DARTS (Differentiable Architecture Search). The search finds a small, fast
image classifier by optimising accuracy and efficiency together, using only software-measurable proxies
(no target device).

## 2. Method
- Architecture loss: L_val(w, alpha) + lambda * E(alpha). Weights w use plain cross-entropy on the
  weight-training subset. First-order DARTS (no Hessian).
- Cost per operation: c(o) = 0.5 * lat(o)/lat_max + 0.5 * FLOPs(o)/FLOPs_max
  - lat(o) is the MEDIAN CPU latency from a lookup table that I measure myself.
  - FLOPs are reported as MACs (fvcore/thop convention); label this in every output.
- Expected cost: E(alpha) = sum over edges, sum over ops of softmax(alpha_e)[o] * c(o), summed over
  both cell types (14 normal + 14 reduction edges). E(alpha) is ALWAYS computed and logged, detached
  when lambda=0 (the lambda=0 loss must not change). Compute E(alpha) in float32.
- Search space (8 ops): sep_conv_3x3, sep_conv_5x5, dil_conv_3x3 (dilation=2), max_pool_3x3,
  avg_pool_3x3, skip_connect, eca_sep_conv_3x3 (3x3 separable conv followed by an ECA block), zero.
  An SE block is the alternative for the attention ablation.
- Cells: normal + reduction; 2 input nodes, 4 intermediate nodes, output = concat of intermediates.
- Genotype derivation: for each intermediate node keep the 2 strongest incoming edges (ranked by the
  strongest non-zero op weight) and that op on each.

## 3. Search configuration
- Supernet: 5 cells, 16 initial channels.
- Data: 10k images for weights + 10k DISJOINT images for alpha, drawn from the 50k training set with a
  fixed seed (30k unused). The 10k test set is never used in search.
- Batch size 48 (32 if out of memory). 20-30 epochs.
- Weights: SGD, lr 0.025 cosine to 0.001, momentum 0.9, weight decay 3e-4.
- Alpha: Adam, lr 3e-4, betas (0.5, 0.999), weight decay 1e-3.
- Cosine T_max = number of epochs in each script.
- Mixed precision (torch.autocast + GradScaler). Order for the weight step: scaler.unscale_(opt) ->
  clip_grad_norm_(5) -> scaler.step. Alpha loss: cast E(alpha) and CE to float32 before lambda * E(alpha).
- Lambda values: 0, 0.1, 1 as the starting plan. The final set is decided AFTER the pilot runs and then
  written back into this file.
- Per-epoch log columns: ce_w, ce_a, eff (E(alpha)), lam_eff (lambda*E(alpha)), train_acc, val_acc,
  skip_count (in the current derived genotype), epoch_time_s, peak_gpu_mb.

## 4. Cost table (latency lookup)
- Profile every op at C=32, 16x16 (middle stage), stride 1 and 2; use this table for both cell types.
- Also profile at 32x32 and 8x8; report the Spearman rank correlation of op costs across sizes.
- CPU timing: torch.set_num_threads(1), eval mode, torch.inference_mode(), 20 warm-up runs, at least 200
  timed runs, report median and std (ms). Flag any op whose std exceeds 10% of its median.
- Also save a FLOPs-only table (c(o) = FLOPs/FLOPs_max) for the ablation.
- Lookup-table latency is an approximation of true end-to-end latency; say so in console output, saved
  JSON metadata, and the report.
- Run it twice and compare; tables must agree within a few percent. Profile with other apps closed,
  charger plugged in, Windows on high performance. Do not run it while a search is running.

## 5. Retraining configuration (ThriftNAS models AND all baselines)
- C_init=36, 8-10 cells. Warn if a ThriftNAS model exceeds 1.5M parameters (warning only for ThriftNAS).
- ONE shared epoch count for all models, stored in configs/shared.yaml and read by every script. Decide
  it after timing one epoch (about 114 s per epoch on my GPU for one run: 100 epochs if it fits, else 50).
- Batch size 96 (64 if out of memory). SGD, lr 0.025 cosine, momentum 0.9, weight decay 3e-4.
- Drop-path only. NO Cutout, NO auxiliary head (keep behind flags, default off).
- Full 50k training set. Same augmentation, optimiser, epochs and seed for every model.

## 6. Baselines
- ResNet-20 (CIFAR version), MobileNetV2 (adapted to 32x32), vanilla DARTS.
- Vanilla DARTS = search with lambda=0, derive the genotype, retrain with the same recipe.

## 7. Metrics
Top-1 accuracy (final epoch, test set), parameters (M), MACs (M), model size (MB) = size of a
weights-only torch.save(model.state_dict()) file (not the resume checkpoint), CPU latency (batch 1,
1 thread, 200+ runs after warm-up, mean +/- std, ms). Also report predicted (lookup-table sum over cell
ops) vs measured end-to-end latency and the percentage error; note that the prediction excludes the
stem, pooling and classifier. Use mean +/- std over at least 3 seeds where compute allows.

## 8. Rules
- The test set is evaluated ONLY once, after the final epoch. Never per epoch, never for checkpoint
  selection or tuning. Per-epoch logs contain training metrics only.
- Hardware: local NVIDIA RTX 3050 laptop GPU, 6 GB VRAM. Not Colab. Batch size and num_workers are CLI
  flags. Catch torch.cuda.OutOfMemoryError with "Lower --batch_size (try 32)".
- Atomic checkpoint writes; resume every epoch; fixed seeds; log to CSV/JSON.
- Clean modular code, type hints, short docstrings, comments for non-obvious choices.
- Repo structure: thriftnas/{data,ops,models,trainers,utils}/ plus root scripts search.py, retrain.py,
  eval_baselines.py; configs/, tests/, results/. One package only. Do not add features, refactor, or
  change anything this file does not mention without asking me.
- Tests must check behaviour that is true. Never write a test or comment claiming a trend (for example
  "skip count increases with lambda") that a short run cannot show.

## 9. Working rules (apply to every step)
1. Do one step at a time. When it is built, run the tests AND the verification listed for that step, and
   show the real output (not a summary).
2. Do not mark anything done, PASS or MATCH that you did not run or read in the code.
3. Then STOP and wait for me to say "continue".
4. List any assumption, stub, hard-coded value or TODO you introduced.

## 10. Addendum (decisions made after the audit)
- Cost normalisation: lat_max and MACs_max are the maxima over ALL (op, stride) entries of the table, so
  stride-1 and stride-2 costs are comparable. Ops are profiled as they appear in the retrained network
  (affine=True, pooling ops WITHOUT BatchNorm).
- MACs convention counts convolution/linear multiply-adds only (pooling, BN, ReLU count as 0 MACs); latency
  captures those other costs. This is why latency and MACs are combined in c(o).
- Edge strides: normal cell = 14 stride-1 edges. Reduction cell = 8 stride-2 edges (those fed by the two input
  nodes) + 6 stride-1 edges (fed by intermediate nodes). E(alpha) uses these strides.
- Seeds: 1 seed for everything; extra seeds only for ResNet-20 and the best ThriftNAS model if time remains.
  The report must state the single-seed limitation.
- Windows: DataLoader num_workers default 2; every entry script uses `if __name__ == "__main__":`.
- Demo: offline on the laptop, CPU inference, no live search. Search is shown by replaying saved logs.
- Lambda decision rule (pilot): r = lam_eff / ce_a at the end of the 5-epoch pilot.
  r < 0.05: too weak; 0.05 <= r <= 1: useful; r > 2: cost dominates, expect collapse (report it as a finding).
