Part 1: Gaps in Existing Literature that ThriftNAS Successfully Addressed
#	Existing Literature Gap (Prior Art)	How ThriftNAS Bridges the Gap
1	Physical Hardware Dependency in HW-NAS
Prior hardware-aware methods (FBNet, ProxylessNAS, Single-Path NAS) require connecting target physical devices (e.g. Raspberry Pi, Android phone) via USB during the search, making NAS expensive and unscalable.	Hardware-Free Empirical Modeling:
Constructs an offline normalized 2D cost table (
𝑐
(
𝑜
)
=
𝜇
⋅
lat
+
(
1
−
𝜇
)
⋅
MACs
c(o)=μ⋅lat+(1−μ)⋅MACs). Enables searching on any standard workstation/laptop GPU without tethering physical edge devices.
2	Performance Collapse from Performance Collapse / Skip-Overload
Standard DARTS suffers from a well-known failure mode where zero-parameter skip connections dominate early in search, ruining representation capacity.	Combined Cost Normalization & 
𝑟
r-Ratio Diagnostic:
Normalizing 
𝑐
(
𝑜
)
c(o) across both strides prevented early operator collapse. Introduced the 
𝑟
=
𝜆
⋅
eff
ce_a
r=
ce_a
λ⋅eff
	​

 pilot diagnostic to verify search health in 5 epochs.
3	Search Spaces Restricted to Conventional Convolutions
Existing DARTS benchmarks (DARTS, PC-DARTS, FairDARTS) solely search over standard convolutions and pooling, completely ignoring attention mechanisms.	Attention-Augmented Primitive Space:
Integrated Efficient Channel Attention (eca_sep_conv_3x3) into the continuous search space, proving that channel attention can be selected under efficiency constraints without exceeding parameter limits.
Part 2: Open Research Gaps & Limitations (Future Scope)

These are the areas where ThriftNAS can be extended for future publications or a master's thesis:

1. The Supernet Discretization Gap (Evaluation Gap)
The Gap: In DARTS, the supernet trains with continuous, weighted mixtures of all operations (
∑
softmax
(
𝛼
𝑖
)
⋅
𝑜
(
𝑥
)
∑softmax(α
i
	​

)⋅o(x)). When the search ends, we discretize the cell by abruptly keeping only the top-2 operations per node.
Impact: Weights trained in a cooperative continuous mixture perform slightly differently when isolated into a discrete standalone network.
Future Solution: Introducing a temperature-annealed Gumbel-Softmax or continuous-to-discrete bridging schedule during the final 5 search epochs.
2. Cross-Architecture Operator Profiling vs. True End-to-End Latency
The Gap: ThriftNAS computes predicted latency as a linear sum over individual operation lookups (
∑
lat
(
𝑜
)
∑lat(o)).
Impact: In actual hardware deployment, memory access overhead (cache misses, kernel launch latencies, memory bandwidth bottlenecks across cell concatenations) is non-linear. The lookup sum underestimates end-to-end latency by 15–25%.
Future Solution: Neural Latency Predictor (a small MLP trained on full subgraph benchmarks) or graph-level execution graph simulators.
3. Macro-Architecture vs. Micro-Architecture Search
The Gap: ThriftNAS uses cell-based (micro) search, where one normal cell and one reduction cell are discovered and then repeated 
𝑁
N times.
Impact: Edge devices have different memory constraints at early layers (high spatial resolution, large activation memory) versus late layers (deep channels, large weight memory). Repeating the identical cell across all layers is not strictly optimal for memory bandwidth.
Future Solution: Layer-wise hierarchical search (searching different cells for low-level, mid-level, and high-level stages).
4. Fixed 
𝜆
λ vs. Dynamic Curriculum Regularization
The Gap: Currently, 
𝜆
λ is held constant throughout the 25 epochs. At high 
𝜆
λ values (e.g. 
𝜆
=
1.0
λ=1.0), applying strong penalty at Epoch 0 causes the network to abandon convolutions before it even learns basic image features.
Future Solution: 
𝜆
λ-Warmup (Curriculum NAS):
𝜆
(
𝑡
)
=
𝜆
target
⋅
(
𝑡
𝑇
warmup
)
λ(t)=λ
target
	​

⋅(
T
warmup
	​

t
	​

)
Letting 
𝜆
=
0
λ=0 for the first 5 epochs allows feature extraction to establish before efficiency pruning begins.
5. Dataset & Task Generalization
The Gap: Search was conducted on CIFAR-10 (
32
×
32
32×32 pixels, 10 classes).
Future Solution: Transferability benchmarking—taking the discovered genotype from CIFAR-10 and testing its transfer learning performance on ImageNet-100, Pascal VOC, or edge medical imaging tasks.