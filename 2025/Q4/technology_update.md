Q4'25: Technology Update – Low Precision and Model Optimization
## Summary

...

## Highlights

- ...

## Papers with notable results
### Quantization
- **INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats** (https://arxiv.org/abs/2510.25602)
*The University of Hong Kong,ByteDance Seed,PicoHeart*
TODO: Nikolai Lialiushkin
- **Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free** (https://arxiv.org/pdf/2505.06708)
TODO: Nikolai Lialiushkin
- **SignRoundV2: Closing the Performance Gap in Extremely Low-Bit Post-Training Quantization for LLMs** (https://arxiv.org/pdf/2512.04746)
TODO: Nikolai Lialiushkin
- **WUSH: Near-Optimal Adaptive Transforms for LLM Quantization** (https://arxiv.org/pdf/2512.00956)
TODO: Nikolai Lialiushkin
- **STaMP: Sequence Transformation and Mixed Precision for Low-Precision Activation Quantization** (https://arxiv.org/abs/2510.26771)
TODO: Nikolai Lialiushkin

- **KVLinC: KV CACHE QUANTIZATION WITH HADAMARD ROTATION AND LINEAR CORRECTION** (https://arxiv.org/pdf/2510.05373v1)
KVLinC is a framework designed to mitigate attention errors arising from KV cache quantization. The authors integrate two complementary techniques to enable robust low-precision caching. First, through a detailed analysis of Hadamard rotation-based quantization strategies, they show that applying channel-wise quantization to raw keys and token-wise quantization to Hadamard-transformed values minimizes quantization error. Second, to address residual errors from quantized keys, they propose lightweight linear correction adapters that explicitly learn to compensate for distortions in attention. Extensive evaluation across the Llama, Qwen2.5, and Qwen3 model families demonstrates that KVLinC consistently matches or surpasses strong baselines under aggressive KV-cache compression. Finally, the authors develop a custom attention kernel that delivers up to 2.55× speedup over FlashAttention, enabling scalable, efficient, and long-context LLM inference.

### Pruning / Sparsity
- **SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference** (https://arxiv.org/pdf/2510.17777).
*NVIDIA, MIT, UC San Diego, University of Toronto*
TODO: Nikolai Lialiushkin

- **THINKV: THOUGHT-ADAPTIVE KV CACHE COMPRESSION FOR EFFICIENT REASONING MODELS** (https://arxiv.org/pdf/2510.01290v1).
ThinKV is a KV cache compression framework for LRMs on tasks like coding and mathematics. It classifies CoT tokens into Reasoning (R), Execution (E), and Transition (T) based on attention sparsity (T > R > E) using an offline calibration phase with Kernel Density Estimation to determine sparsity thresholds. The framework employs two main strategies:

  * Think Before You Quantize: assigns token precision by importance. R/E tokens use 4-bit NVFP4, T tokens use 2-bit ternary, with group quantization (g=16) and shared FP8 (E4M3) scales; keys are quantized per-channel, values per-token. Outlier Transition thoughts are recognized as vital for backtracking and preventing model loops. Token importance is measured via KL divergence of the final answer distribution when a thought segment is removed.
  * Think Before You Evict: a thought-adaptive eviction scheme aligned with PagedAttention. K-means clustering on post-RoPE key embeddings retains cluster centroids and corresponding values for evicted segments.

Experimental results show near-lossless accuracy and up to a 5.8× improvement in inference throughput, while utilizing less than 5% of the original KV cache memory. It operates at an average precision of 3.4–3.8 bits.

- **Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation** (https://arxiv.org/pdf/2505.18875).

SVG2 is a training-free sparse attention method designed to accelerate video generation in DiT-based models. Its core idea is to use semantic-aware permutation to better identify critical tokens and reduce unnecessary sparse-attention overhead.
The method introduces three key techniques:
  * Semantic-aware permutation: k-means clustering is applied to the Q/K/V vectors of each head and layer, and tokens are permuted by cluster to form semantically coherent blocks, improving the accuracy of critical-token detection.
  * Dynamic top-p critical-token selection: Cluster centroids approximate attention scores, and clusters (and their tokens) are selected until the cumulative probability reaches p, enabling dynamic compute budgeting.
  * Customized sparse-attention kernels: Since semantic-aware clusters vary naturally in size, custom kernels are used to support dynamic block sizes, which fixed-size sparse kernels cannot handle efficiently.
    
This approach achieves up to 2.30× and 1.89× speedups while maintaining PSNRs of 30 and 26 on HunyuanVideo and Wan 2.1, respectively.
Code: https://github.com/svg-project/Sparse-VideoGen

- 

### Other
- **LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation** (https://arxiv.org/pdf/2511.00090)
LeMiCa is a novel, training-free, and efficient acceleration framework for diffusion-based video generation. The key idea is to take a global view of caching error using a Global Outcome-Aware error metric, which measures the impact of each cache segment on the final output. This approach removes temporal heterogeneity and mitigates error propagation. Using this metric, LeMiCa builds a Directed Acyclic Graph (DAG) in which each edge represents a potential cache segment, weighted by its global impact on output quality. The DAG is generated offline using multiple prompts and full sampling trajectories. LeMiCa then applies lexicographic minimax optimization to choose the path that minimizes worst-case degradation. Among all feasible paths under a fixed computational budget, it selects the one with the smallest maximum error.
LeMiCa-slow achieves the highest reconstruction quality, reducing LPIPS from 0.134 → 0.05 on Open-Sora and 0.195 → 0.091 on Latte—over 2× improvement compared to TeaCache-slow. LeMiCa-fast improves inference speed from 2.60× → 2.93× on Latte relative to TeaCache-fast, while preserving visual quality. Unlike prior work that relies on online greedy strategies, LeMiCa precomputes its caching policy, eliminating runtime overhead.
Code: https://unicomai.github.io/LeMiCa.

### Software
- [vLLM-Gaudi](https://github.com/vllm-project/vllm-gaudi/releases/tag/v0.11.2) - first production-ready vLLM plugin for Gaudi, developed in close collaboration with the Intel team and fully aligned with upstream vLLM.
- KernelFalcon: Autonomous GPU Kernel Generation via Deep Agents (https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/) - a deep agent architecture for generating GPU kernels that combines hierarchical task decomposition and delegation, a deterministic control plane with early-win parallel search, grounded tool use, and persistent memory/observability. KernelFalcon is the first known open agentic system to achieve 100% correctness across all 250 L1/L2/L3 KernelBench tasks.
- [CacheDiT](https://github.com/vipshop/cache-dit/tree/main): A PyTorch-native and Flexible Inference Engine with Hybrid Cache Acceleration and Parallelism for 🤗 DiTs. It provides a unified cache API that supports features like automatic block adapters, DBCache, and more, covering almost all Diffusers’ DiT-based pipelines.
DBCache is a training-free Dual Block Caching mechanism inspired by the U-Net architecture. It splits the DiT Transformer block stack into three functional segments:
  * Probe (front): performs full computation to capture residual signals and compare them with the previous step.
  * Cache (middle): skips computation and reuses cached outputs when residual changes stay below a configurable threshold.
  * Corrector (rear): always recomputes to refine outputs and correct any accumulated deviations.

This probe → decision → correction loop enables structured, reliable caching that can be applied across DiT models without any retraining.
