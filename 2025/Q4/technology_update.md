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

- **KVLinC: KV CACHE QUANTIZATION WITH HADAMARD ROTATION AND LINEAR CORRECTION** (https://arxiv.org/pdf/2510.05373v1)
KVLinC is a framework designed to mitigate attention errors arising from KV cache quantization. The authors integrate two complementary techniques to enable robust low-precision caching. First, through a detailed analysis of Hadamard rotation-based quantization strategies, they show that applying channel-wise quantization to raw keys and token-wise quantization to Hadamard-transformed values minimizes quantization error. Second, to address residual errors from quantized keys, they propose lightweight linear correction adapters that explicitly learn to compensate for distortions in attention. Extensive evaluation across the Llama, Qwen2.5, and Qwen3 model families demonstrates that KVLinC consistently matches or surpasses strong baselines under aggressive KV-cache compression. Finally, the authors develop a custom attention kernel that delivers up to 2.55× speedup over FlashAttention, enabling scalable, efficient, and long-context LLM inference.

### Pruning / Sparsity
- **SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference** (https://arxiv.org/pdf/2510.17777).
*NVIDIA, MIT, UC San Diego, University of Toronto*
TODO: Nikolai Lialiushkin

### Other
- ...

### Software
- KernelFalcon: Autonomous GPU Kernel Generation via Deep Agents (https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/) - a deep agent architecture for generating GPU kernels that combines hierarchical task decomposition and delegation, a deterministic control plane with early-win parallel search, grounded tool use, and persistent memory/observability. KernelFalcon is the first known open agentic system to achieve 100% correctness across all 250 L1/L2/L3 KernelBench tasks.
