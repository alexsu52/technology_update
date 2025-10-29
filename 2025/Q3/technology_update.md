Q3'25: Technology Update – Low Precision and Model Optimization
## Summary

TBD

## Highlights

- **Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization** ([https://arxiv.org/pdf/2509.23202](https://arxiv.org/pdf/2509.23202))
*ISTA, Red Hat AI, Yandex Research, ETH Zürich*

Authors provide the rigorous analysis of FP4 microscaling formats (MXFP4 and NVFP4) for LLM quantization, introduce Micro-Rotated-GPTQ (MR-GPTQ) and the QuTLASS GPU kernels to bridge the performance gap for such formats. The method uses MSE-optimized grids, static activation reordering, and fused online Hadamard rotations to recover 98-99% of FP16 accuracy while achieving up to 4x inference speedups on modern GPUs. Key insights from the analysis include:
    - The effectiveness of Hadamard transforms is dependent on the quantization group size; while they are beneficial for MXFP4 and INT4, they can actually degrade NVFP4 accuracy.
    - MR-GPTQ consistently improves the accuracy of the lower-performing MXFP4 format, bringing it within 1-2% of NVFP4's performance.
    - On average, NVFP4 and INT4 (with group size 32) offer similar quality.
    - MXFP4 kernels may achieve ~15% higher throughput than NVFP4 on B200 GPUs, likely due to simpler hardware implementation.

<p align="center"><img width="100%" height="100%" src="./figures/MR_GPTQ.png"></p><br/>

- **Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation** ([https://www.arxiv.org/pdf/2506.19852](https://www.arxiv.org/pdf/2506.19852)).

*MIT NVIDIA Princeton UC Berkeley Stanford First Intelligence*

The paper "Radial Attention" introduces a sparse attention mechanism to optimize long video generation. Its core method reduces computational complexity from $O(n^2)$ to $O(n \log n)$ using a static mask inspired by "Spatiotemporal Energy Decay," where attention focuses on spatially and temporally closer tokens. This architecture is highly optimized for inference. It delivers up to a **3.7x speedup** on extended-length videos compared to standard dense attention, without any discernible loss in visual quality. For a concrete 500-frame, 720p video, the mechanism slashes the raw attention computation by a factor of 9x. The industrial impact is significant. Designed as a "plug-and-play" module, Radial Attention can be integrated into powerful pre-trained models like Wan2.1-14B and HunyuanVideo through efficient LoRA-based fine-tuning.


<p align="center"><img width="100%" height="100%" src="./figures/Radial_Attention.png"></p><br/>

- **Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution** ([https://arxiv.org/pdf/2510.00636](https://arxiv.org/pdf/2510.00636))
TODO: Liubov Talamanova

## Papers with notable results
### Quantization

- **Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution** ([https://arxiv.org/pdf/2510.00636](https://arxiv.org/pdf/2510.00636))
TODO: Liubov Talamanova

- **SINQ: Sinkhorn-Normalized Quantization for Calibration-Free Low-Precision LLM Weights** ([https://arxiv.org/abs/2509.22944](https://arxiv.org/abs/2509.22944))
*Huawei*

Authors introduce novel data-free post-training quantization method - SINQ. Instead of the traditional single-scale approach, SINQ employes dual-scales: one for row, another for column. The method adapts the Sinkhorn-Knopp algorithm to normalize the standard deviations of matrix rows and columns. The aglorithm is lightweight - operates at only 1.1x the runtime of basic RTN. The method proves robust across model scales, from small 0.6B parameter models to massive 235B parameter Mixture-of-Experts architectures. SINQ demonstrates orthogonality to other quantization advances. When combined with non-uniform quantization levels (NF4) or activation-aware calibration (A-SINQ with AWQ), it provides additional improvements.

<p align="center"><img width="100%" height="100%" src="./figures/SINQ.png"></p><br/>

- **70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float** ([https://arxiv.org/pdf/2504.11651](https://arxiv.org/pdf/2504.11651))
*Department of Computer Science, Rice University, Department of Computer and Data Sciences, Case Western Reserve University*

The paper presents DFloat11, a dynamic‐length float encoding scheme that exploits the low entropy of BFloat16 weights in large language models to achieve ~30% storage savings (reducing from 100% → ~70% size) without any loss in accuracy (bit‐for‐bit identical outputs). They do this by frequency‐based variable‐length coding of weight values, and couple it with a custom GPU decompression kernel allowing efficient inference. Experiments on large LLMs show major throughput gains and extended context length under fixed GPU memory budgets, making deployment more practical on resource‐constrained hardware.

- **XQUANT: Breaking the Memory Wall for LLM Inference with KV Cache Rematerialization** ([https://arxiv.org/pdf/2508.10395](https://arxiv.org/pdf/2508.10395))
TODO: Liubov Talamanova

- **Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models** ([https://papers-pdfs.assets.alphaxiv.org/2504.04823v2.pdf](https://papers-pdfs.assets.alphaxiv.org/2504.04823v2.pdf))
TODO: Nikolai Lialiushkin

- **KLLM: Fast LLM Inference with K-Means Quantization** ([https://papers-pdfs.assets.alphaxiv.org/2507.23035v3.pdf](https://papers-pdfs.assets.alphaxiv.org/2507.23035v3.pdf))
TODO: Nikolai Lialiushkin

- **Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models**([https://papers-pdfs.assets.alphaxiv.org/2503.22879v3.pdf](https://papers-pdfs.assets.alphaxiv.org/2503.22879v3.pdf))
TODO: Nikolai Lialiushkin

- **Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization** ([https://papers-pdfs.assets.alphaxiv.org/2505.11695v2.pdf](https://papers-pdfs.assets.alphaxiv.org/2505.11695v2.pdf))
TODO: Nikolai Lialiushkin

### Pruning / Sparsity
- **PagedEviction: Structured Block-wise KV Cache Pruning for Efficient Large Language Model Inference** ([https://arxiv.org/pdf/2509.04377](https://arxiv.org/pdf/2509.04377))
TODO: Liubov Talamanova

- **REAP: One-Shot Pruning for Trillion-Parameter Mixture-of-Experts Models** ([https://www.cerebras.ai/blog/reap](https://www.cerebras.ai/blog/reap))
TODO: Nikolai Lialiushkin

- **The Unseen Frontier: Pushing the Limits of LLM Sparsity with Surrogate-Free ADMM** ([https://papers-pdfs.assets.alphaxiv.org/2510.01650v1.pdf](https://papers-pdfs.assets.alphaxiv.org/2510.01650v1.pdf))
TODO: Nikolai Lialiushkin

### Other
- **Stop Spinning Wheels: Mitigating LLM Overthinking via Mining Patterns for Early Reasoning Exit** ([https://arxiv.org/pdf/2508.17627](https://arxiv.org/pdf/2508.17627))
TODO: Liubov Talamanova

### Software
- ...
