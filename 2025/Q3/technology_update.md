Q3'25: Technology Update – Low Precision and Model Optimization
## Summary 

TBD

## Highlights
- **Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation** ([https://www.arxiv.org/pdf/2506.19852](https://www.arxiv.org/pdf/2506.19852)).

*MIT NVIDIA Princeton UC Berkeley Stanford First Intelligence*

The paper "Radial Attention" introduces a sparse attention mechanism to optimize long video generation. Its core method reduces computational complexity from $O(n^2)$ to $O(n \log n)$ using a static mask inspired by "Spatiotemporal Energy Decay," where attention focuses on spatially and temporally closer tokens. This architecture is highly optimized for inference. It delivers up to a **3.7x speedup** on extended-length videos compared to standard dense attention, without any discernible loss in visual quality. For a concrete 500-frame, 720p video, the mechanism slashes the raw attention computation by a factor of 9x. The industrial impact is significant. Designed as a "plug-and-play" module, Radial Attention can be integrated into powerful pre-trained models like Wan2.1-14B and HunyuanVideo through efficient LoRA-based fine-tuning.  

## Papers with notable results 
### Quantization
- **70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float** ([https://arxiv.org/pdf/2504.11651](https://arxiv.org/pdf/2504.11651))

*Department of Computer Science, Rice University, Department of Computer and Data Sciences, Case Western Reserve University*

The paper presents DFloat11, a dynamic‐length float encoding scheme that exploits the low entropy of BFloat16 weights in large language models to achieve ~30% storage savings (reducing from 100% → ~70% size) without any loss in accuracy (bit‐for‐bit identical outputs). They do this by frequency‐based variable‐length coding of weight values, and couple it with a custom GPU decompression kernel allowing efficient inference. Experiments on large LLMs show major throughput gains and extended context length under fixed GPU memory budgets, making deployment more practical on resource‐constrained hardware.

### Pruning / Sparsity
- ...

### Other 
- ...

### Software
- ...
