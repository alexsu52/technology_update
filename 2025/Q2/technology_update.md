Q2'25: Technology Update – Low Precision and Model Optimization
## Summary 
TBD
## Highlights
- **MMInference: Accelerating Pre-filling for Long-Context Visual Language Models via Modality-Aware Permutation Sparse Attention** ([https://arxiv.org/pdf/2502.02631](https://arxiv.org/pdf/2504.16083v1)).

The authors introduce MMInference (Multimodality Million tokens Inference), a **dynamic sparse attention method** that accelerates the prefilling stage for long-context multi-modal inputs. The core ideas stem
from analyzing the attention patterns specific to multi-modal inputs in VLMs: (1) Visual inputs exhibit strong temporal and spatial locality, leading to a unique sparse pattern the authors term the "Grid pattern".
(2) Attention patterns differ significantly within a modality versus across modalities. The authors introduces the permutation-based method for offline searching the optimal sparse patterns for each head based on the input and 
optimized kernels to compute attention much faster. MMInference speeds up the VLM pre-filling stage by up to **8.3x** (at 1 million tokens) **without losing accuracy** and **without needing any model retraining**.
The paper demonstrates maintained performance across various multi-modal benchmarks (like Video QA and Captioning) using state-of-the-art models (LongVila, LlavaVideo, VideoChat-Flash, Qwen2.5-VL).
The code is available at [https://aka.ms/MMInference](https://aka.ms/MMInference).

<p align="center">
  <img width="640" height="568" src="https://github.com/user-attachments/assets/c39d2f43-7198-435a-8679-8a145a78d0a8">
</p>

- **The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs** ([https://arxiv.org/pdf/2504.17768](https://arxiv.org/pdf/2504.17768)).

The authors introduce SparseFrontier, a systematic evaluation of dynamic sparse attention methods aimed at accelerating inference in LLMs for long-context inputs (up to 128K tokens). The core ideas stem from an extensive analysis of sparse attention trade-offs across different inference stages, model scales, and task types: (1) Sparse attention during decoding tolerates higher sparsity than during prefilling, particularly in larger models, due to differences in memory and compute bottlenecks.
(2) No single sparse pattern is optimal across all tasks—retrieval, aggregation, and reasoning tasks each require different units of sparsification (e.g., blocks vs. tokens) and budget strategies.
During prefilling, the best sparsification structure (e.g., blocks or verticals and slashes) is task-dependent, with uniform allocation across layers performing comparably to dynamic allocation.
During decoding, page-level Quest excels by preserving the KV cache structure, avoiding the performance degradation associated with token pruning during generation.
Their isoFLOPS analysis shows that for long contexts, larger sparse models outperform smaller dense ones at the same compute cost. They also establish scaling laws predicting accuracy from model size, sequence length, and compression ratio.
The code is available at: [https://github.com/PiotrNawrot/sparse-frontier](https://github.com/PiotrNawrot/sparse-frontier).

- **XAttention: Block Sparse Attention with Antidiagonal Scoring** ([https://arxiv.org/pdf/2503.16428](https://arxiv.org/pdf/2503.16428)).

The authors introduce XAttention, a training-free block sparse attention method that accelerates the prefill stage of long-context Transformers using a novel antidiagonal scoring strategy. By summing attention values along antidiagonals, XAttention efficiently identifies important blocks while capturing key vertical and slash patterns.
It achieves up to 13.5× prefill speedup at 256K tokens, maintaining accuracy comparable to full attention across benchmarks in language (RULER, LongBench), video understanding (VideoMME), and video generation (VBench). A dynamic thresholding algorithm further improves accuracy-efficiency trade-offs per attention head. For video generation, a short full-attention warmup ensures layout fidelity.
The code is available at: [https://github.com/mit-han-lab/x-attention](https://github.com/mit-han-lab/x-attention).

<p align="center">
  <img src="https://github.com/user-attachments/assets/388094c8-e87c-479a-9704-c5a7202d33e2">
</p>

- **OuroMamba: A Data-Free Quantization Framework for Vision Mamba Models** ([https://www.arxiv.org/pdf/2503.10959](https://www.arxiv.org/pdf/2503.10959)).
  *Intel and Georgia Tech*
  
The authors present OuroMamba, the first data-free post-training quantization (DFQ) method for vision Mamba-based models (VMMs). The authors identify two key challenges in enabling DFQ for VMMs, (1) VMM’s recurrent state transitions
restricts capturing of long-range interactions and leads to semantically weak synthetic data,(2) VMM activations exhibit dynamic outlier variations across time-steps, rendering existing static PTQ techniques ineffective.
To address these challenges,OuroMamba presents a two-stage framework: (1) OuroMamba-Gen to generate semantically rich and meaningful synthetic data. It applies contrastive learning on patch level VMM features generated
through neighborhood interactions in the latent state space, (2) OuroMamba-Quant to employ mixed-precision quantization with lightweight dynamic outlier detection during inference. In specific, the paper presents a thresholding
based outlier channel selection strategy for activations that gets updated every time-step. Extensive experiments across vision and generative tasks show that our data-free OuroMamba surpasses existing data-driven
PTQ techniques, achieving state-of-the-art performance across diverse quantization settings. Additionally, the authors demonstrate the efficacy via implementation of efficient GPU kernels to achieve practical latency speedup of up to 2.36×. 

<p align="center"><img width="100%" height="50%" src="./figures/ouromamba_for_Q2_tech_update.png"></p><br/>

- **Log-Linear Attention** ([https://arxiv.org/pdf/2506.04761](https://arxiv.org/pdf/2506.04761)).

The authors present Log-Linear Attention, a general framework that extends linear attention and state-space models by introducing a logarithmically growing memory structure for efficient long-context modeling. The paper identifies two key limitations in prior linear attention architectures: (1) the use of fixed-size hidden states restricts their ability to model multi-scale temporal dependencies, and (2) their performance degrades on long sequences due to the lack of hierarchical context aggregation.
To address these challenges, Log-Linear Attention places a particular structure on the attention mask, enabling the compute cost to be log-linear and the memory cost to be logarithmic in sequence length (O(TlogT) training time, 
O(logT) inference time and memory). Conceptually, it uses a Fenwick tree–based scheme to hierarchically partition the input into power-of-two-sized segments. Each query attends to a logarithmic number of hidden states, summarizing increasingly coarse ranges of past tokens. This design emphasizes recent context with finer granularity, while efficiently compressing distant information.
The framework is instantiated on top of two representative models: Mamba-2 and Gated DeltaNet, resulting in Log-Linear Mamba-2 and Log-Linear Gated DeltaNet. These variants inherit the expressive recurrence structures of their linear counterparts but benefit from logarithmic memory growth and sub-quadratic training algorithms via a custom chunkwise parallel scan implementation in Triton.
Experiments across language modeling, long-context retrieval, and in-context reasoning benchmarks show that Log-Linear Attention consistently improves long-range recall while achieving competitive or better throughput than FlashAttention-2 at longer sequence lengths (>8K). The code is available at [https://github.com/HanGuo97/log-linear-attention](https://github.com/HanGuo97/log-linear-attention).

<p align="center"><img width="50%" src="https://github.com/user-attachments/assets/8cb7362e-b69f-4953-9ac6-544710456257"></p><br/>

- ...

## Papers with notable results 
### Quantization
- **SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators** ([https://arxiv.org/pdf/2410.10714](https://arxiv.org/pdf/2410.10714)).
  *Apple and Meta*

This paper introduces SeedLM, a novel data-free post-training compression method for Large Language Models (LLMs) that uses seeds of pseudo-random generators and some coefficients to recreate model weights. 
SeedLM aims to reduce memory access and leverage idle compute cycles during inference, effectively speeding up memory-bound tasks by trading compute for fewer memory accesses. 
The method generalizes well across diverse tasks, achieving better zero-shot accuracy retention at 4- and 3-bit compression compared to OmniQuant, AWQ and QuIP#. 
Additionally, FPGA-based tests demonstrate close to 4x speedup for memory-bound tasks such as generation for 4bit per value over an FP16 Llama baseline.

<p align="center"><img width="50%" src="https://github.com/user-attachments/assets/4f0516a5-8b22-459c-8dfe-2225552aa3f3"></p>

- **MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods** ([https://arxiv.org/abs/2501.13484](https://arxiv.org/abs/2501.13484)).
 *Houmo AI, Harbin Institute of Technology (Shenzhen), Nanjing University, Southeast University*

This paper tackles the challenge of post-training quantization for Mamba architectures. Standard quantization techniques adapted from large language models result in substantial accuracy loss when applied to Mamba models, largely due to extreme outliers and inconsistent variances across different channels in weights and activations. To address these issues, the authors propose MambaQuant, introducing two variance alignment techniques: KLT-Enhanced and Smooth-Fused rotations. These methods effectively equalize channel variances, resulting in more uniform data distributions before quantization. Experimental results show that MambaQuant enables Mamba models to be quantized to 8 bits for both weights and activations with less than 1% loss in accuracy, markedly surpassing previous approaches on both vision and language tasks. 

- **SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-bit Training** ([https://arxiv.org/abs/2505.11594](https://arxiv.org/abs/2505.11594)).
 *Tsinghua University*

The authors introduce SageAttention3, a novel FP4 micro-scaling quantization technique for Transformer attention designed to achieve a 5x speedup in inference on NVIDIA GPUs and an 8-bit novel training approach that preserves model accuracy during finetuning while reducing memory demands. The method applies FP4 quantization to the two main attention matrix multiplications, using a microscaling strategy with a group size of 16 elements per scale factor. This fine granularity limits the impact of outlier values that can otherwise cause significant quantization error. To address issues with quantizing the attention map, the authors propose a two-level quantization scheme. First, each row of attention map is scaled into the range 
[0, 448 × 6], which ensures the FP8 scaling factor (required by hardware) fully utilizes its representation range. Then, FP4 quantization is applied at the block level. This two-step process significantly reduces quantization error compared to direct quantization. Empirical results show that SageAttention3 delivers substantial inference speedups with minimal quality loss on language, image, and video generation benchmarks. 

<p align="center"><img width="70%" src=https://github.com/user-attachments/assets/8d08d5e2-d1ff-4dd0-9142-ebea707bf4b8></p>

- **APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers** ([https://arxiv.org/pdf/2504.02508](https://arxiv.org/pdf/2504.02508)).
  *Beihang University*

APHQ-ViT is a PTQ method designed to address the challenges of quantizing Vision Transformers, particularly under ultra-low bit settings. Traditional reconstruction-based PTQ methods, effective for Convolutional Neural Networks, often fail with ViTs due to inaccurate estimation of output importance and significant accuracy degradation when quantizing post-GELU activations.

To overcome these issues, APHQ-ViT introduces an improved Average Perturbation Hessian (APH) loss for better importance estimation. Additionally, it proposes an MLP Reconstruction technique that replaces the GELU activation function with ReLU in the MLP modules and reconstructs them using the APH loss on a small unlabeled calibration set. Experiments demonstrate that APHQ-ViT, utilizing linear quantizers, outperforms existing PTQ methods by substantial margins in 3-bit and 4-bit quantization across various vision tasks.

The source code for APHQ-ViT is available at https://github.com/GoatWu/APHQ-ViT.

- ...
### Pruning / Sparsity
- **SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference** ([https://arxiv.org/pdf/2410.04417](https://arxiv.org/pdf/2410.04417)).

SparseVLM introduces a lightweight, training-free framework for visual token sparsification in vision-language models (VLMs). Unlike text-agnostic approaches, it leverages cross-attention to identify text-relevant visual tokens (“raters”) and adaptively prunes others based on the rank of the attention matrix. Crucially, SparseVLM doesn’t discard all pruned tokens—instead, it recycles the most informative ones (those with high attention relevance scores). These are grouped using a density peak clustering algorithm, and each cluster is compressed into a single representative token. The reconstructed tokens are then reinserted into the model, replacing the larger set of pruned tokens with a compact, information-rich representation. Applied to LLaVA, SparseVLM achieves a 4.5× compression rate with only a 0.9% accuracy drop, reduces CUDA latency by 37%, and saves 67% memory. The code is available at [https://github.com/Gumpest/SparseVLMs](https://github.com/Gumpest/SparseVLMs).

- **Token Sequence Compression for Efficient Multimodal Computing** ([https://arxiv.org/pdf/2504.17892](https://arxiv.org/pdf/2504.17892)).
  
The authors introduce a training-free method for compressing visual token sequences in visual language models (VLMs), significantly reducing computational costs. Instead of relying on attention-based “saliency”—a measure of how much attention a model gives to each token—they use simple clustering to group similar visual tokens and aggregate them. Their “Cluster & Aggregate” approach outperforms prior finetuning-free methods like VisionZip and SparseVLM across 8+ benchmarks, even when retaining as little as 11% of the original tokens. Surprisingly, random and spatial sampling also perform competitively, revealing high redundancy in visual encodings.

<p align="center"><img width="50%" src="https://github.com/user-attachments/assets/b5b99700-ff0c-4f8f-b28a-ca079341feae"></p>

- **TopV: Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal Vision Language Model** ([https://arxiv.org/pdf/2503.18278v2](https://arxiv.org/pdf/2503.18278v2)).

The authors introduce a training-free, optimization-based framework for reducing visual token redundancy in VLMs. Visual tokens often dominate the input sequence—up to 95% in some models. TopV addresses this by pruning unimportant visual tokens once during the prefilling stage, before decoding begins.
Instead of relying on attention scores, TopV estimates the importance of each visual token by solving an optimal transport problem. In this setup:

• Source tokens are the input visual tokens entering a specific transformer layer.

• Target tokens are the output visual tokens after that layer has processed the input—specifically, the output after the Post-LN sublayer.

<p align="center"><img src="https://github.com/user-attachments/assets/52352aef-be92-4060-996f-79d21c86ccbb"></p>

TopV calculates how much each input token contributes to the output using the Sinkhorn algorithm, guided by a cost function that considers:

• How similar the tokens are in content (feature similarity),

• How close they are in the image (spatial proximity),

• How central they are in the image (centrality).

To prevent visual collapse—especially in detail-sensitive tasks like OCR and captioning—TopV includes a lightweight recovery step. From the discarded tokens, TopV uniformly samples a subset at regular intervals (e.g., every 4th or 6th token) and reinserts them into the token sequence alongside the top-k tokens, ensuring spatial diversity and semantic coverage without significant overhead. 
TopV performs pruning once after the prompt and image are processed. The pruned visual token set remains fixed throughout decoding, enabling efficient and consistent inference.

- **Beyond 2:4: exploring V:N:M sparsity for efficient transformer inference on GPUs**
([https://arxiv.org/abs/2410.16135](https://arxiv.org/abs/2410.16135)).
 *Tsinghua University, Beijing Jiaotong University*

This paper introduces and systematically studies V:N:M sparsity as a more efficient and flexible alternative to the industry-standard 2:4 sparsity for accelerating Transformer inference on GPUs. In the V:N:M approach, weight matrices are divided into V×M blocks; within each block, most columns are pruned, and 2:4 sparsity is then applied to the remaining columns. This scheme enables significantly higher and more adaptable sparsity ratios, while remaining compatible with existing GPU sparse tensor core acceleration. The authors propose a comprehensive framework for creating V:N:M-sparse Transformers: it features a heuristic method for selecting V and M values to optimize the accuracy-speedup trade-off, a V:N:M-specific channel permutation method for improving accuracy in low-budget training scenarios, and a three-stage LoRA training process for memory-efficient fine-tuning. Experimental results show that V:N:M-sparse Transformers can achieve much higher sparsity levels - such as 75% parameter reduction, while maintaining nearly lossless accuracy on downstream tasks, and outperform 2:4 sparsity in both speed and flexibility.

<p align="center"><img width="50%" src="https://github.com/user-attachments/assets/63bcfc3b-440f-4395-9d11-9581fedd25b7"></p>

- ...
### Other 
- **MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models** ([https://arxiv.org/pdf/2503.11972](https://arxiv.org/pdf/2503.11972)).
  *Intel and University of Michigan*
 
Diffusion-based text-to-image generation models trade latency for quality: small models are fast but generate lower quality images, while large models produce better images
but are slow. This paper presents MoDM, a novel caching-based serving system for diffusion models that dynamically balances latency and quality through a mixture of diffusion models.
Unlike prior approaches that rely on model-specific internal features, MoDM caches final images, allowing seamless retrieval and reuse across multiple diffusion model families.
This design enables adaptive serving by dynamically balancing latency and image quality: using smaller models for cache-hit requests to reduce latency while reserving larger
models for cache-miss requests to maintain quality. Small model image quality is preserved using retrieved cached images. MoDM has a global monitor that optimally allocates
GPU resources and balances inference workload, ensuring high throughput while meeting Service-Level Objectives (SLOs) under varying request rates. Extensive evaluations show
that MoDM significantly reduces an average serving time by 2.5× while retaining image quality, making it a practical solution for scalable and resource-efficient model deployment.
- ...
### Software
- TBD
- ...
