# Q2'25: Technology Update – Low Precision and Model Optimization
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

- **OuroMamba: A Data-Free Quantization Framework for Vision Mamba Models** ([https://www.arxiv.org/pdf/2503.10959](https://www.arxiv.org/pdf/2503.10959)).
  
The authors present OuroMamba, the first data-free post-training quantization (DFQ) method for vision Mamba-based models (VMMs). The authors identify two key challenges in enabling DFQ for VMMs, (1) VMM’s recurrent state transitions
restricts capturing of long-range interactions and leads to semantically weak synthetic data,(2) VMM activations exhibit dynamic outlier variations across time-steps, rendering existing static PTQ techniques ineffective.
To address these challenges,OuroMamba presents a two-stage framework: (1) OuroMamba-Gen to generate semantically rich and meaningful synthetic data. It applies contrastive learning on patch level VMM features generated
through neighborhood interactions in the latent state space, (2) OuroMamba-Quant to employ mixed-precision quantization with lightweight dynamic outlier detection during inference. In specific, the paper presents a thresholding
based outlier channel selection strategy for activations that gets updated every time-step. Extensive experiments across vision and generative tasks show that our data-free OuroMamba surpasses existing data-driven
PTQ techniques, achieving state-of-the-art performance across diverse quantization settings. Additionally, the authors demonstrate the efficacy via implementation of efficient GPU kernels to achieve practical latency speedup of up to 2.36×. 

<p align="center"><img width="100%" src="./figure/ouromamba_for_Q2_tech_update.png"></p><br/>
- ...
## Papers with notable results 
### Quantization
- TBD
- ...
### Pruning / Sparsity
- TBD
- ...
### Other 
- TBD
- ...
### Software
- TBD
- ...
