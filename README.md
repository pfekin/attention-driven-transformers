# Attention-Driven Transformers for Time-Series Forecasting

Simplified projection architectures driven by attention

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## Overview

This repository introduces **Attention-Driven Transformers (ADTs)** — forecasting models in which strategically placed attention layers drive multiple lightweight projection blocks.

The approach extends the 2025 [*Summation-Based Transformers* (TechRxiv)](https://doi.org/10.36227/techrxiv.175790522.25734653/v2) research and its [accompanying code](https://github.com/pfekin/summation-based-transformers), which first proposed that attention acts as a global representational driver rather than a uniformly repeated mechanism.

In ADTs, most layers are simple linear–activation projections, while strategically placed attention blocks (O(n²) complexity) organizes temporal dependencies across patch embeddings. The design yields models that are faster and more memory-efficient than dense-attention baselines such as PatchTST, while often improving accuracy.

The attention-driven principle—attention globally organizing simpler transformations—remains consistent across scales.
Notably, attention layers can be flexibly positioned - at the end ([proj, proj, attn]), in the middle of projection blocks ([proj, attn, proj]), or as repeating patterns. The key is the interplay between projection and attention, not a rigid architectural template.

## Architecture summary

| Layer type            | Operation                            | Complexity |
| --------------------- | ------------------------------------ | ---------- |
| Projection block (×2) | Linear (no bias) → GELU → Norm → FFN | O(n)       |
| Top attention block   | Multi-head attention → FFN           | O(n²)      |
| Flatten & projection  | Linear mapping to forecast horizon   | O(n)       |

Each variable (channel) in the dataset is handled independently during training for efficiency, following the same pattern as the individual-channel mode in PatchTST.

Projection blocks perform local transformations within patch embeddings, while the top attention block governs longer-range temporal dependencies. The same design principle could be extended to deeper networks by interleaving attention and projection layers.

## Implementation details

### 1. Patching

Time series are segmented into overlapping windows (patches) that serve as input tokens:

```python
class Patching(nn.Module):
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride

    def forward(self, x):
        # x: (batch, seq_len, n_vars)
        num_patches = (x.size(1) - self.patch_len) // self.stride + 1
        patches = torch.zeros(x.size(0), x.size(2), num_patches, self.patch_len, device=x.device)
        for i in range(num_patches):
            start, end = i * self.stride, i * self.stride + self.patch_len
            patches[:, :, i, :] = x[:, start:end, :].transpose(1, 2)
        return patches.reshape(x.size(0) * x.size(2), num_patches, self.patch_len)
```

### 2. Projection blocks

Projection blocks replace most attention layers with a bias-free linear projection followed by GELU activation and normalization:

```python
class ProjectionBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU()
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.proj(x))
        return self.norm2(x + self.ffn(x))
```

These layers are linear in sequence length and capture local transformations within each patch embedding without explicit token mixing across time.

### 3. Multiplicative positional encoding

ADTs use multiplicative positional encoding, scaling features by learned positional weights:

```python
x = patch_embedding(patches)
x = x * self.pos_encoding
```

### 4. Final attention layer

The final multi-head self-attention block processes the sequence of patch embeddings and provides the model’s only mechanism for direct temporal interaction between patches:

```python
class StandardAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ffn(x))
```

### 5. Flattening and prediction

The attention output is flattened and projected to the final prediction:

```python
x = x.reshape(batch_size * n_vars, -1)
pred = self.head(x)
pred = pred.reshape(batch_size, n_vars, pred_len).transpose(1, 2)
```

## Experimental setup

* Environment: Google Colab T4 GPU (16 GB)
* Datasets: ETTh1/2, ETTm1/2, Weather, Traffic
* Training: Adam (lr = 1e-4), GELU activation, multiplicative positional encoding
* Layers: 2 projection + 1 attention

## Benchmark results
| Dataset | TCN* MSE | N-BEATS** MSE | PatchTST MSE | PatchTST ADT MSE | Improvement | Speedup |
| :-----: | :------: | :---------: | :-----------: | :--------------: | :---------: | :-----: |
| Weather | 0.3679 | 0.1737 | 0.1607 | **0.1548** | +3.7 % | × 1.45 |
| Traffic | 0.5141 | 0.3297 | 0.3263 | **0.3206** | +1.8 % | × 1.38 |
| ETTh1 | 1.5799 | 0.4642 | 0.4450 | **0.4387** | +1.4 % | × 1.36 |
| ETTh2 | 1.1139 | 0.2553 | 0.2438 | **0.1941** | +20.4 % | × 1.37 |
| ETTm1 | 0.7694 | 0.3682 | 0.3704 | **0.3295** | +11.0 % | × 1.34 |
| ETTm2 | 0.7570 | 0.1807 | 0.1850 | **0.1751** | +5.4 % | × 1.44 |

\*TCN: Temporal Convolutional Network baseline.

\**N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting (ICLR 2020).

Lightweight implementations were optimized for Colab and are not intended as reference Darts or Hugging Face baselines.

## Installation and usage

```bash
git clone https://github.com/pfekin/attention-driven-transformers
cd attention-driven-transformers
pip install torch numpy pandas scikit-learn darts
python benchmark.py
```

Basic configuration example:

```python
CONFIG = {
    'seq_len': 512,
    'pred_len': 96,
    'patch_len': 16,
    'stride': 8,
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 3,
    'batch_size': 32,
    'n_epochs': 10,
    'lr': 1e-4,
    'dropout': 0.15,
}
```
You can run the full benchmark directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RA2IswgKTHIzMXS9781RA_ZfTLudf9VL?usp=sharing)

## Key findings

1. **Attention-driven architecture** – Attention-driven architecture drives simpler projection layers, structuring representations efficiently without dense attention stacking.
2. **Per-variable temporal modeling** – Each variable is modeled independently across time using shared parameters, enabling efficient parallelization.
3. **Efficient temporal modeling** – Projection layers encode local patch features O(n), the attention layer models cross-patch dependencies O(n²).
4. **Generalizable attention hierarchy** – Forecasting models concentrate attention at the top, while deeper architectures (e.g., LLMs) can interleave attention with projection layers. In both cases, attention functions as the organizational driver of representation.
5. **Architectural continuity** – Extends the Summation-Based Transformer principle—simple projection layers guided by top-level attention—from language modeling to forecasting.

## References

1. Nie et al., [*PatchTST: A Time Series is Worth 64 Words*](https://github.com/yuqinie98/PatchTST), ICLR 2023
2. Ekin, [*Summation-Based Transformers*](https://doi.org/10.36227/techrxiv.175790522.25734653/v2), TechRxiv 2025
3. Vaswani et al., [*Attention Is All You Need*](https://papers.nips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), NeurIPS 2017
4. Oreshkin et al., [*N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting*](https://arxiv.org/abs/1905.10437), ICLR 2020

## Limitations and future work

* Lightweight prototypes optimized for Colab
* Currently independent per-variable modeling
* Future directions:

  * Cross-variable attention extensions
  * Interleaved attention–projection stacks for deeper models
  * Theoretical analysis of attention-driven representation dynamics

## Citation

```bibtex
@article{ekin2025adt,
  title   = {Attention-Driven Transformers for Time-Series Forecasting},
  author  = {Pascal Ekin},
  year    = {2025},
  url     = {https://github.com/pfekin/attention-driven-transformers}
}

@article{Summation_Based_Transformers_2025,
  title={Summation-Based Transformers: A Path Toward Linear Complexity Sequence Modeling},
  author={Pascal Ekin},
  journal={TechRxiv},  
  year={2025},
  doi={10.36227/techrxiv.175790522.25734653/v2},  
  url={https://doi.org/10.36227/techrxiv.175790522.25734653/v2},
}
```

## Language Modeling (Experimental)

This work has been extended to autoregressive language modeling to test whether the attention-driven approach generalizes beyond time series forecasting.

**Important:** These results are preliminary exploration on small NLP datasets (WikiText-2, IMDB, AG News, CMU Book Summaries) rather than definitive benchmarks. Comprehensive evaluation would require substantially more computational resources and larger datasets.

The core principle holds: projection blocks make attention work more effectively. Models with projection blocks combined with strategically placed attention layers achieve comparable or better perplexity with improved inference speed. This can be seen as an ablation of [Summation-Based Transformers](https://github.com/pfekin/summation-based-transformers) - the cumulative summation mechanism is not necessary when combined with attention; attention-driven projection (simple GELU projections + strategic attention placement) is sufficient.

Notably, attention layers can be flexibly positioned - at the end, in the middle of projection blocks, or as repeating patterns (e.g., [proj, proj, attn, proj, proj, attn, ...]). The key is the interplay between projection and attention, not a rigid architectural template.

The `causal_benchmark.py` script is provided as a test bed for those interested in exploring these ideas further.

## Contact and collaboration

Seeking collaborators with access to large-scale compute resources to train attention-driven transformers at language-modeling scale.

* **Email**: [pfekin@gmail.com](mailto:pfekin@gmail.com)

