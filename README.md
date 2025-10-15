# Sparse-Layered Transformers for Time-Series Forecasting

**Attention-driven architecture using simplified projection layers**

## Overview

This repository introduces Sparse-Layered Transformers (SLTs) — forecasting models in which a single top attention layer drives multiple lightweight linear–activation projection blocks.

The approach generalizes the 2025 [*Summation-Based Transformers* (TechRxiv)](https://doi.org/10.36227/techrxiv.175790522.25734653/v2) research and [accompanying code](https://github.com/pfekin/summation-based-transformers):

> Simpler projection layers guided by attention — attention as the global driver rather than the repeated mechanism.

Applied to time-series forecasting, SLTs achieve higher accuracy and faster inference than full-attention models such as PatchTST, while using fewer parameters and lower memory.

## Architecture Summary

| Layer Type            | Operation                            | Complexity |
| --------------------- | ------------------------------------ | ---------- |
| Projection Block (×2) | Linear (no bias) → GELU → Norm → FFN | O(n)       |
| Top Attention Block   | Multi-Head Attention → FFN           | O(n²)      |
| Flatten & Projection  | Linear mapping to forecast horizon   | O(n)       |

### Visual overview 
<div align="center">
  <img src="architecture.png" alt="Sparse-Layered Transformer Diagram" width="800"/>
</div>

## Implementation Details

The implementation builds on PatchTST but simplifies most transformer layers into projection-based blocks and retains a single top attention layer that integrates global context.

### 1. Patching

Time series are segmented into overlapping fixed-length windows (patches), producing a sequence of short subsequences that act as input tokens.

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
        # Flatten variable dimension for embedding
        return patches.reshape(x.size(0) * x.size(2), num_patches, self.patch_len)
```

This reduces the input length and allows local temporal structures to be learned efficiently before applying global attention.


### 2. Projection Blocks

Most transformer layers are replaced by projection blocks, which apply a bias-free linear projection followed by GELU activation and normalization.

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

These layers are linear in sequence length O(n) and preserve local structure efficiently.

### 3. Multiplicative Positional Encoding

Unlike standard additive encodings, the model employs multiplicative positional encoding, which scales features by learned positional weights:

```python
x = patch_embedding(patches)   # (batch*n_vars, num_patches, d_model)
x = x * self.pos_encoding      # multiplicative positional encoding
```

This modulates feature amplitudes by position, helping the model capture smooth temporal variation without inflating parameter count.

### 4. Final Attention Layer

The final layer is a standard multi-head self-attention block, responsible for integrating global dependencies across all patches:

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
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ffn(x))
```

This single top attention layer drives the representational hierarchy, providing global coordination while projection layers encode local context.

### 5. Flattening & Prediction

After the attention stage, outputs are flattened and projected to the desired forecast horizon:

```python
# x shape: (batch*n_vars, num_patches, d_model)
x = x.reshape(batch_size * n_vars, -1)      # (batch*n_vars, num_patches * d_model)
pred = self.head(x)                          # (batch*n_vars, pred_len)
# reshape to (batch, pred_len, n_vars)
pred = pred.reshape(batch_size, n_vars, pred_len).transpose(1, 2)
```

## Experimental Setup

* **Environment:** Google Colab T4 GPU (16 GB)
* **Datasets:** ETTh1/2, ETTm1/2, Weather, Traffic
* **Training:** Adam (lr=1e-4), GELU activation, multiplicative positional encoding
* **Layers:** 2 projection + 1 attention

## Benchmark Results

### Across Datasets

| Dataset | PatchTST MSE | PatchTST SLT MSE | Improvement | Speedup |
| :------ | :----------: | :--------------: | :---------: | :-----: |
| Weather |    0.1607    |    **0.1548**    |    +3.7 %   |  × 1.45 |
| Traffic |    0.3263    |    **0.3206**    |    +1.8 %   |  × 1.38 |
| ETTh1   |    0.4450    |    **0.4387**    |    +1.4 %   |  × 1.36 |
| ETTh2   |    0.2438    |    **0.1941**    |   +20.4 %   |  × 1.37 |
| ETTm1   |    0.3704    |    **0.3295**    |   +11.0 %   |  × 1.34 |
| ETTm2   |    0.1850    |    **0.1751**    |    +5.4 %   |  × 1.44 |

> ⚠️ Lightweight implementations optimized for Colab. Not reference versions of Darts or Hugging Face.


## Installation & Usage

```bash
git clone https://github.com/pfekin/sparse-layered-transformers
cd sparse-layered-transformers
pip install torch numpy pandas scikit-learn darts
python benchmark.py
```

Edit configuration in `benchmark.py`:

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

## Key Findings

1. **Attention-Driven Architecture** — A single attention block *drives* simpler projection layers. Attention becomes the representational organizer, not the primary computational cost.
2. **Efficient Global Modeling** — Projection layers capture local structure efficiently (O(n)); the final attention layer integrates global context (O(n²)).
3. **Architectural Continuity** — Extends the *Summation-Based Transformer* idea — simple projection blocks guided by top-level attention — from language modeling to forecasting.

## References

1. PatchTST: Nie et al., *A Time Series is Worth 64 Words*, ICLR 2023 — [GitHub](https://github.com/yuqinie98/PatchTST)
2. Summation-Based Transformers: [TechRxiv 2025 paper](https://doi.org/10.36227/techrxiv.175790522.25734653/v2)
3. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
4. Oreshkin et al., *N-BEATS*, ICLR 2020


## Limitations & Future Work

* Lightweight, research-focused implementations
* Memory-limited experiments (T4 GPU)
* Future work:

  * Integration with Hugging Face forecasting APIs
  * Theoretical study of attention-driven representational flow

## Citation

```bibtex
@article{fekin2025slt,
  title = {Sparse-Layered Transformers for Time-Series Forecasting},
  author = {Pascal Fekin},
  year = {2025},
  url = {https://github.com/pfekin/sparse-layered-transformers}
}

@article{fekin2025summation,
  title = {Summation-Based Transformers},
  author = {Pascal Fekin},
  journal = {TechRxiv},
  year = {2025},
  doi = {10.36227/techrxiv.175790522.25734653/v2}
}
```

## Contact & Collaboration

I’m seeking collaborators with access to large-scale compute resources to train summation-based / attention-driven transformers at language-modeling scale.

* 📧 Email: [your email]
* 🐙 GitHub: [pfekin](https://github.com/pfekin)
* 📄 Paper: [TechRxiv 2025](https://doi.org/10.36227/techrxiv.175790522.25734653/v2)


## License

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

