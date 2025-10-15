# Sparse-Layered Transformers for Time-Series Forecasting  
**Attention-driven architecture using simplified projection layers**

## Overview

This repository introduces **Sparse-Layered Transformers (SLTs)** — forecasting models in which **a single top attention layer drives multiple lightweight linear–activation projection blocks**.

The approach generalizes the 2025 [*Summation-Based Transformers* (TechRxiv)](https://doi.org/10.36227/techrxiv.175790522.25734653/v2) **research and [accompanying code](https://github.com/pfekin/summation-based-transformers)**:

> **Simpler projection layers guided by attention** — attention as the global driver rather than the repeated mechanism.

Applied to time-series forecasting, SLTs achieve **higher accuracy and faster inference** than full-attention models such as PatchTST, while using fewer parameters and lower memory.

## Architecture Summary

| Layer Type | Operation | Complexity |
|-------------|------------|-------------|
| Projection Block (×2) | Linear (no bias) → GELU → Norm → FFN | O(n) |
| Top Attention Block | Multi-Head Attention → FFN | O(n²) |
| Flatten & Projection | Linear mapping to forecast horizon | O(n) |

See [architecture.md](architecture.md) for full details.

## Experimental Setup

- **Environment:** Google Colab T4 GPU (16 GB)  
- **Datasets:** ETTh1/2, ETTm1/2, Weather, Traffic  
- **Training:** Adam (lr=1e-4), GELU activation, multiplicative positional encoding  
- **Layers:** 2 projection + 1 attention  

## Benchmark Results

| Model | MSE ↓ | MAE ↓ | Speed ↑ (samples/s) |
|:------|:------:|:------:|:------------------:|
| PatchTST (baseline) | 0.4667 | 0.4876 | 2 927 |
| **Sparse-Layered Transformer (ours)** | **0.4430** | **0.4691** | **4 139** |
| N-BEATS | 0.4527 | 0.4782 | 25 392 |
| TFT | 2.1001 | 1.0756 | 6 252 |
| TCN | 1.4645 | 0.9590 | 5 303 |

**ETTh1 Gains:**  
+5 % MSE improvement, +41 % faster, fewer parameters (~450 K vs 470 K)

### Across Datasets

| Dataset | PatchTST MSE | SLT MSE | Improvement | Speedup |
|:---------|:-------------:|:---------:|:-------------:|:---------:|
| Weather | 0.1607 | **0.1548** | +3.7 % | × 1.45 |
| Traffic | 0.3263 | **0.3206** | +1.8 % | × 1.38 |
| ETTh1 | 0.4450 | **0.4387** | +1.4 % | × 1.36 |
| ETTh2 | 0.2438 | **0.1941** | +20.4 % | × 1.37 |
| ETTm1 | 0.3704 | **0.3295** | +11.0 % | × 1.34 |
| ETTm2 | 0.1850 | **0.1751** | +5.4 % | × 1.44 |

> ⚠️ Lightweight implementations optimized for Colab; not reference versions of Darts or Hugging Face.

## Installation & Usage

```bash
git clone https://github.com/pfekin/forecasting-sparse-layered-transformers
cd forecasting-sparse-layered-transformers
pip install torch numpy pandas scikit-learn darts
python benchmark.py
````

Edit configuration in `benchmark.py`:

```python
CONFIG = {
    'seq_len': 512,
    'pred_len': 96,
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 3,
    'batch_size': 32,
    'n_epochs': 10,
    'lr': 1e-4,
}
```

## Key Findings

1. **Attention-Driven Architecture** — A single attention block *drives* simpler projection layers. Attention becomes the representational organizer, not the primary computational cost.
2. **Efficient Global Modeling** — Projection layers capture local structure efficiently (O(n)); the final attention layer integrates global context (O(n²)).
3. **Architectural Continuity** — Extends the *Summation-Based Transformer* idea — simple projection blocks guided by top-level attention — from language modeling to forecasting.

## References

1. **PatchTST**: Nie et al., "A Time Series is Worth 64 Words", ICLR 2023
2. **Summation-Based Transformers**: [Original paper](https://doi.org/10.36227/techrxiv.175790522.25734653/v2)
3. **Attention Is All You Need**: Vaswani et al., NeurIPS 2017
4. **N-BEATS**: Oreshkin et al., ICLR 2020

## Limitations & Future Work

* Lightweight, research-focused implementations
* Memory-limited experiments (T4 GPU)
* Future work:

  * Multi-scale and deeper hybrids
  * Integration with Hugging Face forecasting APIs
  * Theoretical study of attention-driven representational flow

---

## Citation

```bibtex
@article{fekin2025slt,
  title = {Sparse-Layered Transformers for Time-Series Forecasting},
  author = {Pascal Fekin},
  year = {2025},
  url = {https://github.com/pfekin/forecasting-sparse-layered-transformers}
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

I’m seeking collaborators with access to large-scale compute resources to train summation-based transformers at language-modeling scale.

* 📧 Email: [your email]
* 🐙 GitHub: [pfekin](https://github.com/pfekin)
* 📄 Paper: [TechRxiv 2025](https://doi.org/10.36227/techrxiv.175790522.25734653/v2)

---

## License
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Licensed under the **Apache License 2.0** © 2025 Pascal Fekin.

> ⚠️ Research-only code demonstrating relative performance under constrained GPU settings.
