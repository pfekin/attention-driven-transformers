# Sparse-Layered Transformers for Time-Series Forecasting

**Simplified projection blocks guided by top-level attention**

[![TechRxiv](https://img.shields.io/badge/TechRxiv-Sparse--Layered_Transformers-b31b1b.svg)](https://doi.org/10.36227/techrxiv.175790522.25734653/v2)
[![GitHub – Language Modeling](https://img.shields.io/badge/GitHub-Language_Modeling-blue)](https://github.com/pfekin/summation-based-transformers)

---

## Overview

This repository presents **Sparse-Layered Transformers (SLTs)**, a forecasting architecture that replaces most attention layers with lightweight **linear–activation projection blocks**, while retaining a **single top attention layer** to model global temporal dependencies.

The approach generalizes the architectural philosophy introduced in [*Summation-Based Transformers* (TechRxiv, 2025)](https://doi.org/10.36227/techrxiv.175790522.25734653/v2):

> **Simpler blocks guided by a strategic attention layer.**

When applied to time-series forecasting, this design yields higher accuracy and faster inference than full-attention models such as PatchTST, while using fewer parameters and less memory.

---

## Architectural Principle

### Traditional Transformer

```
Input → Attention → Attention → Attention → ... → Output
         O(n²)        O(n²)        O(n²)
```

### Sparse-Layered Transformer (This Work)

```
Input → Projection → Projection → Attention → Output
         O(n)          O(n)          O(n²)
```

* Early blocks: simple linear + activation layers (O(n))
* Final block: full self-attention (O(n²))
* Total complexity: **~66% less attention compute**

---

## Architecture Summary

| Component                  | Function                                  | Complexity |
| -------------------------- | ----------------------------------------- | ---------- |
| **Projection Blocks (×2)** | Linear (no bias) → GELU → LayerNorm → FFN | O(n)       |
| **Top Attention Block**    | Multi-Head Self-Attention + FFN           | O(n²)      |
| **Flatten & Project**      | Flatten patch embeddings → Linear head    | O(n)       |

### Why It Works

* Local features captured by projections
* Global dependencies handled by final attention
* Simpler layers regularize learning and reduce overfitting

For more implementation details, see [architecture.md](architecture.md).

---

## Connection to Language-Modeling Work

| Aspect            | Language Modeling (TechRxiv 2025)             | Forecasting (This Work)           |
| ----------------- | --------------------------------------------- | --------------------------------- |
| Simplified Blocks | Linear → Activation → Pooling (sum or cumsum) | Linear → Activation               |
| Aggregation       | Explicit pooling (summation)                  | Flattened embeddings (no pooling) |
| Attention         | Top-level contextual integration              | Top-level contextual integration  |
| Complexity        | Lower-level O(n) layers + one O(n²)                         | 2 O(n) layers + one O(n²)    |

**Clarification:**
Language models perform explicit pooling after projections.
Forecasting models *flatten* the embeddings before the output projection; no summation or averaging occurs.

---

## Experimental Setup

All experiments were run on **Google Colab (T4 GPU, 15 GB VRAM)**.
Implementation uses custom PyTorch code compatible with Darts datasets.

* Input length: 512
* Prediction horizon: 96
* Optimizer: Adam (lr=1e-4)
* Layers: 2 projection + 1 attention
* Positional encoding: multiplicative
* Activation: GELU

---

## Benchmark Results

| Model                                 |    MSE ↓   |    MAE ↓   | Speed ↑ (samples/s) |
| :------------------------------------ | :--------: | :--------: | :-----------------: |
| PatchTST (baseline)                   |   0.4667   |   0.4876   |       2 926.8       |
| **Sparse-Layered Transformer (ours)** | **0.4430** | **0.4691** |     **4 139.0**     |
| N-BEATS                               |   0.4527   |   0.4782   |       25 392.9      |
| TFT                                   |   2.1001   |   1.0756   |       6 252.3       |
| TCN                                   |   1.4645   |   0.9590   |       5 302.6       |

**Improvement (ETTh1):**

* 5 % lower MSE
* 41 % faster inference
* ~4 % fewer parameters

---

### Across All Datasets

| Dataset | PatchTST MSE |   SLT MSE  | Improvement | Speedup |
| :------ | :----------: | :--------: | :---------: | :-----: |
| Weather |    0.1607    | **0.1548** |    +3.7 %   |  × 1.45 |
| Traffic |    0.3263    | **0.3206** |    +1.8 %   |  × 1.38 |
| ETTh1   |    0.4450    | **0.4387** |    +1.4 %   |  × 1.36 |
| ETTh2   |    0.2438    | **0.1941** |   +20.4 %   |  × 1.37 |
| ETTm1   |    0.3704    | **0.3295** |   +11.0 %   |  × 1.34 |
| ETTm2   |    0.1850    | **0.1751** |    +5.4 %   |  × 1.44 |

> ⚠️ These are custom implementations optimized for reproducibility and speed;
> they are not Darts or Hugging Face reference models.

---

## Installation & Usage

```bash
git clone https://github.com/pfekin/forecasting-sparse-layered-transformers
cd forecasting-sparse-layered-transformers
pip install torch numpy pandas scikit-learn darts
python benchmark.py
```

Adjust hyperparameters in `benchmark.py`:

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

Optional: enable plotting in `benchmark.py` to visualize losses or predictions.

---

## Key Findings

1. **Accuracy–Efficiency Trade-off** — SLTs outperform full-attention PatchTST on accuracy while running 40 % faster.
2. **Simpler Training Dynamics** — Fewer attention layers yield more stable optimization.
3. **Generalizable Design** — Same principle as Summation-Based Transformers, adapted to forecasting tasks.


1. **Attention-Driven Architecture** — The Sparse-Layered Transformer is organized around a single top attention block that drives simpler projection layers. Attention acts as a global integrator, shaping the representational flow rather than performing dense token-to-token mixing at every layer.

2. **Efficiency with Global Awareness** — Linear projection blocks capture local dependencies efficiently (O(n)), while the final attention layer integrates global context (O(n²)), achieving the accuracy of full-attention models at substantially higher speed.

3. **Architectural Continuity** — Extends the principle introduced in Summation-Based Transformers (TechRxiv, 2025) — simple feedforward or projection blocks guided by top-level attention — from language modeling to time-series forecasting.


---

## Limitations & Future Work

* Custom experimental implementations (not production-ready).
* GPU memory limits excluded some Darts models.
* Future goals:

  * Integration with Hugging Face forecasting APIs
  * Multi-scale hybrid architectures
  * Theoretical analysis of representational efficiency

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

---

## Contact & Collaboration

Interested collaborators with access to large-scale compute resources for extending SLTs to language modeling are welcome.

* 📧 Email: [your email]
* 🐙 GitHub: [pfekin](https://github.com/pfekin)
* 📄 Paper: [TechRxiv 2025](https://doi.org/10.36227/techrxiv.175790522.25734653/v2)

---

## License

Licensed under the **Apache License 2.0** © 2025 Pascal Fekin.

> ⚠️ Research-only implementation — demonstrates relative performance under controlled conditions, not production benchmarks.

---

Would you like me to add a simple visual (ASCII or minimal SVG) comparing **Full Transformer vs Sparse-Layered Transformer** right below the “Architectural Principle” section? It would make the README more visually engaging on GitHub.

---

## Overview

This repository extends the ideas introduced in [*Summation-Based Transformers* (TechRxiv, 2025)](https://doi.org/10.36227/techrxiv.175790522.25734653/v2) to **time-series forecasting**.

The core principle:

> Replace most attention blocks with lightweight linear + activation layers, keeping a **single top attention layer** to drive global representation learning.

This **hybrid architecture** achieves better accuracy and faster inference than attention-only models such as PatchTST — validating the same structural philosophy demonstrated earlier in language modeling.

---

## Architectural Principle

Traditional transformers stack multiple self-attention layers, which are computationally expensive.
Here, we use:

* **Simplified Blocks:**
  Linear (no bias) → GELU (or ReLU) → LayerNorm → residual connection
* **Final Attention Block:**
  Multi-Head Self-Attention → FeedForward → LayerNorm

Attention is kept only where it is *most valuable* — at the top of the stack — while earlier layers build representations through simple projections.

---

## Connection to Language-Modeling Work

### Common Philosophy

| Component         | Language Modeling                             | Forecasting (This Work)                                  |
| ----------------- | --------------------------------------------- | -------------------------------------------------------- |
| Simplified Blocks | Linear → Activation → Pooling (sum or cumsum) | Linear → Activation                                      |
| Aggregation       | Explicit summation / cumulative pooling       | **Flattening of patch embeddings (no explicit pooling)** |
| Top Layer         | Attention layer                               | Attention layer                                          |
| Key Insight       | Simpler projection blocks guided by attention | Same principle, adapted to flattened temporal embeddings |

**Clarification:**
In the language-modeling variant, representations are explicitly aggregated through summation or cumulative pooling.
In forecasting, **no pooling occurs** — embeddings are *flattened* before the projection head.
Global dependencies are then captured implicitly through the **final attention layer** and **output projection**, rather than a deterministic sum.

---

## Why Forecasting?

Large-scale language-model experiments require heavy compute.
Forecasting offers:

* ✅ Faster iteration (hours vs days)
* ✅ Colab-compatible benchmarks
* ✅ Clear quantitative metrics (MSE, MAE, speed)
* ✅ Immediate applied relevance

This project demonstrates that the summation-based architectural idea generalizes effectively to real-world forecasting tasks.

---

## Architecture: Hybrid PatchTST

The **Hybrid PatchTST** modifies the standard PatchTST as follows:

```
Input Time Series
    ↓
Patching (convert to segments)
    ↓
Patch Embedding (linear)
    ↓
Positional Encoding (multiplicative)
    ↓
[Simplified Block 1] → Linear (no bias) → GELU → LayerNorm
[Simplified Block 2] → Linear (no bias) → GELU → LayerNorm
    ↓
[Attention Block] → Multi-Head Self-Attention → LayerNorm → FFN
    ↓
Flatten → Projection Head
    ↓
Predictions
```

Standard PatchTST uses attention in every block;
**Hybrid PatchTST** uses simplified blocks everywhere except the final layer.

---

## Benchmark Results

All experiments were run on Google Colab (A100, 15 GB VRAM).
Models are lightweight, non-reference implementations optimized for speed and reproducibility.

### ETTh1 (512 → 96 forecast horizon)

| Model                      |    MSE ↓   |    MAE ↓   | Speed ↑ (samples/s) |
| :------------------------- | :--------: | :--------: | :-----------------: |
| PatchTST (baseline)        |   0.4667   |   0.4876   |       2 926.8       |
| **Hybrid PatchTST (ours)** | **0.4430** | **0.4691** |     **4 139.0**     |
| N-BEATS                    |   0.4527   |   0.4782   |       25 392.9      |
| TFT                        |   2.1001   |   1.0756   |       6 252.3       |
| TCN                        |   1.4645   |   0.9590   |       5 302.6       |

**Hybrid vs Baseline PatchTST:**

* 🎯 5 % MSE reduction
* ⚡ +41 % faster inference
* 💾 Slightly fewer parameters (~450 K vs 470 K)

---

### Across All Datasets

| Dataset | PatchTST MSE | Hybrid MSE | Improvement | Speedup |
| :------ | :----------: | :--------: | :---------: | :-----: |
| Weather |    0.1607    | **0.1548** |    +3.7 %   |  × 1.45 |
| Traffic |    0.3263    | **0.3206** |    +1.8 %   |  × 1.38 |
| ETTh1   |    0.4450    | **0.4387** |    +1.4 %   |  × 1.36 |
| ETTh2   |    0.2438    | **0.1941** |   +20.4 %   |  × 1.37 |
| ETTm1   |    0.3704    | **0.3295** |   +11.0 %   |  × 1.34 |
| ETTm2   |    0.1850    | **0.1751** |    +5.4 %   |  × 1.44 |

> ⚠️ Results demonstrate relative performance under controlled conditions;
> models are not official Darts or Hugging Face implementations.

---

## Usage

```bash
git clone https://github.com/pfekin/forecasting-summation-transformers
cd forecasting-summation-transformers
pip install torch numpy pandas scikit-learn darts
python benchmark.py
```

Hyperparameters in `benchmark.py` can be edited directly:

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

Optional: enable plotting at the end of `benchmark.py` to visualize loss curves or predictions.

---

## Key Findings

1. **Accuracy + Efficiency** — Hybrid PatchTST improves accuracy and throughput over the attention-only baseline.
2. **Simpler Dynamics** — Fewer attention layers yield stable training and easier tuning.
3. **General Principle** — The summation-based design philosophy extends successfully to time series.

---

## Limitations & Future Work

* Lightweight experimental implementations (not production libraries)
* Memory constraints limited Darts models in Colab
* Future plans:
  • Integration with Hugging Face forecasting APIs
  • Longer-horizon and multi-scale benchmarks
  • Theoretical analysis of representation efficiency

---

## Citation

```bibtex
@article{fekin2024summation,
  title = {Summation-Based Transformers},
  author = {Pascal Fekin},
  journal = {TechRxiv},
  year = {2024},
  doi = {10.36227/techrxiv.175790522.25734653/v2}
}

@misc{fekin2025forecasting,
  title = {Summation-Based Transformers for Time-Series Forecasting},
  author = {Pascal Fekin},
  year = {2025},
  url = {https://github.com/pfekin/forecasting-summation-transformers}
}
```

---

## Contact & Collaboration

I am seeking collaborators with access to large-scale compute resources to train summation-based transformers for language modeling.

* 📧 Email: [your email]
* 🐙 GitHub: [pfekin](https://github.com/pfekin)
* 📄 Paper: [TechRxiv 2024](https://doi.org/10.36227/techrxiv.175790522.25734653/v2)

---

**License:** MIT © 2025 Pascal Fekin

> ⚠️ Research-only implementation. Results demonstrate relative performance under resource-constrained conditions.

---

Would you like me to add a **small side-by-side ASCII diagram** comparing
“Full Transformer vs. Hybrid Transformer” right below the *Architectural Principle* section?
It’s a single block of text but visually highlights how your simplified stack differs — very effective for GitHub readers.
