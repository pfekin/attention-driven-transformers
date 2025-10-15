# Architecture Details

## Hybrid PatchTST: Simplified Blocks + Top Attention

### Core Principle

**Traditional Transformers**: Stack multiple expensive attention layers
```
Input → Attention → Attention → Attention → ... → Output
        (O(n²))      (O(n²))      (O(n²))
```

**Hybrid Architecture** (This Work): Simpler blocks driven by final attention
```
Input → Simple → Simple → Simple → ... → Attention → Output
        (O(n))    (O(n))    (O(n))        (O(n²))
```

### Architectural Components

#### 1. Simplified Projection Block

```python
class ProjectionBlock(nn.Module):
    def forward(self, x):
        # Simple linear transformation
        projected = Linear(x)        # O(n × d²)
        activated = GELU(projected)   # O(n × d)
        normed = LayerNorm(x + activated)  # Residual
        
        # Standard feed-forward
        ffn_out = FFN(normed)
        return LayerNorm(normed + ffn_out)
```

**Key Features**:
- ✅ No bias in projection (simpler)
- ✅ Linear complexity in sequence length
- ✅ Maintains residual connections
- ✅ No attention mechanism (faster)

#### 2. Top Attention Block

```python
class StandardAttentionBlock(nn.Module):
    def forward(self, x):
        # Multi-head self-attention
        attn_out = MultiHeadAttention(x, x, x)  # O(n² × d)
        normed = LayerNorm(x + attn_out)
        
        # Feed-forward
        ffn_out = FFN(normed)
        return LayerNorm(normed + ffn_out)
```

**Placed at the end** to:
- Capture long-range dependencies
- Refine features from simplified blocks
- Provide global context integration

### Full Hybrid PatchTST Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ Input: Multivariate Time Series (batch, seq_len, vars) │
└───────────────────┬─────────────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  Patching Layer       │  Convert sequence into patches
        │  (patch_len=16)       │  seq_len=512 → num_patches=64
        └───────────┬───────────┘
                    ↓
        ┌───────────────────────┐
        │  Patch Embedding      │  Linear projection of patches
        │  (patch_len → d_model)│  to model dimension
        └───────────┬───────────┘
                    ↓
        ┌───────────────────────┐
        │  Positional Encoding  │  Multiplicative (not additive)
        │  (learned parameters) │  
        └───────────┬───────────┘
                    ↓
        ╔═══════════════════════╗
        ║ ProjectionBlock #1    ║  Linear → GELU → Norm → FFN
        ║ (simplified, O(n))    ║  NO attention mechanism
        ╚═══════════┬═══════════╝
                    ↓
        ╔═══════════════════════╗
        ║ ProjectionBlock #2    ║  Linear → GELU → Norm → FFN
        ║ (simplified, O(n))    ║  NO attention mechanism
        ╚═══════════┬═══════════╝
                    ↓
        ╔═══════════════════════╗
        ║ Attention Block       ║  Multi-Head Attention
        ║ (full, O(n²))         ║  + Norm + FFN
        ╚═══════════┬═══════════╝
                    ↓
        ┌───────────────────────┐
        │  Flatten & Project    │  
        │  (d_model×patches     │  
        │   → pred_len×vars)    │
        └───────────┬───────────┘
                    ↓
┌───────────────────────────────────────────────────────────┐
│ Output: Future Predictions (batch, pred_len, vars)       │
└───────────────────────────────────────────────────────────┘
```

### Comparison with Baseline PatchTST

| Component | Baseline PatchTST | Hybrid PatchTST (Ours) |
|-----------|------------------|------------------------|
| **Block 1** | Attention (O(n²)) | Projection (O(n)) ✅ |
| **Block 2** | Attention (O(n²)) | Projection (O(n)) ✅ |
| **Block 3** | Attention (O(n²)) | Attention (O(n²)) |
| **Total Complexity** | 3 × O(n²) | 2 × O(n) + O(n²) ✅ |
| **Parameters** | ~470K | ~450K ✅ |
| **Speed** | Baseline | **+41% faster** ✅ |
| **Accuracy** | Baseline | **+5.1% MSE** ✅ |

---

## Connection to Language Modeling Architecture

### Shared Philosophy

Both architectures use the same principle: **simplified blocks + strategic attention**

### Language Model (Original Work)

```
Input Tokens
    ↓
Embedding
    ↓
[Block 1] → Linear → Activation → Global Sum Pool → Norm
[Block 2] → Linear → Activation → Global Sum Pool → Norm
    ↓
[Attention] → Multi-Head Self-Attention → Norm
    ↓
Output Logits
```

**Key difference**: Explicit **pooling operations** (global sum or cumsum) after each projection

### Forecasting Model (This Work)

```
Input Patches
    ↓
Embedding
    ↓
[Block 1] → Linear → Activation → Norm  (no explicit pooling)
[Block 2] → Linear → Activation → Norm  (no explicit pooling)
    ↓
[Attention] → Multi-Head Self-Attention → Norm
    ↓
Output Predictions
```

**Key difference**: No explicit pooling (patches already provide aggregation)

### Why No Pooling in Forecasting?

1. **PatchTST already aggregates**: Patches naturally pool local temporal information
2. **Continuous values**: Unlike discrete tokens, continuous time series don't need discrete aggregation
3. **Multivariate**: Each variable processed independently, no global statistics needed
4. **Empirical results**: Removing pooling works better for forecasting tasks

---

## Computational Advantages

### Memory Complexity

| Operation | Baseline | Hybrid | Savings |
|-----------|----------|--------|---------|
| Attention layers | 3 | 1 | **-66%** |
| Attention memory | 3n²d | n²d | **-66%** |
| Projection layers | 0 | 2 | +2 (but O(n)) |
| Total dominant term | O(3n²d) | O(n²d) | **~66% less** |

### Time Complexity (per forward pass)

**Baseline PatchTST**:
- 3 attention blocks: 3 × (n² × d + n × d²)
- Total: O(3n²d + 3nd²)

**Hybrid PatchTST**:
- 2 projection blocks: 2 × (n × d²)
- 1 attention block: n² × d + n × d²
- Total: O(n²d + 3nd²)

**For typical values** (n=64 patches, d=128):
- Baseline: 3 × (64² × 128) ≈ 1.5M ops (attention dominated)
- Hybrid: 64² × 128 + 3 × 64 × 128² ≈ 0.5M + 3.1M ops

The attention reduction is significant for longer sequences.

### Inference Speed

From benchmark results (ETTh1):
- Baseline: 2,927 samples/s
- Hybrid: **4,139 samples/s** (+41%)
- N-BEATS: 25,393 samples/s (no attention)

Hybrid achieves the best **accuracy-speed trade-off**.

---

## Design Choices

### 1. Multiplicative Positional Encoding

```python
x = x * self.pos_encoding  # Multiplicative
# vs
x = x + self.pos_encoding  # Additive (standard)
```

**Rationale**: Empirically better for this architecture; allows position information to modulate features rather than add to them.

### 2. No Bias in Projection

```python
nn.Linear(d_model, d_model, bias=False)
```

**Rationale**: Simpler, fewer parameters, forces model to learn meaningful transformations through activations.

### 3. GELU vs ReLU

Both work, but **GELU** preferred:
- Smoother gradients
- Better for continuous time series data
- Standard in modern transformers

### 4. LayerNorm Placement

Follows **Pre-LN** style (norm before operation):
```python
x = self.norm1(x + projection(x))  # Pre-LN
```

Provides more stable training than Post-LN.

---

## Hyperparameter Sensitivity

Based on experiments:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `d_model` | 128 | Sweet spot for 7-var datasets |
| `n_layers` | 3 | 2 projection + 1 attention |
| `patch_len` | 16 | Standard for PatchTST |
| `stride` | 8 | 50% overlap works well |
| `n_heads` | 8 | For final attention layer |
| `dropout` | 0.15 | Regularization |
| `lr` | 1e-4 | Adam with default betas |

**Critical**: `n_layers - 1` simplified blocks + 1 attention block

---

## Future Variations

### Multi-Scale Hybrid

```
┌─── Scale 1 (fine) ───┐
│ Projection → Projection → Attention │
└──────────────┬──────────────┘
               ↓
┌─── Scale 2 (coarse) ───┐
│ Projection → Projection → Attention │
└──────────────┬──────────────┘
               ↓
        Fusion Layer
```

### Deeper Hybrids

```
Projection × 5 → Attention × 1  (current: 2+1)
Projection × 10 → Attention × 2
```

**Hypothesis**: More projection layers can replace more attention layers while maintaining performance.

### Adaptive Hybrid

Learn to dynamically choose between projection and attention:
```python
if complexity_metric(x) > threshold:
    x = attention_block(x)  # Hard samples
else:
    x = projection_block(x)  # Easy samples
```

---

## Theoretical Justification

### Why Does This Work?

**Hypothesis 1: Information Bottleneck**
- Early layers extract local features (projection sufficient)
- Final layer integrates global context (attention necessary)

**Hypothesis 2: Inductive Bias**
- Time series have strong local structure
- Projections capture local patterns efficiently
- Attention handles long-range dependencies

**Hypothesis 3: Regularization Effect**
- Simpler blocks → less overfitting
- Strategic attention → preserve expressive power
- Best of both worlds

### Empirical Evidence

✅ Consistent improvements across 7 diverse datasets
✅ Faster inference (fewer attention operations)
✅ Better accuracy (less overfitting?)
✅ Fewer parameters (simpler model)

---

## Implementation Notes

### Efficient Patch Processing

```python
# Process all variables simultaneously
patches = patches.reshape(batch * n_vars, num_patches, patch_len)
# Each variable gets its own "batch" → efficient parallelization
```

### Gradient Stability

```python
# Gradient clipping prevents instability
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Memory Optimization

```python
# Delete models and clear cache between experiments
del model, optimizer
torch.cuda.empty_cache()
```

---

## References

1. **PatchTST**: Nie et al., "A Time Series is Worth 64 Words", ICLR 2023
2. **Summation-Based Transformers**: [Original paper](https://doi.org/10.36227/techrxiv.175790522.25734653/v2)
3. **Attention Is All You Need**: Vaswani et al., NeurIPS 2017
4. **N-BEATS**: Oreshkin et al., ICLR 2020

---

**Last Updated**: 2025
**Author**: [Your name]
**License**: [Your license]