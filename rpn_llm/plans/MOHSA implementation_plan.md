# Implementation Plan: Shared Subspace Transformer (MOHSA)

This document outlines the architectural changes required to implement the "Overlapping Head" design. The goal is to create a Multi-Scale attention mechanism where the embedding dimensions are shared between "Large" (high-resolution) and "Small" (fast) heads.

## Background: The "Overlapping 32 with 64" Concept
Your embedding size is **192**. 
We will split this space in two overlapping ways simultaneously:
1.  **Large Scale**: 3 heads of dimension **64** ($3 \times 64 = 192$).
2.  **Small Scale**: 6 heads of dimension **32** ($6 \times 32 = 192$).

Because both scales sum up to 192, every single dimension in the embedding space will be processed by **one Large Head and one Small Head simultaneously**.

## Proposed Architecture Changes (Option B: Zero-Cost Overlap)

This approach strictly maintains the current parameter count (0.4M) by reusing your existing projections.

### 1. Shared $QKV$ Projection
We will use your existing `c_attn` layer to project $X \rightarrow Q, K, V$ (each of size 192). 
This projection is shared. There are **zero** new parameters added to the model.

### 2. Dual-Scale Branching & RoPE
Immediately after projection, we branch the $Q$ and $K$ tensors:
*   **Large Branch**: We reshape the raw $Q, K, V$ into 3 heads of dimension 64. We apply a 64-dim RoPE rotation to $Q_{large}$ and $K_{large}$.
*   **Small Branch**: We reshape the **same** raw $Q, K, V$ into 6 heads of dimension 32. We apply a 32-dim RoPE rotation to $Q_{small}$ and $K_{small}$.

Even though they start from the exact same raw vectors, applying different RoPE frequencies and computing Softmax over different dimensions will cause the Large and Small branches to generate completely different attention patterns.

### 3. The "Overlapping" Output Fusion
Both the Large Attention and Small Attention branches will perform standard scaled dot-product attention and output tensors of shape `(Batch, Time, 192)`.
We fuse them by summation before the final projection:
$$Y_{combined} = Y_{large} + Y_{small}$$
$$Output = W_{proj}(Y_{combined})$$

### 4. KV Cache Doubling
Because the Large and Small branches apply different RoPE rotations, they must maintain separate KV caches.
*   The `cache_state` will now store a tuple of 4 tensors instead of 2: `(k_cache_large, v_cache_large, k_cache_small, v_cache_small)`.
*   This doubles the memory footprint during inference, but does not affect the parameter count or training speed significantly.

### 5. Implementation in `model_rope.py`
We will:
1. Update `GPT.forward` to generate and pass down two sets of RoPE frequencies (`freqs_cis_large` and `freqs_cis_small`).
2. Rewrite `CausalSelfAttention.forward` to execute the dual-branch logic.
3. Add a `use_mohsa` flag to `GPTConfig` to toggle this on/off for backward compatibility.

## User Review Required

> [!TIP]
> **Experimental Design**
> This Option B is the "Holy Grail" of ablations. If this model beats the 21-digit Law, it proves definitively that **architecture topology**, not raw parameter count, is the key to deep reasoning.

Please approve this updated plan, and I will implement the MOHSA logic in `model_rope.py` immediately!


# Implementation Plan: "True" MOHSA (Option A)

This document outlines the architectural changes to upgrade our current "Zero-Cost Overlap" model into a **"True" Multi-Overlapped-Head Self-Attention** model. 

## The Core Concept
The previous model suffered from "gradient tug-of-war" because the 64-dim Large heads and the 32-dim Small heads were forced to share the exact same $QKV$ projection matrix. 

In **Option A**, we decouple the projections. The model will learn *two completely different ways* to project the exact same embedding space:
1.  A projection optimized for tracking long-range carry bits (Large Heads).
2.  A projection optimized for fast, local syntax tracking (Small Heads).
Because their outputs ($Y_{large}$ and $Y_{small}$) are summed together at the end, the model still achieves the "Holographic Fusion" of logic, but without the bottleneck during backpropagation.

## Proposed Architectural Changes (`model_rope.py`)

### 1. Independent $QKV$ Projections
In `CausalSelfAttention.__init__`, if `use_mohsa` is True, we will initialize a second projection matrix:
*   `self.c_attn_large = nn.Linear(config.n_embd, 3 * config.n_embd)`
*   `self.c_attn_small = nn.Linear(config.n_embd, 3 * config.n_embd)` (Reusing the existing `self.c_attn` variable name for backward compatibility on non-MOHSA models).

### 2. Dual-Branch Forward Pass
In `CausalSelfAttention.forward`, we will compute the projections independently:
```python
qkv_large = self.c_attn_large(x)
q_l, k_l, v_l = qkv_large.split(self.n_embd, dim=2)
# Reshape to 3 heads of 64...

qkv_small = self.c_attn_small(x)  # (reusing c_attn)
q_s, k_s, v_s = qkv_small.split(self.n_embd, dim=2)
# Reshape to 6 heads of 32...
```

### 3. RoPE and Caching
*   The RoPE logic and the doubled KV Cache logic we implemented for Option B will remain completely unchanged, as they already perfectly support dual-resolution attention.

### 4. Output Fusion
*   Both branches will calculate attention and output `(Batch, Time, 192)`.
*   We will sum them: `y = y_large + y_small`.
*   We will project the sum through `self.c_proj`.

## Parameter Cost
*   **Added Parameters**: `192 * (3*192) + (3*192)` = **111,168** parameters.
*   **Total Model Size**: Increases from ~0.38M to **~0.49M** parameters.

## User Review Required

> [!TIP]
> **Checkpoints Compatibility**
> Because we are adding a completely new `Linear` layer (`c_attn_large`), any previous checkpoints (including the `mohsa_400k` checkpoint you just trained) will **NOT** load cleanly into this new architecture. This will be a completely fresh training run.

Please review and approve this plan! Once approved, I'll execute the change and we can start the "True MOHSA" training run.
