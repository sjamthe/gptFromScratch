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
