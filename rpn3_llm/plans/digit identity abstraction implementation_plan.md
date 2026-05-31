# Implementation Plan: Digit-Identity Abstraction

This plan outlines the design and implementation of the **Digit-Identity Abstraction** (content-blind attention routing) inside standard self-attention and coordinate heads in `model_rope.py`, and its integration into the training pipeline in `train_rpn.py`.

## Goal Description
Standard self-attention heads learn a content-matching shortcut during supervised fine-tuning (SFT) to perform digit sequence reversal. When encountering duplicate digits in out-of-distribution (OOD) contexts, positional signals blur, and standard attention heads alias/route to incorrect matching characters (duplicate digits) earlier in the sequence. 

The **Digit-Identity Abstraction** forces the model to ignore digit values when routing attention (specifically during the reversal phase), relying exclusively on positional/coordinate signals (provided by RoPE and the Coordinate Heads). 

During the reversal (`REV`) phase, standard attention $Q$ and $K$ are projected from an *abstracted sequence representation* where all digit face values are mapped to a generic token (e.g., `'0'`, ID 13). Standard attention $V$ is projected from the original un-abstracted representation, allowing the model to copy the actual digit face values to the target positions without content-based attention aliasing.

---

## User Review Required

> [!IMPORTANT]
> **Dual-Stream Execution Design:**
> To prevent copied digit values from leaking into standard attention's $Q$ and $K$ inputs in downstream layers, we propose a dual-stream residual evolution:
> 1. **Standard stream (`x_standard`):** Carries the actual digit face values through the value projections ($V$) and MLPs to ultimately predict the correct output tokens.
> 2. **Abstracted stream (`x_abstracted`):** Synthesizes a digit-blind version of the hidden states at every layer by routing and updating a sequence where all digit tokens are replaced by `'0'`. Standard attention queries ($Q$) and keys ($K$) are projected from this stream for any query position that is in the reversal phase.
>
> This requires no additional parameters, as both streams share the exact same weights (LayerNorms, projection weights, and MLPs).

> [!TIP]
> **Coordinate Scale Freezing:**
> In prior runs, the learnable coordinate head scale parameter decayed from `0.5` to `0.204` at 200k steps because the optimizer favored the standard attention content shortcut. To prevent this, we propose adding a flag `freeze_coord_scale` to keep the scale frozen at `0.5`.

---

## Open Questions

> [!NOTE]
> None. The design is fully specified by the dual-stream weight-tied architecture.

---

## Proposed Changes

### Core Architecture Modifications

#### [MODIFY] [model_rope.py](file:///Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn3_llm/model_rope.py)

1. **Update `GPTConfig`**:
   Add new configurations:
   ```python
   use_digit_abstraction: bool = False  # Enable digit-identity abstraction
   freeze_coord_scale: bool = False  # Freeze coordinate head scale parameter
   ```

2. **Update `CoordinateHead`**:
   If `freeze_coord_scale` is active, freeze the scale parameter:
   ```python
   self.scale = nn.Parameter(
       torch.tensor(0.5), 
       requires_grad=not getattr(config, 'freeze_coord_scale', False)
   )
   ```

3. **Modify `GPT.forward` for Dual-Stream Initialization & Routing**:
   * If `use_digit_abstraction` is enabled:
     * Construct `idx_abstracted` by replacing digit tokens (IDs 13 to 22) in `idx` with the generic digit token ID (13, representing `'0'`).
     * Embed both: `x_standard = self.transformer.wte(idx)` and `x_abstracted = self.transformer.wte(idx_abstracted)`.
     * Extract `is_rev_phase = (global_phase_ids == 1)` (shape `[B, T]`) or retrieve it from `full_phase_ids[:, -1:] == 1` for step-by-step inference.
     * Sequentially feed both `x_standard` and `x_abstracted` to the blocks.
     * Run `CounterHead` and `CoordinateHead` on the abstracted stream `x_abstracted` (for perfect digit-blind counting/relative coordinates) and add their outputs to both streams `x_standard` and `x_abstracted`.
     * Pass the final layer's `x_standard` output to `ln_f` and the `lm_head` for predictions.
   * If disabled, execute the single-stream path exactly as before.

4. **Modify `Block.forward`**:
   Support dual-stream input signatures:
   ```python
   def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, use_cache: bool = False, cache_state: tuple = None, return_attention: bool = False, attn_mask: torch.Tensor = None, head_mask: torch.Tensor = None, x_abstracted: torch.Tensor = None, is_rev_phase: torch.Tensor = None) -> tuple:
   ```
   If `x_abstracted` is provided:
   * Compute independent LayerNorms: `norm_x_std = self.ln_1(x)` and `norm_x_abs = self.ln_1(x_abstracted)`.
   * Pass both streams to `self.attn` to get outputs for both streams: `attn_out_std, attn_out_abs, cache_out, weights`.
   * Apply residuals and project MLPs on both streams independently using shared parameters.

5. **Modify `CausalSelfAttention.forward` for Selection Routing**:
   If `x_abstracted` is provided:
   * Project $Q, K, V$ for standard stream from `x_standard`.
   * Project $Q, K, V$ for abstracted stream from `x_abstracted`.
   * Apply RoPE, tau scaling, and transpose.
   * Compute standard scores `scores_std = Q_std @ K_std^T / sqrt(d)` and abstracted scores `scores_abs = Q_abs @ K_abs^T / sqrt(d)`.
   * Select attention scores: `scores = torch.where(is_rev_phase.view(B, 1, T, 1), scores_abs, scores_std)`.
   * Compute `attn_probs = softmax(scores)` with causal and phase masks.
   * Apply attention probs to values: `y_std = attn_probs @ V_std` and `y_abs = attn_probs @ V_abs`.
   * Project both outputs through `self.c_proj` and return them.
   * *Note:* Update KV Cache caching logic to cache both standard and abstracted keys/values: `cache_state = (k_std, v_std, k_abs, v_abs)`.

---

### Training Pipeline Integration

#### [MODIFY] [train_rpn.py](file:///Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn3_llm/train_rpn.py)

1. **Add CLI Flags**:
   * `--use_digit_abstraction`: Boolean flag to enable the abstraction.
   * `--freeze_coord_scale`: Boolean flag to freeze coordinate head scale at 0.5.

2. **Update GPTConfig Construction**:
   Pass `use_digit_abstraction` and `freeze_coord_scale` to `GPTConfig`.

3. **Update Model Run Name Prefix**:
   Add naming tags `_digitAbs` and `_freezeCoordScale` if enabled.

4. **Update Generation Loop (`run_generation_validation`)**:
   Ensure `is_phase_shift` and `full_phase_ids` logic inside generation accurately passes to the model.

---

## Verification Plan

### Automated Tests

1. **Logits Parity and KV-Cache Correctness**:
   Write a validation script (e.g. `test_digit_abstraction.py`) that runs forward passes with and without KV cache for a generated sequence, verifying that:
   * Logits match exactly (`1e-5` float tolerance) between cache-less and cached modes.
   * Digit face values are indeed abstracted inside standard attention score matrices during the reversal phase.

2. **Gradient Health and Training**:
   Run training for 20 steps locally with `--use_digit_abstraction` and `--freeze_coord_scale` to ensure no shape mismatches or autograd failures, and check that standard attention weights and coordinate head scale parameters behave correctly.

### Manual Verification
* Compare validation curves on WandB to ensure convergence matches or exceeds the baseline model on SFT while showing increased robustness to OOD sequence lengths.
