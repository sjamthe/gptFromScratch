# Design: Sequential Phase Masking for RPN-LLM (v2.0)

## 1. Objective
To improve reasoning in small models (1.3M) by using explicit "Phase Markers" and attention masking to enforce a strict information bottleneck.

## 2. Phase Architecture
The sequence is divided into four distinct phases. A token at Phase $N$ can only attend to tokens in Phase $N$ and Phase $N-1$.

| Phase | Content | Visibility | Logic |
| :--- | :--- | :--- | :--- |
| **0** | **Prompt** | Phase 0 | `[BOS]23 45+?` |
| **1** | **Reversal** | Phase 0, 1 | `[REV]32 54 +=` |
| **2** | **Math** | Phase 1, 2 | `[MATH]3+5=8:2+4=6:86` |
| **3** | **Answer** | Phase 2, 3 | `[ANS]68` |

## 3. Token Implementation
We introduce three new special tokens to the vocabulary:
1.  **`[REV]`**: Signals the start of the Digit Reversal phase.
2.  **`[MATH]`**: Signals the start of the Scratchpad calculation.
3.  **`[ANS]`**: Signals the start of the Final Answer extraction.

### New Format Example:
`[BOS]23 45+?[REV]32 54+=[MATH]3+5=8:2+4=6:86[ANS]68[EOS]`

## 4. Model Logic (`model_rope.py`)
The `forward` pass will calculate `phase_ids` by looking for these three specific token IDs.

```python
# Phase IDs increment on [REV], [MATH], [ANS]
is_phase_shift = (idx == REV_ID) | (idx == MATH_ID) | (idx == ANS_ID)
phase_ids = is_phase_shift.cumsum(dim=-1)

# Difference matrix (B, T, T)
phase_diff = (phase_ids.unsqueeze(1) - phase_ids.unsqueeze(2))

# Mask: Allow if tokens are in same phase (0) or previous phase (1)
phase_mask = (phase_diff == 0) | (phase_diff == 1)

# Combined with Document and Causal masks
full_mask = doc_mask & causal_mask & phase_mask
```

## 5. Information Carryover Rule
Because Phase 2 cannot see Phase 0, the **Operator** (`+` or `-`) **MUST** be copied into the Reversal section (Phase 1) so the Math logic knows what to do.

## 6. Implementation Status
- [x] **Tokenizer**: Added `[REV]`, `[MATH]`, `[ANS]` (IDs 10, 11, 12).
- [x] **Data**: Created `scratch/generate_phase_data.py` for Lean Phase format.
- [x] **Model**: Implemented `phase_mask` logic in `model_rope.py`.
- [/] **Train**: Trial run (3.6M RoPE) in progress.
