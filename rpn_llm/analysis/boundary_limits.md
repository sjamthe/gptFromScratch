# RPN Architectural Boundary Limits

This document defines the performance "ceilings" of the current 2-layer, 6-head Universal Transformer architecture. These benchmarks use **Truly Strict Scoring** (zero partial credit, exact formatting required) to distinguish between true logical grokking and approximate heuristics.

## 1. The Leaderboard

Current champions of the 0.4M - 0.5M parameter class.

| Metric | Ungated (344k) | **Ungated (400k Final)** | Gated (200k) | **Gated (400k Final)** |
|:---|:---|:---|:---|:---|
| **Short Fidelity (4-dig)** | 100.0% | **100.0%** | 96.0% | 96.0% |
| **Long Fidelity (25-dig)** | 76.0% | **72.0%** | 4.0% | 2.0% |
| **22-Digit Reversal** | 30.0% | **65.0%** | 50.0% | 50.0% |
| **Heuristic Fallback** | 0 | 0 | 0 | 0 |

## 2. Circuit Efficiency (Head Recruitment)

This tracks how the model's internal attention heads are utilized. Grokking is defined as the transition from "Noisy/Lazy" heads to "Critical/Logical" heads.

| Model Step | Logic Heads | Noisy/Distractor Heads | Status |
|:---|:---|:---|:---|
| Gated 80k | 0, 1, 4 | 3, 5 | **Partial Grokking**: Circuit is messy. |
| **Gated 400k** | **0, 1, 2, 3, 4, 5** | **None** | **TOTAL CONSOLIDATION**: 100% of bandwidth recruited. |

**Discovery**: By 400k steps, the Gated model has successfully "cleaned its room." Every single attention head (0-5) has become a critical component of the arithmetic circuit. Ablating ANY head now causes the logic to collapse. This proves the model has hit its architectural ceiling.

---

---

## 2. The Architectural "Frontiers"

These are the specific limits that have proven unbreakable for the current 2-layer design.

### A. Test: 22-Digit Consecutive Reversal (Generative Stamina)
The model's internal "counter" drifts after 21 digits, causing it to skip the final digit of the first number.
*   **Ungated 344k**: 30.0% Accuracy
*   **Gated 200k**: **50.0% Accuracy** (Gaining ground)
*   **Target**: 100%
*   **Hypothesis**: The model needs more "internal recurrence" (depth) to maintain the counter for 22+ digits.

### B. Test: 25-Digit Strict Math Fidelity (Scratchpad Capacity)
Tests the ability to solve a 25-digit addition problem perfectly under "gaslighting" conditions.
*   **Ungated 344k**: **76.0% Fidelity** (Mature Pointers)
*   **Gated 200k**: 4.0% Fidelity (Early breakthrough)
*   **Target**: 100%
*   **Hypothesis**: The Ungated model has spent 344k steps "sharpening" its attention heads for long-range connections. The Gated model is still building this "High-Resolution" pointer map.

### C. Test: Positional Shift-Invariance (Padding Test)
Tests if a model can solve a simple 4-digit problem when it is shifted 10+ tokens to the right by padding.
*   **Ungated 344k**: **0.0% Accuracy** (Complete collapse at 10 tokens)
*   **Gated 200k**: 0.0% Accuracy (Fails similarly)
*   **Target**: 100%
*   **Hypothesis**: The model has learned "Absolute Indexing" instead of "Relative Indexing." It is effectively "blind" if digits are not in their expected starting positions.

### D. Test: Asymmetric Capacity (N1=21 Stress Test)
Tests the maximum length of a second number (N2) that can be reversed perfectly when the first number (N1) is already at the 21-digit stamina limit.
*   **Ungated 344k**: **4 Digits** (95% accuracy)
*   **Gated 200k**: 3 Digits (85% accuracy at 4 digits)
*   **Target**: 21+ Digits (Full 21+21 reversal)
*   **Hypothesis**: This measures "Context Switching" stability. Even if a model has the stamina for one long number, the presence of a second number creates "Inter-Pointer Interference."

---

## 3. Proposed Breakthrough Strategies

To break these limits, we are considering the following architectural upgrades:

1.  **3-Layer Expansion**: Adding a third layer to increase the "recurrence resolution" for long sequences.
2.  **RoPE Tuning**: Adjusting the base or frequency of the Rotary Positional Embeddings to prevent "blurring" after 40 tokens.
3.  **Curriculum Randomized Padding**: Training with variable start-offsets to force the model to learn **position-invariant** reversal.
4.  **Wait for 400k**: Seeing if the Gated model naturally overcomes these stamina issues if given double the training time.
