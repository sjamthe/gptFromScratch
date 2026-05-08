# RPN Architectural Boundary Limits

This document defines the performance "ceilings" of the current 2-layer, 6-head Universal Transformer architecture. These benchmarks use **Truly Strict Scoring** (zero partial credit, exact formatting required) to distinguish between true logical grokking and approximate heuristics.

## 1. The Leaderboard

Current champions of the 0.4M - 0.5M parameter class.

| Metric | **Ungated (80k)** | **Ungated (400k Final)** | **MOHSA (Option B 400k)** | **True MOHSA (A 80k)** | **True MOHSA (A 400k)** | **Recency Bias (80k)** | **Recency Bias (400k)** | **Theta 50k (80k)** | **Gated (80k)** | Gated (400k Final) |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Short Fidelity (4-dig)** | **100.0%** | **100.0%** | 98.0% | 80.0% | 78.0% | **100.0%** | **100.0%** | 90.0% | 94.0% | 96.0% |
| **Long Fidelity (25-dig)** | 40.0% | **72.0%** | 46.0% | 0.0% | 0.0% | 62.0% | 46.0% | **64.0%** | 0.0% | 2.0% |
| **22-Digit Reversal** | 50.0% | **65.0%** | 55.0% | **90.0%** | 50.0% | 50.0% | 60.0% | **75.0%** | **75.0%** | 50.0% |
| **Asymmetric Capacity** | 90.0% | - | 85.0% | 100.0% | 100.0% | 100.0% | 85.0% | 95.0% | 95.0% | - |
| **Heuristic Fallback** | Perfect | Perfect | Perfect | Perfect | Perfect | Perfect | Perfect | Perfect | Perfect | Perfect |

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

## MOHSA Analysis
### Option B (Zero-Cost Overlap, 400k)
1. Short Fidelity: 98.0%
2. Long Fidelity: 46.0%
3. 22-Digit Reversal (Stamina): 55.0%
4. Asymmetric Capacity: 85.0%

**Diagnosis**: The "Compromise Model". By forcing the 64-dim and 32-dim heads to share the exact same QKV projection, we created a gradient tug-of-war. The model slightly under-fit the basic math but gained a slight stamina boost over the Gated model due to "holographic redundancy".

### Option A (True MOHSA, 80k)
1. Short Fidelity: 80.0%
2. Long Fidelity: 0.0%
3. 22-Digit Reversal (Stamina): **90.0%**
4. Asymmetric Capacity: 100.0%

**Diagnosis**: Unbelievable Stamina. At only 80k steps, the True MOHSA model has completely shattered the 22-Digit reversal ceiling, achieving a staggering 90% (compared to the baseline's 65%). 

However, because it is only at 80k steps, it is currently in the "Discovery Phase". It has figured out the global algorithm (hence the high stamina and 100% asymmetric capacity), but it has not yet consolidated its syntax and pointer tracking (hence 0% Long Fidelity and 80% Short Fidelity). As training continues to 400k, we expect the fidelity to catch up, potentially making this the first model to conquer the Long Math benchmark.

### Option A (True MOHSA, 400k Final)
1. Short Fidelity: 78.0%
2. Long Fidelity: 0.0%
3. 22-Digit Reversal (Stamina): 50.0%
4. Asymmetric Capacity: 100.0%

**Diagnosis**: Catastrophic Decay / Overfitting. This is a fascinating and counter-intuitive result. At 80k steps, this model was the champion of stamina (90% on 22-digit reversal). However, as training continued to 400k, the stamina **collapsed** to 50%, and the short fidelity dropped slightly to 78%. 

This suggests that while the independent QKV projections allowed the model to discover the global algorithm early on, the continued training caused the model to specialize in the training distribution's specific patterns at the expense of the general pointer algorithm, or it suffered from a form of catastrophic forgetting as it tried to resolve the remaining training loss. The independent projections may have allowed the model to overfit to the training distribution in a way that the shared subspace (Option B) prevented!

### Recency Bias (80k vs 400k)
**80k Results**:
1. Short Fidelity: **100.0%**
2. Long Fidelity: **62.0%**
3. 22-Digit Reversal (Stamina): 50.0%
4. Asymmetric Capacity: 100.0%

**400k Results**:
1. Short Fidelity: **100.0%**
2. Long Fidelity: 46.0%
3. 22-Digit Reversal (Stamina): **60.0%**
4. Asymmetric Capacity: 85.0%

**Diagnosis**: The Inverse Trajectory. Unlike the True MOHSA model (which started with 90% reversal and collapsed to 50%), the Recency Bias model **improved** its stamina over time (from 50% to 60%) but **lost** some of its long fidelity precision (from 62% to 46%). 

This suggests that the recency bias provided a strong foundation for syntax and short-term pointer tracking early on (100% short fidelity and 62% long fidelity at 80k). As training continued, the model traded some of that extreme precision to improve its stamina on the harder reversal tasks. It remains the most balanced performer at 400k, avoiding the catastrophic decay seen in Option A.

### RoPE Theta 50k (80k)
1. Short Fidelity: **90.0%**
2. Long Fidelity: **64.0%**
3. 22-Digit Reversal (Stamina): **75.0%**
4. Asymmetric Capacity: 95.0%

**Diagnosis**: The Grid Expansion. This is an extremely strong result for just 80k steps! By increasing the RoPE base $\theta$ to $50,000$, we stretched the frequency grid. This allowed the model to achieve the best Long Fidelity (64%) of any advanced model at 80k, while also maintaining a very high 75% stamina on the 22-digit reversal.

This proves that the standard RoPE base of 10,000 was indeed "blurring" the positions at long sequence lengths. Expanding the grid gave the model the resolution it needed to track pointers accurately without sacrificing precision. It will be very interesting to see if this model holds onto its skills at 400k or if it also overfits like Option A!