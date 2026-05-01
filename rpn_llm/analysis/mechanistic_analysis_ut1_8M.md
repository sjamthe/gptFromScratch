# Mechanistic Analysis: The Pointer vs. Memory Conflict (UT 1.8M)

**Date:** May 1, 2026  
**Subject:** UT 1.8M Phase-Masked Model (Steps 48,000 - 80,000)  
**Objective:** Determine why simple 2-digit arithmetic accuracy drops as the model trains longer.

---

## 1. Executive Summary
Between 48k and 80k training steps, the model's accuracy on multi-digit problems improved, but its performance on simple 2-digit addition (e.g., `94 + 8`) significantly degraded. We performed a forensic analysis using **Weight Divergence Tracking** and **Counterfactual Inference**. 

**The Verdict:** The model is not "forgetting math." Instead, its MLP layers have developed powerful **Value-Specific Memory Attractors** that override its functional **Attention Pointers**. For certain digits (like 7 and 8), the model "shouts" a memorized pattern, ignoring the correct values it just calculated in its own scratchpad.

---

## 2. Evidence I: Weight Divergence Analysis
We compared the weights of the 48k checkpoint (high 2-digit accuracy) against the 80k checkpoint (low 2-digit accuracy).

| Component | Cosine Similarity | L2 Distance (Movement) |
| :--- | :--- | :--- |
| **MLP (c_fc / c_proj)** | **0.9910** | **2.01** |
| **Attention (c_attn)** | **0.9949** | **1.39** |

**Interpretation:** 
The Attention mechanism (the "Eyes" or "Pointers") remained relatively stable. However, the MLP (the "Calculation Logic" or "Memory Bank") underwent significantly more structural change. This suggests the model's "Retrieval" logic is still intact, but its "Processing" logic has been re-tuned for long-sequence expert math.

---

## 3. Evidence II: Counterfactual Inference (Causal Intervention)
We performed an experiment where we manually "hacked" the scratchpad during inference to see if the model was actually **Pointing** to its calculations or **Memorizing** the final answer string.

**Problem:** `8766 16 + ?` (Correct Answer: `8782`)

| Test Condition | Scratchpad Digits (Calculated) | Model's Final Answer | Verdict |
| :--- | :--- | :--- | :--- |
| **Original (7/8)** | `2, 8, 7, 8` | **`2788`** | **FAIL**: Memory forced a swap of 7/8. |
| **Neutral (0/9)** | `0, 9, 0, 9` | **`0909`** | **PASS**: Pointer correctly copied values. |
| **Descending (9/8/7)**| `2, 9, 8, 7` | **`2987`** | **PASS**: Pointer correctly copied values. |

### The "Smoking Gun"
In the **Neutral** and **Descending** cases, the model's attention heads successfully "pointed" to the scratchpad and copied the digits perfectly. This proves the **Pointer Logic is still functional.**

However, in the **Original** case involving 7s and 8s, the model's MLP "Memory Bank" recognized the numbers and "snapped" the output to a memorized template (`2788`), literally overwriting the signal from the attention heads.

---

## 4. Conclusion: The "Memory Dictator" Effect
As the 1.8M model tries to master 20-digit arithmetic, it "over-memorizes" common digit pairings to reduce loss on long sequences. Because the model has a low parameter count, it has no "spare capacity" to keep these memories separate from its logic.

1.  **At 48k steps**: The model is a **Generalist**. Its pointers are sharp and its memories are weak.
2.  **At 80k steps**: The model is a **Specialist**. It is an expert in long-digit math, but its MLP memories have become "Dictators" that override its pointers for common numbers.

### Recommendation
*   **Capacity Increase**: A 1.8M model is likely too small to hold both "High-Precision Pointers" and "Complex Math Memories." Moving to **3.6M** or **7M** parameters should provide the "residual buffer" needed to prevent memory interference.
*   **Dataset Balancing**: Up-sample short problems to ensure the model doesn't "sacrifice" simple logic to chase the high loss of long sequences.
