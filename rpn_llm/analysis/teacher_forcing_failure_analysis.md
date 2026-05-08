# Teacher Forcing Reversal Analysis

## 1. The Experiment
We created a fully replicated validation logic script using `DataLoaderLite` directly on the `RPNData-1-22_phase_lean_val.txt` validation set. We evaluated the accuracy of next-token predictions in the `[REV]` region under Teacher Forcing (where the model is given the ground truth prefix at every step, meaning it only has to predict $P(x_{t} | x_{<t_{true}})$).

## 2. Key Findings: The 8% Noise Floor (The Mirage)

*   **The `[REV]` Token Transition is Solved:** The model correctly predicts the `[REV]` token transition 100% of the time (0 failures out of 692 validation prompts). It perfectly understands *when* to start the reversal.
*   **Flat Positional Error Rate:** The failures occurred uniformly across the reversed digits. The model exhibited a persistent, baseline **~7-8% probability of outputting the wrong digit at any given step**. It did *not* hit a "positional wall" or "stamina limit" at specific depths.

## 3. Models Tested
Every architectural variation we tested at the 80k step mark hit the exact same ~93% performance ceiling with an identical error profile:

| Model Architecture | Config | Overall Rev Acc | Pos 1-10 Error | Pos 11-20 Error | Pos 21-30 Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ut0.3M mlp2** | `2l, 6h, 192e` | 92.80% | 8.31% | 7.44% | 5.53% |
| **ut0.4M mlp3** (Baseline) | `2l, 6h, 192e` | 92.91% | 8.04% | 7.42% | 5.60% |
| **ut0.4M Recency Bias** | `2l, 6h, 192e` | 92.71% | 8.34% | 7.56% | 5.73% |
| **ut0.4M Theta** | `2l, 6h, 192e` | 92.71% | 8.38% | 7.53% | 5.60% |
| **ut0.5M Gated** | `2l, 6h, 192e` | 92.85% | 8.13% | 7.46% | 5.66% |
| **ut0.5M MOHSA** | `2l, 6h, 192e` | 92.85% | 8.25% | 7.28% | 5.60% |
| **ut0.5M mlp4** | `2l, 6h, 192e` | 92.86% | 8.14% | 7.44% | 5.66% |
| **ut1.8M** (Capacity Scaled) | `2l, 6h, 384e` | 92.70% | 8.34% | 7.56% | 5.73% |
| **rope2.4M** (Standard GPT) | `3l, 4h, 256e` | 92.70% | 8.37% | 7.62% | 5.66% |
| **rope3.6M (No Phase Mask)** | `2l, 6h, 384e` | 92.68% | 8.16% | 7.85% | 5.98% |
| **rope3.6M (Phase Mask)** | `2l, 6h, 384e` | 92.58% | 8.50% | 7.62% | 5.85% |

## 4. The Resolution: The DataLoader Boundary Bug

The failure of the `ut1.8M` model, and the **Standard GPT (`rope`) models**, to break the 93% ceiling was deeply confusing, until we cross-referenced it with the Generative Boundary benchmark, which reported **99.42% sequence-level accuracy**.

It is mathematically impossible for a sequence to have a 99.42% success rate if its component tokens have an 8% failure rate. This led to the discovery of a catastrophic evaluation bug:

### The Bug
1. `DataLoaderLite` slices the data stream into arbitrary 512-token chunks. The *first sequence* in every chunk is almost always sliced in half, meaning its prompt is missing.
2. The Teacher Forcing evaluation mask (`(math_pos == 0)`) accidentally reset to `FALSE` the moment it hit the first `[MATH]` token in the chunk.
3. This forced the metric to **exclusively evaluate the reversal of the very first, chopped sequence**, completely ignoring the 2nd, 3rd, and 4th perfectly intact sequences in the same chunk.

Because the model lacked the full prompt for the chopped sequences, it generated random digits or spaces, creating the illusion of an 8% "noise floor."

### The True Results
After fixing the validation mask to dynamically track sequence boundaries (`has_bos`) and safely evaluate intact sequences across all phases, the true Teacher Forcing accuracies were revealed:

| Phase | Token Accuracy |
| :--- | :--- |
| **Reversal** (`[REV]`) | **99.98%** |
| **Mathematics** (`[MATH]`) | **100.00%** |
| **Final Answer** (`[ANS]`) | **99.99%** |

### Final Conclusion
**The Universal Transformer completely solved the Reversal Curriculum.** 
1. **No Attention Routing Failure:** The model perfectly routes attention across arbitrary distances to execute the dynamic offset `+2` pointer logic.
2. **No Capacity Constraints:** The `192` embedding dimension is fully sufficient to pack both digit identities and complex positional encodings.
3. **No Task Interference:** The model successfully decoupled the pointer logic from the arithmetic logic, achieving functionally 100% precision on both tasks simultaneously. 

The generative benchmark of 99.42% was authentic. We can definitively proceed to the Ten's Complement native arithmetic phase with extreme confidence in the Universal Transformer architecture.
