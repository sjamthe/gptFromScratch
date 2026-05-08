# Teacher Forcing Reversal Analysis

## 1. The Experiment
We created a fully replicated validation logic script using `DataLoaderLite` directly on the `RPNData-1-22_phase_lean_val.txt` validation set. We evaluated the accuracy of next-token predictions in the `[REV]` region under Teacher Forcing (where the model is given the ground truth prefix at every step, meaning it only has to predict $P(x_{t} | x_{<t_{true}})$).

## 2. Key Findings: The 8% Noise Floor

*   **The `[REV]` Token Transition is Solved:** The model correctly predicts the `[REV]` token transition 100% of the time (0 failures out of 692 validation prompts). It perfectly understands *when* to start the reversal.
*   **Flat Positional Error Rate:** The failures occur uniformly across the reversed digits. The model exhibits a persistent, baseline **~7-8% probability of outputting the wrong digit at any given step**. It does *not* hit a "positional wall" or "stamina limit" at specific depths.
*   **The Math of Exposure Bias:** Because Auto-Regressive generation requires getting every token right in sequence, a flat 92% token success rate yields an expected 22-digit sequence success rate of around $0.92^{22} \approx 16\%$. This compounding token-level noise explains why generative boundary tests show much lower scores than the training logs.

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

## 4. Conclusions on the Bottleneck

The failure of the `ut1.8M` model, and now the **Standard GPT (`rope`) models**, to break the 93% ceiling is a definitive result. By testing untied standard GPTs and doubling the embedding dimension to `384` (yielding 3.6M parameters), the performance did not change by even a fraction of a percent.

**This firmly rules out THREE major hypotheses:**
1.  **Not an Attention Routing Failure:** If the model struggled to route information over long distances due to attention collapse, architectural tweaks (Theta, Recency Bias, MOHSA) would have shifted the noise profile. They did not.
2.  **Not a Residual Stream Capacity Constraint:** If the `192` embedding dimension was simply too small to reliably encode digit identities and positional encodings simultaneously, the `384` dimension model would have easily resolved those "collisions" and improved accuracy. It did not.
3.  **Not a Universal Transformer Issue:** Untying the weights across layers (Standard GPT/`rope` models) yielded the exact same noise profile. The forced parameter sharing of UTs is not the culprit.

**Remaining Suspects:**
1.  **Optimization Limits:** Aggressive learning rate schedules, weight decay, or AdamW epsilon values might be trapping the model in a noisy local minimum early in training. Why does every model hit the exact same local minimum? Possibly because the loss landscape is dominated by the same regularizers.
2.  **Dataset / Masking Bugs:** The `1-22_phase_lean` dataset itself, or the masking logic applied during training, might contain a ~7% error rate, strictly capping what the model can learn.
