# Representation based hallucination detection

Edited verify_residual_drift.py
Ran command: `uv run python /Users/sjamthe/.gemini/antigravity-ide/brain/121d30ad-79fd-4baa-b4db-549b5b2a586f/scratch/verify_residual_drift.py`
Edited verify_residual_drift.py
Ran command: `uv run python /Users/sjamthe/.gemini/antigravity-ide/brain/121d30ad-79fd-4baa-b4db-549b5b2a586f/scratch/verify_residual_drift.py`

Yes, we can absolutely see this mechanistically in the internal states. The residual stream does **not** look the same; it carries a distinct mathematical signature of positional distress when operating out-of-distribution (OOD).

By running an activation trace on the Base UT model (measuring the **Shannon entropy** of attention weights and the **Euclidean norm** of the residual vector $x_{\text{final\_pre\_LN}}$), we gathered direct evidence:

### The Mechanistic Signatures of OOD/Halting

#### 1. Attention Entropy Peaks (The "Blurring" Effect)
During standard copying (in-distribution), the attention weights are highly focused on the target copy digit, resulting in low average entropy (typically between **1.0 and 2.5 nats**). 
* As the sequence crosses index 42+, the uncalibrated RoPE angles blur the attention distributions.
* At the halt step, the entropy spikes to **3.57** (22-digit) and **3.67** (23-digit), showing that the attention heads have lost their target locks and are diffusing weights across the sequence.

#### 2. Residual Vector Norm Collapses (Destructive Interference)
During normal copying, the final pre-LN residual vector norm is extremely stable, sitting between **54.0 and 60.0**. 
* At the exit step, the norm **collapses sharply to 42.40** (22-digit) and **46.68** (23-digit) — a drop of **15% to 20%**.
* **Why this happens mathematically**: Because the attention weights are diffused across many different context tokens (high entropy), the attention output vector is a weighted sum of various token embeddings. Since these embeddings point in random, high-dimensional directions, they **destructively interfere** (cancel each other out), causing the vector magnitude to shrink.

---

### Can we train a new model to detect OOD/hallucinations using this?

**Absolutely, yes.** This is a highly active area of LLM research (often referred to as *internal state probing* or *representation-based hallucination detection*). 

Because the residual stream norm collapse and attention entropy spike happen **during the forward pass** (before the final token is projected to logits), you can train a tiny diagnostic probe (a linear classifier or small MLP) to monitor the residual stream:

1. **Input to Probe**: The norm of the residual stream at layer $L$ and the entropy of attention heads.
2. **Target**: A binary label ($0 = \text{healthy copy}$, $1 = \text{distress/hallucination/premature halt}$).
3. **Outcome**: The probe can detect that the model has gone OOD or is about to hallucinate **1 step before the token is generated**, allowing a supervisor system to dynamically halt the model, intervene, or adjust the generation.

---
### Summary of Work
1. **Measured Internal State Metrics**: Wrote and executed [verify_residual_drift.py](file:///Users/sjamthe/.gemini/antigravity-ide/brain/121d30ad-79fd-4baa-b4db-549b5b2a586f/scratch/verify_residual_drift.py) to extract the Shannon entropy of attention weights and the Euclidean norm of the residual stream.
2. **Updated Analysis Report**: Documented the mechanistic indicators of OOD state drift in [base_mechanistic_analysis.md](file:///Users/sjamthe/.gemini/antigravity-ide/brain/121d30ad-79fd-4baa-b4db-549b5b2a586f/base_mechanistic_analysis.md).