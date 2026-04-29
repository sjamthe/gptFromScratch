# RPN-NAT: Non-Autoregressive Transformer for Math

This project implements a Non-Autoregressive Transformer (NAT) designed to generate arithmetic scratchpads in parallel using a **Shifted-Alignment Masked Encoder**.

## The Strategy: Shifted-Alignment & Mixed-Noise

We have moved beyond standard Non-Autoregressive generation to a more robust "Denoising" architecture that separates the prompt from the answer workspace.

### 1. Shifted Alignment
To resolve "Fractal Echoes" and alignment confusion:
- **Input Structure:** `[Prompt Tokens (P)] + [Answer Slots (128 tokens)]`
- **Target Structure:** `[Mask (-100) * P] + [Ground Truth Answer (A)] + [Stop Token ([UNK]) * (128-A)]`
- **Logic:** The model at position `P + i` is explicitly tasked with predicting `Answer[i]`. The prompt acts as a pure context window, and the model uses its bidirectional attention to fill the answer slots.

### 2. Mixed-Noise Training (Denoising)
Instead of a fixed mask probability, we vary the noise for every sample to teach the model different skills:
- **Generation (30%)**: 100% masked. The model builds the answer from the prompt alone.
- **Refinement (40%)**: 50-90% masked. The model learns to fill in missing logic.
- **Polishing (30%)**: 0-20% masked. The model learns to fix small typos and "shut up" (predict stop tokens) in unused slots.

### 3. Fixed Workspace (128 Tokens)
- We use a fixed **128-token workspace** for all problems. 
- This is large enough to handle 5-6 digit subtractions with complex carry/borrow scratchpads.
- The model is trained to predict the `[UNK]` stop token once the math is finished, preventing hallucinations at the end of the line.

### 4. Iterative Refinement Inference
1. **Initialize:** Start with `Prompt + [MASK] * 128`.
2. **Parallel Prediction:** The model predicts all 128 answer slots simultaneously.
3. **Refine:** We feed the 128-token "guess" back into the model for 5-10 iterations.
4. **Denoise:** The model uses its bidirectional attention to "polish" the guess, ensuring mathematical consistency across the entire sequence.

---

## Current Performance (48k Progress)
- **Reversal Accuracy**: Mastery approaching (Gets 3-4 digit reversals almost perfectly).
- **0-Carry Math**: ~97% accurate (Conditional on correct reversal).
- **Carry/Borrow Logic**: Learning rapidly. Model is often off by only 1-2 digits on 5-digit problems.

## To Run Inference
```bash
python rpn_nat/inference_nat.py --prompt "[BOS]689 949-?"
```

## To Train
```bash
python rpn_nat/train_nat.py --dataset 1-22_uniform_BOS
```
