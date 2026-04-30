# Model sizing 

Try to create smallest model that achieves 99% accuracy on the test set.

# Hyperparameter tuning 

## 1st model (7.1M Parameters) - [SUCCESS]
 - n_layer = 4
 - n_head = 6
 - n_embd = 384
 - block_size = 512 
 - Notes: Increasing T to 512 solved the 17-22 digit bottleneck. Reversal is 99.9% solved.

 ### Transformer Weights (7,123,200 Total)
 - **Vocab Embeddings (wte)**: `64 × 384 = 24,576`
 - **Block (Per Layer)**: `1,774,464`
   - Attention (QKV): `384 × 1152 + bias = 443,520`
   - Attention (Proj): `384 × 384 + bias = 147,840`
   - MLP (FC): `384 × 1536 + bias = 591,360`
   - MLP (Proj): `1536 × 384 + bias = 590,208`
   - LayerNorms: `768 + 768 = 1,536`
 - **Total Layers (4)**: `1,774,464 × 4 = 7,097,856`
 - **Final LayerNorm**: `768`

 ### Results:
 - 99.74% accuracy at 48k steps
 - 99.93% accuracy at 56k steps

## 2nd model (1.6M Parameters) - [FAILED - STRUCTURAL]
 - n_layer = 2
 - n_head = 4
 - n_embd = 256
 - block_size = 512
 - Notes: Reached 98.97% accuracy at 80k. Failed on reversals > 15 digits due to ROPE aliasing. 64-dim head capacity is too small for long sequences.
 - We wrote test_reversal_failures.py to test for this and proved that all reversal failures (98.97% - 99.15%) were due to a single failure mode (L1H2).

 ### Transformer Weights (1,601,104 Total)
 - **Vocab Embeddings (wte)**:
   - 'transformer.wte.weight': `64 × 256 = 16,384` # 64 token vocab size, 256-dim embedding
 - **Block (Per Layer)**:
   - 'transformer.h.0.ln_1.weight': `256`
   - 'transformer.h.0.ln_1.bias': `256`
   - 'transformer.h.0.attn.c_attn.weight': `256 × (256*3) = 197,376`
   - 'transformer.h.0.attn.c_attn.bias':  `(256*3) = 768`
   - 'transformer.h.0.attn.c_proj.weight': `256 × 256 = 65,792`
   - 'transformer.h.0.attn.c_proj.bias': `256`
   - 'transformer.h.0.ln_2.weight': `256`
   - 'transformer.h.0.ln_2.bias': `256`
   - 'transformer.h.0.mlp.c_fc.weight': `256 × (256*4) = 263,168`
   - 'transformer.h.0.mlp.c_fc.bias': `(256*4) = 1024`
   - 'transformer.h.0.mlp.c_proj.weight': `1024 × 256 = 262,400`
   - 'transformer.h.0.mlp.c_proj.bias': `256`
  - Sum of weights and biases for a single block: 256+256+197376+768+65792+256+256+256+263168+1024+262400+256 = 792,104
 - **Total Layers (2)**: `792,104 × 2 = 1,584,208`
 - **Final LayerNorm**: 
 - 'transformer.ln_f.weight' shape (256,)
 - 'transformer.ln_f.bias' shape (256,)
 - 'lm_head.weight' shape (64, 256) = 16,384 (these are same weights as wte so don't add again)

Total params = (792,104 × 2) + 256 + 256 + 16,384 = 1,601,104

 ### Results:
 - ~98.97% accuracy at 80k steps
 - Failed on reversals > 15 digits
 - Poor long-term memory (<10 digit carry)
 
## 3rd model (3.6M Parameters) - [SUCCESS]
 - n_layer = 2
 - n_head = 6
 - n_embd = 384
 - block_size = 512
 - Goal: Test if increasing width (d_k=64, but 6 heads and larger stream) solves the reversal capacity limit, isolating whether 2 layers is sufficient for carry math.
 - Goal: Verify if 2 layers are enough to maintain carry logic across 22 digits.

 ### Transformer Weights (3,574,272 Total)
 - **Vocab Embeddings (wte)**: `64 × 384 = 24,576`
 - **Block (Per Layer)**: `1,774,464`
   - Attention (QKV): `384 × 1152 + bias = 443,520`
   - Attention (Proj): `384 × 384 + bias = 147,840`
   - MLP (FC): `384 × 1536 + bias = 591,360`
   - MLP (Proj): `1536 × 384 + bias = 590,208`
   - LayerNorms: `768 + 768 = 1,536`
 - **Total Layers (2)**: `1,774,464 × 2 = 3,548,928`
 - **Final LayerNorm**: `768`

### Results
| Accuracy | Train Steps |
|----------|-------------|
| 66.75%   | 8k          |
| 96.20%   | 32k         |
| 99.77%   | 56k         |
| 99.88%   | 64k         |


### Analysis
Model did very well. Reversal was no problem like it was with 1.6M model.
<pre>
--- Failure Category Breakdown ---
reversal_skipped    : 0 (0.0%)
reversal_failed     : 4 (20.0%)
math_failed         : 6 (30.0%)
only_final_ans_failed: 14 (70.0%)
</pre>

We wrote a few scripts for mechanistic interpretability:
- ./test_reversal_failures.py: Tests for failures in the reversal task.
- ./analyze_layer_attributions.py: Study attribution by layer.

## 4th model (3.6M Parameters) Phased dataset and training- [SUCCESS]

### Dataset: `RPNData-1-22_phase_lean_test.txt`
We created a new dataset with a phased training approach to improve learning efficiency and performance.

Data is split into 4 phases
1. Prompt: [BOS] to [REV]
2. Reversal: [REV] to [MATH]
3. Math: [MATH] to [ANS]
4. Answer: [ANS] to [EOS]

Training shows only two phases at a time and masks others so model only learns from previous phase and predicts next.

**Example Format:**
[BOS]922 560-?[REV]229 065-=[MATH]2-0-0=2:2-6-0=6:9-5-1=3:[BORROW]0|+:263[ANS]362[EOS]

### Results
Model is more efficient and learns each phase better.
| Accuracy | Train Steps |
|----------|-------------|
| 91.67%   | 8k          |
| 98.75%   | 32k         |
| 99.47%   | 56k         |
| 99.38%   | 64k         |

<pre>
--- Failure Category Breakdown ---
reversal_skipped    : 0 (0.0%)
reversal_failed     : 9 (21.4%)
math_failed         : 36 (85.7%)
only_final_ans_failed: 6 (14.3%)
</pre>

### Training model with mask on and off and testing validation with mask on & off.

| Train | Validate | Accuracy |
|-------|----------|----------|
| mask off | mask off | 97.29% |
| mask off | mask on | 85.77% |
| mask on | mask on | 99.44% |
| mask on | mask off |  4.43% |

#### Summary
Model trained with blinders (phase Mask) does well when tested on same phase mask true, but when you test with no phase mask it gets too much information that it has never seen so it fails.  
On the contrary model trained with no phase mask does almost well when tested with no phase mask (expected), but when you test with phase mask true it still does ok, that means it is tried to work with correct logic, but sometimes was cheating by relying on data that was hidden on phase mask.