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