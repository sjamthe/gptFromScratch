# Non-Autoregressive Transformer (NAT) Implementation Plan

I finally understand your vision! You want to treat the model like an **Encoder (like BERT)** rather than a Decoder (like GPT). 

Instead of a continuous text stream, you want to feed **isolated, padded problems**, allowing the model to read the entire prompt simultaneously and attempt to output the answer simultaneously.

## Open Questions
> [!WARNING]
> **Positional Drift**: If `x` is `[BOS]123+?` (length 6), `y` is `<321` (length 6). This means `x[0]` (`[BOS]`) is tasked to predict `<`. 
> But if you add a token to the prompt: `x` is `[BOS]123+?<` (length 7). `y` is `<321=` (length 7). Here, `x[0]` is STILL tasked to predict `<`. 
> This is perfectly fine, but I want to make sure you know that the target `y` is "locked" to the input position. Are you okay with this 1-to-1 position mapping?

## Proposed Changes

### 1. Model Architecture (`rpn_nat/model_nat.py`)
We must **remove the Causal Triangle Mask**. 
- In standard GPT, Token 1 cannot see Token 2. 
- In your NAT, Token 1 *must* see Token 2, 3, and 4 so it understands the full math problem before it outputs its piece of the answer. 
- By removing the mask, the Transformer becomes a Bidirectional Encoder.

### 2. Dataset Generation (`rpn_nat/NATDataset.py`)
We cannot use the packed `DataLoaderLite` because if we remove the Causal Mask on a continuous stream, the model will just "cheat" and look ahead at the answer in the text file!
We will write a custom PyTorch `Dataset` that generates your exact isolated pairs:
- Read the text file and isolate a single problem: `Prompt` (length $P$) and `Answer` (length $A$).
- Pick a random $k \in [0, A]$.
- **Input `x`**: `Prompt + Answer[:k]`, padded to `block_size` with `0`.
- **Target `y`**: `Answer[:P+k]`, padded to `block_size` with `-100`.

### 3. Training Script (`rpn_nat/train_nat.py`)
- Wire the new `NATDataset` into the training loop.
- Use the new Bidirectional `model_nat.py`.

## Verification Plan
1. We will write a small visualization script to print the `x` and `y` arrays to guarantee they match your padding logic exactly.
2. We will run a 1,000-step training loop on a 3M parameter model to see if this NAT architecture is capable of decreasing the loss.
