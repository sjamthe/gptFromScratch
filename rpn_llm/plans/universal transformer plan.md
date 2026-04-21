# Universal Transformer (Weight Sharing) Experiment

Transition the RPN Arithmetic Transformer to a Universal Transformer architecture where all layers share the same weights. This encourages algorithmic abstraction by forcing the model to learn a single "Universal Step" of logic.

## User Review Required

> [!IMPORTANT]
> **Parameter Reduction**: This change will reduce the model's parameter count significantly (from ~25M to ~8M), as we are collapsing 8 layers into 1. This means the model will fit more easily in memory but will rely entirely on recursive refinement.

> [!NOTE]
> **Pass Embeddings**: I am adding "Coordinate Embeddings" (depth signals). Without these, the model is "time-blind" and might struggle to know if it's currently "fetching" or "calculating."

## Proposed Changes

### Model Architecture

#### [MODIFY] [model_rope.py](file:///Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/model_rope.py)
- **GPTConfig**: Add `universal` flag and `n_layer` (now representing "n_passes").
- **GPT Class**:
    - Replace `nn.ModuleList([Block(...)])` with a single `self.shared_block = Block(config)`.
    - Initialize `self.pass_emb = nn.Parameter(torch.zeros(n_layer, n_embd))` to provide a depth signal.
    - Update `forward` to loop `n_layer` times using `self.shared_block` while adding the corresponding `pass_emb`.
- **Diagnostic Compatibility**: Ensure `return_attention` still works by returning the attention weights from every pass in the loop.

## Open Questions

- **Fixed vs. Adaptive Passes**: For this first run, should we fix it at 8 passes (for direct comparison) or allow it to be configurable? (My recommendation is to keep it at 8 to start).

## Verification Plan

### Automated Tests
1. **Model Parameter Check**: Run a script to verify that `named_parameters()` lists only one block.
2. **Attention Verification**: Run `visualize_attention.py` on the untrained model to ensure it can still capture weights from 8 passes.

### Manual Verification
1. **Training Run**: Start a new run using `train_rpn.py` with the `universal` flag enabled and monitor loss convergence compared to the 25M previous run.
