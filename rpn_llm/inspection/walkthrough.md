# Walkthrough: Mechanistic Interpretability Tools

We have successfully added tools to peer into the "brain" of the RPN Transformer. These tools help identify where reversal logic and arithmetic logic reside in the model's layers.

## Key Features

### 1. Diagnostic Hooks in `model_rope.py`
We added `return_attention` parameters to the model's forward pass.
- **Flash Attention Bypass**: Since Flash Attention hides its weights for speed, the model now carries a manual attention implementation that is triggered during diagnostic runs.
- **Weight Extraction**: The model returns a list of attention tensors of shape `[layer, batch, head, query_len, total_key_len]`.

### 2. Zero-Dependency Visualizer
The `visualize_attention.py` script creates a stunning HTML report without needing `matplotlib` or `seaborn` in the environment.
- **Attention Heatmaps**: Visualizes which tokens each layer is "attending to."
- **Logit Lens**: Prints a table of what each layer "thinks" the next token should be.

## How to Inspect the Model

To generate a report for a specific arithmetic problem:

```bash
.venv/bin/python3 rpn_llm/visualize_attention.py "(your_prompt_here)"
```

**Example Output (Logit Lens):**
For the prompt `(123)(456)+?`, you might see the model's guess evolve:
- **Layer 1-2**: Random noise or prompt echoes.
- **Layer 3-4**: Correct reversed digits start appearing in the high-probability tokens.
- **Layer 6-8**: The final mathematical result becomes the dominant guess.

## Files Created/Modified
- [model_rope.py](file:///Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/model_rope.py): Added internal state hooks.
- [visualize_attention.py](file:///Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/visualize_attention.py): The main inspection utility.
- [report.html](file:///Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/inspection/report.html): The generated output.

> [!TIP]
> Use the **Logit Lens** to find the exact layer where the reversal logic "crystallizes." This is often where the model transitions from simply echoing the prompt to performing meaningful computation.
