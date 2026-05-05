import torch
import torch.nn.functional as F
import os
import sys
import matplotlib.pyplot as plt

# Add parent dir to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def visualize_attention(model_path, prompt, output_image="rpn_llm/analysis/attention_map_ood.png", device='cpu'):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Forward pass with return_attention=True
    # Since it's a UT, we'll look at the last pass (or a specific pass)
    with torch.no_grad():
        # Precompute freqs_cis
        T = idx.shape[1]
        freqs_cis = model.freqs_cis[:T].to(device)
        
        # We'll run the model for one step and capture the weights
        # Actually, let's run it until it starts reversing
        # To see the full map, we need a 2D matrix [SeqLen, SeqLen]
        # Our model returns weights for the LAST token by default if use_cache is on
        # So we'll pass the whole sequence at once
        # For the visualization, we want to pass the full sequence (targets=None)
        # to get a 2D attention map
        logits, _, _, all_weights = model(idx, return_attention=True)
        # all_weights is a list of [Batch, Head, T, T] for each layer
        # In UT, all layers are the same, so we just take the first list element
        weights = all_weights[0][0].cpu() # [Head, T, T]

    n_head = weights.shape[0]
    fig, axes = plt.subplots(2, (n_head + 1) // 2, figsize=(20, 10))
    axes = axes.flatten()
    
    token_labels = [tokenizer.decode([t]) for t in tokens]
    
    for h in range(n_head):
        ax = axes[h]
        im = ax.imshow(weights[h].numpy(), cmap="viridis")
        ax.set_title(f"Head {h}")
        # Only show a subset of labels for readability if too long
        if len(token_labels) > 40:
             ax.set_xticks(range(0, len(token_labels), 5))
             ax.set_xticklabels(token_labels[::5], rotation=90)
             ax.set_yticks(range(0, len(token_labels), 5))
             ax.set_yticklabels(token_labels[::5])
        else:
             ax.set_xticks(range(len(token_labels)))
             ax.set_xticklabels(token_labels, rotation=90)
             ax.set_yticks(range(len(token_labels)))
             ax.set_yticklabels(token_labels)

    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Attention map saved to {output_image}")

if __name__ == "__main__":
    # The 25-digit problem that failed
    prompt = "[BOS]9202970213556909275571928 6654310278681447394731217+? [REV]2"
    model_path = "rpn_llm/models/ut1.5M_mlp3_phaseMask_True_1-22_phase_lean_56000.pt"
    
    visualize_attention(model_path, prompt)
