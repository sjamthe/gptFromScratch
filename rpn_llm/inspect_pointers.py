import torch
from model_rope import GPT
from utils import RPNTokenizer

def inspect_pointers(checkpoint_path, prompt_str):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

    tokens = tokenizer.encode(prompt_str)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    token_labels = [f"{i}:{tokenizer.decode([t])}" for i, t in enumerate(tokens)]

    with torch.no_grad():
        _, _, _, all_weights = model(x, return_attention=True)

    print(f"\n--- Pointer Analysis for: {prompt_str} ---")
    
    # We want to look at the attention from the "Scratchpad" tokens back to the "Prompt" tokens
    # Find '<' (scratchpad start)
    lt_id = tokenizer.encode("<")[0]
    try:
        start_idx = tokens.index(lt_id)
    except:
        start_idx = 0

    for layer_idx, layer_weights in enumerate(all_weights):
        print(f"\n[LAYER {layer_idx + 1}]")
        weights = layer_weights[0] # (heads, T, T)
        
        for head_idx in range(weights.shape[0]):
            print(f"  Head {head_idx + 1}:")
            found_pointers = 0
            # Look at tokens in the scratchpad (after <)
            for i in range(start_idx, len(tokens)):
                attn_row = weights[head_idx, i, :i+1]
                top_vals, top_indices = torch.topk(attn_row, min(3, len(attn_row)))
                
                target_label = token_labels[i].split(":")[-1]
                is_last_token = (i == len(tokens) - 1)
                
                # Show diagnostics for last token or digits
                if not (is_last_token or any(c.isdigit() for c in target_label)): 
                    continue

                # Show Top 5 connections for deep diagnostics
                top_vals, top_indices = torch.topk(attn_row, min(5, len(attn_row)))
                
                for val, idx in zip(top_vals, top_indices):
                    source_label = token_labels[idx.item()]
                    print(f"    {token_labels[i]:<12} -- attends to --> {source_label:<12} (Weight: {val.item():.2f})")
                    found_pointers += 1
            
            if found_pointers == 0:
                print("    (No strong pointers to prompt digits found)")

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "rpn_llm/models/rope1.6M_1-22_uniform_BOS_80000.pt"
    prompt = sys.argv[1] if len(sys.argv) > 1 else "[BOS]123 456+?<"
    inspect_pointers(ckpt, prompt)
