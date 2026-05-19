import torch
import torch.nn.functional as F
import os
import sys
import numpy as np

# Ensure working directory is in python path
sys.path.append(os.getcwd())

from rpn3_llm.model_rope import GPT, GPTConfig
from rpn3_llm.utils import RPNTokenizer

def compare_representations(model_path, prompts_file, device='cpu'):
    # 1. Load Checkpoint & Model
    print(f"Loading model checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Enable phase mask config parameter compatibility
    if not hasattr(config, 'use_phase_mask'):
        config.use_phase_mask = True
        
    model = GPT(config)
    
    # Strip unwanted DDP prefixes
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    tokenizer = RPNTokenizer("rpn3_llm/rpn-tokenizer.json")
    
    # 2. Read Prompts
    if not os.path.exists(prompts_file):
        print(f"Prompts file not found: {prompts_file}")
        return
        
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
        
    print(f"Loaded {len(prompts)} test prompts from {prompts_file}.\n")
    
    # 3. Setup hooks to capture intermediate states at the LAST token
    # We will store: layer_name -> list of vectors (one per prompt)
    layer_states = {}
    
    # Temporal holder for hooks during a single forward pass
    current_pass_activations = []
    
    def make_hook(name):
        def hook(module, input, output):
            if "After_Attn" in name:
                # input[0] is the residual stream x after Attention addition but before ln_2/MLP
                val = input[0]
            else:
                # For Block output, it's a tuple: (x, cache_out, weights)
                val = output[0] if isinstance(output, tuple) else output
            
            # Extract last token vector (1, T, D) -> (D,)
            vec = val[0, -1, :].detach().cpu().float()
            current_pass_activations.append((name, vec))
        return hook

    # Register hooks
    hooks = []
    # Bottom layer: after WTE
    hooks.append(model.transformer.wte.register_forward_hook(make_hook("WTE")))
    
    # Intermediate blocks/passes
    if config.universal:
        # We hook ln_2 to capture the state right after Attn addition
        hooks.append(model.transformer.h.ln_2.register_forward_hook(make_hook("After_Attn")))
        # We hook the whole block to capture the state after MLP addition
        hooks.append(model.transformer.h.register_forward_hook(make_hook("After_MLP")))
    else:
        for idx, block in enumerate(model.transformer.h):
            hooks.append(block.ln_2.register_forward_hook(make_hook(f"Block{idx}_After_Attn")))
            hooks.append(block.register_forward_hook(make_hook(f"Block{idx}_After_MLP")))
            
    # Top layer: after LN_F
    hooks.append(model.transformer.ln_f.register_forward_hook(make_hook("LN_F")))
    
    # 4. Run Forward Passes
    predictions = []
    for idx, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        x_in = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # Prepare Phase Masking
        phase_ids = []
        curr = 0
        for t in tokens:
            if t in [10, 11, 12]:
                curr += 1
            phase_ids.append(curr)
        full_phase_ids = torch.tensor(phase_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        # Clear holder
        current_pass_activations.clear()
        
        # Run forward pass
        with torch.no_grad():
            logits, _ = model(x_in, use_cache=False, full_phase_ids=full_phase_ids)
            
        # Get predictions for the last token
        last_token_logits = logits[0, -1, :].detach().cpu().float()
        probs = F.softmax(last_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=10)
        
        pred_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
        pred_probs = [p.item() * 100 for p in top_probs]
        
        predictions.append(list(zip(pred_tokens, pred_probs)))
        
        # Process and store captured activations
        # In Universal Transformer, Block hook is hit multiple times. We rename them sequentially: Block_Pass0, Block_Pass1, etc.
        attn_counter = 0
        mlp_counter = 0
        for name, vec in current_pass_activations:
            final_name = name
            if name == "After_Attn":
                final_name = f"Pass{attn_counter}_After_Attn"
                attn_counter += 1
            elif name == "After_MLP":
                final_name = f"Pass{mlp_counter}_After_MLP"
                mlp_counter += 1
            if final_name not in layer_states:
                layer_states[final_name] = []
            layer_states[final_name].append(vec)
            
    # Clean up hooks
    for h in hooks:
        h.remove()
        
    # 5. Output Prediction Summary
    print("=== Next-Token Prediction Details ===")
    for idx, prompt in enumerate(prompts):
        short_prompt = prompt if len(prompt) < 50 else prompt[:25] + "..." + prompt[-25:]
        pred_str = ", ".join([f"'{token}': {prob:.1f}%" for token, prob in predictions[idx]])
        print(f"Prompt {idx+1}: {short_prompt:<55} -> Top Predictions: {pred_str}")
    print("\n" + "="*80 + "\n")
    
    # 6. Analyze Layer Representations
    layer_names = list(layer_states.keys())
    
    print("=== Layer-by-Layer Representation Cosine Similarity Analysis ===")
    print("This shows the mean cosine similarity between the last-token vectors of all 7 prompts at each layer.")
    print("-" * 80)
    print(f"{'Layer Name':<20} | {'Mean Pairwise Cosine Similarity':<32}")
    print("-" * 80)
    
    for layer in layer_names:
        vectors = layer_states[layer]
        num_prompts = len(vectors)
        
        similarities = []
        for i in range(num_prompts):
            for j in range(i + 1, num_prompts):
                v_i = vectors[i].unsqueeze(0)
                v_j = vectors[j].unsqueeze(0)
                sim = F.cosine_similarity(v_i, v_j).item()
                similarities.append(sim)
                
        mean_sim = np.mean(similarities)
        print(f"{layer:<20} | {mean_sim:^32.4f}")
        
    print("\n" + "="*80 + "\n")
    
    # 7. Detailed Pairwise Cosine Similarity Matrix for all layers
    for layer in layer_names:
        print(f"=== Pairwise Cosine Similarity Matrix for Layer: {layer} ===")
        vectors = layer_states[layer]
        num_prompts = len(vectors)
        
        # Header
        header = f"{'Prompt #':<10}"
        for i in range(num_prompts):
            header += f" | P{i+1:<5}"
        print(header)
        print("-" * len(header))
        
        for i in range(num_prompts):
            row = f"P{i+1:<8}"
            for j in range(num_prompts):
                v_i = vectors[i].unsqueeze(0)
                v_j = vectors[j].unsqueeze(0)
                sim = F.cosine_similarity(v_i, v_j).item()
                row += f" | {sim:.3f}"
            print(row)
        print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint model file (.pt)")
    parser.add_argument("--prompts", type=str, default="rpn3_llm/data/7.txt", help="Path to text file containing test prompts")
    args = parser.parse_args()
    
    compare_representations(args.model, args.prompts)
