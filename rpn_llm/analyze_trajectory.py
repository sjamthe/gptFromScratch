import torch
import torch.nn.functional as F
import os
import sys
import json

# Add current dir to path to import model
sys.path.append(os.getcwd())
from rpn_llm.model_rope import GPT, GPTConfig
from rpn_llm.utils import RPNTokenizer

def analyze_trajectory(model_path, prompt, device='cpu'):
    # 1. Setup
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    
    # Handle state dict (strip 'module.' if it exists from DDP)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    tokens = tokenizer.encode(prompt)
    x_in = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # 2. Phase IDs for masking
    phase_ids = []
    curr = 0
    for t in tokens:
        if t in [10, 11, 12]: curr += 1
        phase_ids.append(curr)
    full_phase_ids = torch.tensor(phase_ids, dtype=torch.long, device=device).unsqueeze(0)

    # 3. Trajectory Tracking
    # We want to capture x at various points
    trajectory = [] # List of (step_name, tensor)
    
    def hook_fn(name):
        def hook(module, input, output):
            # output is often a tuple (x, cache, weights) for blocks
            if isinstance(output, tuple):
                trajectory.append((name, output[0].detach().cpu()))
            else:
                trajectory.append((name, output.detach().cpu()))
        return hook

    # Register hooks
    # Initial embedding
    model.transformer.wte.register_forward_hook(hook_fn("init_wte"))
    
    # Each block
    for i, block in enumerate(model.transformer.h):
        # We need to capture after Attn and after MLP
        # Since the Block.forward does x = x + attn; x = x + mlp,
        # we can hook the internal components and then manually calculate the sum
        # but a cleaner way is to hook the components themselves.
        block.attn.register_forward_hook(hook_fn(f"B{i}_attn_out"))
        block.mlp.register_forward_hook(hook_fn(f"B{i}_mlp_out"))

    # 4. Run Inference
    with torch.no_grad():
        # Pass dummy targets to get full logits if needed, but here we just need trajectory
        _ = model(x_in, full_phase_ids=full_phase_ids)

    # 5. Process Trajectory
    # trajectory[0] is init_wte (1, T, D)
    final_ln = model.transformer.ln_f
    init_emb = trajectory[0][1]
    T = init_emb.shape[1]
  
    # We'll reconstruct the state at each step
    # x_current = init_emb
    # After B0_attn_out: x = x + attn_out
    # After B0_mlp_out: x = x + mlp_out
    
    states = [("Initial",  init_emb)]
    current_x = init_emb.clone()
    
    for name, delta in trajectory[1:]:
        current_x = current_x + delta
        states.append((name, current_x))

    current_x = final_ln(current_x)
    states.append(("Final_ln", current_x.clone()))
    
    logits_pred = model.lm_head(current_x)
    pred_token_ids = torch.argmax(logits_pred[0], dim=-1)

    # 6. Final Logits for the last token prediction
    #with torch.no_grad():
    #    dummy_targets = torch.zeros_like(x_in)
    #    logits, _ = model(x_in, targets=dummy_targets, full_phase_ids=full_phase_ids)
    #    pred_token_ids = torch.argmax(logits[0], dim=-1) # (T,)

    # 7. Identify "Goal" Embeddings
    # For every t, the goal is the embedding of the NEXT token
    # For the very last token, the goal is the embedding of its own prediction
    wte = model.transformer.wte.weight.detach().cpu()
    goal_embs = []
    for t in range(T):
        if t < T - 1:
            target_id = tokens[t+1]
        else:
            target_id = pred_token_ids[-1].item()
        goal_embs.append(final_ln(wte[target_id]))
    goal_embs = torch.stack(goal_embs) # (T, D)

    # 8. Print Results (Transposed for readability)
    if 0:
        # --- TABLE 1: L2 DISTANCE FROM INITIAL ---
        print(f"\n--- TABLE 1: L2 DISTANCE FROM INITIAL EMBEDDING ---")
        header = f"{'Token'.ljust(15)} | {'Target'.rjust(8)} |"
        for name, _ in states:
            header += f"{name.rjust(12)} |"
        header += f"{'Expected shift'.rjust(12)} | {'Predicted'.rjust(8)} |"
        print(header)
        print("-" * len(header))

        for t in range(T):
            token_str = tokenizer.decode([tokens[t]]).replace("\n", "\\n")
            
            if t < T - 1: target_str = tokenizer.decode([tokens[t+1]])
            else: target_str = tokenizer.decode([pred_token_ids[-1].item()])
            target_str = target_str.replace("\n", "\\n")

            row = f"{token_str.ljust(15)} | {target_str.rjust(8)} |"
            for _, state in states:
                with torch.no_grad():
                    norm_curr = state[0, t].to(device)
                    norm_init = init_emb[0, t].to(device)
                    d = torch.linalg.norm(norm_curr - norm_init).item()
                row += f"{d:12.2f} |"
            
            # Shift Distance: How far is Initial[t] from Initial[t+1] in the prediction space?
            with torch.no_grad():
                norm_init = init_emb[0, t].to(device)
                norm_goal = goal_embs[t].to(device)
                d_shift = torch.linalg.norm(norm_goal - norm_init).item()

            # predicted token
            pred_str = tokenizer.decode([pred_token_ids[t].item()])
            pred_str = pred_str.replace("\n", "\\n")
            row += f"{d_shift:14.2f} | {pred_str.rjust(8)} |"
            print(row)

    # --- TABLE 2: COSINE SIMILARITY TO GOAL (Normalized) ---
    print(f"\n--- TABLE 2: COSINE SIMILARITY TO GOAL (via Final LN) ---")
    header = f"{'Token'.ljust(15)} | {'Target'.rjust(8)} |"
    for name, _ in states:
        header += f"{name.rjust(12)} |"
    header += f"{'Shift'.rjust(12)} |"
    print(header)
    print("-" * len(header))

    for t in range(T):
        token_str = tokenizer.decode([tokens[t]]).replace("\n", "\\n")
        
        if t < T - 1: target_str = tokenizer.decode([tokens[t+1]])
        else: target_str = tokenizer.decode([pred_token_ids[-1].item()])
        target_str = target_str.replace("\n", "\\n")

        row = f"{token_str.ljust(15)} | {target_str.rjust(8)} |"
        for _, state in states:
            with torch.no_grad():
                normalized_state = state[0, t].to(device).unsqueeze(0)
                norm_goal = goal_embs[t].to(device).unsqueeze(0)
                sim = F.cosine_similarity(normalized_state, norm_goal).item()
                row += f"{sim:12.4f} |"
        
        # Shift Similarity: How aligned is Initial[t] with Initial[t+1]?
        with torch.no_grad():
            norm_init = init_emb[0, t].to(device).unsqueeze(0)
            norm_goal = goal_embs[t].to(device).unsqueeze(0)
            sim_shift = F.cosine_similarity(norm_init, norm_goal).item()
        
        # predicted token
        pred_str = tokenizer.decode([pred_token_ids[t].item()])
        pred_str = pred_str.replace("\n", "\\n")
        row += f"{sim_shift:12.4f} | {pred_str.rjust(8)} |"
        print(row)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()
    
    analyze_trajectory(args.model, args.prompt)