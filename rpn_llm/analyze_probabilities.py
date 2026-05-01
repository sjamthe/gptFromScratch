import torch
import torch.nn.functional as F
import os
import sys
import json

# Add rpn_llm to path
sys.path.append(os.path.abspath("rpn_llm"))
from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def analyze_probs(checkpoint_path, prompt, force_mask=None):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # 1. Load Model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    if force_mask is not None:
        config.use_phase_mask = force_mask
        
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # 2. Tokenizer
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = RPNTokenizer(os.path.join(script_dir, "rpn-tokenizer.json"))
    
    # 3. Tokenize Prompt
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # 4. Calculate Phase IDs (Internal Logic)
    # We detect phase shifts in the input prompt to maintain mask consistency
    # [REV]=10, [MATH]=11, [ANS]=12
    token_list = tokens
    phase_ids = []
    current_phase = 0
    for t in token_list:
        if t in [10, 11, 12]:
            current_phase += 1
        phase_ids.append(current_phase)
    
    full_phase_ids = torch.tensor(phase_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # 5. Inference
    with torch.no_grad():
        # We pass a dummy targets tensor of the same shape as x 
        # This forces the model to return logits for the FULL sequence (line 311 in model_rope.py)
        # instead of just the last token.
        dummy_targets = torch.zeros_like(x) 
        logits, _ = model(x, targets=dummy_targets, full_phase_ids=full_phase_ids)
        # Shape of logits: (1, T, vocab_size)
        T = logits.shape[1]
        
    # 6. Analyze every position
    for t in range(T):
        probs = F.softmax(logits[0, t, :], dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # The context up to this token
        context_str = tokenizer.decode(tokens[:t+1]).replace("\n", "\\n")
        results = [f"'{context_str}':"]
        
        cum_prob = 0
        for i in range(len(sorted_probs)):
            p = sorted_probs[i].item()
            idx = sorted_indices[i].item()
            token_str = tokenizer.decode([idx])
            token_display = token_str.replace("\n", "\\n")
            
            results.append(f"'{token_display}'")
            results.append(f"{p:.2f}")
            
            cum_prob += p
            if cum_prob >= 0.90:
                break
        
        print(",".join(results))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="The input prompt")
    parser.add_argument("--force_mask", type=str, default=None, choices=["True", "False"], help="Override use_phase_mask")
    
    args = parser.parse_args()
    
    fmask = None
    if args.force_mask == "True": fmask = True
    elif args.force_mask == "False": fmask = False
    
    analyze_probs(args.model, args.prompt, fmask)
