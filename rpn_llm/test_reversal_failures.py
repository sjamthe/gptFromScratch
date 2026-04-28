import torch
import torch.nn.functional as F
from model_rope import GPT
from utils import RPNTokenizer

def test_reversal_failures(checkpoint_path, failures_file):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

    # Collect all reversal failures
    failures = []
    with open(failures_file, "r") as f:
        for line in f:
            if "Q:" in line:
                parts = line.split(" | ")
                prompt = parts[0].replace("Q: ", "").strip()
                full_exp = [p for p in parts if "Full Expected:" in p][0].replace("Full Expected: ", "").strip()
                
                full_pred = [p for p in parts if "Full Pred:" in p]
                if full_pred:
                    pred_str = full_pred[0].replace("Full Pred: ", "").strip()
                    exp_rev = full_exp.split("=")[0]
                    pred_rev = pred_str.split("=")[0]
                    if exp_rev != pred_rev:
                        failures.append((prompt + "<", full_exp))
                        
    if not failures:
        print("No reversal failures found in the file!")
        return

    print(f"Found {len(failures)} reversal failures. Analyzing all...")

    for idx, (failure_prompt, expected_full) in enumerate(failures):
        print(f"\n==================================================")
        print(f"Failure Case {idx+1}/{len(failures)}")
        print(f"Prompt: {failure_prompt}")
        
        tokens = tokenizer.encode(failure_prompt)
        start_idx = len(tokens) - 1 # Index of '<'
        expected_tokens = tokenizer.encode(expected_full)
        
        for step in range(100):
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _, _, all_weights = model(x, return_attention=True)
                
            next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 1. Logit Prediction (What the model actually outputs)
            predicted_token = torch.argmax(next_token_logits).item()
            predicted_char = tokenizer.decode([predicted_token])
            
            # 2. Mechanistic Prediction (Our Hypothesis)
            # Hypothesis: L1H2 acts as the Pointer Head. It looks at the prompt, 
            # and whichever prompt token it attends to the most is what the model copies.
            l1h2_prompt_attn = all_weights[0][0][1, -1, :start_idx] # Restrict to prompt (before <)
            mech_idx = torch.argmax(l1h2_prompt_attn).item()
            mech_token = tokens[mech_idx]
            mech_char = tokenizer.decode([mech_token])
            
            # 3. Expected Prediction (The Ground Truth)
            expected_token = expected_tokens[step + 1] if step + 1 < len(expected_tokens) else None
            exp_char = tokenizer.decode([expected_token]) if expected_token is not None else "None"
            
            # Check if our Mechanistic Interpretation is WRONG
            if mech_token != predicted_token:
                print(f"⚠️ MECHANISTIC MISMATCH at step {step+1}:")
                print(f"  Mechanistic Logic Predicted: '{mech_char}' (L1H2 pointed to index {mech_idx} with weight {l1h2_prompt_attn[mech_idx].item():.2f})")
                print(f"  Actual Logit Predicted   : '{predicted_char}'")
                print(f"  Ground Truth Expected    : '{exp_char}'")
            
            if expected_token is not None and predicted_token != expected_token:
                exp_char = tokenizer.decode([expected_token])
                exp_prob = probs[expected_token].item() * 100
                
                print(f"❌ MISTAKE at step {step+1}: Predicted '{predicted_char}', Expected '{exp_char}'")
                print(f"  Top Preds: '{tokenizer.decode([torch.topk(probs, 2).indices[0].item()])}' ({torch.topk(probs, 2).values[0].item()*100:.1f}%), '{tokenizer.decode([torch.topk(probs, 2).indices[1].item()])}' ({torch.topk(probs, 2).values[1].item()*100:.1f}%)")
                
                # Analyze Attention for the failing token
                token_labels = [f"{i}:{tokenizer.decode([t])}" for i, t in enumerate(tokens)]
                l1_weights = all_weights[0][0] # Layer 1
                
                print("  L1H2 (Pointer) Attends to:")
                attn_row = l1_weights[1, -1, :] # Head 2
                top_vals, top_indices = torch.topk(attn_row, 3)
                for val, i in zip(top_vals, top_indices):
                    print(f"    --> {token_labels[i.item()]:<10} (Weight: {val.item():.2f})")
                
                break # Move to the next failure case
            else:
                tokens.append(predicted_token)
                if predicted_char == "=":
                    break

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/models/rope1.6M_1-22_uniform_BOS_80000.pt"
    failures_file = sys.argv[1] if len(sys.argv) > 1 else "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/results/rope1.6M_1-22_uniform_BOS_80000_failures.txt"
    test_reversal_failures(ckpt, failures_file)
