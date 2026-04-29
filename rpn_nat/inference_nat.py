import torch
import sys
import os

# Add parent directory to path to import model and utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_nat import GPT, GPTConfig
from utils import RPNTokenizer

def run_inference(checkpoint_path, prompt_text, max_gen=100):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer_path = os.path.join(os.path.dirname(__file__), 'rpn-tokenizer.json')
    tokenizer = RPNTokenizer(tokenizer_path)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    prompt_tokens = tokenizer.encode(prompt_text)
    P_len = len(prompt_tokens)
    W_START = 32
    max_answer_len = 128
    
    # Initialize full sequence with [BOS] padding
    # Format: [Prompt] + [Padding] + [Answer Slots]
    current_tokens = [2] * (W_START + max_answer_len)
    current_tokens[:P_len] = prompt_tokens
    
    print(f"\nInitial Prompt: '{prompt_text}' (Length: {P_len})")
    print(f"Fixed Alignment: Answer starts at Index {W_START}")
    print("Starting Iterative Refinement Inference...")
    print("-" * 40)
    
    for i in range(max_gen):
        x = torch.tensor(current_tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, _ = model(x)
                # In Fixed Alignment, Answer[i] is predicted at index W_START + i
                ans_logits = logits[:, W_START:, :]
                new_pred_tokens = torch.argmax(ans_logits, dim=-1)[0].tolist()
            
        # Update only the answer slots
        current_tokens[W_START:] = new_pred_tokens
            
        new_pred_str = tokenizer.decode(new_pred_tokens).strip()
        print(f"\nStep {i+1}: NAT Parallel Answer: '{new_pred_str}'")
        
        # Check if the answer has stabilized or reached a stop token
        if "[UNK]" in new_pred_str:
            print(f"Reached terminator '[UNK]'.")
            # We continue for a few more steps to 'polish', or stop here.
            # For now, let's just update and keep going or break.
            current_answer_tokens = new_pred_tokens
            break
            
        # Update the Answer Guess with the NEW parallel output
        current_answer_tokens = new_pred_tokens
        
        if device == "mps":
            torch.mps.empty_cache()

    print("-" * 40)
    final_ans = tokenizer.decode(current_answer_tokens).split("[UNK]")[0].strip()
    print(f"Final Refined Result: {prompt_text}{final_ans}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="rpn_nat/models/rope3.6M_1-22_uniform_BOS_8000.pt")
    parser.add_argument("--prompt", type=str, default="[BOS]123 456+?")
    parser.add_argument("--max_gen", type=int, default=100)
    args = parser.parse_args()
    
    run_inference(args.checkpoint, args.prompt, args.max_gen)
