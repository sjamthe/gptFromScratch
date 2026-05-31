import torch
import sys
import os

# Add parent directory and current directory to path to import model and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def run_inference(checkpoint_path, prompt_text, max_gen=500):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer_path = os.path.join(os.path.dirname(__file__), 'rpn-tokenizer.json')
    tokenizer = RPNTokenizer(tokenizer_path)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Instantiate model
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    prompt_tokens = tokenizer.encode(prompt_text)
    eos_id = tokenizer.encode("[EOS]")[0]
    unk_id = tokenizer.encode("[UNK]")[0]
    
    print(f"\nInitial Prompt: '{prompt_text}' (Length: {len(prompt_tokens)} tokens)")
    print("Starting Autoregressive Generation...")
    print("-" * 50)
    
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    past_kv = None
    generated_tokens = []
    
    with torch.no_grad():
        for i in range(max_gen):
            # Compute phase shift ids for the entire sequence generated so far
            is_phase_shift = (idx == 10) | (idx == 11) | (idx == 12)
            full_phase_ids = is_phase_shift.cumsum(dim=-1)
            
            # Pass only the last generated token if KV cache is available
            cond_idx = idx[:, -1:] if past_kv is not None else idx
            
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, _, past_kv = model(cond_idx, use_cache=True, past_key_values=past_kv, full_phase_ids=full_phase_ids)
                
            # Focus on the last query output logits
            logits = logits[0, -1, :]
            next_id = torch.argmax(logits).item()
            
            if next_id == eos_id:
                break
                
            generated_tokens.append(next_id)
            idx = torch.cat((idx, torch.tensor([[next_id]], device=device)), dim=1)
            
            # Print current generation progress
            current_str = tokenizer.decode(generated_tokens)
            print(f"\rGenerating: {current_str}", end="", flush=True)
            
            if next_id == unk_id:
                print(f"\nReached terminator '[UNK]'.")
                break
                
    print("\n" + "-" * 50)
    final_output = tokenizer.decode(generated_tokens).strip()
    print(f"Final Answer: {final_output}")
    print(f"Full Sequence: {prompt_text}{final_output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="rpn3_llm/models/ut2.1M_2l_8h_384e_mlp4_phaseMask_True_cnt2_crd2_sft_1-6_4num_BOS_200000.pt")
    parser.add_argument("--prompt", type=str, default="[BOS]123 456+?")
    parser.add_argument("--max_gen", type=int, default=500)
    args = parser.parse_args()
    
    run_inference(args.checkpoint, args.prompt, args.max_gen)
