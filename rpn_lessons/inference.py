import os
import sys
import torch

# Ensure the script can import local modules regardless of where it is run
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="rpn_lessons/models/lesson4_step40000.pt")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
        sys.exit(1)
        
    print(f"Loading model from {args.checkpoint} on {device}...", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Ensure block size limit is set
    config.block_size = 384
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Locate vocab file
    vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rpn-tokenizer.json")
    tokenizer = RPNTokenizer(vocab_path)
    
    bos_id = tokenizer.encode("[BOS]")[0]
    rev_id = tokenizer.encode("[REV]")[0]
    math_id = tokenizer.encode("[MATH]")[0]
    ans_id = tokenizer.encode("[ANS]")[0]
    eos_id = tokenizer.encode("[EOS]")[0]
    
    phase_tensor = torch.tensor([rev_id, math_id, ans_id], device=device)
    stop_ids = {eos_id, ans_id}
    
    print("\n========================================================")
    print("RPN Transformer Interactive Chat Client")
    print("========================================================")
    print("Instructions:")
    print("  - Type any RPN expression (e.g. '12 34 + 56 -') and press Enter.")
    print("  - Or type a raw prompt starting with [BOS], [REV], or [MATH].")
    print("  - Type 'quit' or press Ctrl-D / Cmd-D to exit.")
    print("========================================================\n")
    
    while True:
        try:
            line = input("Prompt > ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break
            
        line = line.strip()
        if not line:
            continue
        if line.lower() == 'quit':
            print("Exiting. Goodbye!")
            break
            
        # Format input automatically if needed
        if not (line.startswith("[BOS]") or line.startswith("[REV]") or line.startswith("[MATH]")):
            import re
            clean_expr = re.sub(r'\s+', ' ', line)
            current_prompt = f"[BOS]{clean_expr}[REV]"
            print(f"Formatted input: {current_prompt}")
        else:
            current_prompt = line
            
        step = 0
        max_phases = 15
        
        while step < max_phases:
            # Determine phase delimiters and stop tokens based on prompt prefix
            if current_prompt.startswith("[BOS]"):
                stop_ids = {math_id}
                max_gen = 150
                phase_name = "Reversal Phase (L2)"
            elif current_prompt.startswith("[REV]"):
                if "[ANS]" in current_prompt:
                    stop_ids = {eos_id}
                    max_gen = 50
                    phase_name = "Final Answer Phase (L1)"
                else:
                    stop_ids = {rev_id}
                    max_gen = 250
                    phase_name = "Math Phase (L3)"
            elif current_prompt.startswith("[MATH]"):
                stop_ids = {math_id, ans_id}
                max_gen = 150
                phase_name = "Transition Phase (L4)"
            else:
                print(f"\nError: Unknown prompt prefix in '{current_prompt}'")
                break
                
            print(f"\n--- {phase_name} ---")
            print(f"Prompt: {current_prompt}")
            print("Output: ", end="", flush=True)
            
            prompt_tokens = tokenizer.encode(current_prompt)
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
            
            past_kv = None
            curr_idx = prompt_tensor
            generated = []
            
            with torch.no_grad():
                for step_count in range(max_gen):
                    is_phase_shift = torch.isin(curr_idx, phase_tensor)
                    full_phase_ids = is_phase_shift.cumsum(dim=-1)
                    
                    cond_idx = curr_idx[:, -1:] if past_kv is not None else curr_idx
                    
                    with torch.autocast(device, dtype=torch.bfloat16):
                        logits, _, past_kv = model(
                            cond_idx, 
                            use_cache=True, 
                            past_key_values=past_kv, 
                            full_phase_ids=full_phase_ids
                        )
                        
                    next_id = torch.argmax(logits[0, -1, :]).item()
                    next_str = tokenizer.decode([next_id])
                    print(next_str, end="", flush=True)
                    
                    generated.append(next_id)
                    curr_idx = torch.cat([curr_idx, torch.tensor([[next_id]], device=device)], dim=1)
                    
                    if next_id in stop_ids:
                        break
                else:
                    print(" [Truncated - reached max token limit]")
                    
            print() # Newline after phase completes
            gen_str = tokenizer.decode(generated).strip()
            
            # Find the trailing delimiter of the prompt to prepend to output for the next phase
            last_delim_idx = current_prompt.rfind("[")
            if last_delim_idx == -1:
                print("\nError: Could not locate delimiter in prompt.")
                break
            last_delim = current_prompt[last_delim_idx:]
            
            current_prompt = last_delim + gen_str
            
            # Check terminal conditions
            if gen_str.endswith("[EOS]"):
                print("\nCalculation Finished successfully!")
                break
            elif gen_str.endswith("[ANS]"):
                # Transition to final Lesson 1 step: [REV]<reversed_ans>[ANS]
                # Prepend [REV]
                current_prompt = "[REV]" + gen_str
                
            step += 1
        print("========================================================\n")

if __name__ == "__main__":
    main()
