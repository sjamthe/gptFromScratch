import os
import sys

# Add project root to sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(base_dir) # rpn3_llm
sys.path.append(project_dir)

from utils import RPNTokenizer

def find_max_widths():
    tokenizer = RPNTokenizer(os.path.join(project_dir, "rpn-tokenizer.json"))
    
    # Get phase shift token IDs
    rev_id = tokenizer.vocab.get("[REV]")
    math_id = tokenizer.vocab.get("[MATH]")
    ans_id = tokenizer.vocab.get("[ANS]")
    phase_shift_ids = {rev_id, math_id, ans_id}
    
    files = [
        "rpn3_llm/data/rpn3_ood_len_val.txt",
    ]
    
    for file_path in files:
        full_path = os.path.join(os.path.dirname(project_dir), file_path)
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            continue
            
        print(f"Analyzing {file_path} (for max consecutive two-phase length)...")
        max_chars = 0
        max_tokens = 0
        max_all_tokens = 0
        longest_char_line = ""
        longest_token_line = ""
        
        with open(full_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                
                # Tokenize the line
                tokens = tokenizer.encode(stripped + "\n")
                if max_all_tokens < len(tokens) :
                    max_all_tokens = len(tokens)
                
                # Split tokens into phases
                phases = []
                current_phase = []
                for tok in tokens:
                    if tok in phase_shift_ids:
                        if current_phase:
                            phases.append(current_phase)
                        current_phase = [tok]
                    else:
                        current_phase.append(tok)
                if current_phase:
                    phases.append(current_phase)
                
                # 1. Compute max consecutive phase token length
                phase_lens = [len(p) for p in phases]
                if len(phase_lens) == 1:
                    token_len = phase_lens[0]
                elif len(phase_lens) > 1:
                    token_len = max(phase_lens[i] + phase_lens[i+1] for i in range(len(phase_lens) - 1))
                else:
                    token_len = 0
                    
                if token_len > max_tokens:
                    max_tokens = token_len
                    longest_token_line = stripped
                
                # 2. Compute max consecutive phase character length
                phase_strs = [tokenizer.decode(p) for p in phases]
                phase_char_lens = [len(s) for s in phase_strs]
                if len(phase_char_lens) == 1:
                    char_len = phase_char_lens[0]
                elif len(phase_char_lens) > 1:
                    char_len = max(phase_char_lens[i] + phase_char_lens[i+1] for i in range(len(phase_char_lens) - 1))
                else:
                    char_len = 0
                    
                if char_len > max_chars:
                    max_chars = char_len
                    longest_char_line = stripped
                    
        print(f"  Max consecutive two-phase characters: {max_chars}")
        print(f"  Max consecutive two-phase tokens:     {max_tokens}")
        print(f"  Longest line (by tokens):   {longest_token_line[:120]}...")
        print("Max all tokens: ", max_all_tokens)
        print()

if __name__ == "__main__":
    find_max_widths()
