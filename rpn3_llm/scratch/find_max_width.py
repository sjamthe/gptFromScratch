import os
import sys

# Add project root to sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(base_dir) # rpn3_llm
sys.path.append(project_dir)

from utils import RPNTokenizer

def find_max_widths():
    tokenizer = RPNTokenizer(os.path.join(project_dir, "rpn-tokenizer.json"))
    
    files = [
        "rpn3_llm/data/sft_1-14_7num_BOS_pre_math_val.txt",
        "rpn3_llm/data/sft_1-14_7num_BOS_pre_math_train.txt"
    ]
    
    for file_path in files:
        full_path = os.path.join(os.path.dirname(project_dir), file_path)
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            continue
            
        print(f"Analyzing {file_path}...")
        max_chars = 0
        max_tokens = 0
        longest_char_line = ""
        longest_token_line = ""
        
        with open(full_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                
                # Character count
                char_len = len(stripped)
                if char_len > max_chars:
                    max_chars = char_len
                    longest_char_line = stripped
                
                # Token count
                tokens = tokenizer.encode(stripped + "\n")
                token_len = len(tokens)
                if token_len > max_tokens:
                    max_tokens = token_len
                    longest_token_line = stripped
                    
        print(f"  Max characters: {max_chars}")
        print(f"  Max tokens:     {max_tokens}")
        print(f"  Longest line:   {longest_token_line[:120]}...")

if __name__ == "__main__":
    find_max_widths()
