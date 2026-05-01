import os
import sys
# Add rpn_llm to path
sys.path.append(os.path.abspath("rpn_llm"))
from utils import RPNTokenizer

def test():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = RPNTokenizer(os.path.join("rpn_llm", "rpn-tokenizer.json"))
    
    prompt = "[BOS]123 21-?[REV]321 123-=[MATH]3-1-0="
    tokens = tokenizer.encode(prompt)
    decoded = tokenizer.decode(tokens)
    
    print(f"Prompt: {prompt}")
    print(f"Token IDs: {tokens}")
    print(f"Token Count: {len(tokens)}")
    print(f"Decoded: {decoded}")
    
    # Check individual tokens
    import re
    tokens_raw = tokenizer.token_pattern.findall(prompt)
    print(f"Raw Matches: {tokens_raw}")

if __name__ == "__main__":
    test()
