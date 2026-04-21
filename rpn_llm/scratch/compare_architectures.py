import os
import sys
import torch
import torch.nn.functional as F

# Add the parent directory (rpn_llm) to the path so we can import model_rope
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def generate_output(model, tokenizer, prompt, max_new_tokens=100, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated = []
    past_kv = None
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1:] if past_kv is not None else idx
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, _, past_kv = model(idx_cond, use_cache=True, past_key_values=past_kv)
            
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            next_id = idx_next.item()
            if next_id == tokenizer.encode("\n")[0]:
                break
            
            generated.append(next_id)
            idx = torch.cat((idx, idx_next), dim=1)
            
    return tokenizer.decode(generated)

def get_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    return model

def run_comparison():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    
    models = [
        {"name": "Standard (25M)", "path": "rpn_llm/models/rope25M_1-22_tens_comp_bracketed_final.pt"},
        {"name": "Universal (3M)", "path": "rpn_llm/models/UT3M_1-22_tens_comp_bracketed_final.pt"},
    ]
    
    test_cases = [
        # 1. In-Distribution (Simple)
        {"desc": "Simple Addition", "prompt": "(123)(456)+?<"},
        
        # 2. Structure/Robustness
        {"desc": "Operator Conflict (Prompt - vs Scratchpad +)", "prompt": "(123)(456)-?<(321)(654)+="},
        
        # 3. OOD Length
        {"desc": "30-Digit Marathon", "prompt": "(" + "9"*30 + ")(" + "1"*30 + ")+?"},
        
        # 4. Multi-Operand (Abstraction Fail points)
        {"desc": "3-Operand Attempt", "prompt": "(1)(2)+?<1+2+3="},
        
        # 5. High Carry
        {"desc": "Carry-2 Test", "prompt": "(9)(9)+?<9+9+2="},
    ]
    
    results = []
    
    for m_info in models:
        print(f"Testing {m_info['name']}...")
        model = get_model(m_info['path'], device)
        
        m_results = []
        for test in test_cases:
            output = generate_output(model, tokenizer, test['prompt'], device=device)
            m_results.append(output)
            
        results.append(m_results)
        # Clean up memory
        del model
        if device == 'cuda': torch.cuda.empty_cache()
        if device == 'mps': torch.mps.empty_cache()
        
    # Generate Markdown Table
    print("\n| Test Case | Standard (25M) | Universal (3M) |")
    print("| :--- | :--- | :--- |")
    for i, test in enumerate(test_cases):
        std_out = results[0][i].replace("|", "\\|").replace("\n", " ")
        uni_out = results[1][i].replace("|", "\\|").replace("\n", " ")
        print(f"| {test['desc']} | `{std_out}` | `{uni_out}` |")

if __name__ == "__main__":
    run_comparison()
