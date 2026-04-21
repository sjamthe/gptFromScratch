import torch
from model_rope import GPT, GPTConfig

def compare():
    conf_std = GPTConfig(universal=False)
    model_std = GPT(conf_std)
    p_std = sum(p.numel() for p in model_std.parameters())
    
    conf_uni = GPTConfig(universal=True)
    model_uni = GPT(conf_uni)
    p_uni = sum(p.numel() for p in model_uni.parameters())
    
    print(f"Standard Model (8 layers): {p_std:,} parameters")
    print(f"Universal Model (8 passes): {p_uni:,} parameters")
    print(f"Reduction: {((p_std - p_uni) / p_std) * 100:.1f}%")

if __name__ == "__main__":
    compare()
