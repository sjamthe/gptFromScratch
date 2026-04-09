import math
from collections import Counter
from train_gpt2 import DataLoaderLite

def calc_entropy():
    # Load your dataset
    loader = DataLoaderLite(B=1, T=1)
    tokens = loader.tokens.tolist()
    total = len(tokens)
    
    # Count occurrences of each token
    counts = Counter(tokens)
    
    entropy_bits = 0.0
    entropy_nats = 0.0
    
    # Apply the Shannon Entropy equation: - Sum( P * log(P) )
    for token, count in counts.items():
        p = count / total
        entropy_bits -= p * math.log2(p)
        entropy_nats -= p * math.log(p)
        
    print("\n--- ENTROPY CALCULATION ---")
    print(f"Total tokens in dataset: {total}")
    print(f"Unique tokens (vocabulary used): {len(counts)}")
    print(f"Unigram Entropy: {entropy_bits:.4f} bits per token")
    print(f"Unigram Entropy (PyTorch Loss equivalent): {entropy_nats:.4f} nats per token")
    
    # Compare against maximum possible entropy (complete randomness)
    print(f"Max possible entropy for full 50257 vocab: {math.log2(50257):.4f} bits")

if __name__ == '__main__':
    calc_entropy()
