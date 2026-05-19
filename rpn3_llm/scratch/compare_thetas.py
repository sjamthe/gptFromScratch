import torch
import numpy as np

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def compare_thetas():
    head_dim = 48  # 384 / 8 heads
    block_size = 2048
    thetas = [10000.0, 2000.0, 1000.0]
    
    print("=" * 70)
    print("      ROPE THETA COMPARISON: NEIGHBORING TOKEN SIMILARITY GRADIENTS")
    print("=" * 70)
    print(f"Head Dimension: {head_dim}")
    
    for theta in thetas:
        freqs_cis = precompute_freqs_cis(head_dim, block_size, theta=theta)
        
        print(f"\n>>> THETA = {theta} <<<")
        print("-" * 50)
        
        # We compare identical tokens at adjacent positions: K=8 and K=9
        v8 = freqs_cis[8]
        v9 = freqs_cis[9]
        
        # Calculate complex cosine similarity: real part of sum(v8 * conj(v9)) / dim
        cos_sim = torch.real(torch.sum(v8 * torch.conj(v9))).item() / (head_dim // 2)
        
        # Real-space Euclidean distance
        euclidean_dist = torch.norm(v8 - v9).item()
        
        print(f"Adjacent Tokens (Pos 8 vs Pos 9):")
        print(f"  * Cosine Similarity: {cos_sim:8.4f} ({cos_sim*100:5.1f}%)")
        print(f"  * Euclidean Distance: {euclidean_dist:8.4f}")
        
        # Calculate similarity across increasing step distances
        print("\n  Similarity decay over distance (Pos 8 vs Pos 8+d):")
        for d in [1, 2, 4, 8, 16, 32]:
            v_other = freqs_cis[8 + d]
            sim = torch.real(torch.sum(v8 * torch.conj(v_other))).item() / (head_dim // 2)
            print(f"    Distance {d:2d} tokens: {sim*100:6.2f}% similarity")
            
        # Calculate Phase Rotation (in degrees) for the first 3 complex dimensions
        print("\n  Phase rotation per step (Dim 0, Dim 1, Dim 2):")
        rel = freqs_cis[1] * torch.conj(freqs_cis[0])
        for dim_idx in range(3):
            angle = np.degrees(np.angle(rel[dim_idx].item()))
            print(f"    Dim {dim_idx}: {angle:+.2f}° shift per step")

if __name__ == "__main__":
    compare_thetas()
