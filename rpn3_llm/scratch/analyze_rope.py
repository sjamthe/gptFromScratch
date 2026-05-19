import os
import sys
import torch
import numpy as np

# Ensure working directory and rpn3_llm are in python path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "rpn3_llm"))

from utils import RPNTokenizer
from model_rope import GPT, precompute_freqs_cis

def analyze_rope():
    checkpoint_path = "rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_sft_1-14_7num_BOS_352000.pt"
    device = 'cpu'
    
    # 1. Load Model Config
    print("Loading model config...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Head dim calculation
    n_embd = config.n_embd
    n_head = config.n_head
    head_dim = n_embd // n_head
    rope_theta = config.rope_theta if hasattr(config, 'rope_theta') else 10000.0
    block_size = config.block_size
    
    print(f"Model settings: n_embd={n_embd}, n_head={n_head}, head_dim={head_dim}, rope_theta={rope_theta}, block_size={block_size}")
    
    # 2. Precompute freqs_cis
    # Shape: (block_size, head_dim // 2)
    freqs_cis = precompute_freqs_cis(head_dim, block_size, theta=rope_theta)
    print(f"freqs_cis shape: {freqs_cis.shape}")
    
    # 3. Duplicate token locations in our Failure #2 sequence:
    # prompt: [BOS]57 313733-?[REV]75[SEP]3
    # Tokens:
    # 0: [BOS]
    # 1: 5
    # 2: 7
    # 3:  (space)
    # 4: 3  (duplicate '3' #1)
    # 5: 1
    # 6: 3  (duplicate '3' #2)
    # 7: 7
    # 8: 3  (duplicate '3' #3 - correct next target)
    # 9: 3  (duplicate '3' #4 - just copied)
    # 10: -
    # 11: ?
    # 12: [REV]
    # 13: 7
    # 14: 5
    # 15: [SEP]
    # 16: 3 (query token at position 16)
    
    dup_indices = [4, 6, 8, 9]
    query_idx = 16
    
    print("\n=== ABSOLUTE ROPE EMBEDDINGS (freqs_cis) COMPARISON ===")
    
    # Let's inspect the absolute freqs_cis at duplicate positions
    for idx in dup_indices:
        vec = freqs_cis[idx]
        print(f"\nPosition {idx} ('3'):")
        # Print first 4 complex dimensions (real and imag, or magnitude and phase)
        for d in range(min(4, vec.shape[0])):
            val = vec[d].item()
            mag = abs(val)
            phase = np.angle(val) # in radians
            print(f"  Dim {d}: Complex Value = {val.real:+.6f} {val.imag:+.6f}j | Mag = {mag:.4f} | Phase = {phase:+.6f} rad ({np.degrees(phase):+.2f}°)")
            
    # Compute similarity metrics between the duplicate positions
    print("\n--- Pairwise Absolute Distance / Similarity Matrix (freqs_cis) ---")
    for idx1 in dup_indices:
        for idx2 in dup_indices:
            if idx1 >= idx2:
                v1 = freqs_cis[idx1]
                v2 = freqs_cis[idx2]
                
                # Real-space Euclidean distance (after treating complex as 2D vectors)
                # Since magnitude is 1, Euclidean distance depends entirely on phase differences.
                dist = torch.norm(v1 - v2).item()
                
                # Complex dot product / cosine similarity
                dot = torch.real(torch.sum(v1 * torch.conj(v2))).item() / (head_dim // 2)
                
                print(f"Pos {idx1} vs Pos {idx2}: Euclidean Dist = {dist:.6f} | Cos Similarity = {dot:.6f}")

    print("\n=== RELATIVE ROTARY EMBEDDINGS COMPARISON ===")
    print(f"Query position: {query_idx} ('3')")
    
    # Under RoPE, the dot product between query at position Q and key at position K
    # depends on relative rotation freqs_cis[K] * conj(freqs_cis[Q])
    for idx in dup_indices:
        rel_freqs = freqs_cis[idx] * torch.conj(freqs_cis[query_idx])
        print(f"\nRelative rotation to target position {idx} (Distance = {idx - query_idx}):")
        for d in range(min(4, rel_freqs.shape[0])):
            val = rel_freqs[d].item()
            mag = abs(val)
            phase = np.angle(val)
            print(f"  Dim {d}: Relative Rotation = {val.real:+.6f} {val.imag:+.6f}j | Phase shift = {phase:+.6f} rad ({np.degrees(phase):+.2f}°)")
            
    # Compute relative similarity: how close is the relative rotation matrix for different positions?
    print("\n--- How close are the relative rotations seen by the Query? ---")
    # Specifically compare key at Pos 8 vs key at Pos 9
    rel_8 = freqs_cis[8] * torch.conj(freqs_cis[query_idx])
    rel_9 = freqs_cis[9] * torch.conj(freqs_cis[query_idx])
    
    rel_dist = torch.norm(rel_8 - rel_9).item()
    rel_sim = torch.real(torch.sum(rel_8 * torch.conj(rel_9))).item() / (head_dim // 2)
    
    print(f"Relative Rotation (Pos 8 vs Pos 9):")
    print(f"  Euclidean distance between their relative rotation vectors: {rel_dist:.6f}")
    print(f"  Cosine similarity of relative rotation vectors: {rel_sim:.6f}")

if __name__ == "__main__":
    analyze_rope()
