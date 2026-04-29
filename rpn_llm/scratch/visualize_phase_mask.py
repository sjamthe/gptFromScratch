import torch

def visualize_mask():
    # tokens: [BOS] P ? [REV] R [MATH] M [ANS] A [EOS]
    # Indices: 0   1 2  3    4   5    6   7    8    9
    tokens = ["[BOS]", "P", "?", "[REV]", "R", "[MATH]", "M", "[ANS]", "A", "[EOS]"]
    idx = torch.tensor([[2, 5, 33, 10, 6, 11, 7, 12, 8, 3]])
    T = idx.size(1)
    
    # 1. Document Mask (Simple 1-doc case)
    is_bos = (idx == 2)
    seq_ids = is_bos.cumsum(dim=-1)
    doc_mask = (seq_ids.unsqueeze(1) == seq_ids.unsqueeze(2))
    
    # 2. Phase IDs
    is_phase_shift = (idx == 10) | (idx == 11) | (idx == 12)
    phase_ids = is_phase_shift.cumsum(dim=-1)
    
    # 3. Phase Mask logic
    # Query should vary with i (row), Key should vary with j (col)
    # phase_ids is (1, T)
    phase_diff = (phase_ids.unsqueeze(-1) - phase_ids.unsqueeze(-2))
    phase_mask = (phase_diff == 0) | (phase_diff == 1)
    
    # 4. Causal Mask
    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    
    # 5. Combined
    full_mask = doc_mask & phase_mask & causal_mask
    
    # --- Visualization ---
    print(f"Phase IDs: {phase_ids.tolist()}")
    i, j = 3, 2
    print(f"Checking (Query={tokens[i]}, Key={tokens[j]})")
    print(f"  phase_diff[{i},{j}] = {phase_diff[0, i, j].item()}")
    print(f"  phase_mask[{i},{j}] = {phase_mask[0, i, j].item()}")
    print(f"  doc_mask[{i},{j}] = {doc_mask[0, i, j].item()}")
    print(f"  causal_mask[{i},{j}] = {causal_mask[i, j].item()}")
    print(f"  full_mask[{i},{j}] = {full_mask[0, i, j].item()}")
    
    print(f"Phase Mask Matrix:\n{phase_mask[0].int()}")
    print("\nAttention Mask Visualization (Row = Query, Col = Key)")
    print(" 'X' = Allowed, '.' = Blocked")
    print("\n      " + " ".join([f"{t:>5}" for t in tokens]))
    
    for i in range(T):
        row_str = f"{tokens[i]:>5} "
        for j in range(T):
            allowed = full_mask[0, i, j].item()
            symbol = "  X  " if allowed else "  .  "
            row_str += symbol
        print(row_str)

    print("\nLogic Check (Full Mask):")
    print(f"- [REV] (idx 3) attending to ? (idx 2): {'Allowed' if full_mask[0,3,2] else 'BLOCKED'}")
    print(f"- M (idx 6) attending to P (idx 1): {'Allowed' if full_mask[0,6,1] else 'BLOCKED'}")
    print(f"- M (idx 6) attending to R (idx 4): {'Allowed' if full_mask[0,6,4] else 'Allowed'}")
    print(f"- A (idx 8) attending to R (idx 4): {'Allowed' if full_mask[0,8,4] else 'BLOCKED'}")

if __name__ == "__main__":
    visualize_mask()
