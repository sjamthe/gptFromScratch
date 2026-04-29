import torch

def test_mask():
    # Sequence of tokens with Phase triggers: [REV]=10, [MATH]=11, [ANS]=12, [BOS]=2
    # Doc 1: [BOS] P P ? [REV] R R [MATH] M M [ANS] A A [EOS]
    # Doc 2: [BOS] P P ? [REV] R R ...
    idx = torch.tensor([
        [2, 5, 5, 33, 10, 6, 6, 11, 7, 7, 12, 8, 8, 3, 2, 5, 33, 10, 6]
    ])
    T = idx.size(1)
    
    # 1. Document Masking (crossing [BOS] ID 2)
    is_bos = (idx == 2)
    seq_ids = is_bos.cumsum(dim=-1)
    doc_mask = (seq_ids.unsqueeze(1) == seq_ids.unsqueeze(2))
    
    # 2. Phase Masking
    is_phase_shift = (idx == 10) | (idx == 11) | (idx == 12)
    global_phase_ids = is_phase_shift.cumsum(dim=-1)
    
    # Within doc 1 (indices 0 to 13), global_phase_ids are 0 to 3
    # At index 14, Doc 2 starts. global_phase_ids is still 3.
    # At index 17, [REV] appears. global_phase_ids becomes 4.
    
    phase_diff = (global_phase_ids.unsqueeze(1) - global_phase_ids.unsqueeze(2))
    phase_mask = (phase_diff == 0) | (phase_diff == 1)
    
    # Let's check Doc 2, index 15 (P) attending to index 14 ([BOS])
    # phase_id[15]=3, phase_id[14]=3. diff=0. Allowed.
    # Let's check Doc 2, index 18 (R) attending to index 15 (P)
    # phase_id[18]=4, phase_id[15]=3. diff=1. Allowed.
    
    # Let's check Doc 2, index 18 (R) attending to Doc 1, index 12 (A)
    # phase_id[18]=4, phase_id[12]=3. diff=1. Allowed by phase_mask.
    # BUT doc_mask[18, 12] should be False.
    
    print("Doc Mask at (18, 12):", doc_mask[0, 18, 12].item())
    
    full_mask = doc_mask & phase_mask
    print("Full Mask at (18, 12):", full_mask[0, 18, 12].item())
    
    # Check Phase 2 cannot see Phase 0 in same doc
    # Doc 1: Phase 2 (M) is index 8. Phase 0 (P) is index 1.
    # phase_id[8]=2, phase_id[1]=0. diff=2.
    print("Phase Mask at (8, 1):", phase_mask[0, 8, 1].item())

if __name__ == "__main__":
    test_mask()
