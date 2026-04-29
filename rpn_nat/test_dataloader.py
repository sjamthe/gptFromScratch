import sys
import os
import torch
from torch.utils.data import DataLoader
from NATDataset import NATDataset, BucketBatchSampler, nat_collate_fn

def test_dataloader():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "rpn_llm", "data", "RPNData-1-22_uniform_BOS_train.txt")
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please check the path.")
        return
        
    print("Initializing NATDataset...")
    dataset = NATDataset(data_path, max_length=128, test_mode=True)
    sampler = BucketBatchSampler(dataset, batch_size=2, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=nat_collate_fn)
    
    for i, (x, y) in enumerate(loader):
        print(f"\nVisualizing NAT DataLoader Batch {i+1} (B={x.size(0)}, max_T={x.size(1)})")
        print("-" * 50)
        print(f"{'Pos':<5} | {'Input (x)':<15} | {'Target (y)':<15}")
        print("-" * 50)
        
        for pos in range(x.size(1)):
            in_tok = x[0, pos].item()
            out_tok = y[0, pos].item()
            
            in_str = repr(dataset.tokenizer.decode([in_tok]))
            if out_tok == -100:
                out_str = "[-100] (MASK)"
            else:
                out_str = repr(dataset.tokenizer.decode([out_tok]))
                
            print(f"{pos:<5} | {in_str:<15} | {out_str:<15}")
            
        if i == 0:
            break

if __name__ == "__main__":
    test_dataloader()
