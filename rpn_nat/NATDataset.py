import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, Sampler
from utils import RPNTokenizer

class NATDataset(Dataset):
    def __init__(self, data_path, max_length=128, test_mode=False, mask_prob=0.8):
        self.max_length = max_length
        self.test_mode = test_mode 
        self.mask_prob = mask_prob
        self.mask_id = 2 # Using [BOS] as the mask token for now
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.tokenizer = RPNTokenizer(os.path.join(base_dir, "rpn-tokenizer.json"))
        
        bin_path = data_path + ".cnt.bin"
        mask_path = data_path + ".mask.bin"
        
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary cache not found at {bin_path}. Run DataLoaderLite once to generate it.")
            
        print("Memory mapping binary files...")
        self.tokens_mmap = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.mask_mmap = np.memmap(mask_path, dtype=np.uint8, mode='r')
        
        print("Finding problem boundaries...")
        bos_id = self.tokenizer.encode("[BOS]")[0]
        is_bos = (self.tokens_mmap == bos_id)
        bos_indices = np.where(is_bos)[0]
        bos_indices = np.append(bos_indices, len(self.tokens_mmap))
        
        lengths = bos_indices[1:] - bos_indices[:-1]
        
        valid_mask = (lengths > 5) & (lengths <= max_length)
        self.starts = bos_indices[:-1][valid_mask]
        self.ends = bos_indices[1:][valid_mask]
        self.lengths = lengths[valid_mask]
        
        print("Sorting by length for bucketing...")
        sort_idx = np.argsort(self.lengths)
        self.starts = self.starts[sort_idx]
        self.ends = self.ends[sort_idx]
        self.lengths = self.lengths[sort_idx]
        
        print(f"Loaded {len(self.starts)} valid problems.")
        
    def __len__(self):
        return len(self.starts)
        
    def __getitem__(self, idx):
        start = self.starts[idx]
        end = self.ends[idx]
        
        tokens = self.tokens_mmap[start:end].astype(np.int64)
        mask = self.mask_mmap[start:end].astype(bool)
        
        P_len = (~mask).sum()
        A_len = mask.sum()
        
        # FIXED ALIGNMENT: Always start the answer at a fixed index (W_START)
        # This solves the "Absolute Position" problem where the model gets confused by varying prompt lengths.
        W_START = 32
        
        # Determine how many answer slots to provide (k)
        if self.test_mode:
            k = A_len
            current_mask_prob = 0.5
        else:
            # 80% of time: A_len + small buffer
            # 20% of time: Up to 128 total
            if random.random() < 0.8:
                k = A_len + random.randint(0, 16)
            else:
                k = random.randint(A_len, 128)
                
            # VARY MASK PROBABILITY:
            r = random.random()
            if r < 0.3:
                current_mask_prob = 1.0
            elif r < 0.7:
                current_mask_prob = random.uniform(0.5, 0.9)
            else:
                current_mask_prob = random.uniform(0.0, 0.2)

        # Input x: [Prompt] + [BOS Padding] + [Answer Slots]
        # Length = W_START + k
        x_len = W_START + k
        x = np.full(x_len, self.mask_id, dtype=np.int64) 
        # Copy original prompt
        x[:P_len] = tokens[:P_len]
        
        # Copy original answer tokens into the workspace at W_START
        available_A = min(A_len, k)
        if available_A > 0:
            x[W_START : W_START + available_A] = tokens[P_len : P_len + available_A]
        
        # Apply Random Masking ONLY to the workspace slots (W_START to x_len-1)
        for i in range(W_START, x_len):
            if random.random() < current_mask_prob:
                x[i] = self.mask_id
        
        # Target y: [-100 (Prompt/Padding)] + [Correct Answer Tokens]
        y = np.full(x_len, -100, dtype=np.int64)
        target_len = min(A_len, k)
        y[W_START : W_START + target_len] = tokens[P_len : P_len + target_len]
        
        # If we provided more slots than the answer length, 
        # train the model to output [PAD] (ID 0) in those extra slots
        pad_id = 0
        if k > A_len:
            y[W_START + A_len : W_START + k] = pad_id
        
        return torch.tensor(x), torch.tensor(y)

class BucketBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.num_batches = len(data_source) // batch_size
        self.batches = []
        for i in range(self.num_batches):
            self.batches.append(list(range(i * batch_size, (i + 1) * batch_size)))
            
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
            
    def __len__(self):
        return self.num_batches

def nat_collate_fn(batch):
    max_len = max(len(x) for x, _ in batch)
    xs, ys = [], []
    
    pad_id = 2 # [BOS]
    
    for x, y in batch:
        pad_len = max_len - len(x)
        if pad_len > 0:
            x_pad = torch.nn.functional.pad(x, (0, pad_len), value=pad_id)
            y_pad = torch.nn.functional.pad(y, (0, pad_len), value=-100)
        else:
            x_pad = x
            y_pad = y
            
        xs.append(x_pad)
        ys.append(y_pad)
        
    return torch.stack(xs), torch.stack(ys)
