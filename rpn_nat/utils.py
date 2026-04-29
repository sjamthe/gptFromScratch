import os
import json
import re
import torch

class RPNTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["model"]["vocab"]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Build a regex that matches the longest tokens first
        # We sort by length descending to ensure "[BORROW]" matches before "["
        sorted_tokens = sorted(self.vocab.keys(), key=len, reverse=True)
        # We escape the tokens (in case they have regex chars like '[' or '+') 
        # and add '|.' at the end to catch any remaining single characters.
        self.token_pattern = re.compile("|".join(re.escape(t) for t in sorted_tokens) + r"|[\s\S]")

    def encode(self, text):
        # findall captures all matched tokens/characters sequentially
        tokens = self.token_pattern.findall(text)
        return [self.vocab.get(t, self.vocab.get("[UNK]", 1)) for t in tokens]

    def decode(self, tokens):
        return "".join([self.inverse_vocab.get(t, "") for t in tokens])

import numpy as np

class DataLoaderLite:
    def __init__(self, B, T, input_path, tokenizer=None):
        self.B = B
        self.T = T
        if tokenizer is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.tokenizer = RPNTokenizer(os.path.join(base_dir, "rpn-tokenizer.json"))
        else:
            self.tokenizer = tokenizer
            
        bin_path = input_path + ".cnt.bin"
        mask_path = input_path + ".mask.bin"
        
        # Staleness check: recreate if txt is newer than bin
        is_stale = False
        if os.path.exists(bin_path):
            txt_mtime = os.path.getmtime(input_path)
            bin_mtime = os.path.getmtime(bin_path)
            if txt_mtime > bin_mtime:
                is_stale = True
        
        if not os.path.exists(bin_path) or not os.path.exists(mask_path) or is_stale:
            if is_stale:
                print(f"Dataset {input_path} has been updated. Recreating binary cache...")
            else:
                print(f"Binary files not found. Creating cache for {input_path}...")
            self._create_binary_cache(input_path, bin_path, mask_path)
            
        # Load using memory mapping for near-instant startup and low RAM usage
        self.tokens_mmap = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.mask_mmap = np.memmap(mask_path, dtype=np.uint8, mode='r')
        self.num_tokens = len(self.tokens_mmap)
        
        print(f"Loaded {self.num_tokens} tokens from binary cache")
        print("1 epoch = ", self.num_tokens // (self.B * self.T), "micro-batches")
        self.current_pos = 0

    def _create_binary_cache(self, input_path, bin_path, mask_path):
        import random
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"{input_path} not found.")
            
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        print(f"Shuffling {len(lines)} lines...")
        random.shuffle(lines)
        
        # Encode line by line to be safer with memory
        all_tokens = []
        all_masks = []
        sep_id = self.tokenizer.encode("?")[0]
        nl_id = self.tokenizer.encode("\n")[0]
        
        print("Tokenizing and building mask...")
        for line in lines:
            # Clean spaces: strip leading/trailing and compress multiple spaces to one
            clean_line = re.sub(r'\s+', ' ', line.strip()) + '\n'
            line_tokens = self.tokenizer.encode(clean_line)
            is_answer = False
            line_mask = []
            for t in line_tokens:
                if is_answer:
                    line_mask.append(1)
                    if t == nl_id: is_answer = False
                else:
                    line_mask.append(0)
                    if t == sep_id: is_answer = True
            
            all_tokens.extend(line_tokens)
            all_masks.extend(line_mask)
            
        # Write to disk
        print(f"Saving to {bin_path}...")
        tokens_arr = np.array(all_tokens, dtype=np.uint16)
        tokens_arr.tofile(bin_path)
        
        print(f"Saving to {mask_path}...")
        mask_arr = np.array(all_masks, dtype=np.uint8)
        mask_arr.tofile(mask_path)
        
        # Free memory
        del all_tokens
        del all_masks
        del lines

    def next_batch(self):
        B, T = self.B, self.T
        if self.current_pos + B * T + 1 > self.num_tokens:
            self.current_pos = 0
            
        # Read ahead by 300 tokens to ensure we have the full answer string for the last problem in the chunk
        read_len = min(B * T + 300, self.num_tokens - self.current_pos)
        
        # Pull slices from memory-mapped files
        buf_np = self.tokens_mmap[self.current_pos : self.current_pos + read_len]
        mask_buf_np = self.mask_mmap[self.current_pos : self.current_pos + read_len]
        
        # Convert to torch tensors (no-copy)
        buf = torch.from_numpy(buf_np.astype(np.int64))
        mask_buf = torch.from_numpy(mask_buf_np.astype(bool))
        
        bos_token_id = self.tokenizer.encode("[BOS]")[0]
        is_bos = (buf == bos_token_id)
        
        bos_indices = torch.where(is_bos)[0]
        # Append the end of the buffer as a dummy next-BOS
        bos_indices = torch.cat([bos_indices, torch.tensor([len(buf)])])
        
        y_full = torch.full_like(buf, -100)
        
        for k in range(len(bos_indices) - 1):
            start_idx = bos_indices[k]
            end_idx = bos_indices[k+1]
            
            tokens_p = buf[start_idx:end_idx]
            is_answer_p = mask_buf[start_idx:end_idx]
            
            P_len = (~is_answer_p).sum().item()
            A_len = is_answer_p.sum().item()
            
            # NAT Target: answer tokens shifted to the front, padded with -100
            target_p = torch.full_like(tokens_p, -100)
            target_p[:A_len] = tokens_p[P_len : P_len + A_len]
            
            # MASK partial prompts: we only start training from the full prompt!
            if P_len > 0:
                target_p[:P_len - 1] = -100
                
            y_full[start_idx:end_idx] = target_p
            
        x = buf[:B*T].view(B, T)
        y = y_full[:B*T].view(B, T)
        
        self.current_pos += B * T
        return x, y
