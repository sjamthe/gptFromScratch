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
    def __init__(self, B, T, input_path, tokenizer=None, delimiter_token="?"):
        self.B = B
        self.T = T
        self.input_path = input_path
        self.delimiter_token = delimiter_token

        if tokenizer is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.tokenizer = RPNTokenizer(os.path.join(base_dir, "rpn-tokenizer.json"))
        else:
            self.tokenizer = tokenizer
            
        delim_clean = delimiter_token.strip("[]")
        bin_path = f"{input_path}.T{T}.{delim_clean}.cnt.bin"
        mask_path = f"{input_path}.T{T}.{delim_clean}.mask.bin"
        
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

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'tokens_mmap' in state:
            del state['tokens_mmap']
        if 'mask_mmap' in state:
            del state['mask_mmap']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'input_path'):
            self.tokens_mmap = None
            self.mask_mmap = None
            return
        delim_clean = getattr(self, 'delimiter_token', '?').strip("[]")
        bin_path = f"{self.input_path}.T{self.T}.{delim_clean}.cnt.bin"
        mask_path = f"{self.input_path}.T{self.T}.{delim_clean}.mask.bin"
        if os.path.exists(bin_path) and os.path.exists(mask_path):
            self.tokens_mmap = np.memmap(bin_path, dtype=np.uint16, mode='r')
            self.mask_mmap = np.memmap(mask_path, dtype=np.uint8, mode='r')
        else:
            self.tokens_mmap = None
            self.mask_mmap = None



    def _create_binary_cache(self, input_path, bin_path, mask_path):
        import random
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"{input_path} not found.")
            
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        print(f"Shuffling {len(lines)} lines...")
        random.shuffle(lines)
        
        all_tokens = []
        all_masks = []
        
        nl_id = self.tokenizer.vocab.get("\n")
        pad_id = self.tokenizer.vocab.get("[PAD]", 0)
        
        print(f"Tokenizing and building full-equation padded cache (T={self.T})...")
        for line in lines:
            # Clean spaces: strip leading/trailing and compress multiple spaces to one
            clean_line = re.sub(r'\s+', ' ', line.strip()) + '\n'
            line_tokens = self.tokenizer.encode(clean_line)
            
            # Dynamically determine delimiter based on prompt prefix and content
            if clean_line.startswith("[BOS]"):
                delim = "[REV]"
            elif clean_line.startswith("[REV]"):
                if "[ANS]" in clean_line:
                    delim = "[ANS]"
                else:
                    delim = "[MATH]"
            elif clean_line.startswith("[MATH]"):
                delim = "[REV]"
            else:
                delim = self.delimiter_token # fallback
                
            sep_id = self.tokenizer.vocab.get(delim)
            if sep_id is None:
                raise ValueError(f"Delimiter token '{delim}' not found in vocabulary.")
            
            # Build mask: 0 for prompt, 1 for solution
            is_solution = False
            line_mask = []
            for t in line_tokens:
                if is_solution:
                    line_mask.append(1)
                    if t == nl_id:
                        is_solution = False
                else:
                    line_mask.append(0)
                    if t == sep_id:
                        is_solution = True
            
            # Pad or truncate to self.T
            if len(line_tokens) > self.T:
                print(f"Warning: Equation length {len(line_tokens)} exceeds T={self.T}! Truncating.")
                chunk_tokens = line_tokens[:self.T]
                chunk_mask = line_mask[:self.T]
            else:
                pad_len = self.T - len(line_tokens)
                chunk_tokens = line_tokens + [pad_id] * pad_len
                chunk_mask = line_mask + [0] * pad_len
                
            all_tokens.extend(chunk_tokens)
            all_masks.extend(chunk_mask)
            
        # Append exactly one extra padding element at the very end to prevent 
        # out-of-bounds or skipping of the final batch by next_batch's "+ 1" lookahead.
        all_tokens.append(pad_id)
        all_masks.append(0)
            
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
            
        # Pull slices from memory-mapped files
        buf_np = self.tokens_mmap[self.current_pos : self.current_pos + B * T + 1]
        mask_buf_np = self.mask_mmap[self.current_pos : self.current_pos + B * T + 1]
        
        # Convert to torch tensors (no-copy)
        buf = torch.from_numpy(buf_np.astype(np.int64))
        mask_buf = torch.from_numpy(mask_buf_np.astype(bool))
        
        x = buf[:-1].view(B, T)
        y = buf[1:].clone()
        # y[~mask_buf[1:]] = -100
        # mask_buf[1:] is a numpy.bool_ array, torch can use it for indexing
        y[~mask_buf[1:]] = -100
        y = y.view(B, T)
        
        self.current_pos += B * T
        return x, y
