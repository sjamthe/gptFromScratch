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
        self.token_pattern = re.compile("|".join(re.escape(t) for t in sorted_tokens) + "|.")

    def encode(self, text):
        # findall captures all matched tokens/characters sequentially
        tokens = self.token_pattern.findall(text)
        # Filter out empty matches if any, and handle unknowns
        return [self.vocab.get(t, self.vocab.get("[UNK]", 1)) for t in tokens if t.strip() or t == ' ']

    def decode(self, tokens):
        return "".join([self.inverse_vocab.get(t, "") for t in tokens])

class DataLoaderLite:
    def __init__(self, B, T, input_path, tokenizer=None):
        self.B = B
        self.T = T
        if tokenizer is None:
            # Look for tokenizer in the same directory as this file by default
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.tokenizer = RPNTokenizer(os.path.join(base_dir, "rpn-tokenizer.json"))
        else:
            self.tokenizer = tokenizer
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"{input_path} not found. Please provide a dataset file.")
                
        import random
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            random.shuffle(lines)
            text = "".join(lines)
            
        raw_tokens = self.tokenizer.encode(text)
        
        eq_id = self.tokenizer.encode("=")[0]
        nl_id = self.tokenizer.encode("\n")[0]
        mask_list = []
        is_answer = False
        for t in raw_tokens:
            if is_answer:
                mask_list.append(True)
                if t == nl_id: 
                    is_answer = False
            else:
                mask_list.append(False)
                if t == eq_id: 
                    is_answer = True
                    
        self.tokens = torch.tensor(raw_tokens, dtype=torch.long)
        self.target_mask = torch.tensor(mask_list, dtype=torch.bool)
        self.num_tokens = len(self.tokens)
        
        print(f"Loaded {self.num_tokens} tokens from disk")
        print("1 epoch = ", self.num_tokens // (self.B * self.T), "micro-batches")
        self.current_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        if self.current_pos + B * T + 1 > self.num_tokens:
            self.current_pos = 0
            
        buf = self.tokens[self.current_pos : self.current_pos + B * T + 1]
        mask_buf = self.target_mask[self.current_pos : self.current_pos + B * T + 1]
        
        x = buf[:-1].view(B, T)
        y = buf[1:].clone()
        y[~mask_buf[1:]] = -100
        y = y.view(B, T)
        
        self.current_pos += B * T
        return x, y
