import os
import json
import time
import math
import torch
import wandb

from model_rope import GPT, GPTConfig

class RPNTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["model"]["vocab"]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        return [self.vocab.get(char, self.vocab.get("[UNK]", 1)) for char in text]
    
    def decode(self, tokens):
        return "".join([self.inverse_vocab.get(t, "") for t in tokens])

class DataLoaderLite:
    def __init__(self, B, T, input_path):
        self.B = B
        self.T = T
        tokenizer = RPNTokenizer("rpn-tokenizer.json")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"{input_path} not found. Please provide a dataset file.")
                
        import random
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            random.shuffle(lines)
            text = "".join(lines)
            
        raw_tokens = tokenizer.encode(text)
        
        eq_id = tokenizer.encode("=")[0]
        nl_id = tokenizer.encode("\n")[0]
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

def train_rpn_llm():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(1337)
    if(torch.cuda.is_available()):
        torch.cuda.manual_seed(1337)
    
    total_batch_size = 8192
    B = 8
    T = 32
    assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by (B * T)"
    grad_accum_steps = total_batch_size // (B * T)
    print(f"Total batch size: {total_batch_size}")
    print(f"Micro batch size: {B}")
    print(f"Context length: {T}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    # Retarget completely UNPADDED data to test RoPE explicitly!
    train_dataset = "data/RPNData-999+-_train.txt"
    val_dataset = "data/RPNData-999+-_val.txt"
    train_loader = DataLoaderLite(B, T, train_dataset)
    val_loader = DataLoaderLite(B, T, val_dataset)

    torch.set_float32_matmul_precision('high')
    
    # Standard 10M Model params
    n_layer = 6
    n_head = 6
    n_embd = 384
    
    # Initialize from our new RoPE Model config limits
    model = GPT(GPTConfig(vocab_size=64, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=1024))
    model.to(device)
    if device == 'cuda':
        model = torch.compile(model)
    print("Model Parameters: ", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    
    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 500
    max_steps = 10000

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + (max_lr - min_lr) * coeff
    
    wandb.init(
        project="rpn-llm",
        name="rpn-rope-baseline",
        config={
            "total_batch_size": total_batch_size,
            "B": B,
            "T": T,
            "grad_accum_steps": grad_accum_steps,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "model_params": sum(p.numel() for p in model.parameters()) / 1e6
        }
    )

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
    for step in range(max_steps):
        if (step+1) % 200 == 0:
            model.eval()
            val_loss_accum = 0.0
            val_loss_steps = 200
            with torch.no_grad():
                for _ in range(val_loss_steps):
                    x_val, y_val = val_loader.next_batch()
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    with torch.autocast(device, dtype=torch.bfloat16):
                        _, loss_val = model(x_val, y_val)
                    val_loss_accum += loss_val.item()
            val_loss_accum /= val_loss_steps
            val_perplexity = math.exp(val_loss_accum)
            print(f"Step {step+1}, Val Loss: {val_loss_accum:.4f}, Val Perplexity: {val_perplexity:.4f}")
            model.train()

        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
            
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
            
        t1 = time.time()
        dt = (t1 - t0)*1000 # ms
        tokens_per_sec = grad_accum_steps * train_loader.B * train_loader.T / (t1 - t0)
        if (step+1) % 50 == 0:
            print(f"Step {step+1}, Loss: {loss_accum.item():.4f}, lr: {lr:.4e}, norm: {norm:.4f}, Time: {dt:.2f}ms, Tokens per second: {tokens_per_sec:.2f}")
        
        if (step+1) % 10 == 0:
            log_dict = {
                "loss": loss_accum.item(),
                "lr": lr,
                "norm": norm,
                "time": dt,
                "tokens_per_sec": tokens_per_sec,
            }
            if (step+1) % 200 == 0:
                log_dict["val_loss"] = val_loss_accum
                log_dict["val_perplexity"] = val_perplexity 
            wandb.log(log_dict, step=step+1)
    
        if (step+1) % 2000 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': model.config,
                'step': step,
                'train_loader': train_loader,
            }
            torch.save(checkpoint, f'rope10M_checkpoint_{step}.pt')
            print(f"Model checkpoint saved to rope10M_checkpoint_{step}.pt")
   
    wandb.finish()

if __name__ == "__main__":
    train_rpn_llm()
