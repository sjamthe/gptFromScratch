from torch.nn.functional import leaky_relu
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
import inspect
import wandb

@dataclass
class GPTConfig:
    block_size: int = 32 # max sequence length
    vocab_size: int = 64 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 2 # number of transformer blocks
    n_head: int = 2 # number of attention heads
    n_embd: int = 32 # embedding dimension

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # scale the weights of c_proj using a flag
        self.c_proj.NANO_GPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                            .view(1, 1, config.block_size, config.block_size))
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Scale factor is 1/sqrt(head_size), where head_size is k.size(-1)

        #Attention calculation
        # att = (q @ k.transpose(-2, -1)) * (k.size(-1) ** -0.5)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        # replace above 4 att lines with flash attention (it is more helpful on NVDIA GPUs but give 5% speedup on MPS)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GPT-2 uses GELU approximation with tanh, but we can remove it later
        # it should not make any difference.
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight sharing scheme between 1st and last layer maps tokens to logits and back to tokens:
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # init params 
        self.apply(self._init_weights)
        
    # init weights, important to see 0.02 used in gpt2 paper
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANO_GPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=.02)
        
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

        pos = torch.arange(T, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters that require grad
        param_dict = dict(self.named_parameters())
        param_dict = {k: v for k, v in param_dict.items() if v.requires_grad}
        # create the two groups of parameters
        # group 1: weight decay (all parameters that are 2D)
        # group 2: no weight decay (all biases, layernorms parameters that are 1D)
        decay_params = [p for k, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for k, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_no_decay = sum(p.numel() for p in no_decay_params)
        print(f"num decayed parameter tensors: {len(decay_params)} with {num_decay} parameters")
        print(f"num non-decayed parameter tensors: {len(no_decay_params)} with {num_no_decay} parameters")  
        # create the optimizer with weight decay and learning rate
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW.__init__).parameters
        use_fused = fused_available and (device == 'cuda' or device == 'mps')
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay, fused=use_fused)
        return optimizer
        
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        # NOTE: The below 3 filters resulted to no reduction in the number of keys
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model, model_hf

import json
import os

class RPNTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["model"]["vocab"]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        # Fallback to [UNK] (ID 1) if character is not found
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
        
        # Build target mask array to ignore prompt tokens (loss only computed on answers)
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

import time
import math

def train_rpn_llm():

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(1337)
    if(torch.cuda.is_available()):
        torch.cuda.manual_seed(1337)
    
    """
    Original GPT2 paper user 0.5M (tokens) batch size so B * T = 500,000
    or B = 500,000/1024 = 488. But this is too high for my GPU so we use gradient accumulation.
    basically we process multiple batches and accumulate the gradients and then update the model parameters.
    """
    total_batch_size = 8192 # vastly reduced for toy RPN model
    B = 8 # micro batch size
    T = 32 # context length (matches config block_size)
    assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by (B * T)"
    grad_accum_steps = total_batch_size // (B * T)
    print(f"Total batch size: {total_batch_size}")
    print(f"Micro batch size: {B}")
    print(f"Context length: {T}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    train_dataset = "data/RPNData-plusminus999padded-train.txt"
    val_dataset = "data/RPNData-plusminus999padded-test.txt"
    train_loader = DataLoaderLite(B, T, train_dataset)
    val_loader = DataLoaderLite(B, T, val_dataset)

    torch.set_float32_matmul_precision('high') # Copied it from Andrej Karpathy's nanoGPT repo. why is this done?
    
    n_layer = 6
    n_head = 6
    n_embd = 384
    model = GPT(GPTConfig(vocab_size=64, n_layer=n_layer, n_head=n_head, n_embd=n_embd)) #use vocab size as power of 2 (matches config defaults)
    model.to(device)
    if device == 'cuda':
        model = torch.compile(model)
    print("Model Parameters: ", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 500
    max_steps = 10000
    # learning rate decay schedule
    def get_lr(it):
        # 1) linear warmup for warmup_steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + (max_lr - min_lr) * coeff
    
    # logging
    wandb.init(
        project="rpn-llm",
        name="rpn-999-padded",
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
            # log model parameters
            "model_params": sum(p.numel() for p in model.parameters()) / 1e6 # in millions
        }
    )

    # Expecected starting loss based on normal distribution of probability of all tokens
    # -ln(1/vocab_size) = -ln(1/64) = 4.1589
    
    # optimizer (betas and eps values from gpt3 paper)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
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
            # Use autocast for mixed precision training. Pytorch decides what gets converted to bfloat16
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # normalize the loss by dividing by the number of gradient accumulation steps
            # so that the magnitude of loss is same as without gradient accumulation as we want mean loss.
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        # clip model parameters to prevent exploding gradients per gpt3 paper
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize() # wait for GPU to finish
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
                log_dict["val_perplexity"] = val_perplexity # Calculated upstream during validation
            # Provide the `step` explicitly so it syncs correctly in the WandB UI timeline!
            wandb.log(log_dict, step=step+1)
    
        # Save the model at every 100 steps 
        if (step+1) % 2000 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': model.config,
                'step': step,
            'train_loader': train_loader,
            }
            torch.save(checkpoint, f'rpn10M_checkpoint_{step}.pt')
            print(f"Model checkpoint saved to rpn10M_checkpoint_{step}.pt")
   
    wandb.finish()

if __name__ == "__main__":
    train_rpn_llm()
