from torch.nn.functional import leaky_relu
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
import inspect

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of transformer blocks
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding dimension

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

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
    
        # at init load tokens from disk and store them in memory
        enc = tiktoken.get_encoding("gpt2")
        with open("input.txt", "r") as f:
            text = f.read()
        self.tokens = enc.encode(text)
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        self.num_tokens = len(self.tokens)
        print(f"Loaded {self.num_tokens} tokens from disk")
        print(" 1 epoch = ", self.num_tokens // self.B // self.T, "batches")

        # state
        self.current_pos = 0

    def next_batch(self):
        # returns a (B, T) batch of input tokens and targets
        B, T = self.B, self.T

        buf = self.tokens[self.current_pos:self.current_pos+B*T+1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets
        # advance the position in the tensor
        self.current_pos += B * T
        # wrap around if we run out of tokens
        if self.current_pos + B * T + 1 > self.num_tokens:
            self.current_pos = 0
        return x, y

import time
import math

def train_gpt2():

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
    total_batch_size = 524288 # 0.5M tokens but multiple of 2^n
    B = 4 # micro batch size
    T = 1024 # context length
    assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by (B * T)"
    grad_accum_steps = total_batch_size // (B * T)
    print(f"Total batch size: {total_batch_size}")
    print(f"Micro batch size: {B}")
    print(f"Context length: {T}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B, T)

    torch.set_float32_matmul_precision('high') # Copied it from Andrej Karpathy's nanoGPT repo. why is this done?
    
    model = GPT(GPTConfig(vocab_size=50304)) #use vocab size as power of 2, more tokens (dummy) but it speeds up training
    model.to(device)
    if device == 'cuda':
        model = torch.compile(model)
    
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 50
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
    
    # optimizer (betas and eps values from gpt3 paper)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # Use autocast for mixed precision training. Pytorch decides what gets converted to bfloat16
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # normalize the loss by dividing by the number of gradient accumulation steps
            # so that the magnitude of loss is same as without gradient accumulation as we want mean loss.
            loss = loss / grad_accum_steps
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
        if step % 1 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}, lr: {lr:.4e}, norm: {norm:.4f}, Time: {dt:.2f}ms, Tokens per second: {tokens_per_sec:.2f}")
        
    
    

#Test run by pre-loading weights
def test_pretrained_model():
    max_length=30
    num_return_sequences=5
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # get input data ready
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I am a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long) # 8,
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)

    print("--- Custom Model Generation ---")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    #Load the pretrained model
    model, model_hf = GPT.from_pretrained('gpt2')
    model.eval()
    model.to(device)

    # generation parameters
    temperature = 0.9
    top_p = 0.9
    top_k_val = 50

    # Save the original batched input so we can use it twice!
    x_input = tokens.to(device)
    print("--- Custom Model Generation ---")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    # Start x with our input
    x = x_input.clone() 
    # generate the output
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, loss = model(x)
            logits = logits[:, -1, :] # (B, vocab_size)
            
            # 1. Apply temperature
            if temperature > 0.0:
                logits = logits / temperature
                
            # 2. Apply top_k
            v, _ = torch.topk(logits, min(top_k_val, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 3. Apply top_p (nucleus sampling)
            if top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # Note: multinomial is non-deterministic, so we check against HF with same seed
            xcol = torch.multinomial(probs, num_samples=1)
            
            # append to seq
            x = torch.cat((x, xcol), dim=1) # (B, T + 1)
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)
    print("\n--- HuggingFace Model Generation ---")
    # RESET the seed before running HF
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    model_hf.eval()
    model_hf.to(device)
    # Start from the SAME input
    x_hf = x_input.clone()
    # generate the output
    # Note: we use max_length=max_length and remove num_return_sequences because x_hf is already batched to 5
    x_hf = model_hf.generate(
        x_hf, 
        max_length=max_length, 
        do_sample=True, 
        top_k=top_k_val, 
        top_p=top_p, 
        temperature=temperature,
        pad_token_id=50256, # explicitly set to eos_token_id to suppress warning
        attention_mask=torch.ones_like(x_hf) # explicitly tell HF there is no padding to suppress warning
    )
    for i in range(num_return_sequences):
        tokens = x_hf[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print("HF >", decoded)

if __name__ == "__main__":
    # test_pretrained_model()
    train_gpt2()
