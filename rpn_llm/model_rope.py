from torch.nn.functional import leaky_relu
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
import inspect
import math

@dataclass
class GPTConfig:
    block_size: int = 2048 # max sequence length
    vocab_size: int = 50 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 8 # number of transformer blocks
    n_head: int = 8 # number of attention heads
    n_embd: int = 512 # embedding dimension

# --- RoPE Implementation ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # No absolute bias mask initialization needed for flash attention
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, use_cache: bool = False, cache_state: tuple = None, return_attention: bool = False) -> tuple:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        
        # Explicit shape for RoPE application: (B, T, n_head, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)

        offset = 0
        if use_cache and cache_state is not None:
            # cache_state[0] is k_cache of shape (B, n_head, T_cache, head_dim)
            offset = cache_state[0].size(2)

        # Apply RoPE
        if freqs_cis is not None:
            # slice freqs_cis exactly aligning to the absolute sequence indices!
            q, k = apply_rotary_emb(q, k, freqs_cis[offset : offset + T])
            
        # Transpose for attention computation: (B, n_head, T_new, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if use_cache:
            if cache_state is not None:
                k_cache, v_cache = cache_state
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)
            cache_state = (k, v)

        # Flash attention handles custom causality correctly when queries run dynamically token-by-token
        is_causal = (T > 1) 
        attn_weights = None
        
        if return_attention:
            # Manual attention to capture weights (Flash Attention hides them)
            # (B, nh, T, hs) @ (B, nh, hs, T_total) -> (B, nh, T, T_total)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if is_causal:
                # Apply causal mask manually
                # Create mask based on actual sequence indices
                # k.size(2) is the total key length (T_old + T_new)
                # T is the new query length
                # We need to mask out tokens where key_index > query_index
                q_idx = torch.arange(offset, offset + T, device=x.device).view(-1, 1)
                k_idx = torch.arange(k.size(2), device=x.device).view(1, -1)
                mask = q_idx < k_idx
                att = att.masked_fill(mask, float('-inf'))
            
            attn_weights = F.softmax(att, dim=-1)
            y = attn_weights @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, cache_state, attn_weights

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
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
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, use_cache: bool = False, cache_state: tuple = None, return_attention: bool = False) -> tuple:
        attn_out, cache_out, weights = self.attn(self.ln_1(x), freqs_cis, use_cache=use_cache, cache_state=cache_state, return_attention=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, cache_out, weights

class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            # 'wpe' is completely removed!
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize RoPE frequencies
        head_dim = config.n_embd // config.n_head
        freqs_cis = precompute_freqs_cis(head_dim, config.block_size)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(self._init_weights)
        
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
        
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, use_cache: bool = False, past_key_values: list = None, return_attention: bool = False) -> tuple:
        B, T = idx.size()

        x = self.transformer.wte(idx)
        
        freqs_cis = self.freqs_cis
        present_key_values = [] if use_cache else None
        
        all_weights = [] if return_attention else None
        for i, block in enumerate(self.transformer.h):
            cache_in = past_key_values[i] if past_key_values is not None else None
            x, cache_out, weights = block(x, freqs_cis, use_cache=use_cache, cache_state=cache_in, return_attention=return_attention)
            if use_cache:
                present_key_values.append(cache_out)
            if return_attention:
                all_weights.append(weights)
            
        x = self.transformer.ln_f(x)
        
        loss = None
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            
        if return_attention:
            return logits, loss, present_key_values, all_weights
            
        if use_cache:
            return logits, loss, present_key_values
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = dict(self.named_parameters())
        param_dict = {k: v for k, v in param_dict.items() if v.requires_grad}
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
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW.__init__).parameters
        use_fused = fused_available and (device == 'cuda' or device == 'mps')
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay, fused=use_fused)
        return optimizer
        
    @classmethod
    def from_pretrained(cls, model_type):
        raise NotImplementedError("from_pretrained is disabled: Custom RoPE implementation incompatible with HF GPT-2 absolute embeddings.")
