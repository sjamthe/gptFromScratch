Transformer Weights
Vocab size V = 50,257
Embedding dim d_model = 768
Number of layers L = 12
Context Length T = 1024

'transformer.wte.weight' shape (50257, 768) = 38,597,376 # V * d_model
'transformer.wpe.weight' shape (1024, 768) = 786,432 # T * d_model

Transformer Layer:
'transformer.h.0.ln_1.weight' shape (768,) # d_model
'transformer.h.0.ln_1.bias' shape (768,) # d_model
'transformer.h.0.attn.c_attn.weight' shape (2304, 768) # 3 * d_model * d_model
'transformer.h.0.attn.c_attn.bias' shape (2304,) # 3 * d_model
'transformer.h.0.attn.c_proj.weight' shape (768, 768) # d_model * d_model
'transformer.h.0.attn.c_proj.bias' shape (768,) # d_model
'transformer.h.0.ln_2.weight' shape (768,) # d_model
'transformer.h.0.ln_2.bias' shape (768,) # d_model
'transformer.h.0.mlp.c_fc.weight' shape (3072, 768) # 4 * d_model * d_model
'transformer.h.0.mlp.c_fc.bias' shape (3072,) # 4 * d_model
'transformer.h.0.mlp.c_proj.weight' shape (768, 3072) # d_model * 4 * d_model
'transformer.h.0.mlp.c_proj.bias' shape (768,) # d_model
Total weights in one block: 2 * 768 + 2304 * 768 + 2304 + 768 * 768 + 3 * 768 + 3072 * 768 + 3072 + 3072*768 + 768 = 7,087,872
# 2*d_model + 3*d_model^2 + 3*d_model + d_model^2 + 3*d_model + 4*d_model^2 + 4*d_model + 4*d_model^2 + d_model 
# = 12*d_model^2 + 13*d_model = 12 * 768*768 + 13 * 768 = 7,087,872

12 transformer blocks = 12 * 7,087,872 = 85,054,464
 
'transformer.ln_f.weight' shape (768,)
'transformer.ln_f.bias' shape (768,)
'lm_head.weight' shape (50257, 768) = 38,597,376
