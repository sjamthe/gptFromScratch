Transformer Weights

'transformer.wte.weight' shape (50257, 768) = 38,597,376
'transformer.wpe.weight' shape (1024, 768) = 786,432

'transformer.h.0.ln_1.weight' shape (768,)
'transformer.h.0.ln_1.bias' shape (768,)
'transformer.h.0.attn.c_attn.weight' shape (2304, 768)
'transformer.h.0.attn.c_attn.bias' shape (2304,)
'transformer.h.0.attn.c_proj.weight' shape (768, 768)
'transformer.h.0.attn.c_proj.bias' shape (768,)
'transformer.h.0.ln_2.weight' shape (768,)
'transformer.h.0.ln_2.bias' shape (768,)
'transformer.h.0.mlp.c_fc.weight' shape (3072, 768)
'transformer.h.0.mlp.c_fc.bias' shape (3072,)
'transformer.h.0.mlp.c_proj.weight' shape (768, 3072)
'transformer.h.0.mlp.c_proj.bias' shape (768,)
Total weights in one block: 2 * 768 + 2304 * 768 + 2304 + 768 * 768 + 3 * 768 + 3072 * 768 + 3072 + 3072*768 + 768 = 7,087,872
...
'transformer.h.11.mlp.c_proj.bias' shape (768,)

12 transformer blocks = 12 * 7,087,872 = 85,054,464
 
'transformer.ln_f.weight' shape (768,)
'transformer.ln_f.bias' shape (768,)
'lm_head.weight' shape (50257, 768) = 38,597,376
