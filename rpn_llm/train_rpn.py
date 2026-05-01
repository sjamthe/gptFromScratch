import os
import json
import time
import math
import torch
import wandb

import sys
from utils import RPNTokenizer, DataLoaderLite

def run_teacher_forcing_validation(model, val_loader, device, step):
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    ans_id = tokenizer.encode("[ANS]")[0]
    unk_id = tokenizer.encode("[UNK]")[0]
    eos_id = tokenizer.encode("[EOS]")[0]
    pad_id = tokenizer.encode("[PAD]")[0]
    bos_id = tokenizer.encode("[BOS]")[0]
    pad_id = 0 # [PAD] is 0

    model.eval()
    val_loss_accum = 0.0
    val_loss_steps = 200
    
    # Equation-level counters (Final Answer only)
    val_correct_accum = 0
    val_target_accum = 0
    
    # Global token-level counters (All valid tokens in sequence)
    val_token_correct_accum = 0
    val_token_target_accum = 0

    with torch.no_grad():
        for _ in range(val_loss_steps):
            x_val, y_val = val_loader.next_batch()
            x_val, y_val = x_val.to(device), y_val.to(device)
            # Mask out BOS (2) only. NL (1) MUST be learned so the model knows when to stop.
            y_val_masked = y_val.clone()
            y_val_masked[y_val_masked == 2] = -100
            
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, loss_val = model(x_val, y_val_masked)
            val_loss_accum += loss_val.item()
            
            # Calculate accuracies
            preds = torch.argmax(logits, dim=-1)
            
            # 1. Global Token-Level Accuracy
            # Mask out non-content tokens (UNK, EOS, PAD, BOS) and non-target tokens (-100)
            valid_mask = (y_val != unk_id) & (y_val != eos_id) & (y_val != pad_id) & (y_val != bos_id) & (y_val != -100)
            val_token_correct_accum += ((preds == y_val) & valid_mask).sum().item()
            val_token_target_accum += valid_mask.sum().item()

            # 2. Equation-Level Exact Match (Final Answer only)
            preds_cpu = preds.cpu().numpy()
            y_cpu = y_val.cpu().numpy()
            
            for b in range(y_val.size(0)):
                gt_pos_y = None
                for t in range(y_val.shape[1] - 1, -1, -1):
                    if y_val[b, t].item() == ans_id:
                        gt_pos_y = t
                        break

                if gt_pos_y is None:
                    continue

                offset = gt_pos_y + 1
                equation_correct = True
                token_count = 0

                while offset < y_val.shape[1]:
                    tok_y = y_cpu[b, offset]
                    tok_p = preds_cpu[b, offset]

                    if tok_y in (unk_id, eos_id, pad_id):
                        break
                    if tok_y != tok_p:
                        equation_correct = False
                    token_count += 1
                    offset += 1

                if token_count > 0:
                    val_target_accum += 1.0
                    if equation_correct:
                        val_correct_accum += 1.0

    val_loss_accum /= val_loss_steps
    val_ans_accuracy_pct = (val_correct_accum / val_target_accum) * 100.0 if val_target_accum > 0 else 0.0
    val_token_accuracy_pct = (val_token_correct_accum / val_token_target_accum) * 100.0 if val_token_target_accum > 0 else 0.0
    val_perplexity = math.exp(val_loss_accum)
    
    print(f"Step {step+1}, Val Loss: {val_loss_accum:.4f}, Val TF Token Acc: {val_token_accuracy_pct:.2f}%, Val TF Ans Acc: {val_ans_accuracy_pct:.2f}%, Val PPL: {val_perplexity:.4f}")
    
    model.train()
    return val_loss_accum, val_perplexity, val_ans_accuracy_pct, val_token_accuracy_pct

def run_generation_validation(model, val_loader, device, step, num_batches=4):
    """
    Performs real auto-regressive generation to measure TRUE accuracy.
    This is slower, so we only run it on a few batches.
    """
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    sep_id = tokenizer.encode("?")[0]
    ans_id = tokenizer.encode("[ANS]")[0]
    eos_id = tokenizer.encode("[EOS]")[0]
    
    model.eval()
    total_correct = 0
    total_equations = 0
    max_new_tokens = 256
    
    with torch.no_grad():
        for _ in range(num_batches):
            x_val, y_val = val_loader.next_batch()
            batch_size = x_val.size(0)
            
            for b in range(batch_size):
                row_x = x_val[b].tolist()
                row_y = y_val[b].tolist()
                try:
                    sep_pos = row_x.index(sep_id)
                except ValueError:
                    continue # No prompt start in this slice
                
                # The prompt is everything up to '?'
                prompt_tokens = torch.tensor(row_x[:sep_pos+1], dtype=torch.long, device=device).unsqueeze(0)
                
                try:
                    # In y_val, the first token is x_val[1], so sep_pos in x is sep_pos-1 in y if we align.
                    # Actually, y_val is just x_val shifted. y[sep_pos] is the token after x[sep_pos].
                    targets_shifted = row_y[sep_pos:]
                    nl_pos_in_targets = targets_shifted.index(nl_id)
                    full_target_seq = targets_shifted[:nl_pos_in_targets]
                except ValueError:
                    continue # Equation truncated by sequence end
                
                # Extract the expected final answer (after last >)
                target_str = tokenizer.decode(full_target_seq)
                if "[ANS]" not in target_str:
                    continue
                expected_answer = target_str.split("[ANS]")[-1].split("[UNK]")[0].split("[EOS]")[0].strip()
                
                # --- GENERATION ---
                idx = prompt_tokens
                past_kv = None
                generated_tokens = []
                
                for _ in range(max_new_tokens):
                    # Track phases for KV cache mask
                    is_phase_shift = (idx == 10) | (idx == 11) | (idx == 12)
                    full_phase_ids = is_phase_shift.cumsum(dim=-1)

                    idx_cond = idx[:, -1:] if past_kv is not None else idx
                    with torch.autocast(device, dtype=torch.bfloat16):
                        logits, _, past_kv = model(idx_cond, use_cache=True, past_key_values=past_kv, full_phase_ids=full_phase_ids)
                    
                    logits = logits[:, -1, :]
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                    next_id = idx_next.item()
                    
                    if next_id == eos_id:
                        break
                    
                    generated_tokens.append(next_id)
                    idx = torch.cat((idx, idx_next), dim=1)
                
                # --- EVALUATION ---
                pred_str = tokenizer.decode(generated_tokens)
                pred_answer = pred_str.split("[ANS]")[-1].split("[UNK]")[0].split("[EOS]")[0].strip() if "[ANS]" in pred_str else "N/A"
                
                if pred_answer == expected_answer:
                    total_correct += 1
                total_equations += 1

    gen_accuracy_pct = (total_correct / total_equations) * 100.0 if total_equations > 0 else 0.0
    print(f"Step {step+1}, True Gen Accuracy: {gen_accuracy_pct:.2f}% ({total_correct}/{total_equations})")
    model.train()
    return gen_accuracy_pct

def train_rpn_llm(start_step=0, checkpoint_path=None, model_type="rdt", max_steps=80000, dataset_prefix="1-22_tens_comp_clean_tiered", use_phase_mask=True):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(1337)
    if(torch.cuda.is_available()):
        torch.cuda.manual_seed(1337)
    
    total_batch_size = 8192
    B = 4
    T = 512  # Increased to 512 to capture the full context (including \n) for 17-22 digit problems
    assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by (B * T)"
    grad_accum_steps = total_batch_size // (B * T)
    print(f"Total batch size: {total_batch_size}")
    print(f"Micro batch size: {B}")
    print(f"Context length: {T}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    # Select dataset
    # dataset_prefix is passed as an argument
    train_dataset = f"rpn_llm/data/RPNData-{dataset_prefix}_train.txt"
    val_dataset = f"rpn_llm/data/RPNData-{dataset_prefix}_val.txt"
    train_loader = DataLoaderLite(B, T, train_dataset)
    val_loader = DataLoaderLite(B, T, val_dataset)

    torch.set_float32_matmul_precision('high')
    
    if model_type == "rdt":
        from model_rdt import GPT, GPTConfig
        n_prelude, n_coda, n_layer = 1, 1, 6
        n_head, n_embd = 8, 512
        config = GPTConfig(vocab_size=64, n_prelude=n_prelude, n_coda=n_coda, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=2048, use_phase_mask=use_phase_mask)
    elif model_type == "ut":
        from model_rope import GPT, GPTConfig
        n_layer, n_head, n_embd = 8, 8, 512
        config = GPTConfig(vocab_size=64, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=2048, universal=True, use_phase_mask=use_phase_mask)
    elif model_type == "rope":
        from model_rope import GPT, GPTConfig
        # Stage: The "Wide" Lite Model to test Reversal Capacity
        n_layer, n_head, n_embd = 3, 4, 256
        config = GPTConfig(vocab_size=64, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=512, universal=False, use_phase_mask=use_phase_mask)

    model = GPT(config)
    model.to(device)
    
    # Calculate parameter count dynamically
    num_params = sum(p.numel() for p in model.parameters())
    param_str = f"{num_params/1e6:.1f}M"
    model_prefix = f"{model_type}{param_str}_phaseMask_{use_phase_mask}"
    
    run_name = f"{model_prefix}_{dataset_prefix}"

    if device == 'cuda':
        model = torch.compile(model)
    print(f"Initialized {model_type.upper()} Model: {model_prefix}")
    print(f"Total Parameters: {num_params:,}")
    
    max_lr = 1e-4
    min_lr = max_lr * 0.1
    warmup_steps = 1000
    lr_decay_steps = 62277
    
    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        if it > lr_decay_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (lr_decay_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + (max_lr - min_lr) * coeff
    
    wandb.init(
        project="rpn-llm",
        name=f"{run_name}",
        config={
            "use_phase_mask": use_phase_mask,
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
            "model_type": model_type,
            "model_params": sum(p.numel() for p in model.parameters()) / 1e6,
            "train_dataset": train_dataset,
        }
    )

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
    
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # advance dataloader to the precise step analytically
        train_loader.current_pos = (start_step * grad_accum_steps * train_loader.B * train_loader.T) % train_loader.num_tokens
        print(f"Resuming from step {start_step}! Dataloader synced to pos {train_loader.current_pos}")

    for step in range(start_step, max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device, dtype=torch.bfloat16):
                # Mask out BOS (2) only. NL (1) MUST be learned.
                # This prevents the model from being penalized for unpredictable sequence transitions
                y_masked = y.clone()
                y_masked[y_masked == 2] = -100
                logits, loss = model(x, y_masked)
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
        if (step+1) % 100 == 0:
            print(f"Step {step+1}, Loss: {loss_accum.item():.4f}, lr: {lr:.4e}, norm: {norm:.4f}, Time: {dt:.2f}ms, Tokens per second: {tokens_per_sec:.2f}")

        if (step+1) % 1000 == 0:
            val_loss_accum, val_perplexity, val_ans_acc, val_tok_acc = run_teacher_forcing_validation(model, val_loader, device, step)
       
        if (step+1) % 10 == 0:
            log_dict = {
                "loss": loss_accum.item(),
                "lr": lr,
                "norm": norm,
                "time": dt,
                "tokens_per_sec": tokens_per_sec,
            }
            if (step+1) % 1000 == 0:
                log_dict["val_loss"] = val_loss_accum
                log_dict["val_perplexity"] = val_perplexity
                log_dict["val_ans_accuracy"] = val_ans_acc
                log_dict["val_token_accuracy"] = val_tok_acc
            wandb.log(log_dict, step=step+1)
    
        if (step+1) % 8000 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': model.config,
                'step': step,
                'train_loader': train_loader,
            }
            torch.save(checkpoint, f'rpn_llm/models/{run_name}_{step+1}.pt')
            print(f"Model checkpoint saved to rpn_llm/models/{run_name}_{step+1}.pt")
   
    wandb.finish()
    if (step+1) % 8000 != 0:
        torch.save(checkpoint, f'rpn_llm/models/{run_name}_{step+1}.pt')
        print(f"Model checkpoint saved to rpn_llm/models/{run_name}_{step+1}.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Positional args
    parser.add_argument("start_step", type=int, nargs="?", default=0, help="Step to resume from")
    parser.add_argument("checkpoint_path", type=str, nargs="?", default=None, help="Path to checkpoint")
    parser.add_argument("--model", type=str, default="rope", choices=["rope", "ut", "rdt"], help="Model architecture to train")
    parser.add_argument("--max_steps", type=int, default=64000, help="Total steps to train for (default 80000)")
    parser.add_argument("--dataset", type=str, default="1-22_uniform_BOS", help="Dataset prefix")
    parser.add_argument("--no_phase_mask", action="store_false", dest="use_phase_mask", help="Disable sequential phase masking")
    parser.set_defaults(use_phase_mask=True)
    
    args = parser.parse_args()
        
    train_rpn_llm(args.start_step, args.checkpoint_path, args.model, args.max_steps, args.dataset, args.use_phase_mask)
