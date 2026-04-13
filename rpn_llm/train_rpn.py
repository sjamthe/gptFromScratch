import os
import json
import time
import math
import torch
import wandb

import sys
from model_rope import GPT, GPTConfig
from utils import RPNTokenizer, DataLoaderLite

def run_validation(model, val_loader, device, step):
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    gt_id  = tokenizer.encode(">")[0]
    unk_id = tokenizer.encode("[UNK]")[0]
    nl_id = tokenizer.encode("\n")[0]

    model.eval()
    val_loss_accum = 0.0
    val_loss_steps = 200
    val_correct_accum = 0
    val_target_accum = 0
    with torch.no_grad():
        for _ in range(val_loss_steps):
            x_val, y_val = val_loader.next_batch()
            x_val, y_val = x_val.to(device), y_val.to(device)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, loss_val = model(x_val, y_val)
            val_loss_accum += loss_val.item()
            
            # Calculate strictly isolated Equation-Level Exact Match accuracy
            preds = torch.argmax(logits, dim=-1)
            preds_cpu = preds.cpu().numpy()
            y_cpu = y_val.cpu().numpy()
            
            for b in range(y_val.size(0)):
                # Find the LAST '>' in ground truth (y_val) to isolate the final answer
                # especially important for subtraction with internal borrow markers.
                gt_pos_y = None
                for t in range(y_val.shape[1] - 1, -1, -1):
                    if y_val[b, t].item() == gt_id:
                        gt_pos_y = t
                        break

                if gt_pos_y is None:
                    continue  # malformed or split sequence, skip

                # In Teacher Forcing, predictions and targets are strictly time-aligned!
                # We only need to check if the model perfectly predicted the true next tokens 
                # after the true '>' occurred in the context window.
                offset = gt_pos_y + 1
                
                equation_correct = True
                token_count = 0

                while offset < y_val.shape[1]:
                    tok_y = y_cpu[b, offset]
                    tok_p = preds_cpu[b, offset]

                    # Stop if we hit \n (or [UNK]) in the ground truth sequence
                    if tok_y in (unk_id, nl_id):
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
    val_accuracy_pct = (val_correct_accum / val_target_accum) * 100.0 if val_target_accum > 0 else 0.0
    val_perplexity = math.exp(val_loss_accum)
    print(f"Step {step+1}, Val Loss: {val_loss_accum:.4f}, Val Accuracy: {val_accuracy_pct:.2f}%, Val Perplexity: {val_perplexity:.4f}")
    model.train()
    return val_loss_accum,val_perplexity,val_accuracy_pct 

def train_rpn_llm(start_step=0, checkpoint_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(1337)
    if(torch.cuda.is_available()):
        torch.cuda.manual_seed(1337)
    
    total_batch_size = 8192
    B = 4
    T = 256  # 256 deeply tracks cross-5-digit logic formats flawlessly!
    assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by (B * T)"
    grad_accum_steps = total_batch_size // (B * T)
    print(f"Total batch size: {total_batch_size}")
    print(f"Micro batch size: {B}")
    print(f"Context length: {T}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    # Phase 8: Disabling padding completely and explicitly reversing native mapping values.
    train_dataset = "rpn_llm/data/RPNData-plusminus99999_tens_complement_compress_train.txt"
    val_dataset = "rpn_llm/data/RPNData-plusminus99999_tens_complement_compress_val.txt"
    train_loader = DataLoaderLite(B, T, train_dataset)
    val_loader = DataLoaderLite(B, T, val_dataset)

    torch.set_float32_matmul_precision('high')
    
    # High-Capacity 35M Model params
    n_layer = 8
    n_head = 8
    n_embd = 512
    
    # Initialize natively tracking deeply scaled logic limits
    model = GPT(GPTConfig(vocab_size=64, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=2048))
    model.to(device)
    if device == 'cuda':
        model = torch.compile(model)
    print("Model Parameters: ", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    
    max_lr = 1e-3
    min_lr = max_lr * 0.1
    warmup_steps = 500
    max_steps = 62277

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

        if (step+1) % 200 == 0:
            val_loss_accum,val_perplexity,val_accuracy_pct = run_validation(model, val_loader, device, step)
       
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
                log_dict["val_accuracy_pct"] = val_accuracy_pct
            wandb.log(log_dict, step=step+1)
    
        if (step+1) % 20000 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': model.config,
                'step': step,
                'train_loader': train_loader,
            }
            torch.save(checkpoint, f'rpn_llm/models/rope25M_tens_complement_compress_{step+1}.pt')
            print(f"Model checkpoint saved to rpn_llm/models/rope25M_tens_complement_compress_{step+1}.pt")
   
    wandb.finish()
    torch.save(checkpoint, f'rpn_llm/models/rope25M_tens_complement_compress_final.pt')
    print(f"Model checkpoint saved to rpn_llm/models/rope25M_tens_complement_compress_final.pt")

if __name__ == "__main__":
    import sys
    start_step = 0
    checkpoint_path = None
    if len(sys.argv) > 2:
        start_step = int(sys.argv[1])
        checkpoint_path = sys.argv[2]
    train_rpn_llm(start_step, checkpoint_path)
