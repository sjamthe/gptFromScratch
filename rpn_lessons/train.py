import os
import time
import math
import argparse
import torch
import wandb

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer, DataLoaderLite
from val import run_lesson_validation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lesson", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--checkpoint", type=str, default=None, help="Warm-start checkpoint path")
    parser.add_argument("--max_steps", type=int, default=None, help="Override default max steps")
    parser.add_argument("--max_lr", type=float, default=None, help="Override default max learning rate")
    parser.add_argument("--min_lr", type=float, default=None, help="Override default min learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Micro batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="rpn-curriculum")
    parser.add_argument("--n_counter", type=int, default=2)
    parser.add_argument("--n_coord", type=int, default=2)
    parser.add_argument("--run_name_suffix", type=str, default="", help="Suffix for checkpoint and wandb run name")
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}", flush=True)
    
    # Tokenizer
    tokenizer = RPNTokenizer("rpn_lessons/rpn-tokenizer.json")
    bos_id = tokenizer.encode("[BOS]")[0]
    rev_id = tokenizer.encode("[REV]")[0]
    math_id = tokenizer.encode("[MATH]")[0]
    ans_id = tokenizer.encode("[ANS]")[0]
    phase_token_ids = [rev_id, math_id, ans_id]
    
    # 1. Lesson defaults
    # LR and step configurations from approved training plan
    defaults = {
        1: {"max_lr": 3e-4, "min_lr": 3e-5, "max_steps": 40000, "delimiter": "[ANS]"},
        2: {"max_lr": 1e-4, "min_lr": 1e-5, "max_steps": 40000, "delimiter": "[REV]"},
        3: {"max_lr": 1.5e-4, "min_lr": 1.5e-5, "max_steps": 80000, "delimiter": "[MATH]"},
        4: {"max_lr": 1e-4, "min_lr": 1e-5, "max_steps": 40000, "delimiter": "[REV]"}
    }
    
    cfg = defaults[args.lesson]
    max_steps = args.max_steps if args.max_steps is not None else cfg["max_steps"]
    max_lr = args.max_lr if args.max_lr is not None else cfg["max_lr"]
    min_lr = args.min_lr if args.min_lr is not None else cfg["min_lr"]
    delimiter = cfg["delimiter"]
    
    # Dataloaders (T = 384 for 380 token cap)
    B = args.batch_size
    T = 384
    grad_accum_steps = args.grad_accum_steps
    
    train_path = f"rpn_lessons/data/lesson{args.lesson}_train.txt"
    val_path = f"rpn_lessons/data/lesson{args.lesson}_val.txt"
    
    print(f"Loading datasets for Lesson {args.lesson} (Delimiter: {delimiter})...", flush=True)
    train_loader = DataLoaderLite(B, T, train_path, tokenizer=tokenizer, delimiter_token=delimiter)
    val_loader = DataLoaderLite(B, T, val_path, tokenizer=tokenizer, delimiter_token=delimiter)
    
    # 2. Initialize Model or Load Checkpoint
    checkpoint = None
    if args.checkpoint is not None:
        print(f"Warm-starting model from checkpoint: {args.checkpoint}", flush=True)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Load config from checkpoint, but update block size and disable digit abstraction
        config = checkpoint['config']
        config.block_size = T
        config.use_digit_abstraction = False
        config.phase_token_ids = phase_token_ids
        config.bos_token_id = bos_id
    else:
        print("Initializing new Universal Transformer model from scratch...", flush=True)
        config = GPTConfig(
            vocab_size=64, 
            n_layer=2, 
            n_head=8, 
            n_embd=384, 
            block_size=T, 
            universal=True,
            use_phase_mask=True, 
            mlp_ratio=4, 
            bos_token_id=bos_id, 
            phase_token_ids=phase_token_ids,
            n_counter=args.n_counter, 
            n_buckets=4, 
            n_coord=args.n_coord, 
            n_coord_heads=4 if args.n_coord > 0 else 0,
            freeze_coord_scale=False,
            tie_weights=False
        )
        
    if config.counter_inject_layers is None and config.n_counter > 0:
        config.counter_inject_layers = [-1, 0] if config.n_counter == 2 else [-1]
    if config.coord_inject_layers is None and config.n_coord > 0:
        config.coord_inject_layers = [-1, 0] if config.n_coord == 2 else [-1]
        
    model = GPT(config)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    
    if device == 'cuda':
        model = torch.compile(model)
        
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params:,}", flush=True)
    
    # LR scheduler
    warmup_steps = 1000
    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + (max_lr - min_lr) * coeff
        
    # Optimizer
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
    
    # Initialize WandB
    run_name = f"lesson{args.lesson}_ut_1.5M_T384"
    if args.run_name_suffix:
        run_name += f"_{args.run_name_suffix}"
    if args.checkpoint is not None:
        run_name += "_warmstart"
    if config.tie_weights is False:
        run_name += "_no_tie"
        
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "lesson": args.lesson,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "B": B,
            "T": T,
            "grad_accum_steps": grad_accum_steps,
            "num_params": num_params,
            **{k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        }
    )
    
    # Output dir for saving checkpoints
    model_dir = "rpn_lessons/models"
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Starting training loop (Total Steps: {max_steps})...", flush=True)
    start_time = time.time()
    
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0
        
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            with torch.autocast(device, dtype=torch.bfloat16):
                # Mask out BOS tokens in loss calculation
                y_masked = y.clone()
                y_masked[y_masked == bos_id] = -100
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
        dt = (t1 - t0) * 1000 # ms
        tokens_per_sec = grad_accum_steps * B * T / (t1 - t0)
        
        # Logging
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{max_steps} | Loss: {loss_accum.item():.5f} | lr: {lr:.5e} | norm: {norm:.4f} | {dt:.1f}ms | {tokens_per_sec:.0f} tok/s", flush=True)
            wandb.log({
                "step": step + 1,
                "loss": loss_accum.item(),
                "lr": lr,
                "grad_norm": norm,
                "tok_per_sec": tokens_per_sec
            })
            
        # Teacher forcing loss and exact-match validation
        if (step + 1) % 1000 == 0:
            # 1. Run quick teacher-forcing validation loss
            val_loss_accum = 0
            model.eval()
            with torch.no_grad():
                for _ in range(10): # 10 batches
                    vx, vy = val_loader.next_batch()
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.autocast(device, dtype=torch.bfloat16):
                        vy_masked = vy.clone()
                        vy_masked[vy_masked == bos_id] = -100
                        _, vloss = model(vx, vy_masked)
                    val_loss_accum += vloss.item()
            val_loss = val_loss_accum / 10
            
            # 2. Run autoregressive exact-match accuracy validation
            # Check 50 samples during training to keep it quick
            em_accuracy = run_lesson_validation(
                model=model,
                tokenizer=tokenizer,
                data_path=val_path,
                lesson=args.lesson,
                device=device,
                num_samples=40
            )
            
            print(f"--> Step {step + 1} Eval: Val Loss: {val_loss:.5f} | Exact Match Accuracy: {em_accuracy:.2f}%", flush=True)
            wandb.log({
                "step": step + 1,
                "val_loss": val_loss,
                "val_perplexity": math.exp(val_loss) if val_loss < 20 else 9999,
                "exact_match_accuracy": em_accuracy
            })
            model.train()
            
        # Save checkpoints
        if max_steps <= 20000:
            step_interval = 5000
        else:
            step_interval = 10000
        if (step + 1) % (step_interval) == 0 or (step + 1) == max_steps:
            suffix_str = f"_{args.run_name_suffix}" if args.run_name_suffix else ""
            checkpoint_path = os.path.join(model_dir, f"lesson{args.lesson}{suffix_str}_step{step + 1}.pt")
            checkpoint = {
                'model': model.state_dict(),
                'config': config,
                'step': step + 1
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}", flush=True)
            
    wandb.finish()
    print(f"Training completed in {((time.time() - start_time) / 60):.2f} minutes.", flush=True)

if __name__ == "__main__":
    main()
