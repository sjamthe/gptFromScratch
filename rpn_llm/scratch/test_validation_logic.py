import torch
import sys
import os
import math
from collections import Counter

sys.path.append("rpn_llm")
from model_rope import GPT, GPTConfig
from utils import RPNTokenizer, DataLoaderLite

def run_teacher_forcing_validation_instrumented(model, val_loader, device, step):
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    ans_id = tokenizer.encode("[ANS]")[0]
    math_id = tokenizer.encode("[MATH]")[0]
    rev_id = tokenizer.encode("[REV]")[0]
    unk_id = tokenizer.encode("[UNK]")[0]
    eos_id = tokenizer.encode("[EOS]")[0]
    pad_id = tokenizer.encode("[PAD]")[0]
    bos_id = tokenizer.encode("[BOS]")[0]
    pad_id = 0 # [PAD] is 0

    model.eval()
    val_loss_accum = 0.0
    val_loss_steps = 50
    
    val_rev_correct = 0; val_rev_target = 0
    val_math_correct = 0; val_math_target = 0
    val_ans_correct = 0; val_ans_target = 0
    failure_positions = [] 
    total_positions = [] 
    failure_examples = []

    with torch.no_grad():
        for _ in range(val_loss_steps):
            x_val, y_val = val_loader.next_batch()
            x_val, y_val = x_val.to(device), y_val.to(device)
            # Mask out BOS (2) only
            y_val_masked = y_val.clone()
            y_val_masked[y_val_masked == 2] = -100
            
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, loss_val = model(x_val, y_val_masked)
            val_loss_accum += loss_val.item()
            
            preds = torch.argmax(logits, dim=-1)
            
            valid_mask = (y_val != unk_id) & (y_val != eos_id) & (y_val != pad_id) & (y_val != bos_id) & (y_val != -100)
            
            B, T = y_val.size()
            valid_rev_mask = torch.zeros_like(y_val, dtype=torch.bool)
            valid_math_mask = torch.zeros_like(y_val, dtype=torch.bool)
            valid_ans_mask = torch.zeros_like(y_val, dtype=torch.bool)
            pos_in_rev = torch.zeros_like(y_val, dtype=torch.long)
            
            x_cpu = x_val.cpu()
            y_cpu = y_val.cpu()
            
            for b in range(B):
                has_bos = False
                phase = None
                curr_pos = 0
                for t in range(T):
                    # Check sequence boundary via x_val
                    if x_cpu[b, t] == bos_id:
                        has_bos = True
                    elif x_cpu[b, t] == eos_id:
                        has_bos = False
                        phase = None
                    
                    if y_cpu[b, t] == rev_id:
                        phase = 'rev'
                        curr_pos = 0
                    elif y_cpu[b, t] == math_id:
                        phase = 'math'
                    elif y_cpu[b, t] == ans_id:
                        phase = 'ans'
                        
                    if has_bos:
                        if phase == 'rev':
                            valid_rev_mask[b, t] = True
                            pos_in_rev[b, t] = curr_pos
                            curr_pos += 1
                        elif phase == 'math':
                            valid_math_mask[b, t] = True
                        elif phase == 'ans':
                            valid_ans_mask[b, t] = True
            
            # Move masks back to device for fast operations if needed, or just do it on CPU
            valid_rev_mask = valid_rev_mask.to(device) & valid_mask
            valid_math_mask = valid_math_mask.to(device) & valid_mask
            valid_ans_mask = valid_ans_mask.to(device) & valid_mask
            
            val_rev_correct += ((preds == y_val) & valid_rev_mask).sum().item()
            val_rev_target += valid_rev_mask.sum().item()
            
            val_math_correct += ((preds == y_val) & valid_math_mask).sum().item()
            val_math_target += valid_math_mask.sum().item()
            
            val_ans_correct += ((preds == y_val) & valid_ans_mask).sum().item()
            val_ans_target += valid_ans_mask.sum().item()
            
            # --- Instrument failures and totals ---
            preds_cpu = preds.cpu()
            valid_rev_mask_cpu = valid_rev_mask.cpu()
            incorrect_mask = valid_rev_mask_cpu & (preds_cpu != y_cpu)
            
            # Track total valid positions
            b_all, t_all = torch.where(valid_rev_mask_cpu)
            for b, t in zip(b_all, t_all):
                total_positions.append(pos_in_rev[b, t].item())
                    
            # Track failure positions
            b_err, t_err = torch.where(incorrect_mask)
            for b, t in zip(b_err, t_err):
                pos = pos_in_rev[b, t].item()
                failure_positions.append(pos)
                
                if len(failure_examples) < 10:
                    b_val = b.item()
                    t_val = t.item()
                    context_str = tokenizer.decode(x_cpu[b_val, max(0, t_val-5):t_val+1].tolist())
                    target = tokenizer.decode([y_cpu[b_val, t_val].item()])
                    pred = tokenizer.decode([preds_cpu[b_val, t_val].item()])
                    failure_examples.append(f"Pos {pos:2d} | Context: '...{context_str}' -> Target: '{target}', Pred: '{pred}'")

    val_loss_accum /= val_loss_steps
    val_rev_accuracy_pct = (val_rev_correct / val_rev_target) * 100.0 if val_rev_target > 0 else 0.0
    val_math_accuracy_pct = (val_math_correct / val_math_target) * 100.0 if val_math_target > 0 else 0.0
    val_ans_accuracy_pct = (val_ans_correct / val_ans_target) * 100.0 if val_ans_target > 0 else 0.0
    
    print("\n--- EXAMPLES OF FAILURES ---")
    for ex in failure_examples:
        print(ex)
    
    dist_failures = Counter(failure_positions)
    dist_total = Counter(total_positions)
    
    # Calculate averages
    def calc_avg(start, end):
        fails = sum(dist_failures.get(p, 0) for p in range(start, end+1))
        totals = sum(dist_total.get(p, 0) for p in range(start, end+1))
        return (fails / totals * 100.0) if totals > 0 else 0.0

    avg_1_10 = calc_avg(1, 10)
    avg_11_20 = calc_avg(11, 20)
    avg_21_30 = calc_avg(21, 30)

    print(f"  Overall Rev Acc: {val_rev_accuracy_pct:.2f}%")
    print(f"  Overall Math Acc: {val_math_accuracy_pct:.2f}%")
    print(f"  Overall Ans Acc: {val_ans_accuracy_pct:.2f}%")
    print(f"  Pos  1-10 Error Rate: {avg_1_10:.2f}%")
    print(f"  Pos 11-20 Error Rate: {avg_11_20:.2f}%")
    print(f"  Pos 21-30 Error Rate: {avg_21_30:.2f}%")
    return val_rev_accuracy_pct

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

models = [
    "rope2.4M_phaseMask_True_1-22_phase_lean_64000.pt",
    "rope3.6M_phaseMask_False_1-22_phase_lean_64000.pt",
    "rope3.6M_phaseMask_True_1-22_phase_lean_64000.pt"
]

for m in models:
    model_path = f"rpn_llm/models/{m}"
    print(f"\nEvaluating: {m}")
    if not os.path.exists(model_path):
        print("  FILE NOT FOUND!")
        continue
        
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        cfg = checkpoint['config']
        print(f"  Model dims: n_layer={cfg.n_layer}, n_head={cfg.n_head}, n_embd={cfg.n_embd}")
        model = GPT(cfg).to(device)
        model.load_state_dict(checkpoint['model'])
    except Exception as e:
        print(f"  FAILED TO LOAD: {e}")
        continue
    
    B = 16
    T = 512
    val_dataset = "rpn_llm/data/RPNData-1-22_phase_lean_val.txt"
    val_loader = DataLoaderLite(B, T, val_dataset)
    
    run_teacher_forcing_validation_instrumented(model, val_loader, device, step=0)
