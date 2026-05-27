import os
import re
import time
import torch
import json
from collections import defaultdict
from utils import RPNTokenizer

def validate_reversal(checkpoint_path, test_file_path, output_fail_path, ratio=0.03, max_batch_size=256):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Model Checkpoint
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Determine model architecture
    if getattr(config, 'universal', False):
        detected_arch = "ut"
    else:
        detected_arch = "rope"
    print(f"Detected Architecture: {detected_arch.upper()}")
        
    from model_rope import GPT
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = RPNTokenizer(os.path.join(base_dir, "rpn-tokenizer.json"))
    
    rev_id = tokenizer.encode("[REV]")[0]
    math_id = tokenizer.encode("[MATH]")[0]
    ans_id = tokenizer.encode("[ANS]")[0]
    eos_id = tokenizer.encode("[EOS]")[0]
    nl_id = tokenizer.encode("\n")[0]

    # 2. Parse and Group Test File
    print(f"Parsing test file {test_file_path}...")
    with open(test_file_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
        
    # Group by prompt length to enable batching without padding
    length_groups = defaultdict(list)
    total_lines = 0
    
    for line in raw_lines:
        line_clean = line.strip()
        if not line_clean or "?" not in line_clean:
            continue
            
        parts = line_clean.split("?", 1)
        prompt_str = parts[0] + "?"
        expected_str = parts[1].strip()
        
        prompt_tokens = tokenizer.encode(prompt_str)
        prompt_len = len(prompt_tokens)
        
        length_groups[prompt_len].append((prompt_tokens, expected_str, line_clean))
        total_lines += 1
        
    print(f"Grouped {total_lines} lines into {len(length_groups)} prompt token lengths.")

    # 3. Setup outputs
    os.makedirs(os.path.dirname(output_fail_path), exist_ok=True)
    with open(output_fail_path, "w", encoding="utf-8") as f:
        f.write(f"--- Reversal Validation Failures - Model: {checkpoint_path} ---\n\n")

    summary_path = output_fail_path.replace("_failures.txt", "_summary.txt") if "_failures.txt" in output_fail_path else output_fail_path + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"--- Reversal Validation Summary - Model: {checkpoint_path} ---\n\n")

    # Evaluate a sample ratio of the dataset (e.g. 3%)
    max_rows_per_group = max(1, int(ratio * total_lines / len(length_groups)))
    print(f"Evaluating up to {max_rows_per_group} lines per group (ratio={ratio:.2%})...")

    total_processed = 0
    total_correct = 0
    total_tokens_count = 0
    total_tokens_correct = 0
    
    # Reversal-specific diagnostics
    operand_failures = defaultdict(int)
    digit_failures = defaultdict(int)
    
    start_time = time.time()
    
    group_idx = 1
    for length, items in sorted(length_groups.items()):
        items_to_eval = items[:min(len(items), max_rows_per_group)]
        if not items_to_eval:
            continue
            
        print(f"\n--- Group {group_idx}/{len(length_groups)} (prompt length {length}) - Eval {len(items_to_eval)} items ---")
        group_idx += 1
        
        for idx_start in range(0, len(items_to_eval), max_batch_size):
            batch_items = items_to_eval[idx_start : idx_start + max_batch_size]
            B = len(batch_items)
            
            batch_prompts = []
            expected_strs = []
            expected_tokens_list = []
            
            for prompt_tokens, expected_str, _ in batch_items:
                batch_prompts.append(prompt_tokens)
                expected_strs.append(expected_str)
                # expected tokens to match against
                expected_tokens_list.append(tokenizer.encode(expected_str))
                
            max_expected_len = max(len(t) for t in expected_tokens_list) if expected_tokens_list else 0
            max_new_tokens = max_expected_len + 5
            
            prompts_tensor = torch.tensor(batch_prompts, dtype=torch.long, device=device)
            idx = prompts_tensor
            past_kv = None
            
            # Autoregressive generation
            for step in range(max_new_tokens):
                # Calculate full_phase_ids for UT phase masking
                is_phase_shift = (idx == rev_id) | (idx == math_id) | (idx == ans_id)
                full_phase_ids = is_phase_shift.cumsum(dim=-1)

                idx_cond = idx[:, -1:] if past_kv is not None else idx
                
                with torch.no_grad():
                    with torch.autocast(device, dtype=torch.bfloat16):
                        logits, _, past_kv = model(idx_cond, use_cache=True, past_key_values=past_kv, full_phase_ids=full_phase_ids)
                
                logits = logits[:, -1, :]
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, idx_next), dim=1)
                
                if device == "mps" and (step + 1) % 32 == 0:
                    torch.mps.empty_cache()

            if device == "mps":
                torch.mps.empty_cache()

            # Process generated outputs
            for b in range(B):
                prompt_len = len(batch_prompts[b])
                gen_tokens = idx[b, prompt_len:].tolist()
                
                # Truncate at newline or EOS
                for stop_val in [nl_id, eos_id]:
                    if stop_val in gen_tokens:
                        gen_tokens = gen_tokens[:gen_tokens.index(stop_val)]
                
                pred_str = tokenizer.decode(gen_tokens).strip()
                exp_str = expected_strs[b]
                
                # Token-level stats
                exp_toks = expected_tokens_list[b]
                # Strip NL/EOS from expected tokens for matching
                for stop_val in [nl_id, eos_id]:
                    if stop_val in exp_toks:
                        exp_toks = exp_toks[:exp_toks.index(stop_val)]
                        
                match_count = sum(1 for t1, t2 in zip(gen_tokens, exp_toks) if t1 == t2)
                total_tokens_count += len(exp_toks)
                total_tokens_correct += match_count
                
                is_correct = (pred_str == exp_str)
                total_processed += 1
                
                if is_correct:
                    total_correct += 1
                else:
                    # Diagnose which operand failed in the reversal
                    exp_operands = re.split(r'\[SEP\]', exp_str.replace("[REV]", ""))
                    pred_operands = re.split(r'\[SEP\]', pred_str.replace("[REV]", ""))
                    
                    failed_op_indices = []
                    for i in range(max(len(exp_operands), len(pred_operands))):
                        e_op = exp_operands[i].strip() if i < len(exp_operands) else None
                        p_op = pred_operands[i].strip() if i < len(pred_operands) else None
                        if e_op != p_op:
                            failed_op_indices.append(i)
                            operand_failures[f"operand_{i+1}"] += 1
                            if e_op:
                                num_digits = len([c for c in e_op if c.isdigit()])
                                digit_failures[num_digits] += 1
                                
                    op_fail_str = f"op_idx:{failed_op_indices}" if failed_op_indices else "length_mismatch"
                    
                    fail_log = (
                        f"FAIL | Prompt: {tokenizer.decode(batch_prompts[b]).strip()}\n"
                        f"     | Exp:    {exp_str}\n"
                        f"     | Pred:   {pred_str}\n"
                        f"     | Diagnose: {op_fail_str}\n"
                    )
                    with open(output_fail_path, "a", encoding="utf-8") as f:
                        f.write(fail_log + "\n")
                        
            # Print intermediate stats
            acc = (total_correct / total_processed) * 100
            tok_acc = (total_tokens_correct / total_tokens_count) * 100 if total_tokens_count > 0 else 0
            print(f"Batch Progress: {total_processed} items | Running Acc: {acc:.2f}% | Running Token Acc: {tok_acc:.2f}%")

    # 4. Final Diagnostics & Summaries
    elapsed = time.time() - start_time
    final_acc = (total_correct / total_processed) * 100 if total_processed > 0 else 0
    final_tok_acc = (total_tokens_correct / total_tokens_count) * 100 if total_tokens_count > 0 else 0
    
    summary_txt = (
        f"Validation Complete in {elapsed:.1f}s\n"
        f"Total Evaluated:  {total_processed}\n"
        f"Sequence Accuracy: {final_acc:.2f}% ({total_correct}/{total_processed})\n"
        f"Token Accuracy:    {final_tok_acc:.2f}% ({total_tokens_correct}/{total_tokens_count})\n\n"
    )
    
    if operand_failures:
        summary_txt += "--- Failure Distribution by Operand Index ---\n"
        total_op_fails = sum(operand_failures.values())
        for op, count in sorted(operand_failures.items()):
            pct = (count / total_op_fails) * 100
            summary_txt += f"  {op:<12}: {count:4d} failures ({pct:.1f}%)\n"
            
    if digit_failures:
        summary_txt += "\n--- Failure Distribution by Operand Digit Length ---\n"
        total_dig_fails = sum(digit_failures.values())
        for length, count in sorted(digit_failures.items()):
            pct = (count / total_dig_fails) * 100
            summary_txt += f"  {length:2d} digits  : {count:4d} failures ({pct:.1f}%)\n"
            
    print("\n==================================")
    print(summary_txt)
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(summary_txt)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate Reversal Only Model")
    parser.add_argument("--model", type=str, default="rpn3_llm/models/checkpoint.pt", help="Path to checkpoint")
    parser.add_argument("--test_file", type=str, default="rpn3_llm/data/sft_1-14_7num_BOS_pre_math_val.txt", help="Path to validation text file")
    parser.add_argument("--output_file", type=str, default="rpn3_llm/results/reversal_val_failures.txt", help="Path to output failure file")
    parser.add_argument("--ratio", type=float, default=0.03, help="Fraction of the dataset to validate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for parallel generation")
    
    args = parser.parse_args()
    validate_reversal(args.model, args.test_file, args.output_file, args.ratio, args.batch_size)
