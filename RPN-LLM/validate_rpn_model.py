import os
import torch
import json
import time
from collections import defaultdict
from model_rope import GPT, GPTConfig
from train_rpn import RPNTokenizer, DataLoaderLite

def calculate_carries(a_str, b_str, op):
    a, b = int(a_str), int(b_str)
    if op == '-' and a < b:
        a_str, b_str = b_str, str(a).zfill(len(b_str))
        
    a_digits = [int(x) for x in reversed(a_str)]
    b_digits = [int(x) for x in reversed(b_str)]
    
    max_len = max(len(a_digits), len(b_digits))
    a_digits += [0] * (max_len - len(a_digits))
    b_digits += [0] * (max_len - len(b_digits))
    
    carries = 0
    carry_val = 0
    if op == '+':
        for i in range(max_len):
            if a_digits[i] + b_digits[i] + carry_val > 9:
                carries += 1
                carry_val = 1
            else:
                carry_val = 0
    elif op == '-':
        for i in range(max_len):
            if a_digits[i] - b_digits[i] - carry_val < 0:
                carries += 1
                carry_val = 1
            else:
                carry_val = 0
    return carries

def validate_model(checkpoint_path, test_file_path, output_fail_path):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    tokenizer = RPNTokenizer("rpn-tokenizer.json")
    
    # 2. Parse Test File
    print(f"Parsing test file {test_file_path}...")
    with open(test_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # 3. Group by prompt token length to avoid padding issues
    # A prompt is everything up to and including the '=' sign and the trailing space. 
    length_groups = defaultdict(list)
    eq_id = tokenizer.encode("=")[0]
    nl_id = tokenizer.encode("\n")[0]
    
    print("Grouping by prompt length...")
    for line in lines:
        if "=" not in line:
            continue
        
        # Tokenize the entire line at once exactly as training does
        line_tokens = tokenizer.encode(line)
        
        # Find index of '='
        try:
            eq_idx = line_tokens.index(eq_id)
        except ValueError:
            continue
            
        prompt_tokens = line_tokens[:eq_idx + 1]
        expected_ans_tokens = line_tokens[eq_idx + 1:]
        
        length_groups[len(prompt_tokens)].append({
            'prompt_tokens': prompt_tokens,
            'expected_tokens': expected_ans_tokens,
            'full_str': line.strip()
        })
        
    print(f"Grouped into {len(length_groups)} different prompt lengths.")
    
    # 4. Batched Generation
    max_batch_size = 32 # Dropped surgically to prevent Apple MPS fragmenting the KV Cache memory allocation bounds!
    failures = []
    total_processed = 0
    total_correct = 0
    
    total_by_carry = defaultdict(int)
    correct_by_carry = defaultdict(int)
    
    edge_stats = {
        'zero_operand': {'total': 0, 'correct': 0},
        'negative_result': {'total': 0, 'correct': 0},
        'normal': {'total': 0, 'correct': 0}
    }
    
    # Max generation steps for an answer
    # A padded 5-digit math evaluates to ~112 tokens dynamically, pushing past the bounds of 96 natively!
    max_new_tokens = 128

    total_items = sum(len(items) for items in length_groups.values())
    start_time = time.time()
    overall_tokens_gen = 0

    # Ensure output file is freshly clean before append looping!
    with open(output_fail_path, "w", encoding="utf-8") as f:
        f.write("--- Real-time Validation Failures ---\n\n")

    print("Beginning batched evaluation...")
    group_idx = 1
    for length, items in length_groups.items():
        print(f"\n--- Evaluating group {group_idx}/{len(length_groups)} (prompt length {length}) - {len(items)} items ---")
        group_idx += 1
        
        # Process in chunks of max_batch_size
        for i in range(0, len(items), max_batch_size):
            batch_items = items[i:i+max_batch_size]
            B = len(batch_items)
            
            # Construct (B, L) tensor natively (no padding required because all logic lengths identically match)
            prompts = torch.tensor([item['prompt_tokens'] for item in batch_items], dtype=torch.long, device=device)
            
            # Generate max_new_tokens sequentially using Argmax
            idx = prompts
            
            t0 = time.time()
            
            past_kv = None
            for step in range(max_new_tokens):
                # If KV Cache is fully loaded, ONLY pass the raw isolated new single token forward!
                idx_cond = idx[:, -1:] if past_kv is not None else idx
                
                with torch.no_grad():
                    with torch.autocast(device, dtype=torch.bfloat16):
                        logits, _, past_kv = model(idx_cond, use_cache=True, past_key_values=past_kv)
                
                logits = logits[:, -1, :] # Pluck final step logits
                idx_next = torch.argmax(logits, dim=-1, keepdim=True) # Deterministic greedy decision
                idx = torch.cat((idx, idx_next), dim=1) # Append
                
                # Apple MPS has a structural bug where tight internal `torch.cat` matrices 
                # heavily defer Garbage Collection! Natively sync the caches securely!
                if device == "mps":
                    torch.mps.empty_cache()
            
            t1 = time.time()
            batch_time = t1 - t0
            overall_tokens_gen += B * max_new_tokens
            tokens_per_sec = (B * max_new_tokens) / batch_time if batch_time > 0 else 0
            
            total_processed += B
            pct_complete = (total_processed / total_items) * 100
                            
            # Extract and verify the generated characters
            for b in range(B):
                full_generated_tokens = idx[b].tolist()
                gen_answer_tokens = full_generated_tokens[length:]
                
                # Truncate string sequence heavily at newline (to match exactly)
                if nl_id in gen_answer_tokens:
                    nl_idx = gen_answer_tokens.index(nl_id)
                    gen_answer_tokens = gen_answer_tokens[:nl_idx+1]
                
                # Compare
                expected = batch_items[b]['expected_tokens']
                expected_str = tokenizer.decode(expected).strip()
                predicted_str = tokenizer.decode(gen_answer_tokens).strip() 
                prompt_str = tokenizer.decode(batch_items[b]['prompt_tokens']).strip()
                
                expected_ans_str = expected_str.split('>')[-1].strip() if '>' in expected_str else expected_str
                predicted_ans_str = predicted_str.split('>')[-1].strip() if '>' in predicted_str else predicted_str
                
                # Intercept trailing artifacts generated forcefully beyond equation bounds!
                predicted_ans_str = predicted_ans_str.split('[UNK]')[0].split('\n')[0].strip()
                expected_ans_str = expected_ans_str.split('[UNK]')[0].split('\n')[0].strip()
                
                def flip_ans(s):
                    s = s.replace(" ", "")
                    if s.startswith("-"):
                        return "-" + s[1:][::-1]
                    return s[::-1]
                
                expected_ans_flipped = flip_ans(expected_ans_str)
                predicted_ans_flipped = flip_ans(predicted_ans_str)
                
                parts = prompt_str.split(' ')
                carries = 0
                is_zero = False
                if len(parts) >= 3:
                    is_zero = (int(parts[0]) == 0 or int(parts[1]) == 0)
                    carries = calculate_carries(parts[0], parts[1], parts[2])
                    
                is_neg = expected_ans_flipped.startswith('-')
                is_normal = not is_zero and not is_neg
                
                total_by_carry[carries] += 1
                if is_zero: edge_stats['zero_operand']['total'] += 1
                if is_neg: edge_stats['negative_result']['total'] += 1
                if is_normal: edge_stats['normal']['total'] += 1
                
                # Check absolute truth of final numerical output!
                if expected_ans_str != predicted_ans_str:
                    # Document failure
                    fail_str = f"Q: {prompt_str} | Expected: {expected_ans_flipped} | Predicted: {predicted_ans_flipped} | Full Pred: {predicted_str}"
                    failures.append(fail_str)
                    # Stream directly to disk live!
                    with open(output_fail_path, "a", encoding="utf-8") as f:
                        f.write(fail_str + "\n")
                else:
                    total_correct += 1
                    correct_by_carry[carries] += 1
                    if is_zero: edge_stats['zero_operand']['correct'] += 1
                    if is_neg: edge_stats['negative_result']['correct'] += 1
                    if is_normal: edge_stats['normal']['correct'] += 1

            # print accuracy
            accuracy = (total_correct / total_processed) * 100
            print(f"Progress: {total_processed}/{total_items} ({pct_complete:.2f}%) | Accuracy: {accuracy:.2f}% | Batch inference time: {batch_time:.2f}s | Tokens/sec: {tokens_per_sec:.1f}")

    # 5. Output Results
    accuracy = (total_correct / total_processed) * 100
    print(f"\n=====================")
    print(f"Validation Complete!")
    print(f"Total Evaluated: {total_processed}")
    print(f"Total Correct: {total_correct}")
    print(f"Total Failures: {len(failures)}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    with open(output_fail_path, "a", encoding="utf-8") as f:
        f.write(f"\nValidation Accuracy: {accuracy:.2f}% ({total_correct}/{total_processed})\n")
        f.write("=========================================\n\n")
            
        f.write("\n--- Breakdown by Carry Operations ---\n")
        f.write("Carries | Total   | Correct | Failures | Accuracy\n")
        
    print("\n--- Breakdown by Carry Operations ---")
    print("Carries | Total   | Correct | Failures | Accuracy")
    for c in sorted(total_by_carry.keys()):
        tot = total_by_carry[c]
        cor = correct_by_carry[c]   
        fls = tot - cor
        acc = (cor / tot) * 100 if tot > 0 else 0
        stats = f"{c:<8}| {tot:<8}| {cor:<8}| {fls:<8}| {acc:.2f}%"
        print(stats)
        with open(output_fail_path, "a", encoding="utf-8") as f:
            f.write(stats + "\n")
            
    with open(output_fail_path, "a", encoding="utf-8") as f:
        f.write("\n--- Edge Case Analysis ---\n")
        f.write(f"{'Category':<16} | {'Total':<8} | {'Correct':<8} | Accuracy\n")
        for cat, stats in edge_stats.items():
            tot = stats['total']
            cor = stats['correct']
            acc = (cor / tot) * 100 if tot > 0 else 0
            f.write(f"{cat:<16} | {tot:<8} | {cor:<8} | {acc:.2f}%\n")
            
    print("\n--- Edge Case Analysis ---")
    print(f"{'Category':<16} | {'Total':<8} | {'Correct':<8} | Accuracy")
    for cat, stats in edge_stats.items():
        tot = stats['total']
        cor = stats['correct']
        acc = (cor / tot) * 100 if tot > 0 else 0
        print(f"{cat:<16} | {tot:<8} | {cor:<8} | {acc:.2f}%")
        
    print(f"\nFailures dumped to {output_fail_path}")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "rope25M_reversed_checkpoint_9999.pt"
    validate_model(model_path, "data/RPNData-plusminus99999_fully_reversed_nopad-_test.txt", "rope_validation_failures_reversed.txt")
