import os
import re
import torch
import json
import time
from collections import defaultdict
from utils import RPNTokenizer, DataLoaderLite

VALIDATION_SET_RATIO = 0.03

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

def get_prompt_for_phase(line, phase):
    """
    Returns the string representing the prompt for the specified testing phase.
    """
    if phase == "MATH1":
        parts = line.split("[MATH]")
        if len(parts) > 1:
            return parts[0] + "[MATH]"
    elif phase == "REV2":
        parts = line.split("[REV]")
        if len(parts) > 2:
            return parts[0] + "[REV]" + parts[1] + "[REV]"
    elif phase == "MATH2":
        parts = line.split("[MATH]")
        if len(parts) > 2:
            return parts[0] + "[MATH]" + parts[1] + "[MATH]"
    elif phase == "ANS":
        parts = line.split("[ANS]")
        if len(parts) > 1:
            return parts[0] + "[ANS]"
    
    # Default to REV1 (split at '?')
    parts = line.split("?")
    return parts[0] + "?" if len(parts) > 1 else line

def validate_model(checkpoint_path, test_file_path, output_fail_path, arch=None, num_passes=None, early_stop=None, force_mask=None, test_phase="REV1"):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Override mask if requested
    if force_mask is not None:
        print(f"Overriding use_phase_mask to: {force_mask}")
        config.use_phase_mask = force_mask
    
    # Auto-detect architecture from config to prevent mismatches
    if hasattr(config, 'n_prelude'):
        detected_arch = "rdt"
        from model_rdt import GPT
    else:
        detected_arch = "ut" if getattr(config, 'universal', False) else "rope"
        from model_rope import GPT
        
    print(f"Detected Architecture: {detected_arch.upper()}")
        
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = RPNTokenizer(os.path.join(base_dir, "rpn-tokenizer.json"))
    
    # 2. Parse Test File
    print(f"Parsing test file {test_file_path}...")
    with open(test_file_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
        
    lines = []
    for line in raw_lines:
        clean_line = re.sub(r'\s+', ' ', line.strip()) + '\n'
        lines.append(clean_line)
    del raw_lines
        
    # 3. Group by prompt token length to avoid padding issues
    # A prompt is everything up to and including the '=' sign and the trailing space. 
    length_groups = defaultdict(list)
    eq_id = tokenizer.encode("?")[0]
    eos_id = tokenizer.encode("[EOS]")[0]
    ans_id = tokenizer.encode("[ANS]")[0]
    
    print("Grouping by prompt length...")
    for line in lines:
        if "?" not in line:
            continue
            
        line_clean = line.rstrip()
        prompt_str = get_prompt_for_phase(line_clean, test_phase)
        
        # We must group by the exact number of tokens in the prompt
        prompt_tokens = tokenizer.encode(prompt_str)
        prompt_len = len(prompt_tokens)
        
        length_groups[prompt_len].append((prompt_tokens, line_clean))
        
        
    # free lines memory
    del lines
    
    print(f"Grouped into {len(length_groups)} different prompt lengths.")
    
    # 4. Batched Generation
    max_batch_size = 256 # Dropped surgically to prevent Apple MPS fragmenting the KV Cache memory allocation bounds!
    failures = []
    total_processed = 0
    total_correct = 0
    
    total_by_carry = defaultdict(int)
    correct_by_carry = defaultdict(int)
    
    edge_stats = {
        'zero_operand': {'total': 0, 'correct': 0},
        'negative_result': {'total': 0, 'correct': 0},
        'neg_intermediate': {'total': 0, 'correct': 0},
        'normal': {'total': 0, 'correct': 0}
    }
    # Enhanced Failure Categorization
    failure_categories = {
        'reversal_skipped': 0,
        'reversal_failed': 0,
        'math_failed': 0,
        'only_final_ans_failed': 0
    }
    
    # Stats by problem type (N-num)
    type_stats = defaultdict(lambda: defaultdict(int))
    reversal_fail_by_spaces = defaultdict(int) # Key: (s1_len, s2_len)
    reversal_pos_failures = defaultdict(int) # Dynamically tracks num1, num2, etc.
    reversal_digit_failures = defaultdict(int) # Key: num_digits

    total_items = sum(len(items) for items in length_groups.values())
    start_time = time.time()
    overall_tokens_gen = 0
    total_tokens_count = 0
    total_tokens_correct = 0
    fail_tokens_count = 0
    fail_tokens_correct = 0

    # Setup output paths
    if "_failures.txt" in output_fail_path:
        summary_path = output_fail_path.replace("_failures.txt", "_summary.txt")
    else:
        summary_path = output_fail_path + "_summary.txt"
        
    print(f"Logging failures to: {output_fail_path}")
    print(f"Logging summary to: {summary_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_fail_path), exist_ok=True)

    # Ensure output files are freshly clean before looping!
    with open(output_fail_path, "w", encoding="utf-8") as f:
        f.write(f"--- Real-time Validation Failures - Model: {checkpoint_path} ---\n\n")
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"--- Validation Summary - Model: {checkpoint_path} ---\n\n")

    max_rows = int(VALIDATION_SET_RATIO*total_items/len(length_groups))
    print(f"Beginning batched evaluation on {max_rows} rows per group...")
    accuracy_by_length = {}
    processed_by_length = {}
    group_idx = 1
    for length, items in length_groups.items():
        print(f"\n--- Evaluating group {group_idx}/{len(length_groups)} (prompt length {length}) - {len(items)} items ---")
        group_total_processed = 0
        group_total_correct = 0
        group_idx += 1
        
        # Process in chunks of max_batch_size. only validate a fraction of the data
        for i in range(0, min(max_rows, len(items)), max_batch_size):
            batch_items = items[i : min(i + max_batch_size, max_rows)]
            B = len(batch_items)
            
            batch_prompt_tokens = []
            expected_strs = []
            expected_tokens_list = []
            prompt_strs = []
            
            for prompt_tokens, line_str in batch_items:
                line_tokens = tokenizer.encode(line_str)
                prompt_len = len(prompt_tokens)
                
                expected_tokens_list.append(line_tokens[prompt_len:])
                batch_prompt_tokens.append(prompt_tokens)
                prompt_strs.append(tokenizer.decode(prompt_tokens).strip())
                expected_strs.append(line_str.split('?', 1)[1].strip())
            
            # Dynamically calculate the maximum tokens needed for this specific batch
            # We add a small buffer (+10) to allow the model to fully finish its hallucination and close EOS
            max_batch_expected = max(len(t) for t in expected_tokens_list) if expected_tokens_list else 0
            max_new_tokens = max_batch_expected + 10
            
            # Construct (B, L) tensor natively (no padding required because all logic lengths identically match)
            prompts = torch.tensor(batch_prompt_tokens, dtype=torch.long, device=device)
            
            # Generate max_new_tokens sequentially using Argmax
            idx = prompts
            
            t0 = time.time()
            
            past_kv = None
            for step in range(max_new_tokens):
                # Calculate full_phase_ids for masking
                is_phase_shift = (idx == 10) | (idx == 11) | (idx == 12)
                full_phase_ids = is_phase_shift.cumsum(dim=-1)

                # If KV Cache is fully loaded, ONLY pass the raw isolated new single token forward!
                idx_cond = idx[:, -1:] if past_kv is not None else idx
                
                with torch.no_grad():
                    with torch.autocast(device, dtype=torch.bfloat16):
                        logits, _, past_kv = model(idx_cond, use_cache=True, past_key_values=past_kv, num_passes=num_passes, halt_on_logit_stability=early_stop, full_phase_ids=full_phase_ids)
                
                logits = logits[:, -1, :] # Pluck final step logits
                idx_next = torch.argmax(logits, dim=-1, keepdim=True) # Deterministic greedy decision
                idx = torch.cat((idx, idx_next), dim=1) # Append
                
                # Apple MPS has a structural bug where tight internal `torch.cat` matrices 
                # heavily defer Garbage Collection! Natively sync the caches securely every N steps.
                if device == "mps" and (step + 1) % 32 == 0:
                    torch.mps.empty_cache()
            
            # Final clear after each batch
            if device == "mps":
                torch.mps.empty_cache()

            t1 = time.time()
            batch_time = t1 - t0
            overall_tokens_gen += B * max_new_tokens
            tokens_per_sec = (B * max_new_tokens) / batch_time if batch_time > 0 else 0
            
            total_processed += B
            group_total_processed += B
            pct_complete = (total_processed / total_items) * 100
                            
            # Extract and verify the generated characters
            for b in range(B):
                # Find the separator token index in the prompt (up to the original prompt length)
                # This ensures we align perfectly with the first generated token
                current_prompt_tokens = batch_prompt_tokens[b]
                prompt_len = len(current_prompt_tokens)
                
                full_generated_tokens = idx[b].tolist()
                gen_answer_tokens = full_generated_tokens[prompt_len:]
                
                # Truncate string sequence at [EOS] (to match exactly)
                if eos_id in gen_answer_tokens:
                    eos_idx = gen_answer_tokens.index(eos_id)
                    gen_answer_tokens = gen_answer_tokens[:eos_idx]
                
                actual_str = tokenizer.decode(gen_answer_tokens).strip()
                
                prompt_after_q = prompt_strs[b].split('?', 1)[1] if '?' in prompt_strs[b] else ""
                predicted_str = prompt_after_q + actual_str
                expected_str = expected_strs[b]
                prompt_str = prompt_strs[b]
                total_length = len(prompt_str) + len(expected_str)
                
                clean_prompt = prompt_str.replace("[BOS]", "").replace("?", "").strip()
                prompt_tokens = clean_prompt.split()
                num_operands = len(prompt_tokens)
                p_type = f"{num_operands}-num"
                
                if p_type not in type_stats:
                    type_stats[p_type] = defaultdict(int)
                
                expected_ans_str = expected_str.split("[ANS]")[-1].split("[UNK]")[0].split("[EOS]")[0].strip() if "[ANS]" in expected_str else ""
                predicted_ans_str = predicted_str.split("[ANS]")[-1].split("[UNK]")[0].split("[EOS]")[0].strip() if "[ANS]" in predicted_str else ""
                
                # Use the pre-sliced tokens from the original line to avoid re-encoding shifts
                expected_tokens = expected_tokens_list[b]
                
                # Truncate expected_tokens at [EOS] to match the generation limit
                if eos_id in expected_tokens:
                    expected_tokens = expected_tokens[:expected_tokens.index(eos_id)+1]

                match_count = 0
                for t_idx in range(min(len(gen_answer_tokens), len(expected_tokens))):
                    if gen_answer_tokens[t_idx] == expected_tokens[t_idx]:
                        match_count += 1
                
                total_tokens_count += len(expected_tokens)
                total_tokens_correct += match_count
                
                is_correct = (expected_ans_str == predicted_ans_str)
                if not is_correct:
                    fail_tokens_count += len(expected_tokens)
                    fail_tokens_correct += match_count

                carries = 0
                is_zero = False
                if len(prompt_tokens) >= 2:
                    n1_str = prompt_tokens[0]
                    n2_op = prompt_tokens[1]
                    n2_str = n2_op[:-1]
                    op = n2_op[-1]
                    try:
                        p0_val = int(n1_str); p1_val = int(n2_str)
                        is_zero = (p0_val == 0 or p1_val == 0)
                        carries = calculate_carries(n1_str, n2_str, op)
                    except ValueError: pass
                
                is_neg = expected_ans_str.startswith('-')
                is_neg_intermed = "[REV]-" in expected_str
                is_normal = not is_zero and not is_neg and not is_neg_intermed
                
                total_by_carry[carries] += 1
                if is_zero: edge_stats['zero_operand']['total'] += 1
                if is_neg: edge_stats['negative_result']['total'] += 1
                if is_neg_intermed: edge_stats['neg_intermediate']['total'] += 1
                if is_normal: edge_stats['normal']['total'] += 1

                # Helper to split RPN parts for Phase format
                def split_rpn(s):
                    parts = s.split("[ANS]")
                    if len(parts) >= 2:
                        ans = parts[-1].split("[EOS]")[0].strip()
                        content = parts[0]
                    else:
                        ans = ""
                        content = s
                    
                    # Split by the first [MATH] to separate reversal from math steps
                    math_parts = content.split("[MATH]", 1)
                    if len(math_parts) < 2:
                        return "", "", ans
                    
                    # Everything before first [MATH] contains the prompt and reversal
                    pre_math = math_parts[0]
                    rev_parts = pre_math.split("[REV]")
                    rev = rev_parts[1] if len(rev_parts) > 1 else ""
                    
                    # Everything after first [MATH] is the math trace!
                    math = math_parts[1]
                    
                    return rev, math, ans

                exp_pre, exp_math, exp_ans_final = split_rpn(expected_str)
                pred_pre, pred_math, pred_ans_final = split_rpn(predicted_str)
                
                # Check absolute truth of final numerical output!
                if expected_ans_str != predicted_ans_str:
                    # Categorize failure
                    space_key = (0, 0)

                    is_reversal_skip = "[REV]" not in predicted_str or "[MATH]" not in predicted_str
                    is_reversal_fail = not is_reversal_skip and pred_pre.strip() != exp_pre.strip()
                    is_math_fail = pred_math.strip() != exp_math.strip()
                    
                    matched = False
                    math1_fail = False
                    rev2_fail = False
                    math2_fail = False
                    if is_reversal_skip:
                        failure_categories['reversal_skipped'] += 1
                        matched = True
                    
                    elif is_reversal_fail:
                        failure_categories['reversal_failed'] += 1
                        matched = True
                        reversal_fail_by_spaces[space_key] += 1
                        fail_str = "" # Initialize to avoid UnboundLocalError
                        
                        # Analyze WHICH number failed in the reversal
                        try:
                            exp_parts = exp_pre.strip().rstrip('=').split('[SEP]')
                            pred_parts = pred_pre.strip().rstrip('=').split('[SEP]')
                            
                            fail_count = 0
                            n_fails = []
                            for i in range(max(len(exp_parts), len(pred_parts))):
                                e = exp_parts[i] if i < len(exp_parts) else None
                                p = pred_parts[i] if i < len(pred_parts) else None
                                if e != p:
                                    n_fails.append(i)
                                    fail_count += 1
                                    if e:
                                        # e looks like "51" or "51+"
                                        num_digits = len([c for c in e if c.isdigit()])
                                        reversal_digit_failures[num_digits] += 1
                            
                            if fail_count > 1:
                                reversal_pos_failures['multiple'] += 1
                                fail_str = "MULTIPLE"
                            elif len(n_fails) == 1:
                                pos = f"num{n_fails[0] + 1}"
                                reversal_pos_failures[pos] += 1
                                fail_str = str(n_fails[0] + 1)
                            else:
                                reversal_pos_failures['malformed'] += 1
                                fail_str = "M"
                        except Exception as e: 
                            reversal_pos_failures['malformed'] += 1
                    
                    elif is_math_fail:
                        failure_categories['math_failed'] += 1
                        matched = True
                        
                        exp_m_parts = exp_math.split("[MATH]")
                        pred_m_parts = pred_math.split("[MATH]")
                        
                        fail_idx = -1
                        for i in range(max(len(exp_m_parts), len(pred_m_parts))):
                            e = exp_m_parts[i].strip() if i < len(exp_m_parts) else None
                            p = pred_m_parts[i].strip() if i < len(pred_m_parts) else None
                            if e != p:
                                fail_idx = i
                                break
                                
                        math_fail_step = -1
                        rev_fail_step = -1

                        if fail_idx != -1:
                            e = exp_m_parts[fail_idx].strip() if fail_idx < len(exp_m_parts) else ""
                            p = pred_m_parts[fail_idx].strip() if fail_idx < len(pred_m_parts) else ""
                            
                            if "[REV]" in e:
                                e_sub = e.split("[REV]")
                                p_sub = p.split("[REV]") if p else []
                                
                                e_m = e_sub[0].strip()
                                p_m = p_sub[0].strip() if len(p_sub) > 0 else ""
                                
                                if e_m != p_m:
                                    math_fail_step = fail_idx + 1
                                else:
                                    rev_fail_step = fail_idx + 2
                            else:
                                math_fail_step = fail_idx + 1
                                
                        if math_fail_step != -1:
                            type_stats[p_type][f'math{math_fail_step}_failed'] += 1
                        elif rev_fail_step != -1:
                            type_stats[p_type][f'rev{rev_fail_step}_failed'] += 1
                    
                    elif not matched:
                        failure_categories['only_final_ans_failed'] += 1

                    # Update type stats
                    type_stats[p_type]['total'] += 1
                    if is_reversal_skip or is_reversal_fail:
                        type_stats[p_type]['reversal_failed'] += 1
                    elif is_math_fail:
                        type_stats[p_type]['math_failed'] += 1
                    elif not matched:
                        type_stats[p_type]['only_final_ans_failed'] += 1

                    # Document failure
                    reason = "REV_SKIP" if is_reversal_skip else f"REV_FAIL_{fail_str}" if is_reversal_fail else f"MATH{math_fail_step}_FAIL" if (is_math_fail and math_fail_step != -1) else f"REV{rev_fail_step}_FAIL" if (is_math_fail and rev_fail_step != -1) else "MATH_FAIL" if is_math_fail else "ANS_FAIL"
                    fail_str = f"Q_Len:{total_length} | [{reason}]:{prompt_str} | Exp Rev:{exp_pre} | Pred Rev:{pred_pre} | Exp Math:{exp_math} | Pred Math:{pred_math} | Exp Ans:{exp_ans_final} | Pred Ans:{pred_ans_final}"
                    failures.append(fail_str)
                    # Stream directly to disk live!
                    with open(output_fail_path, "a", encoding="utf-8") as f:
                        f.write(fail_str + "\n")
                else:
                    total_correct += 1
                    group_total_correct += 1
                    correct_by_carry[carries] += 1
                    if is_zero: edge_stats['zero_operand']['correct'] += 1
                    if is_neg: edge_stats['negative_result']['correct'] += 1
                    if is_neg_intermed: edge_stats['neg_intermediate']['correct'] += 1
                    if is_normal: edge_stats['normal']['correct'] += 1

                    # Update type stats
                    type_stats[p_type]['total'] += 1
                    type_stats[p_type]['correct'] += 1

            # print accuracy
            accuracy = (total_correct / total_processed) * 100
            group_accuracy = (group_total_correct / group_total_processed) * 100 if group_total_processed > 0 else 0
            accuracy_by_length[length] = group_accuracy
            processed_by_length[length] = group_total_processed
            
            running_token_acc = (total_tokens_correct / total_tokens_count) * 100 if total_tokens_count > 0 else 0
            fail_token_acc = (fail_tokens_correct / fail_tokens_count) * 100 if fail_tokens_count > 0 else 0
            
            print(f"Progress: {total_processed}/{total_items} ({pct_complete:.2f}%) | Acc: {accuracy:.2f}% | Token Acc: {running_token_acc:.2f}% | Fail Token Acc: {fail_token_acc:.2f}% | Group {group_idx-1} Acc: {group_accuracy:.2f}% | Tokens/sec: {tokens_per_sec:.1f}")
            for cat, count in failure_categories.items():
                print(f"{cat:<20}: {count}")

    # 5. Output Results
    accuracy = (total_correct / total_processed) * 100
    final_token_acc = (total_tokens_correct / total_tokens_count) * 100 if total_tokens_count > 0 else 0
    final_fail_token_acc = (fail_tokens_correct / fail_tokens_count) * 100 if fail_tokens_count > 0 else 0

    print(f"\n=====================")
    print(f"Validation Complete!")
    print(f"Total Evaluated: {total_processed}")
    print(f"Total Correct: {total_correct}")
    print(f"Total Failures: {len(failures)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Global Token Accuracy: {final_token_acc:.2f}%")
    print(f"Failure Token Accuracy: {final_fail_token_acc:.2f}%")
    
    # Write to Summary File
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"\nValidation Accuracy: {accuracy:.2f}% ({total_correct}/{total_processed})\n")
        f.write("=========================================\n\n")

        f.write("\n--- Breakdown by Prompt Length ---\n")
        f.write("Token Length | Processed Items | Accuracy\n")
        for g in sorted(accuracy_by_length.keys()):
            stats = f"{g:2d} | {processed_by_length[g]:<15} | {accuracy_by_length[g]:.2f}%"
            f.write(stats + "\n")
            
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
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(stats + "\n")
            
    with open(summary_path, "a", encoding="utf-8") as f:
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

    # Detailed Failure Analysis Output
    print("\n--- Failure Category Breakdown ---")
    total_fails = len(failures)
    for cat, count in failure_categories.items():
        pct = (count / total_processed * 100) if total_processed > 0 else 0
        print(f"{cat:<20}: {count} ({pct:.1f}%)")

        print("\n--- Reversal Failure Analysis (Position & Length) ---")
        total_pos_fails = sum(reversal_pos_failures.values())
        for pos, count in reversal_pos_failures.items():
            pct = (count / total_pos_fails * 100) if total_pos_fails > 0 else 0
            print(f"Failed on {pos:<8}: {count} ({pct:.1f}%)")
            
        print("\nReversal Failures by Digit Length:")
        total_digit_fails = sum(reversal_digit_failures.values())
        for length in sorted(reversal_digit_failures.keys()):
            count = reversal_digit_failures[length]
            pct = (count / total_digit_fails * 100) if total_digit_fails > 0 else 0
            print(f"{length:2d} digits: {count} failures ({pct:.1f}%)")

    print("\n--- Problem Type Breakdown ---")
    for p_type, stats in type_stats.items():
        tot = stats['total']
        cor = stats['correct']
        acc = (cor / tot) * 100 if tot > 0 else 0
        print(f"{p_type}: Total: {tot} | Correct: {cor} | Accuracy: {acc:.2f}%")
        detail_items = []
        for k, v in sorted(stats.items()):
            if k not in ['total', 'correct', 'reversal_failed', 'math_failed', 'only_final_ans_failed']:
                detail_items.append(f"{k.replace('_failed', '')}: {v}")
        detail_str = f" ({', '.join(detail_items)})" if detail_items else ""
        print(f"  Failures -> Rev: {stats.get('reversal_failed', 0)}, Math: {stats.get('math_failed', 0)}{detail_str}, Ans: {stats.get('only_final_ans_failed', 0)}")

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n--- Failure Category Breakdown ---\n")
        for cat, count in failure_categories.items():
            pct = (count / total_processed * 100) if total_processed > 0 else 0
            f.write(f"{cat:<20}: {count} ({pct:.1f}%)\n")
        
        f.write("\n--- Problem Type Breakdown ---\n")
        for p_type, stats in type_stats.items():
            tot = stats['total']
            cor = stats['correct']
            acc = (cor / tot) * 100 if tot > 0 else 0
            f.write(f"{p_type}: Total: {tot} | Correct: {cor} | Accuracy: {acc:.2f}%\n")
            detail_items = []
            for k, v in sorted(stats.items()):
                if k not in ['total', 'correct', 'reversal_failed', 'math_failed', 'only_final_ans_failed']:
                    detail_items.append(f"{k.replace('_failed', '')}: {v}")
            detail_str = f" ({', '.join(detail_items)})" if detail_items else ""
            f.write(f"  Failures -> Rev: {stats.get('reversal_failed', 0)}, Math: {stats.get('math_failed', 0)}{detail_str}, Ans: {stats.get('only_final_ans_failed', 0)}\n")

        f.write("\n--- Reversal Failure Analysis (Position & Length) ---\n")
        total_pos_fails = sum(reversal_pos_failures.values())
        for pos, count in reversal_pos_failures.items():
            pct = (count / total_pos_fails * 100) if total_pos_fails > 0 else 0
            f.write(f"Failed on {pos:<8}: {count} ({pct:.1f}%)\n")
            
        f.write("\nReversal Failures by Digit Length:\n")
        total_digit_fails = sum(reversal_digit_failures.values())
        for length in sorted(reversal_digit_failures.keys()):
            count = reversal_digit_failures[length]
            pct = (count / total_digit_fails * 100) if total_digit_fails > 0 else 0
            f.write(f"{length:2d} digits: {count} failures ({pct:.1f}%)\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate RPN Transformer")
    parser.add_argument("--model", type=str, default="rpn3_llm/models/checkpoint.pt", help="Path to checkpoint")
    parser.add_argument("--test_file", type=str, default="rpn3_llm/data/rpn3_val.txt", help="Path to test file")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output failure file (auto-generated if not provided)")
    parser.add_argument("--arch", type=str, default="rope", choices=["rope", "ut", "rdt"], help="Architecture of the checkpoint (rope, ut, rdt)")
    parser.add_argument("--num_passes", type=int, default=None, help="Force number of universal passes")
    parser.add_argument("--early_stop", type=int, default=None, help="Stop if logit is stable for N passes")
    parser.add_argument("--force_mask", type=str, default=None, choices=["True", "False"], help="Force enable/disable phase mask")
    parser.add_argument("--test_phase", type=str, default="REV1", choices=["REV1", "MATH1", "REV2", "MATH2", "ANS"], help="Phase to begin generation from")
    
    args = parser.parse_args()
    
    fmask = None
    if args.force_mask == "True": fmask = True
    elif args.force_mask == "False": fmask = False

    if args.output_file is None:
        model_basename = os.path.basename(args.model)
        model_name = model_basename.replace(".pt", "")
        fmask_str = f"_fmask_{fmask}" if fmask is not None else ""
        phase_str = f"_phase_{args.test_phase}" if args.test_phase != "REV1" else ""
        args.output_file = f"rpn3_llm/results/{model_name}{fmask_str}{phase_str}_failures.txt"

    validate_model(args.model, args.test_file, args.output_file, arch=args.arch, num_passes=args.num_passes, early_stop=args.early_stop, force_mask=fmask, test_phase=args.test_phase)
