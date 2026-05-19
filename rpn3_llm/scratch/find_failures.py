import os
import re
import sys
import torch
import time
from collections import defaultdict

# Ensure working directory and rpn3_llm are in python path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "rpn3_llm"))

from utils import RPNTokenizer
from model_rope import GPT

def calculate_carries(a_str, b_str, op):
    try:
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
    except Exception:
        return 0

def find_failures(checkpoint_path, test_file_path, output_fail_path, num_failures_needed=5):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Auto-detect architecture
    config.universal = True # 1.8M checkpoint is UT
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    tokenizer = RPNTokenizer("rpn3_llm/rpn-tokenizer.json")
    
    # 2. Parse Test File
    print(f"Parsing test file {test_file_path}...")
    with open(test_file_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
        
    lines = []
    for line in raw_lines:
        clean_line = re.sub(r'\s+', ' ', line.strip()) + '\n'
        lines.append(clean_line)
    del raw_lines
        
    # Group by prompt token length to avoid padding issues
    length_groups = defaultdict(list)
    eos_id = tokenizer.encode("[EOS]")[0]
    
    print("Grouping by prompt length...")
    for line in lines:
        if "?" not in line:
            continue
            
        line_clean = line.rstrip()
        parts = line_clean.split("?")
        prompt_str = parts[0] + "?"
        
        prompt_tokens = tokenizer.encode(prompt_str)
        prompt_len = len(prompt_tokens)
        
        length_groups[prompt_len].append((prompt_tokens, line_clean))
        
    del lines
    
    print(f"Grouped into {len(length_groups)} different prompt lengths.")
    
    failures = []
    total_processed = 0
    total_correct = 0
    
    os.makedirs(os.path.dirname(output_fail_path), exist_ok=True)
    with open(output_fail_path, "w", encoding="utf-8") as f:
        f.write(f"--- failure Case Analysis - Model: {checkpoint_path} ---\n\n")

    # Sort length groups: we want to process shorter sequences first to get fast validation, then longer ones
    sorted_lengths = sorted(length_groups.keys())
    
    for length in sorted_lengths:
        items = length_groups[length]
        print(f"\n--- Evaluating prompt length group {length} ({len(items)} items) ---")
        
        # Dynamically scale batch size based on prompt length to prevent Apple MPS memory thrashing
        if length > 500:
            max_batch_size = 16
        elif length > 250:
            max_batch_size = 32
        elif length > 100:
            max_batch_size = 64
        else:
            max_batch_size = 128
            
        for i in range(0, len(items), max_batch_size):
            batch_items = items[i : i + max_batch_size]
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
            
            max_batch_expected = max(len(t) for t in expected_tokens_list) if expected_tokens_list else 0
            max_new_tokens = max_batch_expected + 10
            
            prompts = torch.tensor(batch_prompt_tokens, dtype=torch.long, device=device)
            idx = prompts
            
            t0 = time.time()
            past_kv = None
            for step in range(max_new_tokens):
                is_phase_shift = (idx == 10) | (idx == 11) | (idx == 12)
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

            t1 = time.time()
            batch_time = t1 - t0
            total_processed += B
            
            for b in range(B):
                current_prompt_tokens = batch_prompt_tokens[b]
                prompt_len = len(current_prompt_tokens)
                
                full_generated_tokens = idx[b].tolist()
                gen_answer_tokens = full_generated_tokens[prompt_len:]
                
                if eos_id in gen_answer_tokens:
                    eos_idx = gen_answer_tokens.index(eos_id)
                    gen_answer_tokens = gen_answer_tokens[:eos_idx]
                
                actual_str = tokenizer.decode(gen_answer_tokens).strip()
                
                prompt_after_q = prompt_strs[b].split('?', 1)[1] if '?' in prompt_strs[b] else ""
                predicted_str = prompt_after_q + actual_str
                expected_str = expected_strs[b]
                prompt_str = prompt_strs[b]
                total_length = len(prompt_str) + len(expected_str)
                
                expected_ans_str = expected_str.split("[ANS]")[-1].split("[UNK]")[0].split("[EOS]")[0].strip() if "[ANS]" in expected_str else ""
                predicted_ans_str = predicted_str.split("[ANS]")[-1].split("[UNK]")[0].split("[EOS]")[0].strip() if "[ANS]" in predicted_str else ""
                
                is_correct = (expected_ans_str == predicted_ans_str)
                if not is_correct:
                    def split_rpn(s):
                        parts = s.split("[ANS]")
                        ans = parts[-1].split("[EOS]")[0].strip() if len(parts) >= 2 else ""
                        content = parts[0]
                        math_parts = content.split("[MATH]", 1)
                        if len(math_parts) < 2:
                            return "", "", ans
                        pre_math = math_parts[0]
                        rev_parts = pre_math.split("[REV]")
                        rev = rev_parts[1] if len(rev_parts) > 1 else ""
                        math = math_parts[1]
                        return rev, math, ans

                    exp_pre, exp_math, exp_ans_final = split_rpn(expected_str)
                    pred_pre, pred_math, pred_ans_final = split_rpn(predicted_str)
                    
                    fail_str = f"Failure #{len(failures)+1} | Q_Len:{total_length} | Prompt:{prompt_str} | Exp Rev:{exp_pre} | Pred Rev:{pred_pre} | Exp Math:{exp_math} | Pred Math:{pred_math} | Exp Ans:{exp_ans_final} | Pred Ans:{pred_ans_final}"
                    print(f"\n>>> FOUND FAILURE:\n{fail_str}\n")
                    failures.append(fail_str)
                    
                    with open(output_fail_path, "a", encoding="utf-8") as f:
                        f.write(fail_str + "\n")
                        
                    if len(failures) >= num_failures_needed:
                        print(f"\nCollected {num_failures_needed} failures. Stopping early!")
                        return
                else:
                    total_correct += 1
            
            print(f"Processed: {total_processed} | Correct: {total_correct} | Failures: {len(failures)} | Batch time: {batch_time:.2f}s")

if __name__ == "__main__":
    find_failures(
        checkpoint_path="rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_sft_1-14_7num_BOS_352000.pt",
        test_file_path="rpn3_llm/data/sft_1-14_7num_BOS_val.txt",
        output_fail_path="rpn3_llm/results/ut1.8M_failures_subset.txt",
        num_failures_needed=5
    )
