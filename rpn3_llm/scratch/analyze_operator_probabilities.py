import os
import re
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from utils import RPNTokenizer

def analyze_operator_probabilities():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Paths
    model_path = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_theta_tau0.7_sft_1-14_7num_BOS_500000.pt"
    val_file_path = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn3_llm/data/sft_1-14_7num_BOS_val.txt"
    failures_file_path = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn3_llm/results/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_theta_tau0.7_sft_1-14_7num_BOS_500000_7num_ratio_1.0_failures.txt"
    tokenizer_path = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn3_llm/rpn-tokenizer.json"

    # 1. Load Tokenizer
    tokenizer = RPNTokenizer(tokenizer_path)

    # 2. Load Failures to identify incorrect equations
    failed_prompts = set()
    if os.path.exists(failures_file_path):
        with open(failures_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Q_Len:"):
                    # Extract prompt
                    parts = line.split(" | ")
                    if len(parts) >= 2:
                        prompt_part = parts[1]
                        prompt_str = prompt_part.split(":", 1)[1].strip()
                        # Clean to match the clean prompt
                        failed_prompts.add(prompt_str)
        print(f"Loaded {len(failed_prompts)} failed prompts.")
    else:
        print(f"Warning: Failures file not found at {failures_file_path}")

    # 3. Load Model
    print(f"Loading model checkpoint {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Force use_phase_mask to match validation
    config.use_phase_mask = True
    
    # Auto-detect architecture from config
    if getattr(config, 'universal', False):
        from model_rope import GPT
        print("Using Universal Transformer (model_rope)")
    else:
        from model_rope import GPT
        print("Using standard RoPE Transformer (model_rope)")

    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # 4. Parse Val File using the exact selection logic of validate_rpn_model.py
    print(f"Parsing val file {val_file_path}...")
    with open(val_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Replicate grouping by prompt token length
    length_groups = defaultdict(list)
    total_items = 0
    for line in lines:
        if "?" not in line:
            continue
        line_clean = re.sub(r'\s+', ' ', line.strip()) + '\n'
        prompt_str = line_clean.split("?", 1)[0] + "?"
        prompt_tokens = tokenizer.encode(prompt_str)
        prompt_len = len(prompt_tokens)
        length_groups[prompt_len].append((prompt_tokens, line_clean))
        total_items += 1

    print(f"Total validation items: {total_items}")
    print(f"Grouped into {len(length_groups)} prompt lengths.")
    
    VALIDATION_SET_RATIO = 1.0
    max_rows = int(VALIDATION_SET_RATIO * total_items / len(length_groups))
    print(f"Using max_rows = {max_rows} per group (matching validation script)")

    selected_items = []
    for length, items in length_groups.items():
        subset = items[:max_rows]
        for prompt_tokens, line_clean in subset:
            prompt_str = tokenizer.decode(prompt_tokens)
            clean_prompt = prompt_str.replace("[BOS]", "").replace("?", "").strip()
            prompt_words = clean_prompt.split()
            if len(prompt_words) == 7:
                selected_items.append((line_clean, prompt_str))

    print(f"Found {len(selected_items)} selected 7-num equations.")

    correct_pred_data = []
    incorrect_pred_data = []
    other_pred_data = []
    failed_file_data = []
    correct_file_data = []

    # Token IDs
    # + is 23, - is 24, [SEP] is 5, [REV] is 10, [MATH] is 11, [ANS] is 12
    plus_id = 23
    minus_id = 24

    for line_idx, (clean_line, prompt_str) in enumerate(selected_items):
        prompt_tokens = tokenizer.encode(prompt_str)
        prompt_len = len(prompt_tokens)
        
        full_tokens = tokenizer.encode(clean_line)
        expected_tokens = full_tokens[prompt_len:]
        
        # Find the sign operator after the first [SEP] in REV2
        # REV2 starts at the second [REV] (token ID 10) in expected_tokens
        rev_indices = [i for i, x in enumerate(expected_tokens) if x == 10]
        if len(rev_indices) < 2:
            continue
        rev2_idx = rev_indices[1]
        
        # Find first [SEP] after rev2_idx
        sep_idx = -1
        for i in range(rev2_idx + 1, len(expected_tokens)):
            if expected_tokens[i] == 5:
                sep_idx = i
                break
        if sep_idx == -1:
            continue
            
        # Find first operator (+ or -) after sep_idx
        op_idx = -1
        for i in range(sep_idx + 1, len(expected_tokens)):
            if expected_tokens[i] in (plus_id, minus_id):
                op_idx = i
                break
        if op_idx == -1:
            continue
            
        # The index in the full sequence is prompt_len + op_idx
        target_token_idx = prompt_len + op_idx
        target_op = full_tokens[target_token_idx]
        
        # We pass full_tokens[:target_token_idx] as input
        input_tokens = torch.tensor(full_tokens[:target_token_idx], dtype=torch.long, device=device).unsqueeze(0)
        
        # Compute full_phase_ids for masking
        is_phase_shift = (input_tokens == 10) | (input_tokens == 11) | (input_tokens == 12)
        full_phase_ids = is_phase_shift.cumsum(dim=-1)
        
        with torch.no_grad():
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, _ = model(input_tokens, use_cache=False, full_phase_ids=full_phase_ids)
        
        # Pluck final step logits
        last_step_logits = logits[0, -1, :].float()
        
        # Compute probabilities over vocab
        probs = F.softmax(last_step_logits, dim=-1)
        
        prob_correct_op = probs[target_op].item()
        pred_token = torch.argmax(last_step_logits).item()
        
        is_failed_overall = (prompt_str in failed_prompts)
        
        info = {
            'prob': prob_correct_op,
            'is_failed_overall': is_failed_overall,
            'pred_token': pred_token,
            'target_op': target_op
        }
        
        # 1. Group by model's step prediction
        if pred_token == target_op:
            correct_pred_data.append(info)
        elif pred_token in (plus_id, minus_id):
            incorrect_pred_data.append(info)
        else:
            other_pred_data.append(info)
            
        # 2. Group by failures file classification
        if is_failed_overall:
            failed_file_data.append(info)
        else:
            correct_file_data.append(info)

    # Function to print stats
    def print_group_stats(name, data):
        probs = [x['prob'] for x in data]
        print(f"\n{name} (total: {len(data)}):")
        if probs:
            print(f"  Mean probability of correct operator:   {np.mean(probs)*100:.2f}%")
            print(f"  Median probability of correct operator: {np.median(probs)*100:.2f}%")
            print(f"  Min probability:                        {np.min(probs)*100:.2f}%")
            print(f"  Max probability:                        {np.max(probs)*100:.2f}%")
        else:
            print("  None")

    print("\n================ 1. MODEL PREDICTION STEP BREAKDOWN ================")
    print("Classified strictly by what token the model predicted at this step (teacher-forced prefix)")
    print_group_stats("Model Predicted Correct Operator", correct_pred_data)
    print_group_stats("Model Predicted Incorrect Operator", incorrect_pred_data)
    print_group_stats("Model Predicted Other Token", other_pred_data)

    print("\n================ 2. FAILURES FILE BREAKDOWN ================")
    print("Classified by whether the prompt was in the failures file (which mixes untested equations and other failure types)")
    print_group_stats("Correct in Failures File (includes untested)", correct_file_data)
    print_group_stats("Incorrect/Failed in Failures File", failed_file_data)

    # Let's do a cross-tabulation of failures file vs actual prediction!
    print("\n================ 3. CROSS-TABULATION ================")
    
    # Of the equations in the failures file:
    failed_op_failures = [x for x in failed_file_data if x['pred_token'] != x['target_op']]
    failed_other_failures = [x for x in failed_file_data if x['pred_token'] == x['target_op']]
    print(f"Of the {len(failed_file_data)} failures in failures file:")
    print(f"  - Failed on this operator (wrong prediction at this step): {len(failed_op_failures)}")
    print(f"  - Passed this operator step, but failed later in sequence:  {len(failed_other_failures)}")
    
    # Of the equations classified as "Correct" (untested + truly correct):
    correct_but_pred_wrong = [x for x in correct_file_data if x['pred_token'] != x['target_op']]
    correct_and_pred_right = [x for x in correct_file_data if x['pred_token'] == x['target_op']]
    print(f"\nOf the {len(correct_file_data)} classified as 'Correct' (no failure logged):")
    print(f"  - Model predicted operator correctly at this step:           {len(correct_and_pred_right)}")
    print(f"  - Model predicted WRONG operator (these are untested rows):   {len(correct_but_pred_wrong)}")

if __name__ == "__main__":
    analyze_operator_probabilities()
