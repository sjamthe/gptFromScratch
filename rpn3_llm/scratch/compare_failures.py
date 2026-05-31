import re
import os
from collections import Counter

def parse_failures(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return None
        
    failures = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("Q_Len:"):
                continue
            
            parts = line.split(" | ")
            if len(parts) < 8:
                continue
                
            q_len = int(parts[0].split(":")[1])
            
            reason_prompt = parts[1].split("]:", 1)
            reason = reason_prompt[0].strip("[]")
            prompt = reason_prompt[1] if len(reason_prompt) > 1 else ""
            
            exp_rev = parts[2].split(":", 1)[1] if ":" in parts[2] else parts[2]
            pred_rev = parts[3].split(":", 1)[1] if ":" in parts[3] else parts[3]
            exp_math = parts[4].split(":", 1)[1] if ":" in parts[4] else parts[4]
            pred_math = parts[5].split(":", 1)[1] if ":" in parts[5] else parts[5]
            exp_ans = parts[6].split(":", 1)[1] if ":" in parts[6] else parts[6]
            pred_ans = parts[7].split(":", 1)[1] if ":" in parts[7] else parts[7]
            
            failures[prompt] = {
                "q_len": q_len,
                "reason": reason,
                "exp_rev": exp_rev,
                "pred_rev": pred_rev,
                "exp_math": exp_math,
                "pred_math": pred_math,
                "exp_ans": exp_ans,
                "pred_ans": pred_ans,
                "raw": line
            }
    return failures

def main():
    file_cnt_crd = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn3_llm/results/ut2.1M_2l_8h_384e_mlp4_phaseMask_True_cnt2_crd2_sft_1-6_4num_BOS_200000_ratio_1.0_failures.txt"
    file_base = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn3_llm/results/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_sft_1-6_4num_BOS_200000_ratio_1.0_failures.txt"
    
    fails_cnt_crd = parse_failures(file_cnt_crd)
    fails_base = parse_failures(file_base)
    
    if not fails_cnt_crd or not fails_base:
        return
        
    print("--- Detailed REV_FAIL_3 Analysis ---")
    rev3_prompts = [p for p, f in fails_cnt_crd.items() if f["reason"] == "REV_FAIL_3"]
    print(f"Total REV_FAIL_3 prompts in New Model: {len(rev3_prompts)}")
    
    base_reasons_for_rev3 = [fails_base[p]["reason"] for p in rev3_prompts]
    reasons_counter = Counter(base_reasons_for_rev3)
    
    print("\nBaseline failure categories for these same prompts:")
    for reason, count in reasons_counter.items():
        print(f"  - {reason:<20}: {count}")
        
    print("\nExamples of prompts migrating to REV_FAIL_3:")
    shown = 0
    for p in rev3_prompts:
        f_cnt = fails_cnt_crd[p]
        f_base = fails_base[p]
        if f_base["reason"] == "REV_FAIL_MULTIPLE":
            shown += 1
            print(f"\nMigration Example {shown}:")
            print(f"Prompt: {p}")
            print(f"Exp Rev:       {f_cnt['exp_rev']}")
            print(f"New Pred Rev:  {f_cnt['pred_rev']}")
            print(f"Base Pred Rev: {f_base['pred_rev']}")
            if shown >= 3:
                break

if __name__ == "__main__":
    main()
