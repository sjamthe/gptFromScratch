import sys
import os
from collections import Counter

if len(sys.argv) > 1:
    failures_file = sys.argv[1]
else:
    # Default to the most recent run for convenience
    failures_file = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn3_llm/results/ut1.5M_2l_8h_384e_mlp3_phaseMask_True_sft_1-14_7num_BOS_560000_failures.txt"

if not os.path.exists(failures_file):
    print(f"File not found: {failures_file}")
    exit()

total_lines = 0
phase_failures = Counter()

with open(failures_file, "r", encoding="utf-8") as f:
    for line in f:
        total_lines += 1
        
        # Extract parts
        parts = line.split(" | ")
        exp_rev = ""
        pred_rev = ""
        exp_math = ""
        pred_math = ""
        exp_ans = ""
        pred_ans = ""
        
        for p in parts:
            if p.startswith("Exp Rev:"):
                exp_rev = p[len("Exp Rev:"):]
            elif p.startswith("Pred Rev:"):
                pred_rev = p[len("Pred Rev:"):]
            elif p.startswith("Exp Math:"):
                exp_math = p[len("Exp Math:"):]
            elif p.startswith("Pred Math:"):
                pred_math = p[len("Pred Math:"):]
            elif p.startswith("Exp Ans:"):
                exp_ans = p[len("Exp Ans:"):]
            elif p.startswith("Pred Ans:"):
                pred_ans = p[len("Pred Ans:"):]
                
        # 1. Check 1st REV
        if exp_rev.strip() != pred_rev.strip():
            phase_failures["1st REV"] += 1
        else:
            exp_m_parts = exp_math.split("[MATH]")
            pred_m_parts = pred_math.split("[MATH]")
            
            math_fail_step = -1
            rev_fail_step = -1
            failed_idx = -1
            
            for i in range(max(len(exp_m_parts), len(pred_m_parts))):
                e = exp_m_parts[i].strip() if i < len(exp_m_parts) else None
                p = pred_m_parts[i].strip() if i < len(pred_m_parts) else None
                if e != p:
                    failed_idx = i
                    break
                    
            if failed_idx != -1:
                e = exp_m_parts[failed_idx].strip() if failed_idx < len(exp_m_parts) else ""
                p = pred_m_parts[failed_idx].strip() if failed_idx < len(pred_m_parts) else ""
                
                e_sub = e.split("[REV]")
                p_sub = p.split("[REV]") if p else []
                
                e_m = e_sub[0].strip()
                p_m = p_sub[0].strip() if len(p_sub) > 0 else ""
                
                if e_m != p_m:
                    math_fail_step = failed_idx + 1
                    phase_failures[f"{math_fail_step}th MATH"] += 1
                else:
                    rev_fail_step = failed_idx + 2
                    phase_failures[f"{rev_fail_step}th REV"] += 1
                    
                    # Sort comparison for REV failures
                    e_rev = e_sub[1].strip() if len(e_sub) > 1 else ""
                    p_rev = p_sub[1].strip() if len(p_sub) > 1 else ""
                    
                    ex_parts = e_rev.split("[SEP]")
                    pred_parts = p_rev.split("[SEP]")
                    
                    # Strip '=' from tails for fair comparison
                    ex_parts = [part.rstrip('=') for part in ex_parts]
                    pred_parts = [part.rstrip('=') for part in pred_parts]
                    
                    sorted_ex = sorted(ex_parts)
                    sorted_pred = sorted(pred_parts)
                    
                    if sorted_ex == sorted_pred:
                        print(f"REV{rev_fail_step} matches after sorting (permutation error)!")
            else:
                # 3. Check ANS
                if exp_ans.strip() != pred_ans.strip():
                    phase_failures["ANS"] += 1
                    ex_sorted = "".join(sorted(exp_ans))
                    pred_sorted = "".join(sorted(pred_ans))
                    if ex_sorted == pred_sorted:
                        print("ANS matches after sorting ")
                else:
                    phase_failures["UNKNOWN"] += 1

print(f"Total lines in failure file: {total_lines}")
print("\n--- Failures by Phase (First Point of Failure) ---")
print(f"{'Phase':<12} | {'Count':<6} | {'%':<6}")
print("-" * 32)
total_classified = sum(phase_failures.values()) - phase_failures["UNKNOWN"]

# Custom sort keys to order mathematically
def phase_sort_key(phase):
    if phase == "1st REV": return 0
    if phase == "ANS": return 999
    if phase == "UNKNOWN": return 1000
    digits = ''.join(filter(str.isdigit, phase))
    if not digits: return 1000
    num = int(digits)
    if "MATH" in phase: return num * 2 - 1
    if "REV" in phase: return num * 2
    return 1000

for phase in sorted(phase_failures.keys(), key=phase_sort_key):
    if phase == "UNKNOWN": continue
    count = phase_failures[phase]
    pct = (count / total_classified) * 100 if total_classified > 0 else 0
    print(f"{phase:<12} | {count:<6} | {pct:>5.1f}%")
