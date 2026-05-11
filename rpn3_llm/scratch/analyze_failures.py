import os
import re
from collections import Counter

failures_file = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn3_llm/results/ut0.7M_2l_8h_256e_mlp3_phaseMask_True_rpn3_3num_104000_failures.txt"


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
        if exp_rev != pred_rev:
            phase_failures["1st REV"] += 1
        else:
            # 2. Check 1st MATH
            exp_m_parts = exp_math.split("[MATH]")
            pred_m_parts = pred_math.split("[MATH]")
            
            exp_m1 = exp_m_parts[0] if len(exp_m_parts) >= 1 else ""
            pred_m1 = pred_m_parts[0] if len(pred_m_parts) >= 1 else ""
            
            exp_steps1 = exp_m1.split("[REV]")[0]
            pred_steps1 = pred_m1.split("[REV]")[0]
            
            if exp_steps1 != pred_steps1:
                phase_failures["1st MATH"] += 1
            else:
                # 3. Check 2nd REV
                exp_rev2 = exp_m1.split("[REV]")[1] if "[REV]" in exp_m1 else ""
                pred_rev2 = pred_m1.split("[REV]")[1] if "[REV]" in pred_m1 else ""
                
                if exp_rev2 != pred_rev2:
                    phase_failures["2nd REV"] += 1
                else:
                    # 4. Check 2nd MATH
                    exp_m2 = exp_m_parts[1] if len(exp_m_parts) >= 2 else ""
                    pred_m2 = pred_m_parts[1] if len(pred_m_parts) >= 2 else ""
                    
                    exp_steps2 = exp_m2.split("[REV]")[0]
                    pred_steps2 = pred_m2.split("[REV]")[0]
                    
                    if exp_steps2 != pred_steps2:
                        phase_failures["2nd MATH"] += 1
                    else:
                        # 5. Check ANS
                        if exp_ans != pred_ans:
                            phase_failures["ANS"] += 1
                        else:
                            # It might be a line that is NOT a failure but was in the file?
                            # Or my parsing missed something!
                            phase_failures["UNKNOWN"] += 1

print(f"Total lines in failure file: {total_lines}")
print("\n--- Failures by Phase (First Point of Failure) ---")
print(f"{'Phase':<10} | {'Count':<6} | {'%':<6}")
print("-" * 30)
total_classified = sum(phase_failures.values())
for phase in ["1st REV", "1st MATH", "2nd REV", "2nd MATH", "ANS", "UNKNOWN"]:
    count = phase_failures[phase]
    pct = (count / total_classified) * 100 if total_classified > 0 else 0
    print(f"{phase:<10} | {count:<6} | {pct:>5.1f}%")
