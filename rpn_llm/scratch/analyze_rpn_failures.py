import re
from collections import defaultdict, Counter

def analyze_failures(file_path):
    category_counts = Counter()
    total_failures = 0
    regex_fails = 0
    
    # regex to parse a failure line: 
    # Q: {prompt} | Expected: {ans} | Predicted: {pred} | Full Expected: {full_exp} | Full Pred: {full_pred}
    line_re = re.compile(r"Q: (.*?) \| Expected: (.*?) \| Predicted: (.*?) \| Full Expected: (.*?) \| Full Pred: (.*)")

    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith("Q:"):
                continue
            
            total_failures += 1
            match = line_re.match(line)
            if not match:
                regex_fails += 1
                continue
                
            prompt, expected_ans, pred_ans, full_exp, full_pred = match.groups()
            
            # --- Prefix Analysis ---
            # Prefix format: <a_rev b_rev op=:
            exp_prefix_match = re.search(r"<(.*?)=:", full_exp)
            pred_prefix_match = re.search(r"<(.*?)=:", full_pred)

            if not pred_prefix_match:
                if ":" in full_pred and "<" not in full_pred:
                    category_counts["Skipped Prefix (started with :)"] += 1
                elif "BORROW" in full_pred and "<" not in full_pred:
                    category_counts["Skipped Prefix (started with BORROW)"] += 1
                else:
                    category_counts["Missing Prefix (Malformed/Early Stop)"] += 1
                continue

            exp_prefix = exp_prefix_match.group(1).strip()
            pred_prefix = pred_prefix_match.group(1).strip()
            
            # Use regex to find (num) groups
            prefix_regex = re.compile(r"\((.*?)\)")
            exp_parts = prefix_regex.findall(exp_prefix)
            pred_parts = prefix_regex.findall(pred_prefix)
            
            # More granular num1/num2 analysis
            num1_fail = False
            num2_fail = False
            
            if len(pred_parts) > 0 and len(exp_parts) > 0:
                e1 = exp_parts[0]
                p1 = pred_parts[0]
                if e1 != p1:
                    num1_fail = True
                    if len(p1) > len(e1):
                        category_counts["num1: Hallucinated (Too long)"] += 1
                    elif len(p1) < len(e1):
                        category_counts["num1: Truncated (Too short)"] += 1
                    else:
                        category_counts["num1: Digit substitution error"] += 1
            
            if not num1_fail: # Only look at num2 if num1 was okay
                if len(pred_parts) > 1 and len(exp_parts) > 1:
                    e2 = exp_parts[1]
                    p2 = pred_parts[1]
                    if e2 != p2:
                        num2_fail = True
                        if len(p2) > len(e2):
                            category_counts["num2: Hallucinated (Too long)"] += 1
                        elif len(p2) < len(e2):
                            category_counts["num2: Truncated (Too short)"] += 1
                        else:
                            category_counts["num2: Digit substitution/Op error"] += 1
                elif len(pred_parts) < len(exp_parts):
                    category_counts["num2: Missing entirely"] += 1
                    num2_fail = True

            if not num1_fail and not num2_fail:
                # Check for operator or formatting mismatch in the rest of the prefix
                if exp_prefix != pred_prefix:
                    category_counts["Prefix Format/Op Mismatch"] += 1
                else:
                    # Prefix matches! Why did it fail?
                    # Check the math steps
                    category_counts["Prefix Perfect, Math Step Error"] += 1

    print(f"Total Failures: {total_failures}")
    if regex_fails:
        print(f"Lines failed regex (should be 0): {regex_fails}")
    
    print("\nDetailed Failure Categories:")
    # Calculate percentages relative to total_failures
    for cat, count in category_counts.most_common():
        pct = (count / total_failures) * 100
        print(f"  {cat:<40}: {count} ({pct:.1f}%)")
    
    total_categorized = sum(category_counts.values())
    print(f"\nTotal Categorized: {total_categorized} / {total_failures} ({100*total_categorized/total_failures:.1f}%)")

if __name__ == "__main__":
    analyze_failures("/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/results/rope25M_1-22_tens_comp_bracketed_final_failures.txt")
