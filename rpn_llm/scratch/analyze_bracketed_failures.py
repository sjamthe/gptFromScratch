import re
from collections import defaultdict, Counter

def analyze_bracketed_failures(file_path):
    category_counts = Counter()
    len_pair_stats = defaultdict(lambda: {"total": 0, "failures": 0})
    total_failures = 0
    
    # Q: (n1)(n2)op= | Expected: ... | Predicted: ... | Full Expected: ... | Full Pred: ...
    line_re = re.compile(r"Q: (.*?) \| Expected: (.*?) \| Predicted: (.*?) \| Full Expected: (.*?) \| Full Pred: (.*)")

    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith("Q:"):
                continue
            
            total_failures += 1
            match = line_re.match(line)
            if not match:
                continue
                
            prompt, expected_ans, pred_ans, full_exp, full_pred = match.groups()
            
            # Get operand lengths
            m = re.search(r"\((\d+)\)\((\d+)\)([+\-])=", prompt)
            if m:
                l1, l2 = len(m.group(1)), len(m.group(2))
                len_pair_stats[(l1, l2)]["failures"] += 1
                prompt_len = l1 + l2
            else:
                prompt_len = 0

            # Categorize the skip
            if full_pred.startswith(":"):
                category_counts["Skipped entirely to math steps (:)"] += 1
            elif full_pred.startswith("<") and ":" not in full_pred:
                category_counts["Started prefix but never finished (no :)"] += 1
            elif "<" in full_pred and ":" in full_pred:
                # It didn't skip the prefix, so why did it fail?
                # Likely a math error or prefix hallucination
                category_counts["Hybrid/Math Error (Did NOT skip prefix)"] += 1
            else:
                category_counts["Other/Garbage"] += 1

    print(f"Total Failures: {total_failures}")
    print("\nDetailed Failure Categories:")
    for cat, count in category_counts.most_common():
        print(f"  {cat:<45}: {count} ({100*count/total_failures:.1f}%)")

    print("\nTop Failing Length Pairs (num1_len, num2_len):")
    sorted_pairs = sorted(len_pair_stats.items(), key=lambda x: x[1]["failures"], reverse=True)
    for pair, stats in sorted_pairs[:10]:
        print(f"  {pair}: {stats['failures']} failures")

if __name__ == "__main__":
    analyze_bracketed_failures("/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/results/rope25M_1-22_tens_comp_bracketed_final_failures.txt")
