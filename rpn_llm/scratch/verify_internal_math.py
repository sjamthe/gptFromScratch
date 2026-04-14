import re

log_path = "rpn_llm/results/tens_complement_compress_failures_large_digit.txt"

# Regex explanation:
# Full Pred: <       -- literal start
# ([^ ]+)             -- Group 1: characters until space (rev_a)
# \s+                 -- one or more spaces
# ([^ ]+)([+\-])=:   -- Group 2: rev_b, Group 3: op (+ or -), followed by =:
# .*?                 -- non-greedy match for scratchpad steps
# >                   -- literal closing marker
# ([^\[\n ]+)         -- Group 4: the final answer tokens until [UNK] or space/newline
pattern = re.compile(r"Full Pred: <([^ ]+)\s+([^ ]+)([+\-])=:.*?>([^\[\n ]+)")

total = 0
math_correct = 0

try:
    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                total += 1
                rev_a = match.group(1)
                rev_b = match.group(2)
                op = match.group(3)
                pred_ans_str = match.group(4)
                
                try:
                    # Reverse the numbers back to normal order
                    a = int(rev_a[::-1])
                    b = int(rev_b[::-1])
                    
                    # Calculate correct math based on internal operands
                    if op == "+":
                        expected = a + b
                    else:
                        expected = a - b
                    
                    # Clean predicted answer
                    pred_ans = int(pred_ans_str)
                    
                    if pred_ans == expected:
                        math_correct += 1
                    # else:
                    #     print(f"Mismatch: {a} {op} {b} = {expected}, but model said {pred_ans}")
                        
                except ValueError:
                    continue
except FileNotFoundError:
    print(f"Error: {log_path} not found.")
    exit(1)

if total > 0:
    print(f"Total analyzed rows: {total}")
    print(f"Internal Math Correct: {math_correct}")
    print(f"Internal Accuracy: {(math_correct / total) * 100:.2f}%")
else:
    print("No matches found. Check regex or file content.")
