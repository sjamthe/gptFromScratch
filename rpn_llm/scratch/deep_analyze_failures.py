
import re
from collections import defaultdict

def analyze_math_failures(file_path):
    failures = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                if "Q:" in line:
                    failures.append(line.strip())
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    stats = {
        'total': len(failures),
        'ops': defaultdict(int),
        'lengths': defaultdict(int),
        'error_types': defaultdict(int),
        'off_by_one': 0,
    }

    for fail in failures:
        # Extract prompt and operands
        # Q: [BOS]50 73+? | Expected: 123 | Predicted: 12321...
        # Note: Predicted might be messy, so we take the first sequence of digits
        m = re.search(r"Q: (?:\[BOS\])?(\d+) (\d+)([+\-])\? \| Expected: ([\d\-]+) \| Predicted: ([\d\-]+)", fail)
        if not m: continue
        
        n1, n2, op, expected, predicted = m.groups()
        stats['ops'][op] += 1
        stats['lengths'][max(len(n1), len(n2))] += 1
        
        try:
            # We strip any trailing garbage from predicted
            clean_pred = re.match(r"-?\d+", predicted).group()
            exp_val = int(expected)
            pred_val = int(clean_pred)
            diff = abs(exp_val - pred_val)
            
            if diff == 0:
                # If values match but it still failed, it might be the scratchpad
                stats['error_types']['scratchpad_only_error'] += 1
            elif diff == 1:
                stats['off_by_one'] += 1
                stats['error_types']['off_by_one'] += 1
            elif diff % 10 == 0 and diff > 0:
                stats['error_types']['carry_borrow_misalignment'] += 1
            else:
                stats['error_types']['structural_logic_error'] += 1
        except:
            stats['error_types']['malformed_output'] += 1

    print(f"--- Deep Failure Analysis (n={stats['total']}) ---")
    print(f"Operations: Addition: {stats['ops']['+']}, Subtraction: {stats['ops']['-']}")
    print(f"Off-by-one errors: {stats['off_by_one']} ({stats['off_by_one']/max(1,stats['total'])*100:.1f}%)")
    
    print("\nError Type Breakdown:")
    for t, c in sorted(stats['error_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"{t:<25}: {c}")

    print("\nFailures by Max Digit Length:")
    for l in sorted(stats['lengths'].keys()):
        print(f"{l:2d} digits: {stats['lengths'][l]}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/results/rope7.1M_1-22_uniform_BOS_32000_failures.txt'
    analyze_math_failures(path)
