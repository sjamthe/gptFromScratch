import re
import os
import argparse
from collections import defaultdict

def parse_fail_file(fail_file):
    fail_counts = defaultdict(int)
    total_evaluated = 0
    
    if not os.path.exists(fail_file):
        return None
        
    with open(fail_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
        m_acc = re.search(r"Validation Accuracy:\s+[\d.]+%\s+\(\d+/(\d+)\)", content)
        if m_acc:
            total_evaluated = int(m_acc.group(1))
            
        section = re.search(r"Reversal Failures by Digit Length:\n(.*?)(?:\n\n|\Z)", content, re.DOTALL)
        if section:
            lines = section.group(1).strip().split('\n')
            for line in lines:
                m = re.search(r"(\d+)\s+digits:\s+(\d+)\s+failures", line)
                if m:
                    length = int(m.group(1))
                    count = int(m.group(2))
                    fail_counts[length] = count
                    
    if total_evaluated == 0:
        return None
        
    # Derive accuracy
    num_possible_lengths = 22
    total_per_length = total_evaluated / num_possible_lengths
    
    accuracies = {}
    for l in range(1, 23):
        fails = fail_counts.get(l, 0)
        acc = max(0, (total_per_length - fails) / total_per_length) * 100
        accuracies[l] = acc
    return accuracies

def main():
    files = [
        ("ROPE_16k", "rpn_llm/results/rope25M_1-22_uniform_BOS_16000_failures.txt"),
        ("ROPE_32k", "rpn_llm/results/rope25M_1-22_uniform_BOS_32000_failures.txt"),
        ("RDT_16k", "rpn_llm/results/RDT9M_1-22_uniform_BOS_16000_failures.txt"),
        ("RDT_32k", "rpn_llm/results/RDT9M_1-22_uniform_BOS_32000_failures.txt"),
        ("RDT_48k", "rpn_llm/results/RDT9M_1-22_uniform_BOS_48000_failures.txt"),
        ("RDT_64k", "rpn_llm/results/RDT9M_1-22_uniform_BOS_64000_failures.txt"),
    ]
    
    results = {}
    valid_cols = []
    for label, path in files:
        accs = parse_fail_file(path)
        if accs:
            results[label] = accs
            valid_cols.append(label)
            
    # Print Markdown Table
    header = "| Num1 Length | " + " | ".join(valid_cols) + " |"
    sep = "|---|" + "|---" * len(valid_cols) + "|"
    print(header)
    print(sep)
    
    for l in range(1, 23):
        row = f"| {l:2d} | "
        cells = []
        for col in valid_cols:
            acc = results[col][l]
            cells.append(f"{acc:.2f}%")
        row += " | ".join(cells) + " |"
        print(row)

if __name__ == "__main__":
    main()
