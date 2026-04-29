import os
import re

def convert_to_lean_phase_format(line):
    # Original: [BOS]8 89+?<8 98 +=8+9+0=7:0+8+1=9:79>97
    # Desired: [BOS]8 89+?[REV]8 98+=[MATH]8+9+0=7:0+8+1=9:79[ANS]97[EOS]
    line = line.strip()
    if not line: return ""
    
    # 1. Split by ? (Prompt end)
    if '?' not in line: return line
    prompt_part, rest = line.split('?', 1)
    prompt = prompt_part.replace('(', '').replace(')', '').replace('[BOS]', '').strip()
    # Remove space before operator in prompt
    prompt = re.sub(r'\s+([+\-*\/])', r'\1', prompt)
    prompt = prompt + '?'
    
    # 2. Split by < (Reversal start)
    if '<' not in rest: return line
    _, rest = rest.split('<', 1)
    
    # 3. Split by the first math-operator equals sign (Math start)
    m = re.search(r'([+\-])=', rest)
    if not m: return line
    op_eq_pos = m.end()
    reversal = rest[:op_eq_pos].strip()
    # Remove space before + or - and the following =
    reversal = re.sub(r'\s+([+\-])=', r'\1=', reversal)
    # Also remove space before the operator itself if it's there
    reversal = re.sub(r'\s+([+\-])', r'\1', reversal)
    
    math_and_ans = rest[op_eq_pos:].strip()
    
    # 4. Split by > (Answer start)
    if '>' not in math_and_ans: return line
    math, answer = math_and_ans.split('>', 1)
    math = math.strip()
    answer = answer.strip()
    
    # Assemble
    return f"[BOS]{prompt}[REV]{reversal}[MATH]{math}[ANS]{answer}[EOS]"

def main():
    # Training Set
    input_train = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/data/RPNData-1-22_uniform_BOS_train.txt"
    output_train = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/data/RPNData-1-22_phase_lean_train.txt"
    
    # Val Set
    input_val = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/data/RPNData-1-22_uniform_BOS_val.txt"
    output_val = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/data/RPNData-1-22_phase_lean_val.txt"

    # Test Set
    input_test = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/data/RPNData-1-22_uniform_BOS_test.txt"
    output_test = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/data/RPNData-1-22_phase_lean_test.txt"

    for input_file, output_file, limit in [(input_train, output_train, 100000), (input_val, output_val, 10000), (input_test, output_test, 10000)]:
        if not os.path.exists(input_file):
            print(f"Input file {input_file} not found.")
            continue

        print(f"Reading from {input_file}...")
        with open(input_file, "r") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= limit: break
                lines.append(line)
            
        print(f"Converting {len(lines)} lines...")
        new_lines = [convert_to_lean_phase_format(l) for l in lines]
        
        print(f"Saving to {output_file}...")
        with open(output_file, "w") as f:
            for nl in new_lines:
                if nl:
                    f.write(nl + "\n")
                
    print("Done. Sample:")
    print(new_lines[0])

if __name__ == "__main__":
    main()
