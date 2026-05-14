import sys

def parse_file(filename):
    incorrect_count = 0
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                ans_idx = line.rindex('[ANS]')
                rev_idx = line.rindex('[REV]', 0, ans_idx)
                
                num_before_ans = line[rev_idx + 5: ans_idx]
                
                eos_idx = line.index('[EOS]', ans_idx)
                num_after_ans = line[ans_idx + 5: eos_idx]
                
                is_negative = False
                if num_before_ans.startswith('-'):
                    is_negative = True
                    num_to_reverse = num_before_ans[1:]
                else:
                    num_to_reverse = num_before_ans
                    
                reversed_num = num_to_reverse[::-1]
                if is_negative:
                    expected_ans = '-' + reversed_num
                else:
                    expected_ans = reversed_num
                    
                if expected_ans != num_after_ans:
                    print(f"Line {i+1}:")
                    print(f"  Before [ANS]: {num_before_ans}")
                    print(f"  After [ANS] : {num_after_ans}")
                    print(f"  Expected    : {expected_ans}")
                    incorrect_count += 1
            except ValueError:
                print(f"Line {i+1}: Malformed line, missing [REV], [ANS] or [EOS]")
                incorrect_count += 1
                
    print(f"Total incorrect: {incorrect_count}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse_file(sys.argv[1])
    else:
        parse_file('rpn3_llm/data/rpn3_3num_val.txt')
