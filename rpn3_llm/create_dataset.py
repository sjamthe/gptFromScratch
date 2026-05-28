import random
import os
import argparse
from utils import RPNTokenizer

def generate_number(length):
    if length <= 0:
        return ""
    if length == 1:
        return str(random.randint(0, 9))
    else:
        return str(random.randint(1, 9)) + "".join([str(random.randint(0, 9)) for _ in range(length - 1)])

def reverse_string(s):
    return s[::-1]

def generate_math_steps(a_str, b_str, op):
    a_true = int(a_str)
    b_true = int(b_str)
    is_a_negative = a_true < 0
    is_b_negative = b_true < 0
    
    a_mag_rev = a_str.lstrip('-')[::-1]
    b_mag_rev = b_str.lstrip('-')[::-1]
    
    """ If number is negative find 1st non zero digit (in reverse string)
        for that digit replace it with 10s complement and for remaining digits 
        replace them with 9s complement
    """
    def to_10s_comp_rev(mag_rev):
        comp_rev = []
        found_nonzero = False
        for d_str in mag_rev:
            d = int(d_str)
            if not found_nonzero:
                if d == 0:
                    comp_rev.append('0')
                else:
                    comp_rev.append(str(10 - d))
                    found_nonzero = True
            else:
                comp_rev.append(str(9 - d))
        return comp_rev

    a_rev = to_10s_comp_rev(a_mag_rev) if is_a_negative else list(a_mag_rev)
    b_rev = to_10s_comp_rev(b_mag_rev) if is_b_negative else list(b_mag_rev)
    
    max_len = max(len(a_rev), len(b_rev))
    
    if op == '+':
        ans = a_true + b_true
    elif op == '-':
        ans = a_true - b_true
        
    steps = []
    carry = 0
    derived_digits = []
    
    i = 0
    while True:
        is_exhausted = i >= max_len
        
        d_a = int(a_rev[i]) if i < len(a_rev) else (9 if is_a_negative else 0)
        d_b = int(b_rev[i]) if i < len(b_rev) else (9 if is_b_negative else 0)
        
        if op == '+':
            res = d_a + d_b + carry
            digit = res % 10
            new_carry = res // 10
            
            if is_exhausted and new_carry == carry and (len(derived_digits) > 0 and derived_digits[-1] == str(digit)):
                break
                
            steps.append(f"{d_a}+{d_b}+{carry}={digit}")
            
        elif op == '-':
            res = (d_a - carry) - d_b
            if res < 0:
                res += 10
                new_carry = 1
            else:
                new_carry = 0
            digit = res
            
            if is_exhausted and new_carry == carry and (len(derived_digits) > 0 and derived_digits[-1] == str(digit)):
                break
                
            steps.append(f"{d_a}-{d_b}-{carry}={digit}")
            
        derived_digits.append(str(digit))
        carry = new_carry
        i += 1
        
    scratchpad_math = ":".join(steps)
    
    if digit == 0:
        # Positive answer
        steps_str = scratchpad_math + ":[BORROW]0|+"
        ans_rev = "".join(derived_digits)
        ans_str = str(int(ans_rev[::-1]))
        full_math_str = f"[MATH]{steps_str}[REV]{ans_rev}"
    else:
        # Negative answer
        steps_part2 = ["[BORROW]1|-"]
        tens_comp_digits = []
        found_nonzero = False
        for d_str in derived_digits:
            d = int(d_str)
            if not found_nonzero:
                if d == 0:
                    steps_part2.append("[PASS]0=0")
                    tens_comp_digits.append("0")
                else:
                    comp = 10 - d
                    steps_part2.append(f"10-{d}={comp}")
                    tens_comp_digits.append(str(comp))
                    found_nonzero = True
            else:
                comp = 9 - d
                steps_part2.append(f"9-{d}={comp}")
                tens_comp_digits.append(str(comp))
        
        steps_str = ":".join([scratchpad_math] + steps_part2)
        ans_rev = "-" + "".join(tens_comp_digits)
        ans_str = str(int("-" + "".join(tens_comp_digits)[::-1]))
        full_math_str = f"[MATH]{steps_str}[REV]{ans_rev}"
        
    assert int(ans_str) == ans, f"Math logic failure: scratchpad derived {ans_str}, expected {ans}"
    return full_math_str, ans_str

def generate_example(max_numbers=3, max_digits=22):
    num_count = random.randint(2, max_numbers)
    
    # Generate random lengths from 1 to max_digits
    lengths = [random.randint(1, max_digits) for _ in range(num_count)]
    numbers = [generate_number(l) for l in lengths]
    ops = [random.choice(['+', '-']) for _ in range(num_count - 1)]
    
    # Phase 1: Prompt & Reversal
    prompt = f"[BOS]{numbers[0]}"
    for i in range(1, num_count):
        prompt += f" {numbers[i]}{ops[i-1]}"
    prompt += "?"
    
    rev_numbers = [reverse_string(n) for n in numbers]
    reversal = f"[REV]{rev_numbers[0]}"
    if num_count > 1:
        reversal += f"[SEP]{rev_numbers[1]}{ops[0]}="
    for i in range(2, num_count):
        reversal += f"[SEP]{rev_numbers[i]}{ops[i-1]}"
    
    # Phase 2: Math Loop
    current_val_str = numbers[0]
    math_phases = []
    
    for i in range(1, num_count):
        next_num_str = numbers[i]
        op = ops[i-1]
        
        math_str, current_true_str = generate_math_steps(current_val_str, next_num_str, op)
        current_val_str = current_true_str # For the next step!
        
        # Build tail
        tail = ""
        if i + 1 < num_count:
            tail += f"[SEP]{rev_numbers[i+1]}{ops[i]}="
        for j in range(i+2, num_count):
            tail += f"[SEP]{rev_numbers[j]}{ops[j-1]}"
        
        if tail == "":
            transition = "[ANS]"
        else:
            transition = tail
            
        math_phases.append(math_str + transition)
        
    full_math_sequence = "".join(math_phases)
    
    # Phase 3: Final Answer
    ans_line = f"{current_val_str}[EOS]"
    
    full_line = f"{prompt}{reversal}{full_math_sequence}{ans_line}"
    return full_line, num_count, len(prompt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--output_dir", type=str, default="rpn3_llm/data")
    parser.add_argument("--split_type", action="store_true", help="Split into num_count files")
    parser.add_argument("--split_length", action="store_true", help="Split into le25 and gt25 files")
    parser.add_argument("--max_numbers", type=int, default=3, help="Maximum number of operands")
    parser.add_argument("--max_digits", type=int, default=22, help="Maximum digits per operand")
    parser.add_argument("--prefix", type=str, default="rpn3", help="Prefix for output files")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    handles = {}
    handles["train"] = open(os.path.join(args.output_dir, f"{args.prefix}_train.txt"), "w", encoding="utf-8")
    handles["val"] = open(os.path.join(args.output_dir, f"{args.prefix}_val.txt"), "w", encoding="utf-8")
    
    print(f"Generating {args.samples} samples...")
    train_pct = 0.9
    num_train = int(args.samples * train_pct)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = RPNTokenizer(os.path.join(script_dir, "rpn-tokenizer.json"))
    max_tokens = 2040

    for i in range(args.samples):
        while True:
            example, num_count, _ = generate_example(args.max_numbers, args.max_digits)
            tokens = tokenizer.encode(example)
            if len(tokens) <= max_tokens:
                break
        length = len(tokenizer.decode(tokenizer.encode(example.split('?')[0] + '?')))
        is_train = i < num_train
        
        prefix = "train" if is_train else "val"
        handles[prefix].write(example + "\n")
        
        if args.split_type:
            key = f"{prefix}_{num_count}num"
            if key not in handles:
                handles[key] = open(os.path.join(args.output_dir, f"rpn3_{num_count}num_{prefix}.txt"), "w", encoding="utf-8")
            handles[key].write(example + "\n")
            
        if args.split_length:
            len_suffix = "le25" if length <= 25 else "gt25"
            key = f"{prefix}_{len_suffix}"
            if key not in handles:
                handles[key] = open(os.path.join(args.output_dir, f"rpn3_{len_suffix}_{prefix}.txt"), "w", encoding="utf-8")
            handles[key].write(example + "\n")
            
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i+1}/{args.samples}...")
            
    for h in handles.values():
        h.close()
        
    print(f"Done! Datasets saved to {args.output_dir}")

if __name__ == "__main__":
    main()
