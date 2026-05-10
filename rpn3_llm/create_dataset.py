import random
import os
import argparse

def generate_number(length):
    return "".join([str(random.randint(1, 9)) for _ in range(length)])

def reverse_string(s):
    return s[::-1]

def generate_math_steps(a_str, b_str, op):
    a_rev, b_rev = a_str[::-1], b_str[::-1]
    max_len = max(len(a_rev), len(b_rev))
    
    steps = []
    carry = 0
    
    if op == '+':
        for i in range(max_len):
            d_a = int(a_rev[i]) if i < len(a_rev) else 0
            d_b = int(b_rev[i]) if i < len(b_rev) else 0
            res = d_a + d_b + carry
            new_carry = res // 10
            digit = res % 10
            steps.append(f"{d_a}+{d_b}+{carry}={digit}")
            carry = new_carry
            
        if carry > 0:
            steps.append(f"0+0+{carry}={carry}")
            
        ans = int(a_str) + int(b_str)
        ans_rev = str(ans)[::-1]
        return ":".join(steps), ans_rev, str(ans), str(ans)
        
    elif op == '-':
        derived_digits = []
        for i in range(max_len):
            d_a = int(a_rev[i]) if i < len(a_rev) else 0
            d_b = int(b_rev[i]) if i < len(b_rev) else 0
            
            res = (d_a - carry) - d_b
            if res < 0:
                res += 10
                new_carry = 1
            else:
                new_carry = 0
                
            steps.append(f"{d_a}-{d_b}-{carry}={res}")
            derived_digits.append(str(res))
            carry = new_carry
            
        scratchpad_math = ":".join(steps)
        
        if carry == 0:
            # Positive answer
            final_scratchpad = scratchpad_math + ":[BORROW]0|+" + ":" + "".join(derived_digits)
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
            
            steps_part2.append("".join(tens_comp_digits))
            final_scratchpad = ":".join([scratchpad_math] + steps_part2)
            
        ans = int(a_str) - int(b_str)
        ans_rev = str(ans)[::-1]
        ans_val_str = str(ans)
        if ans < 0:
            # For negative numbers, the reversed string in the scratchpad is the complement part
            ans_rev = "".join(tens_comp_digits) # Use the complement as the revans
            ans_val_str = ans_rev[::-1] # Use unreversed complement for next step
            
        return final_scratchpad, ans_rev, ans_val_str, str(ans)

def generate_example():
    # 50% chance of 2 numbers or 3 numbers
    is_3_number = random.random() < 0.5
    
    # Generate random lengths from 1 to 22
    l1 = random.randint(1, 22)
    l2 = random.randint(1, 22)
    
    n1 = generate_number(l1)
    n2 = generate_number(l2)
    op1 = random.choice(['+', '-'])
    
    if is_3_number:
        l3 = random.randint(1, 22)
        n3 = generate_number(l3)
        op2 = random.choice(['+', '-'])
        
        # Prompt: num1 num2+num3-?
        prompt = f"[BOS]{n1} {n2}{op1}{n3}{op2}?"
        
        # Reversal: [REV]revnum1 revnum2+revnum3-=
        rev1 = reverse_string(n1)
        rev2 = reverse_string(n2)
        rev3 = reverse_string(n3)
        reversal = f"[REV]{rev1} {rev2}{op1}{rev3}{op2}="
        
        # Math 1
        math1_steps, rev_ans1, ans1_val, ans1_true = generate_math_steps(n1, n2, op1)
        math1 = f"[MATH]{math1_steps}[REV]{rev_ans1}"
        
        # Transition: Transition shouldn't contain rev_ans1 as it is already in math1!
        transition = f" {rev3}{op2}="
        
        # Math 2
        math2_steps, rev_ans2, ans2_val, ans2_true = generate_math_steps(ans1_val, n3, op2)
        math2 = f"[MATH]{math2_steps}={rev_ans2}"
        
        # Final Answer
        ans_line = f"[ANS]{ans2_true}[EOS]"
        
        full_line = f"{prompt}{reversal}{math1}{transition}{math2}{ans_line}"
        return full_line, True, len(prompt)
    else:
        # 2 numbers
        prompt = f"[BOS]{n1} {n2}{op1}?"
        
        # Reversal
        rev1 = reverse_string(n1)
        rev2 = reverse_string(n2)
        reversal = f"[REV]{rev1} {rev2}{op1}="
        
        math1_steps, rev_ans1, ans1_val, ans1_true = generate_math_steps(n1, n2, op1)
        math1 = f"[MATH]{math1_steps}={rev_ans1}"
        
        ans_line = f"[ANS]{ans1_true}[EOS]"
        
        full_line = f"{prompt}{reversal}{math1}{ans_line}"
        return full_line, False, len(prompt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--split_type", action="store_true", help="Split into 2num and 3num files")
    parser.add_argument("--split_length", action="store_true", help="Split into le25 and gt25 files")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Base files
    files_to_open = {
        "train": os.path.join(args.output_dir, "rpn3_train.txt"),
        "val": os.path.join(args.output_dir, "rpn3_val.txt")
    }
    
    if args.split_type:
        files_to_open.update({
            "train_2num": os.path.join(args.output_dir, "rpn3_2num_train.txt"),
            "train_3num": os.path.join(args.output_dir, "rpn3_3num_train.txt"),
            "val_2num": os.path.join(args.output_dir, "rpn3_2num_val.txt"),
            "val_3num": os.path.join(args.output_dir, "rpn3_3num_val.txt")
        })
        
    if args.split_length:
        files_to_open.update({
            "train_le25": os.path.join(args.output_dir, "rpn3_le25_train.txt"),
            "train_gt25": os.path.join(args.output_dir, "rpn3_gt25_train.txt"),
            "val_le25": os.path.join(args.output_dir, "rpn3_le25_val.txt"),
            "val_gt25": os.path.join(args.output_dir, "rpn3_gt25_val.txt")
        })
        
    if args.split_type and args.split_length:
        files_to_open.update({
            "train_3num_le25": os.path.join(args.output_dir, "rpn3_3num_le25_train.txt"),
            "train_3num_gt25": os.path.join(args.output_dir, "rpn3_3num_gt25_train.txt"),
            "val_3num_le25": os.path.join(args.output_dir, "rpn3_3num_le25_val.txt"),
            "val_3num_gt25": os.path.join(args.output_dir, "rpn3_3num_gt25_val.txt")
        })
        
    # Open all needed files
    handles = {}
    for key, path in files_to_open.items():
        handles[key] = open(path, "w", encoding="utf-8")
        
    print(f"Generating {args.samples} samples...")
    
    train_pct = 0.9
    num_train = int(args.samples * train_pct)
    
    for i in range(args.samples):
        example, is_3, length = generate_example()
        is_train = i < num_train
        
        # Write to base files
        prefix = "train" if is_train else "val"
        handles[prefix].write(example + "\n")
        
        # Write to type files
        if args.split_type:
            type_suffix = "3num" if is_3 else "2num"
            handles[f"{prefix}_{type_suffix}"].write(example + "\n")
            
        # Write to length files
        if args.split_length:
            len_suffix = "le25" if length <= 25 else "gt25"
            handles[f"{prefix}_{len_suffix}"].write(example + "\n")
            
        # Write to specific 3num + length files
        if args.split_type and args.split_length and is_3:
            len_suffix = "le25" if length <= 25 else "gt25"
            handles[f"{prefix}_3num_{len_suffix}"].write(example + "\n")
            
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i+1}/{args.samples}...")
            
    # Close all files
    for handle in handles.values():
        handle.close()
        
    print(f"Done! Datasets saved to {args.output_dir}")

if __name__ == "__main__":
    main()
