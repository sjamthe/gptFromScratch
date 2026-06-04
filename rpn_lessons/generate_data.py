import os
import random
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
    if s.startswith('-'):
        return '-' + s[1:][::-1]
    return s[::-1]

def generate_math_steps(a_str, b_str, op):
    a_true = int(a_str)
    b_true = int(b_str)
    is_a_negative = a_true < 0
    is_b_negative = b_true < 0
    
    a_mag_rev = a_str.lstrip('-')[::-1]
    b_mag_rev = b_str.lstrip('-')[::-1]
    
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
        
    assert int(ans_str) == ans, f"Math logic failure: scratchpad derived {ans_str}, expected {ans}"
    return steps_str, ans_rev, ans_str

# --- Helper to wrap magnitudes ---
def wrap_num(s):
    if s.startswith('-'):
        return '-<num>' + s[1:] + '</num>'
    return '<num>' + s + '</num>'

# --- Lesson generators ---

def generate_lesson1_sample(max_digits=22):
    L = random.randint(1, max_digits)
    num = generate_number(L)
    is_negative = random.random() < 0.5 and num != "0"
    
    if is_negative:
        return f"[REV]-<num>{num}</num>[ANS]-<num>{num[::-1]}</num>[EOS]"
    else:
        return f"[REV]<num>{num}</num>[ANS]<num>{num[::-1]}</num>[EOS]"

def generate_lesson2_sample(max_numbers=6, max_digits=9):
    N = random.randint(1, max_numbers)
    numbers = []
    for _ in range(N):
        L = random.randint(1, max_digits)
        n = generate_number(L)
        if random.random() < 0.5 and n != "0":
            n = "-" + n
        numbers.append(n)
        
    ops = [random.choice(['+', '-']) for _ in range(N - 1)]
    
    # Prompt construction
    prompt_parts = [f"[BOS]{wrap_num(numbers[0])}"]
    for i in range(1, N):
        prompt_parts.append(f"{wrap_num(numbers[i])}{ops[i-1]}")
    prompt = " ".join(prompt_parts) + "[REV]"
    
    # Target construction
    rev_nums = [reverse_string(n) for n in numbers]
    target_parts = [wrap_num(rev_nums[0])]
    for i in range(1, N):
        target_parts.append(f"{wrap_num(rev_nums[i])}{ops[i-1]}")
    target = "[SEP]".join(target_parts) + "[MATH]"
    
    return f"{prompt}{target}"

def generate_lesson3_sample(max_numbers=6, max_digits=9):
    # Lesson 3 requires at least 2 numbers to perform math
    N = random.randint(2, max_numbers)
    numbers = []
    for _ in range(N):
        L = random.randint(1, max_digits)
        n = generate_number(L)
        if random.random() < 0.5 and n != "0":
            n = "-" + n
        numbers.append(n)
        
    ops = [random.choice(['+', '-']) for _ in range(N - 1)]
    rev_nums = [reverse_string(n) for n in numbers]
    
    # Input prompt: [REV]rev_n1[SEP]rev_n2<op1>[SEP]...[MATH]
    prompt_parts = [wrap_num(rev_nums[0])]
    for i in range(1, N):
        prompt_parts.append(f"{wrap_num(rev_nums[i])}{ops[i-1]}")
    prompt = "[REV]" + "[SEP]".join(prompt_parts) + "[MATH]"
    
    # Perform math steps on the first two numbers
    steps_str, ans_rev, ans_str = generate_math_steps(numbers[0], numbers[1], ops[0])
    
    # Target: steps_str [SEP] rev_n3<op2>... [REV]
    target = steps_str
    if N > 2:
        tail_parts = []
        for i in range(2, N):
            tail_parts.append(f"{wrap_num(rev_nums[i])}{ops[i-1]}")
        target += "[SEP]" + "[SEP]".join(tail_parts)
    target += "[REV]"
    
    return f"{prompt}{target}"

def generate_lesson4_sample(max_numbers=6, max_digits=9):
    # Lesson 4 takes MATH steps + tail and outputs answer + tail + transition
    N = random.randint(2, max_numbers)
    numbers = []
    for _ in range(N):
        L = random.randint(1, max_digits)
        n = generate_number(L)
        if random.random() < 0.5 and n != "0":
            n = "-" + n
        numbers.append(n)
        
    ops = [random.choice(['+', '-']) for _ in range(N - 1)]
    rev_nums = [reverse_string(n) for n in numbers]
    
    # Generate math steps for numbers[0] and numbers[1]
    steps_str, ans_rev, ans_str = generate_math_steps(numbers[0], numbers[1], ops[0])
    
    # Input prompt: [MATH]steps_str[SEP]rev_n3<op2>...[REV]
    prompt = "[MATH]" + steps_str
    if N > 2:
        tail_parts = []
        for i in range(2, N):
            tail_parts.append(f"{wrap_num(rev_nums[i])}{ops[i-1]}")
        prompt += "[SEP]" + "[SEP]".join(tail_parts)
    prompt += "[REV]"
    
    # Target: ans_rev[SEP]rev_n3<op2>...[MATH] (or [ANS] if N=2)
    target = wrap_num(ans_rev)
    if N > 2:
        tail_parts = []
        for i in range(2, N):
            tail_parts.append(f"{wrap_num(rev_nums[i])}{ops[i-1]}")
        target += "[SEP]" + "[SEP]".join(tail_parts) + "[MATH]"
    else:
        target += "[ANS]"
        
    return f"{prompt}{target}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_samples", type=int, default=100000)
    parser.add_argument("--val_samples", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="rpn_lessons/data")
    parser.add_argument("--max_token_len", type=int, default=380)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = RPNTokenizer("rpn_lessons/rpn-tokenizer.json")
    
    lessons = {
        1: generate_lesson1_sample,
        2: generate_lesson2_sample,
        3: generate_lesson3_sample,
        4: generate_lesson4_sample
    }
    
    for lesson_idx, generator in lessons.items():
        print(f"Generating data for Lesson {lesson_idx}...")
        for split, num_samples in [("train", args.train_samples), ("val", args.val_samples)]:
            output_path = os.path.join(args.output_dir, f"lesson{lesson_idx}_{split}.txt")
            count = 0
            with open(output_path, "w", encoding="utf-8") as f:
                while count < num_samples:
                    r = random.random()
                    if lesson_idx == 1:
                        sampled_lesson = 1
                    elif lesson_idx == 2:
                        sampled_lesson = 2 if r < 0.85 else 1
                    elif lesson_idx == 3:
                        if r < 0.80:
                            sampled_lesson = 3
                        elif r < 0.90:
                            sampled_lesson = 2
                        else:
                            sampled_lesson = 1
                    elif lesson_idx == 4:
                        if r < 0.80:
                            sampled_lesson = 4
                        elif r < 0.90:
                            sampled_lesson = 3
                        elif r < 0.95:
                            sampled_lesson = 2
                        else:
                            sampled_lesson = 1
                            
                    sample = lessons[sampled_lesson]()
                    tokens = tokenizer.encode(sample + "\n")
                    if len(tokens) <= args.max_token_len:
                        f.write(sample + "\n")
                        count += 1
                        if count % 20000 == 0:
                            print(f"  Lesson {lesson_idx} {split}: {count}/{num_samples} generated.")
                    # If it exceeds, we silently loop and generate another one
            print(f"Finished Lesson {lesson_idx} {split} -> {output_path}")

if __name__ == "__main__":
    main()
