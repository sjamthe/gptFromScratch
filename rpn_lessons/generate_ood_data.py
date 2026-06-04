import os
import sys
import random

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_data import generate_number, reverse_string, generate_math_steps, wrap_num
from utils import RPNTokenizer

def main():
    tokenizer = RPNTokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rpn-tokenizer.json"))
    ood_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/ood")
    os.makedirs(ood_dir, exist_ok=True)
    
    num_samples = 100
    max_token_len = 380
    
    print("Generating gradual OOD datasets under rpn_lessons/data/ood/...")
    
    # ----------------------------------------------------
    # 1. Lesson 1: Reversal gradual digit increments (23 to 30)
    # ----------------------------------------------------
    for digits in range(23, 31):
        filename = os.path.join(ood_dir, f"lesson1_ood_digits_{digits}.txt")
        count = 0
        with open(filename, "w", encoding="utf-8") as f:
            while count < num_samples:
                num = generate_number(digits)
                is_negative = random.random() < 0.5 and num != "0"
                if is_negative:
                    sample = f"[REV]-<num>{num}</num>[ANS]-<num>{num[::-1]}</num>[EOS]\n"
                else:
                    sample = f"[REV]<num>{num}</num>[ANS]<num>{num[::-1]}</num>[EOS]\n"
                
                # Verify token limit
                if len(tokenizer.encode(sample)) <= max_token_len:
                    f.write(sample)
                    count += 1
        print(f"  Generated {filename}")
        
    # ----------------------------------------------------
    # 2. Lesson 2: Multi-operand Reversal
    # ----------------------------------------------------
    # OOD Digits (10 to 13), using 2 operands
    for digits in range(10, 14):
        filename = os.path.join(ood_dir, f"lesson2_ood_digits_{digits}.txt")
        count = 0
        with open(filename, "w", encoding="utf-8") as f:
            while count < num_samples:
                numbers = []
                for _ in range(2):
                    n = generate_number(digits)
                    if random.random() < 0.5 and n != "0":
                        n = "-" + n
                    numbers.append(n)
                ops = [random.choice(['+', '-'])]
                
                prompt = f"[BOS]{wrap_num(numbers[0])} {wrap_num(numbers[1])}{ops[0]}[REV]"
                target = f"{wrap_num(reverse_string(numbers[0]))}[SEP]{wrap_num(reverse_string(numbers[1]))}{ops[0]}[MATH]"
                sample = f"{prompt}{target}\n"
                
                if len(tokenizer.encode(sample)) <= max_token_len:
                    f.write(sample)
                    count += 1
        print(f"  Generated {filename}")
        
    # OOD Operands (7 to 10), using 4 digits
    for operands in range(7, 11):
        filename = os.path.join(ood_dir, f"lesson2_ood_operands_{operands}.txt")
        count = 0
        with open(filename, "w", encoding="utf-8") as f:
            while count < num_samples:
                numbers = []
                for _ in range(operands):
                    n = generate_number(4)
                    if random.random() < 0.5 and n != "0":
                        n = "-" + n
                    numbers.append(n)
                ops = [random.choice(['+', '-']) for _ in range(operands - 1)]
                
                prompt_parts = [f"[BOS]{wrap_num(numbers[0])}"]
                for i in range(1, operands):
                    prompt_parts.append(f"{wrap_num(numbers[i])}{ops[i-1]}")
                prompt = " ".join(prompt_parts) + "[REV]"
                
                rev_nums = [reverse_string(n) for n in numbers]
                target_parts = [wrap_num(rev_nums[0])]
                for i in range(1, operands):
                    target_parts.append(f"{wrap_num(rev_nums[i])}{ops[i-1]}")
                target = "[SEP]".join(target_parts) + "[MATH]"
                sample = f"{prompt}{target}\n"
                
                if len(tokenizer.encode(sample)) <= max_token_len:
                    f.write(sample)
                    count += 1
        print(f"  Generated {filename}")
        
    # ----------------------------------------------------
    # 3. Lesson 3: Step-by-Step Math
    # ----------------------------------------------------
    # OOD Digits (10 to 13), using 2 operands
    for digits in range(10, 14):
        filename = os.path.join(ood_dir, f"lesson3_ood_digits_{digits}.txt")
        count = 0
        with open(filename, "w", encoding="utf-8") as f:
            while count < num_samples:
                numbers = []
                for _ in range(2):
                    n = generate_number(digits)
                    if random.random() < 0.5 and n != "0":
                        n = "-" + n
                    numbers.append(n)
                op = random.choice(['+', '-'])
                
                rev_n1 = reverse_string(numbers[0])
                rev_n2 = reverse_string(numbers[1])
                prompt = f"[REV]{wrap_num(rev_n1)}[SEP]{wrap_num(rev_n2)}{op}[MATH]"
                
                steps_str, _, _ = generate_math_steps(numbers[0], numbers[1], op)
                target = f"{steps_str}[REV]"
                sample = f"{prompt}{target}\n"
                
                if len(tokenizer.encode(sample)) <= max_token_len:
                    f.write(sample)
                    count += 1
        print(f"  Generated {filename}")
        
    # OOD Operands (7 to 10), using 4 digits
    for operands in range(7, 11):
        filename = os.path.join(ood_dir, f"lesson3_ood_operands_{operands}.txt")
        count = 0
        with open(filename, "w", encoding="utf-8") as f:
            while count < num_samples:
                numbers = []
                for _ in range(operands):
                    n = generate_number(4)
                    if random.random() < 0.5 and n != "0":
                        n = "-" + n
                    numbers.append(n)
                ops = [random.choice(['+', '-']) for _ in range(operands - 1)]
                rev_nums = [reverse_string(n) for n in numbers]
                
                prompt_parts = [wrap_num(rev_nums[0])]
                for i in range(1, operands):
                    prompt_parts.append(f"{wrap_num(rev_nums[i])}{ops[i-1]}")
                prompt = "[REV]" + "[SEP]".join(prompt_parts) + "[MATH]"
                
                steps_str, _, _ = generate_math_steps(numbers[0], numbers[1], ops[0])
                
                target = steps_str
                tail_parts = []
                for i in range(2, operands):
                    tail_parts.append(f"{wrap_num(rev_nums[i])}{ops[i-1]}")
                target += "[SEP]" + "[SEP]".join(tail_parts) + "[REV]"
                sample = f"{prompt}{target}\n"
                
                if len(tokenizer.encode(sample)) <= max_token_len:
                    f.write(sample)
                    count += 1
        print(f"  Generated {filename}")
        
    # ----------------------------------------------------
    # 4. Lesson 4: Result Reversal & Phase Transition
    # ----------------------------------------------------
    # OOD Digits (10 to 13), using 2 operands
    for digits in range(10, 14):
        filename = os.path.join(ood_dir, f"lesson4_ood_digits_{digits}.txt")
        count = 0
        with open(filename, "w", encoding="utf-8") as f:
            while count < num_samples:
                numbers = []
                for _ in range(2):
                    n = generate_number(digits)
                    if random.random() < 0.5 and n != "0":
                        n = "-" + n
                    numbers.append(n)
                op = random.choice(['+', '-'])
                
                steps_str, ans_rev, _ = generate_math_steps(numbers[0], numbers[1], op)
                prompt = f"[MATH]{steps_str}[REV]"
                target = f"{wrap_num(ans_rev)}[ANS]"
                sample = f"{prompt}{target}\n"
                
                if len(tokenizer.encode(sample)) <= max_token_len:
                    f.write(sample)
                    count += 1
        print(f"  Generated {filename}")
        
    # OOD Operands (7 to 10), using 4 digits
    for operands in range(7, 11):
        filename = os.path.join(ood_dir, f"lesson4_ood_operands_{operands}.txt")
        count = 0
        with open(filename, "w", encoding="utf-8") as f:
            while count < num_samples:
                numbers = []
                for _ in range(operands):
                    n = generate_number(4)
                    if random.random() < 0.5 and n != "0":
                        n = "-" + n
                    numbers.append(n)
                ops = [random.choice(['+', '-']) for _ in range(operands - 1)]
                rev_nums = [reverse_string(n) for n in numbers]
                
                steps_str, ans_rev, _ = generate_math_steps(numbers[0], numbers[1], ops[0])
                
                prompt = "[MATH]" + steps_str
                tail_parts = []
                for i in range(2, operands):
                    tail_parts.append(f"{wrap_num(rev_nums[i])}{ops[i-1]}")
                prompt += "[SEP]" + "[SEP]".join(tail_parts) + "[REV]"
                
                target = wrap_num(ans_rev) + "[SEP]" + "[SEP]".join(tail_parts) + "[MATH]"
                sample = f"{prompt}{target}\n"
                
                if len(tokenizer.encode(sample)) <= max_token_len:
                    f.write(sample)
                    count += 1
        print(f"  Generated {filename}")
        
    # ----------------------------------------------------
    # 5. End-to-End Joint Evaluation (State Machine)
    # ----------------------------------------------------
    # OOD Digits (10 to 12), using 3 operands
    for digits in range(10, 13):
        filename = os.path.join(ood_dir, f"e2e_ood_digits_{digits}.txt")
        count = 0
        with open(filename, "w", encoding="utf-8") as f:
            while count < num_samples:
                numbers = []
                for _ in range(3):
                    n = generate_number(digits)
                    if random.random() < 0.5 and n != "0":
                        n = "-" + n
                    numbers.append(n)
                ops = [random.choice(['+', '-']) for _ in range(2)]
                
                stack = [int(numbers[0])]
                for i in range(1, 3):
                    op = ops[i-1]
                    val = int(numbers[i])
                    if op == '+':
                         res = stack.pop() + val
                    else:
                         res = stack.pop() - val
                    stack.append(res)
                gt_ans = stack[0]
                
                prompt = f"[BOS]{wrap_num(numbers[0])} {wrap_num(numbers[1])}{ops[0]} {wrap_num(numbers[2])}{ops[1]}[REV]"
                sample = f"{prompt}|||{gt_ans}\n" # delimiter for parsing in verification
                
                if len(tokenizer.encode(prompt)) <= max_token_len:
                    f.write(sample)
                    count += 1
        print(f"  Generated {filename}")
        
    # OOD Operands (7 to 10), using 4 digits
    for operands in range(7, 11):
        filename = os.path.join(ood_dir, f"e2e_ood_operands_{operands}.txt")
        count = 0
        with open(filename, "w", encoding="utf-8") as f:
            while count < num_samples:
                numbers = []
                for _ in range(operands):
                    n = generate_number(4)
                    if random.random() < 0.5 and n != "0":
                        n = "-" + n
                    numbers.append(n)
                ops = [random.choice(['+', '-']) for _ in range(operands - 1)]
                
                stack = [int(numbers[0])]
                for i in range(1, operands):
                    op = ops[i-1]
                    val = int(numbers[i])
                    if op == '+':
                         res = stack.pop() + val
                    else:
                         res = stack.pop() - val
                    stack.append(res)
                gt_ans = stack[0]
                
                prompt_parts = [f"[BOS]{wrap_num(numbers[0])}"]
                for i in range(1, operands):
                    prompt_parts.append(f"{wrap_num(numbers[i])}{ops[i-1]}")
                prompt = " ".join(prompt_parts) + "[REV]"
                sample = f"{prompt}|||{gt_ans}\n"
                
                if len(tokenizer.encode(prompt)) <= max_token_len:
                    f.write(sample)
                    count += 1
        print(f"  Generated {filename}")
        
    print("Done generating OOD datasets.")

if __name__ == "__main__":
    main()
