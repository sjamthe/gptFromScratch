import torch
from torch.utils.data import Dataset
import random
from typing import Tuple
import itertools

class RPNDataset(Dataset):
    def __init__(self, num_samples=-1, max_operands=2, operations=('+', '-'), 
                 max_number=100, file_path=None, tokenizer=None, max_seq_len=256):
        """
        Dataset class for RPN mathematical expressions.
        """
        self.examples = []
        
        if file_path:
            self._load_from_file(file_path)
        else:
            self._generate_random_scratchpad_examples(num_samples, operations, max_number, tokenizer, max_seq_len)
    
    def _load_from_file(self, file_path: str):
        """Load RPN expressions and answers from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    parts = line.strip().split('=')
                    if len(parts) == 2:
                        rpn_expr = parts[0].strip()
                        answer = parts[1].strip()
                        self.examples.append((rpn_expr, answer))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        rpn_expr, answer = self.examples[index]      
        example = {
            'rpn': rpn_expr,
            'answer': answer
        }
        return example
        
    def _generate_scratchpad(self, a_str: str, b_str: str, op: str) -> str:
        # reverse the numbers for the scratchpad    
        a_rev, b_rev = a_str[::-1], b_str[::-1]
        max_len = max(len(a_rev), len(b_rev))
        
        steps = []
        carry = 0
        
        if op == '+':
            prefix = f"<{a_rev} {b_rev}{op}=:"
            derived_digits = []
            for i in range(max_len):
                d_a = int(a_rev[i]) if i < len(a_rev) else 0
                d_b = int(b_rev[i]) if i < len(b_rev) else 0
                res = d_a + d_b + carry
                new_carry = res // 10
                digit = res % 10
                steps.append(f"{d_a}+{d_b}+{carry}={digit}")
                derived_digits.append(str(digit))
                carry = new_carry
                
            if carry > 0:
                steps.append(f"0+0+{carry}={carry}")
                derived_digits.append(str(carry))
                
            ans = int(a_str) + int(b_str)
            ans_rev = str(ans)[::-1]
            ans_str = str(ans) 
            scratchpad_math = ":".join(steps)
            return f"{prefix}{scratchpad_math}:{ans_rev}>{ans_str}"
            
        elif op == '-':
            prefix = f"<{a_rev} {b_rev}{op}=:"
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
                # Positive answer!
                steps_part2 = [f"[BORROW]0|+", "".join(derived_digits)]
                final_scratchpad = ":".join([scratchpad_math] + steps_part2)
            else:
                # Negative answer => Ten's Complement Pass!
                steps_part2 = [f"[BORROW]1|-"]
                tens_comp_digits = []
                found_nonzero = False
                for d_str in derived_digits: # Going LSB to MSB
                    d = int(d_str)
                    if not found_nonzero:
                        if d == 0:
                            steps_part2.append(f"[PASS]0=0")
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
            ans_str = str(ans)
            return f"{prefix}{final_scratchpad}>{ans_str}"
        return ""

    def _generate_random_scratchpad_examples(self, num_samples: int, operations: Tuple[str, ...], max_number: int, tokenizer, max_seq_len):
        def get_uniform_length_number(max_num):
            num_digits = random.randint(1, len(str(max_num)))
            curr_max = min(max_num, (10 ** num_digits) - 1)
            curr_min = 0 if num_digits == 1 else (10 ** (num_digits - 1))
            return random.randint(curr_min, curr_max)
            
        def rs0(): return " " * random.randint(0, 2)
        def lpad0(): return "0" * random.randint(0, 2)
        
        while len(self.examples) < num_samples:
            a = get_uniform_length_number(max_number)
            b = get_uniform_length_number(max_number)
            op = random.choice(operations)
            if op == '/' and b == 0: continue
                
            str_a = str(a)
            str_b = str(b)
            prompt_str = f"{rs0()}{str_a} {rs0()}{str_b}{rs0()}{op}{rs0()}={rs0()}"
            result_str = self._generate_scratchpad(str_a, str_b, op)
            full_line = f"{prompt_str}{result_str}"

            if tokenizer is not None:
                tokens = tokenizer.encode(full_line)
                if len(tokens) > max_seq_len: continue
            
            self.examples.append((prompt_str, result_str))
            if len(self.examples) % 100000 == 0:
                print(f"Generated {len(self.examples)}/{num_samples} samples...")

if __name__ == "__main__":
    from utils import RPNTokenizer
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

    max_number = (10 ** 22) - 1
    num_samples = 6_000_000
    print(f"Initializing full {num_samples} mixed-scale dataset...")
    dataset = RPNDataset(
        num_samples=num_samples,
        max_operands=2,
        operations=('+','-',),
        max_number=max_number,
        tokenizer=tokenizer,
        max_seq_len=256
    )
    
    print("Generating Training Data (Mixed-Scale 1-22 digits)...")
    file_path_prefix = "rpn_llm/data/RPNData-mixed-1-22_tens_comp"
    train_f = open(file_path_prefix + "_train.txt", 'w', encoding='utf-8')
    val_f = open(file_path_prefix + "_val.txt", 'w', encoding='utf-8')
    test_f = open(file_path_prefix + "_test.txt", 'w', encoding='utf-8')

    train_pct, val_pct = 80, 10
    for rpn_expr, answer in dataset.examples:
        rand_num = random.randint(0, 99)
        line = f"{rpn_expr}{answer}\n"
        if rand_num < train_pct: train_f.write(line)
        elif rand_num < train_pct + val_pct: val_f.write(line)
        else: test_f.write(line)

    train_f.close(); val_f.close(); test_f.close()
    print("Training data generation complete.")