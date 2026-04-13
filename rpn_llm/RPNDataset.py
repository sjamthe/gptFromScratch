import torch
from torch.utils.data import Dataset
import random
from typing import Tuple
import itertools

class RPNDataset(Dataset):
    def __init__(self, num_samples=-1, max_operands=5, operations=('+',), 
                 max_number=100, file_path=None):
        """
        Dataset class for RPN mathematical expressions.
        
        Args:
            num_samples: Number of samples to generate (if not loading from file)
            max_operands: Maximum number of operands in a generated expression
            operations: Tuple of operations to use ('+', '-', '*', '/')
            max_number: Maximum value for generated numbers
            file_path: Optional path to load existing data from
        """
        self.examples = []
        
        if file_path:
            self._load_from_file(file_path)
        else:
            self._generate_random_scratchpad_examples(num_samples, operations, max_number)
    
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
        # Create example
        example = {
            'rpn': rpn_expr,
            'answer': answer
        }
        
        return example
        
    # Simple algorithm to create valid RPN
    def _calc_rpn_expr(self, operands, ops):
        """Generate RPN expressions and calculate their answers."""
        operators = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b
        }
        
        # Create RPN expression
        rpn_tokens = []
        stack = []
            
        num_operands = len(operands)
        for i in range(num_operands + len(ops)):
            if i < num_operands:
                # Add an operand
                rpn_tokens.append(str(operands[i]))
                stack.append(operands[i])
            
            # If we have at least 2 values on the stack, we can apply an operation
            if len(stack) >= 2 and ops:
                # Pop the operation to use
                op = ops.pop(0)
                rpn_tokens.append(op)
                
                # Calculate the result of this operation
                b = stack.pop()
                a = stack.pop()
                
                # Skip division by zero or operations that lead to negative results
                if op == '/' and b == 0:
                    continue
                
                result = operators[op](a, b)
                stack.append(result)
        
        # Create the expression and result
        rpn_expr = ' '.join(rpn_tokens)
        result = stack[0]
        # Remove decimal point for whole numbers
        if result == int(result):
            answer = str(int(result))
        else:
            answer = str(result)

        return rpn_expr, answer
        
    def _generate_examples(self, num_samples: int, max_operands: int, 
                          operations: Tuple[str, ...], max_number: int):
           
        for _ in range(num_samples):
            # Decide number of operands (at least 2)
            num_operands = random.randint(2, max_operands)
            
            # Phase 6: Uniform dimensional bias completely erasing operand scarcity!
            operands = []
            for _ in range(num_operands):
                num_digits = random.randint(1, len(str(max_number)))
                curr_max = min(max_number, (10 ** num_digits) - 1)
                curr_min = 0 if num_digits == 1 else (10 ** (num_digits - 1))
                operands.append(random.randint(curr_min, curr_max))
            
            # Generate operations (need num_operands-1 operations)
            ops = [random.choice(operations) for _ in range(num_operands-1)]
            
            rpn_expr, answer = self._calc_rpn_expr(operands, ops)
            self.examples.append((rpn_expr, answer))
    
    """
    Step 1: reverse both numbers in scratchpad. for example "123 45 + = " becomes "<321 54 + = :>"
    Step 2: perform the operation digit by digit on scratchpad "<321 54 + = : 3 + 5 + 0 = 8 : 2 + 4 + 0 = 6 : 1 + 0 + 0 = 1 :>"
    Step 3: Add the final result to the scratchpad and reverse it outside a final answer"<321 54 + = : 3 + 5 + 0 = 8 : 2 + 4 + 0 = 6 : 1 + 0 + 0 = 1 : 861 : > 168"
    Ste 4: Sanity check. Do real calc to check the answer we got before we return the scratchpad and answer
    """
    def _generate_scratchpad(self, a: int, b: int, op: str) -> str:
        # reverse the numbers for the scratchpad    
        a_str, b_str = str(a)[::-1], str(b)[::-1]
        max_len = max(len(a_str), len(b_str))
        
        steps = []
        carry = 0
        
        if op == '+':
            prefix = f"<{a_str} {b_str}+=:"
            derived_digits = []
            for i in range(max_len):
                d_a = int(a_str[i]) if i < len(a_str) else 0
                d_b = int(b_str[i]) if i < len(b_str) else 0
                res = d_a + d_b + carry
                new_carry = res // 10
                digit = res % 10
                steps.append(f"{d_a}+{d_b}+{carry}={digit}")
                derived_digits.append(str(digit))
                carry = new_carry
                
            if carry > 0:
                steps.append(f"0+0+{carry}={carry}")
                derived_digits.append(str(carry))
                
            ans = a + b
            
            # Sanity check digit aggregation correctly resolves structural bounds matching true answer algorithm!
            derived_ans_str = "".join(derived_digits)[::-1].lstrip('0') or '0'
            assert derived_ans_str == str(ans), f"Addition derived structural output {derived_ans_str} != True Math {ans}!"

            ans_rev = str(ans)[::-1]
            ans_str = str(ans) 
            scratchpad_math = ":".join(steps)
            return f"{prefix}{scratchpad_math}:{ans_rev}>{ans_str}"
            
        elif op == '-':
            # Ten's Complement Two-pass Subtraction completely eradicating zero-shot magnitude prediction!
            prefix = f"<{a_str} {b_str}-=:"
            derived_digits = []
            for i in range(max_len):
                d_a = int(a_str[i]) if i < len(a_str) else 0
                d_b = int(b_str[i]) if i < len(b_str) else 0
                
                res = (d_a - carry) - d_b
                if res < 0:
                    res += 10
                    new_carry = 1
                else:
                    new_carry = 0
                    
                steps.append(f"{d_a}-{d_b}-{carry}={res}")
                derived_digits.append(str(res))
                carry = new_carry
                
            ans = a - b
            ans_str = str(ans)
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
                
                # Check arithmetic truth
                derived_val = "".join(tens_comp_digits)[::-1].lstrip('0') or '0'
                assert derived_val == str(abs(ans)), f"Tens Comp {derived_val} != Abs({ans})"
                
                steps_part2.append("".join(tens_comp_digits))
                final_scratchpad = ":".join([scratchpad_math] + steps_part2)

            return f"{prefix}{final_scratchpad}>{ans_str}"
        else:
            return ""

    def _generate_random_scratchpad_examples(self, num_samples: int, operations: Tuple[str, ...], max_number: int):
        """Generate extremely sparse variable-length math combinations!"""
        
        generated_examples = set()
        
        # Helper to force uniform digit lengths gracefully overcoming population bloat!
        def get_uniform_length_number(max_num):
            num_digits = random.randint(1, len(str(max_num)))
            curr_max = min(max_num, (10 ** num_digits) - 1)
            curr_min = 0 if num_digits == 1 else (10 ** (num_digits - 1))
            return random.randint(curr_min, curr_max)
            
        def rs():
            # Inject jitter sequence whitespace randomness (1 to 3 spaces)
            return " " * random.randint(1, 3)
        
        while len(self.examples) < num_samples:
            a = get_uniform_length_number(max_number)
            b = get_uniform_length_number(max_number)
            op = random.choice(operations)
            
            # Skip division by zero
            if op == '/' and b == 0:
                continue
                
            result_str = self._generate_scratchpad(a, b, op)
            rpn_expr = f"{rs()}{a}{rs()}{b}{rs()}{op}"
            
            # Removed the unique_key constraint. Because 1-digit numbers only have ~200 combinations, 
            # the unique constraint immediately exhausted them and skipped all future short samples! 
            # Allowing natural duplicates balances the training loss geometry perfectly across lengths.
            self.examples.append((rpn_expr, result_str))
                    
# Example usage:
from tokenizers import Tokenizer

if __name__ == "__main__":

    max_number = 99999
    # Tagging Phase 10 model-driven scale mapping alignments
    file_path_prefix = "rpn_llm/data/RPNData-plusminus" + str(max_number) + "_tens_complement_compress"

    dataset = RPNDataset(
        num_samples=6000000,
        max_operands=2,
        operations=('+','-',),  # Start with just addition
        max_number=max_number
    )
    print("max_number=",max_number, "datalen=",len(dataset))

    # Save the dataset to files with train, validation, and test sets
    train_pct = 80
    val_pct = 10
    test_pct = 100 - train_pct - val_pct

    """
    Generate random number between 0-99 assign 0-79 to trian, 80-89 to val, 90-99 to test
    Open three files, write the data to the files
    """
    train_f = open(file_path_prefix + "_train.txt", 'w', encoding='utf-8')
    val_f = open(file_path_prefix + "_val.txt", 'w', encoding='utf-8')
    test_f = open(file_path_prefix + "_test.txt", 'w', encoding='utf-8')

    import collections
    train_length_counts = collections.defaultdict(int)

    for example in dataset:
        rand_num = random.randint(0, 99)
        def rs(): return " " * random.randint(1, 3)
        
        # Add random spaces ONLY before the `=` sign so the target deterministic format starts clean!
        eq_str = f"{rs()}="
        
        if rand_num < train_pct:
            prompt_str = example['rpn'] + eq_str
            train_length_counts[len(prompt_str[:prompt_str.find('=')+1])] += 1
            train_f.write(example['rpn'] + eq_str + example['answer'] + "\n")
        elif rand_num < train_pct + val_pct:
            val_f.write(example['rpn'] + eq_str + example['answer'] + "\n")
        else:
            test_f.write(example['rpn'] + eq_str + example['answer'] + "\n")

    train_f.close()
    val_f.close()
    test_f.close()

    print("Data saved to files")
    
    print("\n--- Training Set Prompt Length Distribution ---")
    print("Length | Count     | Percentage")
    total_train_items = sum(train_length_counts.values())
    for length in sorted(train_length_counts.keys()):
        count = train_length_counts[length]
        pct = (count / total_train_items) * 100 if total_train_items > 0 else 0
        print(f"{length:6d} | {count:<9d} | {pct:.2f}%")