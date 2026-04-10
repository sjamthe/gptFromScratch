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
    
    def _generate_scratchpad(self, a: int, b: int, op: str) -> str:
        # Inputs are ALREADY structurally reversed mapping characters
        a_str, b_str = str(a)[::-1], str(b)[::-1]
        max_len = max(len(a_str), len(b_str))
        
        steps = []
        carry = 0
        
        if op == '+':
            for i in range(max_len):
                d_a = int(a_str[i]) if i < len(a_str) else 0
                d_b = int(b_str[i]) if i < len(b_str) else 0
                res = d_a + d_b + carry
                new_carry = res // 10
                steps.append(f"{d_a} + {d_b} + {carry} = {res}")
                carry = new_carry
                
            ans = a + b
            ans_str = str(ans)[::-1] 
            scratchpad = " : ".join(steps)
            return f"< {scratchpad} > {ans_str}"
            
        elif op == '-':
            is_negative = a < b
            # Top is ALWAYS the structurally larger integer to evaluate clean sub-loops!
            top_str = b_str if is_negative else a_str
            bot_str = a_str if is_negative else b_str

            for i in range(max_len):
                d_t = int(top_str[i]) if i < len(top_str) else 0
                d_b = int(bot_str[i]) if i < len(bot_str) else 0
                
                res = (d_t - carry) - d_b
                if res < 0:
                    res += 10
                    new_carry = 1
                else:
                    new_carry = 0
                    
                steps.append(f"{d_t} - {d_b} - {carry} = {res}")
                carry = new_carry
                
            ans = a - b
            # Result naturally appends `-` to strings representing mathematically negative strings backwards.
            ans_str = str(abs(ans))[::-1] + "-" if ans < 0 else str(ans)[::-1]
            scratchpad = " : ".join(steps)
            
            prefix = "< - : " if is_negative else "< "
            return f"{prefix}{scratchpad} > {ans_str}"
        else:
            return ""

    def _generate_random_scratchpad_examples(self, num_samples: int, operations: Tuple[str, ...], max_number: int):
        """Generate extremely sparse variable-length math combinations!"""
        
        generated_examples = set()
        
        while len(self.examples) < num_samples:
            a = random.randint(0, max_number)
            b = random.randint(0, max_number)
            op = random.choice(operations)
            
            # Skip division by zero
            if op == '/' and b == 0:
                continue
                
            result_str = self._generate_scratchpad(a, b, op)
            a_rev, b_rev = str(a)[::-1], str(b)[::-1]
            rpn_expr = f"{a_rev} {b_rev} {op}"
            
            example_key = (rpn_expr, result_str)
            if example_key not in generated_examples:
                generated_examples.add(example_key)
                self.examples.append(example_key)
                    
# Example usage:
from tokenizers import Tokenizer

if __name__ == "__main__":

    max_number = 99999
    # Tagging Phase 8 fully reversed scale-invariant baseline payload limits natively!
    file_path_prefix = "data/RPNData-plusminus" + str(max_number) + "_fully_reversed_nopad"

    dataset = RPNDataset(
        num_samples=2000000,
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
    train_f = open(file_path_prefix + "-_train.txt", 'w', encoding='utf-8')
    val_f = open(file_path_prefix + "-_val.txt", 'w', encoding='utf-8')
    test_f = open(file_path_prefix + "-_test.txt", 'w', encoding='utf-8')

    for example in dataset:
        rand_num = random.randint(0, 99)
        if rand_num < train_pct:
            train_f.write(example['rpn'] + " = " + example['answer'] + "\n")
        elif rand_num < train_pct + val_pct:
            val_f.write(example['rpn'] + " = " + example['answer'] + "\n")
        else:
            test_f.write(example['rpn'] + " = " + example['answer'] + "\n")

    train_f.close()
    val_f.close()
    test_f.close()

    print("Data saved to files")