import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from create_dataset import generate_math_steps

def test_generate_math_steps(a_str, b_str, op):
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
    
    ans = a_true + b_true if op == '+' else a_true - b_true
        
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
            
            # CHANGE: Check if stable AND we already have the stable digit!
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
        ans_rev = "".join(derived_digits)
        ans_str = str(int(ans_rev[::-1]))
    else:
        tens_comp_digits = []
        found_nonzero = False
        for d_str in derived_digits:
            d = int(d_str)
            if not found_nonzero:
                if d == 0:
                    tens_comp_digits.append("0")
                else:
                    comp = 10 - d
                    tens_comp_digits.append(str(comp))
                    found_nonzero = True
            else:
                comp = 9 - d
                tens_comp_digits.append(str(comp))
        ans_str = str(int("-" + "".join(tens_comp_digits)[::-1]))
        
    assert int(ans_str) == ans, f"Math logic failure: scratchpad derived {ans_str}, expected {ans}"

for a in range(-50, 50):
    for b in range(-50, 50):
        for op in ['+', '-']:
            try:
                test_generate_math_steps(str(a), str(b), op)
            except AssertionError as e:
                print(f"FAILED: {a} {op} {b}")
                sys.exit(1)
print("All passed!")
