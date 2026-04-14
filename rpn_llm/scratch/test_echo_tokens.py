from utils import RPNTokenizer
import random

tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

def estimate_tokens(a_val, b_val, op):
    a_str_orig = str(a_val)
    b_str_orig = str(b_val)
    a_str_rev = a_str_orig[::-1]
    b_str_rev = b_str_orig[::-1]
    
    # Prompt (with brackets)
    # Adding some spaces for jitter
    prompt = f" ({a_str_orig})   ({b_str_orig})   {op} ="
    
    # Echo-then-Reverse Scratchpad Parts
    echo_part = f"{a_str_orig} {b_str_orig} |"
    rev_part = f"{a_str_rev} {b_str_rev} |"
    
    # Math steps (Worst case: Tens Complement Subtraction)
    steps = []
    max_len = max(len(a_str_rev), len(b_str_rev))
    carry = 0
    ans = 0
    
    if op == '+':
        ans = a_val + b_val
        for i in range(max_len):
            d_a = int(a_str_rev[i]) if i < len(a_str_rev) else 0
            d_b = int(b_str_rev[i]) if i < len(b_str_rev) else 0
            res = d_a + d_b + carry
            new_carry = 1 if res > 9 else 0
            steps.append(f"{d_a}+{d_b}+{carry}={res}")
            carry = new_carry
        if carry > 0:
            steps.append(f"0+0+{carry}={carry}")
        math_part = ":".join(steps)
        ans_rev = str(ans)[::-1]
        scratchpad = f"<{echo_part} {rev_part} {math_part}:{ans_rev}>"
    else:
        ans = a_val - b_val
        for i in range(max_len):
            d_a = int(a_str_rev[i]) if i < len(a_str_rev) else 0
            d_b = int(b_str_rev[i]) if i < len(b_str_rev) else 0
            res = (d_a - carry) - d_b
            if res < 0:
                res += 10
                new_carry = 1
            else:
                new_carry = 0
            steps.append(f"{d_a}-{d_b}-{carry}={res}")
            carry = new_carry
        
        math_base = ":".join(steps)
        if carry == 0:
            scratchpad = f"<{echo_part} {rev_part} {math_base}:[BORROW]0|+:{str(abs(ans))[::-1]}>"
        else:
            # Tens Complement Pass (Worst Case)
            steps_tc = ["[BORROW]1|-"]
            found_nonzero = False
            derived_digits = [str((int(a_str_rev[i]) if i < len(a_str_rev) else 0) - (int(b_str_rev[i]) if i < len(b_str_rev) else 0)) for i in range(max_len)] # Simplification for token counting
            # Let's just simulate 9 steps of "9-x=y"
            for _ in range(max_len):
                steps_tc.append("9-9=0")
            steps_tc.append("000000000") # Final reversed digits
            math_part = math_base + ":" + ":".join(steps_tc)
            scratchpad = f"<{echo_part} {rev_part} {math_part}>"

    full_seq = f"{prompt}{scratchpad}{ans}"
    tokens = tokenizer.encode(full_seq)
    return len(tokens), full_seq

# Test Cases
print(f"{'Case':<40} | {'Tokens':<6}")
print("-" * 50)

# 5-digit Addition
len5_add, s5a = estimate_tokens(99999, 99999, '+')
print(f"5-digit Add (Worst Case)                 | {len5_add}")

# 5-digit Subtraction (Negative)
len5_sub, s5s = estimate_tokens(10000, 99999, '-')
print(f"5-digit Sub (Negative, Worst Case)       | {len5_sub}")

# 9-digit Addition
len9_add, s9a = estimate_tokens(999999999, 999999999, '+')
print(f"9-digit Add (Worst Case)                 | {len9_add}")

# 9-digit Subtraction (Negative)
len9_sub, s9s = estimate_tokens(100000000, 999999999, '-')
print(f"9-digit Sub (Negative, Worst Case)       | {len9_sub}")

print("\nSample 9-digit Sub Sequence (Tokens: {}):".format(len9_sub))
print(s9s)
