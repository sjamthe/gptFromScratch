import random
import os

def generate_rpn_with_scratchpad(digits):
    a_str = "".join([str(random.randint(1, 9)) if i == 0 else str(random.randint(0, 9)) for i in range(digits)])
    b_str = "".join([str(random.randint(1, 9)) if i == 0 else str(random.randint(0, 9)) for i in range(digits)])
    
    a_rev = a_str[::-1]
    b_rev = b_str[::-1]
    
    # Calculate step by step
    carry = 0
    steps = []
    res_rev = ""
    
    for i in range(digits):
        d1 = int(a_rev[i])
        d2 = int(b_rev[i])
        
        if i == 0:
            s = d1 + d2
            step_str = f"{d1}+{d2}={s%10}"
        else:
            s = d1 + d2 + carry
            step_str = f"{d1}+{d2}+{carry}={s%10}"
            
        carry = s // 10
        res_rev += str(s % 10)
        steps.append(step_str)
        
    if carry > 0:
        res_rev += str(carry)
        steps.append(f"C={carry}")
        
    scratchpad = ":".join(steps) + ":"
    full_line = f"[BOS]{a_str} {b_str}+? [REV]{a_rev} {b_rev}+={scratchpad}[ANS]{res_rev}[EOS]"
    return full_line

def create_long_dataset(output_path):
    random.seed(42)
    lines = []
    
    # 50 samples each for 25, 30, 35 digits
    for length in [25, 30, 35]:
        for _ in range(50):
            lines.append(generate_rpn_with_scratchpad(length))
            
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
        
    print(f"Created OOD Long Dataset with {len(lines)} samples at {output_path}")

if __name__ == "__main__":
    os.makedirs("rpn_llm/data", exist_ok=True)
    create_long_dataset("rpn_llm/data/RPNData-long_ood_test.txt")
