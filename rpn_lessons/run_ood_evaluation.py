from IPython.core import extensions
import os
import sys
import torch

# Ensure local imports work
lessons_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(lessons_dir)

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer
from val import run_lesson_validation

def run_e2e_state_machine_eval(model, tokenizer, data_path, device):
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    correct = 0
    total = len(lines)
    
    bos_id = tokenizer.encode("[BOS]")[0]
    rev_id = tokenizer.encode("[REV]")[0]
    math_id = tokenizer.encode("[MATH]")[0]
    ans_id = tokenizer.encode("[ANS]")[0]
    eos_id = tokenizer.encode("[EOS]")[0]
    
    phase_tensor = torch.tensor([rev_id, math_id, ans_id], device=device)
    
    for idx_sample, line in enumerate(lines):
        parts = line.split("|||")
        if len(parts) < 2:
            continue
        prompt = parts[0]
        gt_ans = parts[1].strip()
        
        current_prompt = prompt
        step = 0
        # Calculate dynamic max_phases: 2 * (number of space-separated tokens/operands) + 2
        expr_clean = prompt.replace("[BOS]", "").replace("[REV]", "").strip()
        num_operands = len(expr_clean.split())
        max_phases = 2 * num_operands + 2
        success = False
        final_decoded = "[UNK]"
        
        while step < max_phases:
            if current_prompt.startswith("[BOS]"):
                stop_ids = {math_id}
                max_gen = 150
            elif current_prompt.startswith("[REV]"):
                if "[ANS]" in current_prompt:
                    stop_ids = {eos_id}
                    max_gen = 50
                else:
                    stop_ids = {rev_id}
                    max_gen = 250
            elif current_prompt.startswith("[MATH]"):
                stop_ids = {math_id, ans_id}
                max_gen = 150
            else:
                break
                
            prompt_tokens = tokenizer.encode(current_prompt)
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
            
            past_kv = None
            curr_idx = prompt_tensor
            generated = []
            
            with torch.no_grad():
                for step_count in range(max_gen):
                    is_phase_shift = torch.isin(curr_idx, phase_tensor)
                    full_phase_ids = is_phase_shift.cumsum(dim=-1)
                    
                    cond_idx = curr_idx[:, -1:] if past_kv is not None else curr_idx
                    
                    with torch.autocast(device, dtype=torch.bfloat16):
                        logits, _, past_kv = model(
                            cond_idx, 
                            use_cache=True, 
                            past_key_values=past_kv, 
                            full_phase_ids=full_phase_ids
                        )
                        
                    next_id = torch.argmax(logits[0, -1, :]).item()
                    generated.append(next_id)
                    curr_idx = torch.cat([curr_idx, torch.tensor([[next_id]], device=device)], dim=1)
                    
                    if next_id in stop_ids:
                        break
                        
            gen_str = tokenizer.decode(generated).strip()
            
            last_delim_idx = current_prompt.rfind("[")
            if last_delim_idx == -1:
                break
            last_delim = current_prompt[last_delim_idx:]
            current_prompt = last_delim + gen_str
            
            if gen_str.endswith("[EOS]"):
                final_decoded = gen_str.replace("[EOS]", "").replace("<num>", "").replace("</num>", "").strip()
                # Compare as integers to handle leading zero padding
                try:
                    if int(final_decoded) == int(gt_ans):
                        success = True
                except ValueError:
                    if final_decoded == gt_ans:
                        success = True
                break
            elif gen_str.endswith("[ANS]"):
                current_prompt = "[REV]" + gen_str
                
            step += 1
            
        if success:
            correct += 1
            
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return accuracy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(lessons_dir, "models/lesson4_wrappedNum_step40000.pt")
    
    print(f"Loading checkpoint {checkpoint_path} on {device}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    config.block_size = 768
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    tokenizer = RPNTokenizer(os.path.join(lessons_dir, "rpn-tokenizer.json"))
    ood_dir = os.path.join(lessons_dir, "data/ood")
    
    results = {}
    
    print("\n--- Running Lesson 1 OOD Tests (Reversal Digits) ---")
    results["L1"] = {}
    for digits in range(23, 31):
        path = os.path.join(ood_dir, f"lesson1_ood_digits_{digits}.txt")
        acc = run_lesson_validation(model, tokenizer, path, lesson=1, device=device, num_samples=100)
        results["L1"][digits] = acc
        
    print("\n--- Running Lesson 2 OOD Tests ---")
    results["L2_digits"] = {}
    for digits in range(10, 14):
        path = os.path.join(ood_dir, f"lesson2_ood_digits_{digits}.txt")
        acc = run_lesson_validation(model, tokenizer, path, lesson=2, device=device, num_samples=100)
        results["L2_digits"][digits] = acc
        
    results["L2_operands"] = {}
    for operands in range(7, 11):
        path = os.path.join(ood_dir, f"lesson2_ood_operands_{operands}.txt")
        acc = run_lesson_validation(model, tokenizer, path, lesson=2, device=device, num_samples=100)
        results["L2_operands"][operands] = acc

    print("\n--- Running Lesson 3 OOD Tests ---")
    results["L3_digits"] = {}
    for digits in range(10, 14):
        path = os.path.join(ood_dir, f"lesson3_ood_digits_{digits}.txt")
        acc = run_lesson_validation(model, tokenizer, path, lesson=3, device=device, num_samples=100)
        results["L3_digits"][digits] = acc
        
    results["L3_operands"] = {}
    for operands in range(7, 11):
        path = os.path.join(ood_dir, f"lesson3_ood_operands_{operands}.txt")
        acc = run_lesson_validation(model, tokenizer, path, lesson=3, device=device, num_samples=100)
        results["L3_operands"][operands] = acc

    print("\n--- Running Lesson 4 OOD Tests ---")
    results["L4_digits"] = {}
    for digits in range(10, 14):
        path = os.path.join(ood_dir, f"lesson4_ood_digits_{digits}.txt")
        acc = run_lesson_validation(model, tokenizer, path, lesson=4, device=device, num_samples=100)
        results["L4_digits"][digits] = acc
        
    results["L4_operands"] = {}
    for operands in range(7, 11):
        path = os.path.join(ood_dir, f"lesson4_ood_operands_{operands}.txt")
        acc = run_lesson_validation(model, tokenizer, path, lesson=4, device=device, num_samples=100)
        results["L4_operands"][operands] = acc

    print("\n--- Running End-to-End OOD Tests ---")
    results["E2E_digits"] = {}
    for digits in range(10, 13):
        path = os.path.join(ood_dir, f"e2e_ood_digits_{digits}.txt")
        acc = run_e2e_state_machine_eval(model, tokenizer, path, device=device)
        results["E2E_digits"][digits] = acc
        
    results["E2E_operands"] = {}
    for operands in range(7, 11):
        path = os.path.join(ood_dir, f"e2e_ood_operands_{operands}.txt")
        acc = run_e2e_state_machine_eval(model, tokenizer, path, device=device)
        results["E2E_operands"][operands] = acc

    # ----------------------------------------------------
    # Generate the Markdown Report
    # ----------------------------------------------------
    report_path = "/Users/sjamthe/.gemini/antigravity-ide/brain/121d30ad-79fd-4baa-b4db-549b5b2a586f/curriculum_ood_report_no_tie.md"
    
    print(f"\nWriting Markdown Report to {report_path}...")
    
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# Curriculum Learning Out-of-Distribution (OOD) Performance Report\n\n")
        rf.write("This report compiles performance data of the final Universal Transformer model (`lesson4_step40000.pt`) across gradual OOD scale shifts. In compliance with variables isolation, we tested only one OOD dimension (Digit Length or Operand Count) at a time in steps of 1 above the training thresholds.\n\n")
        
        # 1. Lesson 1 Table
        rf.write("## 1. Lesson 1: Reversal Generalization (Digit-scale)\n")
        rf.write("* **In-Distribution limit**: 22 digits.\n\n")
        rf.write("| Digit Length | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        for digits in sorted(results["L1"].keys()):
            rf.write(f"| {digits} | {results['L1'][digits]:.2f}% |\n")
        rf.write("\n")
        
        # 2. Lesson 2 Tables
        rf.write("## 2. Lesson 2: Multi-operand Reversal Generalization (Workspace capacity)\n")
        rf.write("* **In-Distribution limits**: 9 digits, 6 operands.\n\n")
        
        rf.write("### Digit Length Scaling (with 2 operands)\n")
        rf.write("| Digit Length | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        for digits in sorted(results["L2_digits"].keys()):
            rf.write(f"| {digits} | {results['L2_digits'][digits]:.2f}% |\n")
        rf.write("\n")
        
        rf.write("### Operand Count Scaling (with 4 digits)\n")
        rf.write("| Operand Count | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        for operands in sorted(results["L2_operands"].keys()):
            rf.write(f"| {operands} | {results['L2_operands'][operands]:.2f}% |\n")
        rf.write("\n")
        
        # 3. Lesson 3 Tables
        rf.write("## 3. Lesson 3: Step-by-Step Math Generalization (Alignment & Carry)\n")
        rf.write("* **In-Distribution limits**: 9 digits, 6 operands.\n\n")
        
        rf.write("### Digit Length Scaling (with 2 operands)\n")
        rf.write("| Digit Length | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        for digits in sorted(results["L3_digits"].keys()):
            rf.write(f"| {digits} | {results['L3_digits'][digits]:.2f}% |\n")
        rf.write("\n")
        
        rf.write("### Operand Count Scaling (with 4 digits)\n")
        rf.write("| Operand Count | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        for operands in sorted(results["L3_operands"].keys()):
            rf.write(f"| {operands} | {results['L3_operands'][operands]:.2f}% |\n")
        rf.write("\n")
        
        # 4. Lesson 4 Tables
        rf.write("## 4. Lesson 4: Result Reversal & Phase Transitions\n")
        rf.write("* **In-Distribution limits**: 9 digits, 6 operands.\n\n")
        
        rf.write("### Digit Length Scaling (with 2 operands)\n")
        rf.write("| Digit Length | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        for digits in sorted(results["L4_digits"].keys()):
            rf.write(f"| {digits} | {results['L4_digits'][digits]:.2f}% |\n")
        rf.write("\n")
        
        rf.write("### Operand Count Scaling (with 4 digits)\n")
        rf.write("| Operand Count | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        for operands in sorted(results["L4_operands"].keys()):
            rf.write(f"| {operands} | {results['L4_operands'][operands]:.2f}% |\n")
        rf.write("\n")
        
        # 5. End-to-End Tables
        rf.write("## 5. End-to-End State Machine Integration\n")
        rf.write("* **In-Distribution limits**: 9 digits, 6 operands.\n\n")
        
        rf.write("### Digit Length Scaling (with 3 operands)\n")
        rf.write("| Digit Length | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        for digits in sorted(results["E2E_digits"].keys()):
            rf.write(f"| {digits} | {results['E2E_digits'][digits]:.2f}% |\n")
        rf.write("\n")
        
        rf.write("### Operand Count Scaling (with 4 digits)\n")
        rf.write("| Operand Count | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        for operands in sorted(results["E2E_operands"].keys()):
            rf.write(f"| {operands} | {results['E2E_operands'][operands]:.2f}% |\n")
        rf.write("\n")
        
        # 6. Deep-Dive Analysis section
        rf.write("## 6. Failure Analysis & Interpretations\n\n")
        
        rf.write("### A. Reversal Scaling (Lesson 1 & 2)\n")
        rf.write("> [!NOTE]\n")
        rf.write("> **Observation**: Reversal pointer logic uses Coordinate heads to track token spaces. Explain below where performance starts degrading as digits increase from 23 to 30.\n\n")
        
        rf.write("### B. Step-by-Step Math Scaling (Lesson 3 & 4)\n")
        rf.write("> [!NOTE]\n")
        rf.write("> **Observation**: Math execution requires both alignment tracking and carry tracking. Increasing operands requires longer tail-copy sequences. Explain below the impact of digit length vs. operand counts on math logic.\n\n")
        
        rf.write("### C. End-to-End Generalization\n")
        rf.write("> [!NOTE]\n")
        rf.write("> **Observation**: E2E combines all lessons. Explain if compounding errors or coordinate drift dominates OOD failures.\n")
        
    print("Done writing report.")

if __name__ == "__main__":
    main()
