from IPython.core import extensions
import os
import sys
import torch
import datetime
import json

# Ensure local imports work
lessons_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(lessons_dir)

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer
from val import run_lesson_validation

def tprint(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

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
            if current_prompt.startswith("[REV]") and "[ANS]" in current_prompt:
                target_tokens = tokenizer.encode(f"<num>{gt_ans}</num> [EOS]")
                max_gen = min(max_gen, len(target_tokens) + 5)
            else:
                max_gen = min(max_gen, len(prompt_tokens) + 15)
                
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
            
        if idx_sample > 0 and (idx_sample + 1) % max(1, total // 10) == 0:
            accuracy = (correct / (idx_sample + 1)) * 100
            tprint(f"    E2E Sample Progress: {int((idx_sample+1) / total * 100)}% ({idx_sample+1}/{total} samples) Correct {correct}/{idx_sample+1} -> Accuracy: {accuracy:.2f}%")
            
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return accuracy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results_file", type=str, default=None, help="Path to save/load JSON results for resume capability")
    args = parser.parse_args()
    
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(lessons_dir, "models/lesson4_wrappedNumUA_step40000.pt")
    
    tprint(f"Loading checkpoint {checkpoint_path} on {device}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    config.block_size = 768
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    tokenizer = RPNTokenizer(os.path.join(lessons_dir, "rpn-tokenizer.json"))
    ood_dir = os.path.join(lessons_dir, "data/ood")
    
    # Create results folder
    results_dir = os.path.join(lessons_dir, "reports")
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    
    # Resolve results_file path
    if args.results_file is not None:
        results_file = args.results_file
    else:
        results_file = os.path.join(results_dir, f"{checkpoint_name}_ood_results.json")
        
    results = {}
    if os.path.exists(results_file):
        tprint(f"Found existing results file: {results_file}. Loading to resume...")
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
                results = loaded_data.get("results", {})
        except Exception as e:
            tprint(f"Warning: Failed to load results file: {e}. Starting fresh.")
            results = {}
            
    def save_results():
        try:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump({
                    "checkpoint": checkpoint_path,
                    "results": results
                }, f, indent=2)
            tprint(f"Progress saved to {results_file}")
        except Exception as e:
            tprint(f"Warning: Failed to save progress to {results_file}: {e}")

    # Helper function to run/retrieve validation for standard lessons
    def evaluate_lesson_cached(category, key, lesson, path):
        if category not in results:
            results[category] = {}
        if str(key) in results[category]:
            tprint(f"  Skipping {category} for {key} (cached accuracy: {results[category][str(key)]:.2f}%)")
            return results[category][str(key)]
        
        tprint(f"  Evaluating {category} for {key} on {path}...")
        acc = run_lesson_validation(model, tokenizer, path, lesson=lesson, device=device, num_samples=100)
        results[category][str(key)] = acc
        save_results()
        return acc

    # Helper function to run/retrieve validation for E2E
    def evaluate_e2e_cached(category, key, path):
        if category not in results:
            results[category] = {}
        if str(key) in results[category]:
            tprint(f"  Skipping {category} for {key} (cached accuracy: {results[category][str(key)]:.2f}%)")
            return results[category][str(key)]
            
        tprint(f"  Evaluating {category} for {key} on {path}...")
        acc = run_e2e_state_machine_eval(model, tokenizer, path, device=device)
        results[category][str(key)] = acc
        save_results()
        return acc

    tprint("\n--- Running Lesson 1 OOD Tests (Reversal Digits) ---")
    for digits in range(23, 31):
        path = os.path.join(ood_dir, f"lesson1_ood_digits_{digits}.txt")
        evaluate_lesson_cached("L1", digits, lesson=1, path=path)
        
    tprint("\n--- Running Lesson 2 OOD Tests ---")
    for digits in range(10, 14):
        path = os.path.join(ood_dir, f"lesson2_ood_digits_{digits}.txt")
        evaluate_lesson_cached("L2_digits", digits, lesson=2, path=path)
        
    for operands in range(7, 11):
        path = os.path.join(ood_dir, f"lesson2_ood_operands_{operands}.txt")
        evaluate_lesson_cached("L2_operands", operands, lesson=2, path=path)

    tprint("\n--- Running Lesson 3 OOD Tests ---")
    for digits in range(10, 14):
        path = os.path.join(ood_dir, f"lesson3_ood_digits_{digits}.txt")
        evaluate_lesson_cached("L3_digits", digits, lesson=3, path=path)
        
    for operands in range(7, 11):
        path = os.path.join(ood_dir, f"lesson3_ood_operands_{operands}.txt")
        evaluate_lesson_cached("L3_operands", operands, lesson=3, path=path)

    tprint("\n--- Running Lesson 4 OOD Tests ---")
    for digits in range(10, 14):
        path = os.path.join(ood_dir, f"lesson4_ood_digits_{digits}.txt")
        evaluate_lesson_cached("L4_digits", digits, lesson=4, path=path)
        
    for operands in range(7, 11):
        path = os.path.join(ood_dir, f"lesson4_ood_operands_{operands}.txt")
        evaluate_lesson_cached("L4_operands", operands, lesson=4, path=path)

    tprint("\n--- Running End-to-End OOD Tests ---")
    for digits in range(10, 13):
        path = os.path.join(ood_dir, f"e2e_ood_digits_{digits}.txt")
        evaluate_e2e_cached("E2E_digits", digits, path=path)
        
    for operands in range(7, 11):
        path = os.path.join(ood_dir, f"e2e_ood_operands_{operands}.txt")
        evaluate_e2e_cached("E2E_operands", operands, path=path)

    # ----------------------------------------------------
    # Generate the Markdown Report
    # ----------------------------------------------------
    report_path = os.path.join(results_dir, f"{checkpoint_name}_ood_report.md")
    
    tprint(f"\nWriting Markdown Report to {report_path}...")
    
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# Curriculum Learning Out-of-Distribution (OOD) Performance Report\n\n")
        rf.write(f"This report compiles performance data of the Universal Transformer model (`{os.path.basename(checkpoint_path)}`) across gradual OOD scale shifts. In compliance with variables isolation, we tested only one OOD dimension (Digit Length or Operand Count) at a time in steps of 1 above the training thresholds.\n\n")
        
        # 1. Lesson 1 Table
        rf.write("## 1. Lesson 1: Reversal Generalization (Digit-scale)\n")
        rf.write("* **In-Distribution limit**: 22 digits.\n\n")
        rf.write("| Digit Length | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        if "L1" in results:
            for digits in sorted(int(k) for k in results["L1"].keys()):
                rf.write(f"| {digits} | {results['L1'][str(digits)]:.2f}% |\n")
        rf.write("\n")
        
        # 2. Lesson 2 Tables
        rf.write("## 2. Lesson 2: Multi-operand Reversal Generalization (Workspace capacity)\n")
        rf.write("* **In-Distribution limits**: 9 digits, 6 operands.\n\n")
        
        rf.write("### Digit Length Scaling (with 2 operands)\n")
        rf.write("| Digit Length | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        if "L2_digits" in results:
            for digits in sorted(int(k) for k in results["L2_digits"].keys()):
                rf.write(f"| {digits} | {results['L2_digits'][str(digits)]:.2f}% |\n")
        rf.write("\n")
        
        rf.write("### Operand Count Scaling (with 4 digits)\n")
        rf.write("| Operand Count | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        if "L2_operands" in results:
            for operands in sorted(int(k) for k in results["L2_operands"].keys()):
                rf.write(f"| {operands} | {results['L2_operands'][str(operands)]:.2f}% |\n")
        rf.write("\n")
        
        # 3. Lesson 3 Tables
        rf.write("## 3. Lesson 3: Step-by-Step Math Generalization (Alignment & Carry)\n")
        rf.write("* **In-Distribution limits**: 9 digits, 6 operands.\n\n")
        
        rf.write("### Digit Length Scaling (with 2 operands)\n")
        rf.write("| Digit Length | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        if "L3_digits" in results:
            for digits in sorted(int(k) for k in results["L3_digits"].keys()):
                rf.write(f"| {digits} | {results['L3_digits'][str(digits)]:.2f}% |\n")
        rf.write("\n")
        
        rf.write("### Operand Count Scaling (with 4 digits)\n")
        rf.write("| Operand Count | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        if "L3_operands" in results:
            for operands in sorted(int(k) for k in results["L3_operands"].keys()):
                rf.write(f"| {operands} | {results['L3_operands'][str(operands)]:.2f}% |\n")
        rf.write("\n")
        
        # 4. Lesson 4 Tables
        rf.write("## 4. Lesson 4: Result Reversal & Phase Transitions\n")
        rf.write("* **In-Distribution limits**: 9 digits, 6 operands.\n\n")
        
        rf.write("### Digit Length Scaling (with 2 operands)\n")
        rf.write("| Digit Length | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        if "L4_digits" in results:
            for digits in sorted(int(k) for k in results["L4_digits"].keys()):
                rf.write(f"| {digits} | {results['L4_digits'][str(digits)]:.2f}% |\n")
        rf.write("\n")
        
        rf.write("### Operand Count Scaling (with 4 digits)\n")
        rf.write("| Operand Count | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        if "L4_operands" in results:
            for operands in sorted(int(k) for k in results["L4_operands"].keys()):
                rf.write(f"| {operands} | {results['L4_operands'][str(operands)]:.2f}% |\n")
        rf.write("\n")
        
        # 5. End-to-End Tables
        rf.write("## 5. End-to-End State Machine Integration\n")
        rf.write("* **In-Distribution limits**: 9 digits, 6 operands.\n\n")
        
        rf.write("### Digit Length Scaling (with 3 operands)\n")
        rf.write("| Digit Length | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        if "E2E_digits" in results:
            for digits in sorted(int(k) for k in results["E2E_digits"].keys()):
                rf.write(f"| {digits} | {results['E2E_digits'][str(digits)]:.2f}% |\n")
        rf.write("\n")
        
        rf.write("### Operand Count Scaling (with 4 digits)\n")
        rf.write("| Operand Count | Exact Match Accuracy |\n")
        rf.write("|:---:|:---:|\n")
        if "E2E_operands" in results:
            for operands in sorted(int(k) for k in results["E2E_operands"].keys()):
                rf.write(f"| {operands} | {results['E2E_operands'][str(operands)]:.2f}% |\n")
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
        
    tprint("Done writing report.")

if __name__ == "__main__":
    main()
