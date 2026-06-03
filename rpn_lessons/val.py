import os
import argparse
import torch
from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def run_lesson_validation(model, tokenizer, data_path, lesson, device, num_samples=500):
    model.eval()
    
    # Define tokenizer token IDs we need
    bos_id = tokenizer.encode("[BOS]")[0]
    rev_id = tokenizer.encode("[REV]")[0]
    math_id = tokenizer.encode("[MATH]")[0]
    ans_id = tokenizer.encode("[ANS]")[0]
    eos_id = tokenizer.encode("[EOS]")[0]
    sep_id = tokenizer.encode("[SEP]")[0]
    
    # Validate lesson
    if lesson not in [1, 2, 3, 4]:
        raise ValueError(f"Invalid lesson: {lesson}")
        
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Validation file {data_path} not found.")
        
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    if num_samples > 0:
        lines = lines[:num_samples]
        
    correct = 0
    total = len(lines)
    
    # Warm-up phase token IDs list for the sequential mask cumsum
    phase_tensor = torch.tensor([rev_id, math_id, ans_id], device=device)
    
    print(f"Running exact-match validation on {total} samples for Lesson {lesson}...")
    
    failures_printed = 0
    
    for idx_sample, line in enumerate(lines):
        # Dynamically determine delimiter based on prompt prefix and content
        if line.startswith("[BOS]"):
            curr_delimiter = "[REV]"
            curr_stop_ids = {math_id}
            curr_max_gen = 100
        elif line.startswith("[REV]"):
            if "[ANS]" in line:
                curr_delimiter = "[ANS]"
                curr_stop_ids = {eos_id}
                curr_max_gen = 50
            else:
                curr_delimiter = "[MATH]"
                curr_stop_ids = {rev_id}
                curr_max_gen = 250
        elif line.startswith("[MATH]"):
            curr_delimiter = "[REV]"
            curr_stop_ids = {math_id, ans_id}
            curr_max_gen = 100
        else:
            # Fallback based on lesson passed to CLI
            if lesson == 1:
                curr_delimiter = "[ANS]"
                curr_stop_ids = {eos_id}
                curr_max_gen = 50
            elif lesson == 2:
                curr_delimiter = "[REV]"
                curr_stop_ids = {math_id}
                curr_max_gen = 100
            elif lesson == 3:
                curr_delimiter = "[MATH]"
                curr_stop_ids = {rev_id}
                curr_max_gen = 250
            elif lesson == 4:
                curr_delimiter = "[REV]"
                curr_stop_ids = {math_id, ans_id}
                curr_max_gen = 100

        # Split line at delimiter
        parts = line.split(curr_delimiter)
        if len(parts) < 2:
            print(f"Warning: line '{line}' missing delimiter '{curr_delimiter}'")
            continue
            
        prompt = parts[0] + curr_delimiter
        target = line[len(prompt):].strip() # Everything after delimiter
        
        prompt_tokens = tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
        past_kv = None
        generated = []
        curr_idx = prompt_tensor
        
        with torch.no_grad():
            for _ in range(curr_max_gen):
                # Compute dynamic phase IDs for phase masking
                is_phase_shift = torch.isin(curr_idx, phase_tensor)
                full_phase_ids = is_phase_shift.cumsum(dim=-1)
                
                # If KV Cache is active, we only pass the single last token
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
                
                if next_id in curr_stop_ids:
                    break
                    
        gen_str = tokenizer.decode(generated).strip()
        
        if gen_str == target:
            correct += 1
        else:
            if failures_printed < 5:
                print(f"Sample {idx_sample} Failure:", flush=True)
                print(f"  Prompt:   {prompt}", flush=True)
                print(f"  Expected: {target}", flush=True)
                print(f"  Got:      {gen_str}", flush=True)
                failures_printed += 1

        if idx_sample > 0 and (idx_sample+1) % (total//10) == 0:
            accuracy = (correct / (idx_sample+1)) * 100
            print(f"Lesson {lesson} Progress: {int((idx_sample+1) / total * 100)}% ({idx_sample+1}/{total} samples) Correct {correct}/{idx_sample+1} -> Accuracy: {accuracy:.2f}%", flush=True)

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"Lesson {lesson} Validation: Correct {correct}/{total} -> Accuracy: {accuracy:.2f}%", flush=True)
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lesson", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=500)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    tokenizer = RPNTokenizer("rpn_lessons/rpn-tokenizer.json")
    
    if args.data_path is None:
        args.data_path = f"rpn_lessons/data/lesson{args.lesson}_val.txt"
        
    print(f"Loading checkpoint {args.checkpoint} on {device}...", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    
    accuracy = run_lesson_validation(model, tokenizer, args.data_path, args.lesson, device, args.num_samples)
    
    # Exit with code 0 if accuracy matches the gate requirement, 1 otherwise
    gates = {1: 99.0, 2: 98.0, 3: 98.0, 4: 99.0}
    if accuracy >= gates[args.lesson]:
        print(f"SUCCESS: Lesson {args.lesson} passed the accuracy gate ({gates[args.lesson]}%)!", flush=True)
        exit(0)
    else:
        print(f"FAILURE: Lesson {args.lesson} failed the accuracy gate ({gates[args.lesson]}%)!", flush=True)
        exit(1)
