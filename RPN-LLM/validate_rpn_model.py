import os
import torch
import json
from collections import defaultdict
from train_rpn_llm import GPT, GPTConfig, RPNTokenizer, DataLoaderLite

def validate_model(checkpoint_path, test_file_path, output_fail_path):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    tokenizer = RPNTokenizer("rpn-tokenizer.json")
    
    # 2. Parse Test File
    print(f"Parsing test file {test_file_path}...")
    with open(test_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # 3. Group by prompt token length to avoid padding issues
    # A prompt is everything up to and including the '=' sign and the trailing space. 
    length_groups = defaultdict(list)
    eq_id = tokenizer.encode("=")[0]
    nl_id = tokenizer.encode("\n")[0]
    
    print("Grouping by prompt length...")
    for line in lines:
        if "=" not in line:
            continue
        
        # Tokenize the entire line at once exactly as training does
        line_tokens = tokenizer.encode(line)
        
        # Find index of '='
        try:
            eq_idx = line_tokens.index(eq_id)
        except ValueError:
            continue
            
        prompt_tokens = line_tokens[:eq_idx + 1]
        expected_ans_tokens = line_tokens[eq_idx + 1:]
        
        length_groups[len(prompt_tokens)].append({
            'prompt_tokens': prompt_tokens,
            'expected_tokens': expected_ans_tokens,
            'full_str': line.strip()
        })
        
    print(f"Grouped into {len(length_groups)} different prompt lengths.")
    
    # 4. Batched Generation
    max_batch_size = 1024
    failures = []
    total_processed = 0
    total_correct = 0
    
    # Max generation steps for an answer (e.g. " <space> -1998 \n" -> ~7 tokens)
    max_new_tokens = 8

    print("Beginning batched evaluation...")
    group_idx = 1
    for length, items in length_groups.items():
        print(f"Evaluating group {group_idx}/{len(length_groups)} (prompt length {length}) - {len(items)} items...")
        group_idx += 1
        
        # Process in chunks of max_batch_size
        for i in range(0, len(items), max_batch_size):
            batch_items = items[i:i+max_batch_size]
            B = len(batch_items)
            total_processed += B
            
            # Construct (B, L) tensor natively (no padding required because all logic lengths identically match)
            prompts = torch.tensor([item['prompt_tokens'] for item in batch_items], dtype=torch.long, device=device)
            
            # Generate max_new_tokens sequentially using Argmax
            idx = prompts
            
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size:]
                with torch.no_grad():
                    with torch.autocast(device, dtype=torch.bfloat16):
                        logits, _ = model(idx_cond)
                
                logits = logits[:, -1, :] # Pluck final step logits
                idx_next = torch.argmax(logits, dim=-1, keepdim=True) # Deterministic greedy decision
                idx = torch.cat((idx, idx_next), dim=1) # Append
                
            # Extract and verify the generated characters
            for b in range(B):
                full_generated_tokens = idx[b].tolist()
                gen_answer_tokens = full_generated_tokens[length:]
                
                # Truncate string sequence heavily at newline (to match exactly)
                if nl_id in gen_answer_tokens:
                    nl_idx = gen_answer_tokens.index(nl_id)
                    gen_answer_tokens = gen_answer_tokens[:nl_idx+1]
                
                # Compare
                expected = batch_items[b]['expected_tokens']
                expected_str = tokenizer.decode(expected).strip()
                predicted_str = tokenizer.decode(gen_answer_tokens).strip() 
                
                if expected != gen_answer_tokens:
                    # Document failure
                    prompt_str = tokenizer.decode(batch_items[b]['prompt_tokens']).strip()
                    failures.append(f"Q: {prompt_str} | Expected: {expected_str} | Predicted: {predicted_str}")
                else:
                    total_correct += 1

    # 5. Output Results
    accuracy = (total_correct / total_processed) * 100
    print(f"\n=====================")
    print(f"Validation Complete!")
    print(f"Total Evaluated: {total_processed}")
    print(f"Total Correct: {total_correct}")
    print(f"Total Failures: {len(failures)}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    with open(output_fail_path, "w", encoding="utf-8") as f:
        f.write(f"Validation Accuracy: {accuracy:.2f}% ({total_correct}/{total_processed})\n")
        f.write("=========================================\n\n")
        for fail in failures:
            f.write(fail + "\n")
            
    print(f"Failures dumped to {output_fail_path}")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "gpt2_10M_checkpoint_9999.pt"
    validate_model(model_path, "data/RPNData-999+-_test.txt", "validation_failures.txt")
