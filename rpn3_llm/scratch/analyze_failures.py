import re
import os

def analyze_failures():
    failures_file = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn3_llm/results/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_theta_tau0.7_sft_1-14_7num_BOS_500000_7num_ratio_1.0_failures.txt"
    if not os.path.exists(failures_file):
        print(f"Error: {failures_file} does not exist.")
        return

    with open(failures_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Phase name mapping
    phase_names = {
        2: "MATH1",
        3: "REV2",
        4: "MATH2",
        5: "REV3",
        6: "MATH3",
        7: "REV4",
        8: "MATH4",
        9: "REV5",
        10: "MATH5",
        11: "REV6",
        12: "MATH6",
        13: "REV7",
        14: "ANS"
    }

    results = []
    
    # Summary counters
    total_failures = 0
    phase_counts = {}
    type_counts = {}
    copy_source_counts = {}
    char_mismatch_counts = {}

    for line_idx, line in enumerate(lines):
        if not line.startswith("Q_Len:"):
            continue
        total_failures += 1
        
        parts = line.split(" | ")
        if len(parts) < 6:
            continue
            
        metadata = parts[0]
        prompt_str = parts[1]
        exp_rev_part = parts[2]
        pred_rev_part = parts[3]
        exp_math_part = parts[4]
        pred_math_part = parts[5]
        
        prompt = prompt_str.split(":", 1)[1].strip()
        exp_rev = exp_rev_part.split(":", 1)[1].strip()
        pred_rev = pred_rev_part.split(":", 1)[1].strip()
        exp_math = exp_math_part.split(":", 1)[1].strip()
        pred_math = pred_math_part.split(":", 1)[1].strip()
        
        # Check if the failure is in the initial REV1 phase
        if exp_rev != pred_rev:
            # Reversal 1 failure
            phase = "REV1"
            fail_type = "initial_reversal"
            copy_source = "PROMPT"
            # Find mismatch in REV1
            min_len = min(len(exp_rev), len(pred_rev))
            diff_idx = -1
            for idx in range(min_len):
                if exp_rev[idx] != pred_rev[idx]:
                    diff_idx = idx
                    break
            if diff_idx == -1:
                diff_idx = min_len
            exp_char = exp_rev[diff_idx] if diff_idx < len(exp_rev) else "<EOF>"
            pred_char = pred_rev[diff_idx] if diff_idx < len(pred_rev) else "<EOF>"
        else:
            # Failure is in MATH or subsequent REV phases
            min_len = min(len(exp_math), len(pred_math))
            diff_idx = -1
            for idx in range(min_len):
                if exp_math[idx] != pred_math[idx]:
                    diff_idx = idx
                    break
            #if diff_idx == -1:
            #    diff_idx = min_len #Why are we doing this? this should not happen so assert.
            assert diff_idx != -1
            exp_char = exp_math[diff_idx] if diff_idx < len(exp_math) else "<EOF>"
            pred_char = pred_math[diff_idx] if diff_idx < len(pred_math) else "<EOF>"
            
            # Determine Phase
            sub_str = exp_math[:diff_idx]
            current_phase_num = 2  # Starts at phase 2 (MATH1)
            pos = 0
            while pos < len(sub_str):
                if sub_str[pos:].startswith("[REV]"):
                    current_phase_num += 1
                    pos += len("[REV]")
                elif sub_str[pos:].startswith("[MATH]"):
                    current_phase_num += 1
                    pos += len("[MATH]")
                elif sub_str[pos:].startswith("[ANS]"):
                    current_phase_num += 1
                    pos += len("[ANS]")
                else:
                    pos += 1
            
            phase = phase_names.get(current_phase_num, f"PHASE_{current_phase_num}")
            
            # Find the substring for this phase
            last_token_idx = 0
            for token in ["[REV]", "[MATH]", "[ANS]"]:
                idx_find = sub_str.rfind(token)
                if idx_find > last_token_idx:
                    last_token_idx = idx_find
            
            # Find which token it was
            for token in ["[REV]", "[MATH]", "[ANS]"]:
                if sub_str[last_token_idx:].startswith(token):
                    start_idx = last_token_idx + len(token)
                    break
            else:
                start_idx = 0
                
            end_idx = len(exp_math)
            for token in ["[REV]", "[MATH]", "[ANS]"]:
                idx_find = exp_math.find(token, start_idx)
                if idx_find != -1 and idx_find < end_idx:
                    end_idx = idx_find
                    
            phase_text = exp_math[start_idx:end_idx]
            rel_diff_idx = diff_idx - start_idx
            
            if "REV" in phase:
                # Inside a reversal phase (REV2, REV3, etc.)
                # Split by [SEP] to find which sub-block failed
                sep_positions = [0]
                for m in re.finditer(r"\[SEP\]", phase_text):
                    sep_positions.append(m.start())
                    sep_positions.append(m.end())
                sep_positions.append(len(phase_text))
                
                sub_block_idx = -1
                is_separator = False
                for i in range(0, len(sep_positions) - 1, 2):
                    start_b = sep_positions[i]
                    end_b = sep_positions[i+1]
                    if start_b <= rel_diff_idx < end_b:
                        sub_block_idx = i // 2
                        break
                    if i + 2 < len(sep_positions):
                        start_s = sep_positions[i+1]
                        end_s = sep_positions[i+2]
                        if start_s <= rel_diff_idx < end_s:
                            is_separator = True
                            break
                            
                if is_separator:
                    fail_type = "separator_[SEP]"
                    copy_source = "CONSTANT"
                elif sub_block_idx == 0:
                    fail_type = "intermediate_math_result"
                    copy_source = "MATH"
                elif sub_block_idx > 0:
                    fail_type = f"after_sep{sub_block_idx}_failed"
                    copy_source = "REV"
                else:
                    fail_type = "unknown_reversal_block"
                    copy_source = "UNKNOWN"
            elif "MATH" in phase:
                fail_type = "math_computation"
                copy_source = "REV_AND_MATH"
            else:
                fail_type = "final_answer_generation"
                copy_source = "MATH"
                
        # Record results
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        type_counts[fail_type] = type_counts.get(fail_type, 0) + 1
        copy_source_counts[copy_source] = copy_source_counts.get(copy_source, 0) + 1
        
        mismatch_pair = f"'{exp_char}' -> '{pred_char}'"
        char_mismatch_counts[mismatch_pair] = char_mismatch_counts.get(mismatch_pair, 0) + 1
        
        results.append({
            "line": line_idx + 1,
            "phase": phase,
            "rel_diff_idx": rel_diff_idx,
            "fail_type": fail_type,
            "copy_source": copy_source,
            "expected_char": exp_char,
            "predicted_char": pred_char
        })
        
    print(f"Total analyzed failures: {total_failures}")
    print("\n--- Failure Count by Phase ---")
    for phase, count in sorted(phase_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {phase:<10}: {count} ({count/total_failures*100:.1f}%)")
        
    print("\n--- Failure Count by Copy Source ---")
    for src, count in sorted(copy_source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {src:<15}: {count} ({count/total_failures*100:.1f}%)")

    print("\n--- Failure Count by Failure Type ---")
    for t, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {t:<30}: {count} ({count/total_failures*100:.1f}%)")

    print("\n--- Top Character Mismatches (Expected -> Predicted) ---")
    for pair, count in sorted(char_mismatch_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {pair:<12}: {count} ({count/total_failures*100:.1f}%)")
        
    print("\n--- Detailed List of Failures ---")
    for r in results:
        print(f"Line {r['line']}: Phase {r['phase']} | idx: {r['rel_diff_idx']} | Type: {r['fail_type']} | Copied from: {r['copy_source']} | Mismatch: '{r['expected_char']}' -> '{r['predicted_char']}'")

if __name__ == "__main__":
    analyze_failures()
