import os

def strip_math(input_file, output_file):
    print(f"Processing {input_file} -> {output_file}")
    if not os.path.exists(input_file):
        print(f"  Error: {input_file} not found.")
        return
        
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
         
        processed = 0
        for line in f_in:
            if "[MATH]" in line:
                # Split at [MATH] and keep the first part (up to and including =)
                part = line.split("[MATH]")[0]
                # Append [EOS] and newline
                f_out.write(part + "[EOS]\n")
                processed += 1
            else:
                # If no [MATH], write as is (shouldn't happen in this dataset)
                f_out.write(line)
                
    print(f"  Done. Processed {processed} lines.")

if __name__ == "__main__":
    train_in = "rpn_llm/data/RPNData-1-22_phase_lean_train.txt"
    train_out = "rpn_llm/data/RPNData-1-22_rev_only_train.txt"
    
    val_in = "rpn_llm/data/RPNData-1-22_phase_lean_val.txt"
    val_out = "rpn_llm/data/RPNData-1-22_rev_only_val.txt"
    
    strip_math(train_in, train_out)
    strip_math(val_in, val_out)
    
    print("\nDataset creation complete! The DataLoaderLite will automatically generate binary caches on first load.")
