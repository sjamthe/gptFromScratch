import os
import glob

def split_result_file(file_path):
    print(f"Processing {file_path}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    split_idx = -1
    for i, line in enumerate(lines):
        if "Validation Accuracy:" in line:
            split_idx = i
            break
    
    if split_idx == -1:
        print(f"  No summary found in {file_path}. Skipping.")
        return

    failures_content = lines[:split_idx]
    summary_content = lines[split_idx:]
    
    summary_path = file_path.replace("_failures.txt", "_summary.txt")
    
    # Write summary
    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(summary_content)
    
    # Overwrite original failures file with ONLY failures
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(failures_content)
        
    print(f"  Split into:")
    print(f"    - {file_path}")
    print(f"    - {summary_path}")

def main():
    results_dir = "rpn_llm/results"
    pattern = os.path.join(results_dir, "ut1.8M_phaseMask_True_1-22_phase_lean_*_failures.txt")
    files = glob.glob(pattern)
    
    if not files:
        print("No files found matching the pattern.")
        return
        
    print(f"Found {len(files)} files to process.")
    for f in files:
        split_result_file(f)

if __name__ == "__main__":
    main()
