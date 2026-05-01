import re

def parse_failures(file_path):
    questions = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Q: "):
                # Extract the question part up to the '?'
                match = re.search(r"Q: (.*?\?)", line)
                if match:
                    questions.add(match.group(1))
    return questions

def analyze_diff(file48, file80):
    q48 = parse_failures(file48)
    q80 = parse_failures(file80)
    
    forgotten = q80 - q48
    learned = q48 - q80
    both_failed = q48 & q80
    
    print(f"Total Failures at 48k: {len(q48)}")
    print(f"Total Failures at 80k: {len(q80)}")
    print(f"Common Failures: {len(both_failed)}")
    print(f"Newly Failed (Forgotten): {len(forgotten)}")
    print(f"Newly Correct (Learned): {len(learned)}")
    
    print("\n--- Samples of Forgotten Questions (80k failed, 48k passed) ---")
    for i, q in enumerate(list(forgotten)[:10]):
        print(f"{i+1}. {q}")
        
    print("\n--- Samples of Learned Questions (48k failed, 80k passed) ---")
    for i, q in enumerate(list(learned)[:10]):
        print(f"{i+1}. {q}")

if __name__ == "__main__":
    analyze_diff(
        "rpn_llm/results/ut1.8M_phaseMask_True_1-22_phase_lean_48000_failures.txt",
        "rpn_llm/results/ut1.8M_phaseMask_True_1-22_phase_lean_80000_failures.txt"
    )
