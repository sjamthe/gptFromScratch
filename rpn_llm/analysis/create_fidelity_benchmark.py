import json
import random
import os

def generate_random_problem(digits):
    a = "".join([str(random.randint(1, 9)) if i == 0 else str(random.randint(0, 9)) for i in range(digits)])
    b = "".join([str(random.randint(1, 9)) if i == 0 else str(random.randint(0, 9)) for i in range(digits)])
    return f"[BOS]{a} {b}+? [REV]{a[::-1]} {b[::-1]}+="

def create_benchmark(output_path):
    random.seed(42) # Ensure the benchmark itself is reproducible if recreated
    
    benchmark = {
        "short": [generate_random_problem(4) for _ in range(100)],
        "long": [generate_random_problem(25) for _ in range(100)]
    }
    
    with open(output_path, "w") as f:
        json.dump(benchmark, f, indent=4)
    
    print(f"Created benchmark file with {len(benchmark['short'])} short and {len(benchmark['long'])} long problems at {output_path}")

if __name__ == "__main__":
    os.makedirs("rpn_llm/analysis", exist_ok=True)
    create_benchmark("rpn_llm/analysis/fidelity_benchmark.json")
