import subprocess
import sys
import os

def run_command(cmd, cwd=None):
    print(f"\n========================================================")
    print(f"Running: {' '.join(cmd)}")
    print(f"========================================================")
    
    # We use stream execution so we can see the logs in real time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=cwd
    )
    
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(line, end='', flush=True)
        
    process.wait()
    if process.returncode != 0:
        print(f"\nCommand failed with exit code: {process.returncode}")
        sys.exit(process.returncode)
    else:
        print(f"\nCommand completed successfully.")

def main():
    lessons_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(lessons_dir)
    
    cmd_l1 = [
        "python", "rpn_lessons/train.py",
        "--lesson", "1",
        "--run_name_suffix", "wrappedNum"
    ]
    run_command(cmd_l1, cwd=project_root)

    # 1. Lesson 2 Training (40k steps)
    # Warm-start from lesson1_wrappedNum_step20000.pt
    cmd_l2 = [
        "python", "rpn_lessons/train.py",
        "--lesson", "2",
        "--checkpoint", "rpn_lessons/models/lesson1_wrappedNum_step20000.pt",
        "--run_name_suffix", "wrappedNum"
    ]
    run_command(cmd_l2, cwd=project_root)
    
    # 2. Lesson 3 Training (80k steps)
    # Warm-start from lesson2_wrappedNum_step40000.pt
    cmd_l3 = [
        "python", "rpn_lessons/train.py",
        "--lesson", "3",
        "--checkpoint", "rpn_lessons/models/lesson2_wrappedNum_step40000.pt",
        "--run_name_suffix", "wrappedNum"
    ]
    run_command(cmd_l3, cwd=project_root)
    
    # 3. Lesson 4 Training (40k steps)
    # Warm-start from lesson3_wrappedNum_step80000.pt
    cmd_l4 = [
        "python", "rpn_lessons/train.py",
        "--lesson", "4",
        "--checkpoint", "rpn_lessons/models/lesson3_wrappedNum_step80000.pt",
        "--run_name_suffix", "wrappedNum"
    ]
    run_command(cmd_l4, cwd=project_root)
    
    # 4. Final OOD Evaluation
    # Evaluate the resulting lesson 4 model
    cmd_eval = [
        "python", "rpn_lessons/run_ood_evaluation.py",
        "--checkpoint", "rpn_lessons/models/lesson4_wrappedNum_step40000.pt",
        "--device", "cpu"
    ]
    run_command(cmd_eval, cwd=project_root)
    
    print("\n========================================================")
    print("ALL THREE TRAINING RUNS AND EVALUATIONS COMPLETED SUCCESSFULLY!")
    print("========================================================")

if __name__ == "__main__":
    main()
