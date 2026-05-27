import os
import csv

def parse_csv():
    csv_path = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn3_llm/scratch/wandb_export_2026-05-26T22_47_55.472-07_00.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        rows = []
        for r in reader:
            try:
                step = int(r[0])
                loss = float(r[1]) if r[1] else None
                loss_min = float(r[2]) if r[2] else None
                loss_max = float(r[3]) if r[3] else None
                rows.append((step, loss, loss_min, loss_max))
            except ValueError:
                continue
                
        rows.sort(key=lambda x: x[0])
        
        print("Run 1 (Batch size 8) loss at various steps:")
        for step, loss, l_min, l_max in rows:
            if step in [36000, 40000, 72000, 80000]:
                print(f"  Step {step}: Avg={loss}, Min={l_min}, Max={l_max}")

if __name__ == "__main__":
    parse_csv()
