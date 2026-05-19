with open("rpn3_llm/data/sft_1-14_7num_BOS_val.txt", "r", encoding="utf-8") as f:
    lines = [f.readline() for _ in range(15)]

with open("rpn3_llm/data/mini_val.txt", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Created mini_val.txt with 15 lines successfully!")
