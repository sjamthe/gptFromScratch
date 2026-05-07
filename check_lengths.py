import json

with open("rpn_llm/analysis/fidelity_benchmark.json", "r") as f:
    benchmark = json.load(f)

print(f"Short prompt sample length (BOS to REV): {len(benchmark['short'][0].split('[REV]')[0])}")
print(f"Long prompt sample length (BOS to REV): {len(benchmark['long'][0].split('[REV]')[0])}")

# Let's count digits in the first long prompt
long_prompt = benchmark['long'][0]
numbers = long_prompt.split('[BOS]')[1].split('+')[0].split(' ')
print(f"Number 1 length: {len(numbers[0])}")
print(f"Number 2 length: {len(numbers[1])}")
