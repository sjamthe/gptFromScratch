from rpn_llm.RPNDataset import RPNDataset
from rpn_llm.utils import RPNTokenizer

tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
dataset = RPNDataset(
    num_samples=5,
    max_operands=2,
    operations=('+','-',),
    max_number=100,
    tokenizer=tokenizer,
    max_seq_len=256
)

print("Verifying Clean Format:")
for prompt, answer in dataset.examples:
    print(f"Prompt: {prompt}")
    print(f"Answer: {answer}")
    print("-" * 20)
