from utils import RPNTokenizer
        
tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
#text = "< 6 5 - = : 6 - 5 - 0 = 1 : [BORROW] 0 > + : 1 : > 1"
#print(tokenizer.encode(text))
text = "[PAD][BOS][BORROW]"
print(tokenizer.encode(text))
