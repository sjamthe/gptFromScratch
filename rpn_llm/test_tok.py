from utils import RPNTokenizer
        
tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
#text = "< 6 5 - = : 6 - 5 - 0 = 1 : [BORROW] 0 > + : 1 : > 1"
#print(tokenizer.encode(text))
#text = "[PAD][BOS][BORROW]"
text = "<7 22584-=:7-4-0=3:0-8-0=2:0-5-1=4:0-2-1=7:0-2-1=7:[BORROW]1|-:10-3=7:9-2=7:9-4=5:9-7=2:9-7=2:77522>-22577"
print(f"Text: {text}")
tokens = tokenizer.encode(text)
print(f"Token count: {len(tokens)}")
print(f"Tokens: {tokens}")
print(f"Decoded: {tokenizer.decode(tokens)}")
