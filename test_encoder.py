import re

def test_cleaning():
    test_cases = [
        " 541   2    + ? | Expected: 543 | Predicted: 5543\n",
        "   12    34  -?  \n",
        "1 2 +?\n"
    ]
    
    print("Testing Encoder Space Sanitization:")
    print("-" * 50)
    for original in test_cases:
        clean = re.sub(r'\s+', ' ', original.strip()) + '\n'
        print(f"ORIGINAL: {repr(original)}")
        print(f"CLEANED : {repr(clean)}")
        print("-" * 50)

if __name__ == "__main__":
    test_cleaning()
