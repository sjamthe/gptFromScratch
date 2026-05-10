import sys
import os

# Add rpn3_llm to path so we can import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
rpn_dir = os.path.dirname(script_dir) # parent of scratch is rpn3_llm
sys.path.append(rpn_dir)

from utils import RPNTokenizer

tokenizer = RPNTokenizer(os.path.join(rpn_dir, "rpn-tokenizer.json"))

# Let's find which tokens map to 10, 11, 12
# We can just iterate over all possible tokens if we know them,
# or inspect the vocab if RPNTokenizer exposes it.

tokens = ["[BOS]", "[REV]", "[MATH]", "[ANS]", "[EOS]", "[BORROW]", "[PASS]", "?", "=", ":"]
for t in tokens:
    try:
        idx = tokenizer.encode(t)[0]
        print(f"Token {t} -> ID {idx}")
    except:
        pass

# Let's also print the mapping for 10, 11, 12 if we can find it
# We can't easily invert the tokenizer without access to vocab.
# But let's see what we get from the above!
