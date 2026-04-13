# Data manipulation techniques

## fully_reversed_nopad
Rewrite the expression in reverse in the dataset and show in scratchpad \<scratch\> the operation on each digit (from units to 100000s) and carry. the final answer is also reversed and if negative - is in the end.
 - file **data/RPNData-plusminus99999_fully_reversed_nopad-train.txt** 

 - **sample 1**: `38666 81333 + = 119999`
    rewritten in reverse in the data file. 

    `66683 33318 + = < 6 + 3 + 0 = 9 : 6 + 3 + 0 = 9 : 6 + 3 + 0 = 9 : 8 + 1 + 0 = 9 : 3 + 8 + 0 = 11 > 999911`
 - **sample 2**: `85220 86551 - = -1331`
    rewritten in reverse in the data file. 

    `02258 15568 - = < - : 1 - 0 - 0 = 1 : 5 - 2 - 0 = 3 : 5 - 2 - 0 = 3 : 6 - 5 - 0 = 1 : 8 - 8 - 0 = 0 > 1331-`

### Results (total test rows 599,918)
**Total Accuracy: 89.24%**
- *Group 1: (prompt length 15) 485,973 items* group accuracy 99.01%
- *Group 2: (prompt length 14) 97,345 items* group accuracy 52.33%, group 1+2 accurary 91.22%
- *Group 3: (prompt length 13) 14,438 items* group accuracy 22.49%, group 1+2+3 accurary 89.56%
- *Group 4: (prompt length 12) 1,891 items* group accuracy 0%, group 1+2+3+4 accurary 89.28%
- *Group 5: (prompt length 11) 245 items* group accuracy 0%, group 1+2+3+4 accurary 89.28%
- *Group 6: (prompt length 10) 23 items* group accuracy 0%, group 1+2+3+4 accurary 89.24%
- *Group 7: (prompt length 9) 3 items* group accuracy 0%, group 1+2+3+4 accurary 89.24%

** Summary:** Model probably has less data for lengths 12-14 so couldn't learn. sensitive to position even with ROPE.

## model_driven_reversals
The above technique (fully_reversed_nopad) reversed the input data. This technique builds on it and puts reversal inside scratchpad. input expression is as user entered.
We also add random spaces between numbers, operators and equals sign. This is done so model doesn't learn exact positions of =. The dataset was recreated with better distribution of numbers which was a problem in previous technique (fully_reversed_nopad) where 85% data had 15 tokens.

### Total Accuracy: 95.87% (15265/15922)
- 94% failures are due to eager start of incorrect scratchpad (missing < and reverse numbers)

<pre>
--- Breakdown by Prompt Length ---
Token Length | Total Items | Accuracy
 8 | 281        | 81.85%
 9 | 1806       | 70.02%
10 | 6286       | 89.65%
11 | 15261      | 94.43%
12 | 30085      | 95.02%
13 | 49125      | 97.66%
14 | 68967      | 98.54%
15 | 83385      | 99.32%
16 | 89193      | 98.93%
17 | 83404      | 99.90%
18 | 68545      | 99.90%
19 | 49047      | 100.00%
20 | 30118      | 99.22%
21 | 15334      | 99.80%
22 | 6284       | 99.32%
23 | 1772       | 99.51%
24 | 281        | 98.58%
</pre>

 ## Ten's complement two pass subtraction
 This plan rewrites the generated subtraction format in RPNDataset.py to completely eradicate the zero-shot $A < B$ global sign prediction. The model will now blindly evaluate $A - B$ right to left cleanly taking borrows, and run a fast local 10s-complement correction string on the output if the final sequence borrow is 1.
 The addition is same as model driven reversals.

### Validation Accuracy: 96.03% (15310/15943)
- Of the 633 failures 309 are for subtraction and 324 for addition, so pretty even split.
- 17 failures are where second number **dropped a last digit while reversing**, math is correct.
- 616 failures are **eager start** where model starts scratchpad without < and reverses numbers.
<pre>

--- Breakdown by Prompt Length ---
Token Length | Total Items | Accuracy
 8 | 295        | 96.95%
 9 | 1735       | 73.83%
10 | 6294       | 87.70%
11 | 15320      | 93.16%
12 | 29994      | 92.77%
13 | 49246      | 95.41%
14 | 68872      | 97.46%
15 | 83480      | 99.41%
16 | 90008      | 99.51%
17 | 83515      | 100.00%
18 | 68567      | 100.00%
19 | 49094      | 100.00%
20 | 30155      | 100.00%
21 | 15484      | 100.00%
22 | 6264       | 100.00%
23 | 1817       | 99.90%
24 | 288        | 99.65%

--- Breakdown by Carry Operations ---
Carries | Total   | Correct | Failures | Accuracy
0       | 5638    | 5359    | 279     | 95.05%
1       | 5169    | 4897    | 272     | 94.74%
2       | 3130    | 3066    | 64      | 97.96%
3       | 1463    | 1448    | 15      | 98.97%
4       | 490     | 487     | 3       | 99.39%
5       | 53      | 53      | 0       | 100.00%

--- Edge Case Analysis ---
Category         | Total    | Correct  | Accuracy
zero_operand     | 832      | 784      | 94.23%
negative_result  | 3934     | 3781     | 96.11%
normal           | 11387    | 10939    | 96.07%
</pre>

 ## Compressed scratchpad on Ten's complement two pass subtraction
### What was changed:
New Marker: Added | to rpn-tokenizer.json at ID 9. This is used as the internal borrow separator (e.g., [BORROW]0|+).

- Compressed Logic: Removed spaces around all operators (+, -, =) and separators (:) in RPNDataset.py
.
- Strict Bracketing: The scratchpad now strictly adheres to the format \<scratchpad\>answer with no trailing spaces after total closure.
