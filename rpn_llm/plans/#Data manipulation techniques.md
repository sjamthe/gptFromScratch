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

### Validation Accuracy: 99.97%
<pre>
Total Evaluated: 354782
Total Correct: 354658
Total Failures: 124

--- Breakdown by Carry Operations ---
Carries | Total   | Correct | Failures | Accuracy
0       | 120369  | 120290  | 79      | 99.93%
1       | 128564  | 128525  | 39      | 99.97%
2       | 70148   | 70143   | 5       | 99.99%
3       | 28096   | 28095   | 1       | 100.00%
4       | 6948    | 6948    | 0       | 100.00%
5       | 657     | 657     | 0       | 100.00%

--- Edge Case Analysis ---
Category         | Total    | Correct  | Accuracy
zero_operand     | 15176    | 15157    | 99.87%
negative_result  | 88326    | 88326    | 100.00%
normal           | 255055   | 254950   | 99.96%

</pre>

### Extended data test
On a test file that contains atleast one number > 99999 the model fails spectacularly (0% accuracy).
But on closer examination the failures were only in reversing the numbers. The math after reversal was 99.95% accurate
So model knows the math is just confused about the length of numbers. The next variation on test file tries to fix that by bracketing the numbers.

## Brackated input numbers and Compressed scratchpad on Ten's complement (current best)
Put the input digits in bracketes () so model knows how to identify them. finetune the final model on this dataset.

### Results
Model trains well as earlier (> 99%) but fails on large digit test set (0% accuracy).

## Echo-then-Reverse Scratchpad
1. Echo: Copy the input string exactly. Transformers are incredibly good at identity operations (copying) and this forces the numbers into the active local KV-cache of the scratchpad.
2. Reverse Locally: Once copied, reverse the numbers from the scratchpad copy, not the prompt.
<pre>
PROMPT: 608951119 89519 - =
SCRATCHPAD: <608951119 89519 | 911159806 91598 | 9-9-0=0...
</pre>

### 100% Accuracy on dataset with numbers less than 100,000
<pre>
Total Evaluated: 15941
Total Correct: 15941
Total Failures: 0
Accuracy: 100.00%

--- Breakdown by Carry Operations ---
Carries | Total   | Correct | Failures | Accuracy
0       | 5577    | 5577    | 0       | 100.00%
1       | 5330    | 5330    | 0       | 100.00%
2       | 3006    | 3006    | 0       | 100.00%
3       | 1496    | 1496    | 0       | 100.00%
4       | 492     | 492     | 0       | 100.00%
5       | 40      | 40      | 0       | 100.00%

--- Edge Case Analysis ---
Category         | Total    | Correct  | Accuracy
zero_operand     | 804      | 804      | 100.00%
negative_result  | 3991     | 3991     | 100.00%
normal           | 11343    | 11343    | 100.00%
</pre>

### Model fails on large digit test set 0% accuracy

**What is happening?**
- The Good: The model correctly copies the first operand (54874) perfectly.
- The Bad (The Echo Truncation): For the second operand (818591913), it only copies the first 5 digits (81859) and then immediately writes the | phase separator.
- The Ugly (The Reverse Truncation): In the second phase, it reverses what it just memorized in the Echo phase (81859 reversed becomes 95818). It does this entirely correctly based on its truncated copy!
- The Math: It then performs perfect Tens Complement math on these truncated digits.

## 9-9's training data on Compressed scratchpad on Ten's complement
The new training dataset (rpn_llm/data/RPNData-mixed-1-9_tens_comp_train.txt) contains 6 million samples with operand lengths ranging uniformly from 1 to 9 digits. This will force the model to learn logical delimiters instead of expecting a fixed 5-digit length.

### Results
trains well but suffers from same problem as 5-9s. if you give a number > 9-9s it cannot reverse it properly.

## 1-22 length numbers on Compressed scratchpad on Ten's complement
Increase length of numbers to 22. 0-3 spaces added before numbers and operator. 
Trained from scratch. 

## Results:
<pre>
81% accuracy. 
Total Evaluated: 18864
Total Correct: 15292
Total Failures: 3572
Accuracy: 81.06%

--- Breakdown by Carry Operations ---
Carries | Total   | Correct | Failures | Accuracy
0       | 11623   | 8858    | 2765    | 76.21%
1       | 1355    | 975     | 380     | 71.96%
2       | 1032    | 817     | 215     | 79.17%
3       | 850     | 750     | 100     | 88.24%
4       | 794     | 731     | 63      | 92.07%
5       | 703     | 678     | 25      | 96.44%
6       | 677     | 668     | 9       | 98.67%
7       | 618     | 612     | 6       | 99.03%
8       | 466     | 463     | 3       | 99.36%
9       | 352     | 350     | 2       | 99.43%
10      | 212     | 211     | 1       | 99.53%
11      | 113     | 112     | 1       | 99.12%
12      | 46      | 46      | 0       | 100.00%
13      | 21      | 19      | 2       | 90.48%
14      | 2       | 2       | 0       | 100.00%

--- Edge Case Analysis ---
Category         | Total    | Correct  | Accuracy
zero_operand     | 254      | 148      | 58.27%
negative_result  | 2380     | 1433     | 60.21%
normal           | 16287    | 13742    | 84.37%
</pre>


## Replace all spaces and bracket numbers in prompt. 1-22 length numbers on Compressed scratchpad on Ten's complement
example:  (10920155782176418)(30417522173842868)+=

## Results: 96.06% 
Only failures are for 16, 17 length where model skipped prefix (started with :)
Happens as it is sees too many = signs.
<pre>
Validation Accuracy: 96.06% (16134/16795)
=========================================


--- Breakdown by Prompt Length ---
Token Length | Total Items | Accuracy
 8 | 2161       | 100.00%
 9 | 4308       | 100.00%
10 | 6559       | 100.00%
11 | 8719       | 100.00%
12 | 10601      | 99.80%
13 | 12782      | 100.00%
14 | 15212      | 100.00%
15 | 17507      | 100.00%
16 | 19528      | 99.80%
17 | 21928      | 100.00%
18 | 23989      | 99.80%
19 | 26017      | 99.80%
20 | 27444      | 99.80%
21 | 28199      | 100.00%
22 | 29300      | 99.80%
23 | 30473      | 99.61%
24 | 31366      | 99.80%
25 | 32777      | 100.00%
26 | 33622      | 99.61%
27 | 31787      | 99.41%
28 | 28495      | 99.61%
29 | 26139      | 99.22%
30 | 23787      | 99.02%
31 | 21814      | 99.02%
32 | 18168      | 96.48%
33 | 16256      | 92.77%
34 | 14791      | 88.09%
35 | 11897      | 87.30%
36 | 10396      | 83.79%
37 | 6603       | 80.08%
38 | 4797       | 80.08%
39 | 3128       | 80.66%
40 | 411        | 84.43%
</pre>

## Replace = sign with ? in prommpt
**Example:** 
<pre>
(3037913)(48)+?<(3197303)(84)+=:3+8+0=1:1+4+1=6:9+0+0=9:7+0+0=7:3+0+0=3:0+0+0=0:3+0+0=3:1697303>3037961
</pre>

### Results: Accuracy: 99.99% (16827/16828)
only 1 failure. 


# Universal Transformer
Use the same dataset as above but train universal transformer.

**Universal Transfoemer** has same weights for all the blocks. 
This reduces the number of parameters and allows the model to generalize better.

## Results:
<pre>
Validation Accuracy: 99.45% (16736/16828)
=========================================


--- Breakdown by Prompt Length ---
Token Length | Total Items | Accuracy
 8 | 2114       | 88.87%
 9 | 4345       | 100.00%
10 | 6619       | 99.80%
11 | 8755       | 100.00%
12 | 11120      | 100.00%
13 | 13158      | 100.00%
14 | 15269      | 100.00%
15 | 17176      | 100.00%
16 | 19768      | 100.00%
17 | 21752      | 100.00%
18 | 23904      | 100.00%
19 | 26167      | 100.00%
20 | 27082      | 100.00%
21 | 28470      | 100.00%
22 | 29484      | 99.41%
23 | 30488      | 100.00%
24 | 31290      | 99.80%
25 | 32436      | 99.61%
26 | 33603      | 99.61%
27 | 31632      | 99.22%
28 | 28228      | 99.80%
29 | 26066      | 99.80%
30 | 23298      | 99.80%
31 | 21551      | 100.00%
32 | 18018      | 99.80%
33 | 16374      | 99.80%
34 | 14604      | 100.00%
35 | 12006      | 99.80%
36 | 10305      | 99.80%
37 | 6491       | 99.61%
38 | 4918       | 100.00%
39 | 3000       | 99.61%
40 | 444        | 97.52%
</pre>

## Train Reverseing first.
Create a separate dataset that stops after reversing the numbers. Train the model on this dataset first. Then train the model on the full dataset.

### Results of reversing
Only 16,000 steps were enough to train to 100.00% accuracy

At 16k training with full dataset accuracy dropped to 37% and almost all failures are related to reversal. looks like model forgot how to reverse.
<pre>
Validation Complete!
Total Evaluated: 18233
Total Correct: 6869
Total Failures: 11364
Accuracy: 37.67%

--- Breakdown by Carry Operations ---
Carries | Total   | Correct | Failures | Accuracy
0       | 3218    | 403     | 2815    | 12.52%
1       | 3451    | 528     | 2923    | 15.30%
2       | 2469    | 626     | 1843    | 25.35%
3       | 1909    | 697     | 1212    | 36.51%
4       | 1526    | 727     | 799     | 47.64%
5       | 1421    | 818     | 603     | 57.57%
6       | 1200    | 806     | 394     | 67.17%
7       | 1047    | 743     | 304     | 70.96%
8       | 845     | 634     | 211     | 75.03%
9       | 516     | 401     | 115     | 77.71%
10      | 344     | 282     | 62      | 81.98%
11      | 164     | 121     | 43      | 73.78%
12      | 76      | 51      | 25      | 67.11%
13      | 34      | 25      | 9       | 73.53%
14      | 12      | 6       | 6       | 50.00%
15      | 1       | 1       | 0       | 100.00%

--- Edge Case Analysis ---
Category         | Total    | Correct  | Accuracy
zero_operand     | 418      | 31       | 7.42%
negative_result  | 2378     | 7        | 0.29%
normal           | 15542    | 6831     | 43.95%

--- Failure Category Breakdown ---
reversal_skipped    : 3 (0.0%)
reversal_failed     : 11141 (98.0%)
math_failed         : 199 (1.8%)
final_ans_failed    : 21 (0.2%)
other               : 0 (0.0%)

--- Reversal Failures vs Spaces (S1, S2) ---
Spaces (1, 0): 1856 failures (16.7%)
Spaces (1, 1): 1448 failures (13.0%)
Spaces (1, 2): 1139 failures (10.2%)
Spaces (2, 0): 1301 failures (11.7%)
Spaces (2, 1): 1134 failures (10.2%)
Spaces (2, 2): 1021 failures (9.2%)
Spaces (3, 0): 1137 failures (10.2%)
Spaces (3, 1): 1071 failures (9.6%)
Spaces (3, 2): 1034 failures (9.3%)
</pre>

## Modify dataset to handle short numbers
1. **Biased Sampling**: Updated RPNDataset.py to sample shorter numbers (1–5 digits) 40% of the time. This forces the model to constantly practice "early exits" during the reversal phase.
2. **Space Variation**: Increased the random space range from (0, 2) to (0, 5). This breaks any internal "fixed token clock" the model might have been using to track number length, forcing it to attend specifically to the operator/questions marks as decorators.

Fine-tuning the model on the top of the above model.
### Results for each 16,000 steps
1.  Accuracy 77.8% at 16,000 steps. 
<pre>
reversal_failed     : 4368 (97.7%)
math_failed         : 102 (2.3%)
final_ans_failed    : 0 (0.0%)
</pre>
2. Accuracy 81.11% at 32,000 steps.
<pre>
reversal_failed     : 3795 (99.3%)
math_failed         : 22 (0.6%)
final_ans_failed    : 5 (0.1%)
</pre>
3. Accuracy 84.05% at 48,000 steps.
<pre>
reversal_failed     : 3215 (99.6%)
math_failed         : 13 (0.4%)
final_ans_failed    : 0 (0.0%)
</pre>
4. Accuracy 88.81% at 62,500 steps.
<pre>
reversal_failed     : 2259 (99.8%)
math_failed         : 5 (0.2%)
final_ans_failed    : 0 (0.0%)
</pre>

Results for prompt length 18-29 is lowest.

## Reverse mixed refresh
1. **Interleaved Dataset Generation**: Modified RPNDataset.py to generate a "Mixed Refresh" dataset.
- **30% Reversal-Only samples**: These force the model to prioritize the scratchpad setup (the bottleneck).
- **70% Full Math samples**: Keeps the 99%+ arithmetic logic fresh.
- *Biased Sampling**: Maintained the 40% bias for short numbers (1–5 digits) to specifically target the weak prompt lengths (18–29 tokens).
1. Training Script Update: Updated train_rpn.py to:
Point to the new rpn_llm/data/RPNData-1-22_mixed_refresh_train.txt.
Resume from your last checkpoint (UT3M_1-22_boundary_refinement_final.pt).

Fine-tuning the model on the top of the above model.
### Results for each 16,000 steps
1.  Accuracy 75.32% at 16,000 steps. 
2.  Accuracy 80.14% at 32,000 steps. 

** This training is not helping much. trying next strategy **

## The Valley Fix
The "Mixed Refresh" curriculum (30% truncated reversal sequences) caused a regression in accuracy from 88.8% to 80.1%. Analysis reveals a "Valley of Failures" in the 6-10 digit range (prompt lengths 18-29) which is currently under-represented in the training data. This plan pivots to a "Balanced Refinement" strategy.

We will resume from the 88.8% checkpoint (UT3M_1-22_boundary_refinement_final.pt) rather than the current mixed-refresh model. This discards the regressed weights and builds directly on the model's best known state.

** New Plan ** 
1. **Update `get_uniform_length_number` logic** to target 1-12 digits when force_short is true (broadening the bias from 1-5).
2. **Revert the main block** to generate 100% full math sequences (remove the mix_reversal_ratio logic).
3. **Set `bias_short=0.6` (60%)** to heavily prioritize the "Valley" and short ranges.

### Results : Accuract 74%

Did worse than mixed dataset.
<pre>
Token Length | Total Items | Accuracy
 5 | 109        | 0.00%
 6 | 447        | 0.00%
 7 | 1002       | 1.95%
 8 | 2058       | 9.96%
 9 | 3469       | 19.53%
10 | 5792       | 24.22%
11 | 8187       | 36.91%
12 | 11173      | 39.45%
13 | 14608      | 48.05%
14 | 18152      | 52.54%
15 | 21740      | 59.18%
16 | 25141      | 66.60%
17 | 28554      | 66.99%
18 | 31712      | 74.41%
19 | 34339      | 80.47%
20 | 36556      | 84.18%
21 | 37434      | 84.77%
22 | 37142      | 88.09%
</pre>

# Switching to RDT - Recurrent depth Tranaformer 
