# Compare ROPE vs UT vs RDT
## Data: RPNData-1-22_tens_comp_clean_tiered
1. **Tiered Curriculum**: We will enforce a perfectly balanced 33/33/33 split. One-third of the dataset will be 1-4 digit numbers, one-third will be 5-12 digit numbers, and one-third will be 13-22 digit numbers. This guarantees the model learns exactly how to handle short numbers without hallucinating.
2. **True Clean Spaces**: We will strip all rs0() noise logic out of RPNDataset.py directly. The prompt will natively be formatted exactly as requested: 123 456+?. No extra spaces, no unexpected spacing bounds.
3. **Examples:**
   -  `"123 456+?"`

### Accuracy
|Models/Steps|16000|32000|48000|64000|80000|96000|100000|
|---|---|---|---|---|---|---|---|
|ROPE|57.52%|77.11%|82.38%|79.31%|81.24%|85.16%|83.35%|
|UT|37.88%|70.00%|69.68%|68.33%|77.96%|82.81%|81.41%|
|RDT|49.07%|70.28%|71.49%|74.38%|75.72%|80.24%|81.15%|

### Analysis
All 3 models show a strange pattern of accuracy graph.
1. Lower prompt lengths (5-7) and particularly 5 has 0 accuracy for all models.
2. There is a valley around length 10-14 created because of our tiered dataset split.
3. All reverse failures happen for number 1 and 50% of those are for digit length 1.

![rope25M accuracy by prompt length](../results/rope25M_accuracy_by_length.png)
![UT3M accuracy by prompt length](../results/UT3M_accuracy_by_length.png)
![RDT9M accuracy by prompt length](../results/RDT9M_accuracy_by_length.png)

#### All models have similar accuracy graph pattern.
![All models accuracy by prompt length](../results/100000steps_accuracy_by_length.png)

## Data: 1-22_uniform_BOS
To solve the three problems identified before we create a new dataset with following features
1. Uniform distribution of number of digits. This should solve valley for length 10-14
2. [BOS] token before prompt begins. This should solve the problem of reversal of 1st number.
3. **Example**
    -  "[BOS]123 456+?"

### Accuracy
|Models/Steps|16000|32000|48000|64000|
|---|---|---|---|---|
|ROPE|51.27%|76.00%|
|RDT|26.98%|80.46%|75.26%|82.61%|
|UT|


### Analysis
Uniform distribution helped. BOS is not enough to stop hallucinations for smaller 1st number reversal.

#### Comparative Failure % by num1 Length

 Num1 Length | ROPE_16k | ROPE_32k | RDT_16k | RDT_32k | RDT_48k | RDT_64k |
|---|---|---|---|---|---|---|
|  1 | 21.58% | 22.01% | 13.73% | 47.45% | 40.71% | 53.06% |
|  2 | 11.82% | 6.17% | 11.39% | 14.86% | 4.13% | 1.47% |
|  3 | 6.16% | 4.02% | 9.54% | 0.98% | 0.02% | 0.24% |
|  4 | 11.97% | 5.08% | 9.07% | 1.27% | 3.19% | 0.27% |
|  5 | 6.89% | 10.97% | 4.63% | 5.30% | 11.55% | 10.98% |
|  6 | 7.11% | 11.47% | 4.77% | 7.32% | 13.60% | 18.82% |
|  7 | 6.31% | 6.37% | 4.23% | 1.92% | 11.96% | 11.84% |
|  8 | 5.81% | 2.50% | 4.57% | 0.68% | 7.15% | 0.89% |
|  9 | 4.00% | 2.10% | 4.79% | 0.55% | 0.77% | 0.07% |
| 10 | 2.16% | 2.88% | 4.65% | 0.81% | 0.31% | 0.10% |
| 11 | 1.22% | 2.70% | 4.12% | 0.55% | 0.12% | 0.03% |
| 12 | 1.27% | 3.41% | 4.05% | 0.65% | 0.17% | 0.07% |
| 13 | 1.64% | 2.45% | 2.90% | 0.46% | 0.17% | 0.03% |
| 14 | 1.20% | 2.91% | 2.39% | 0.65% | 0.19% | 0.03% |
| 15 | 1.33% | 3.39% | 2.31% | 1.53% | 0.27% | 0.21% |
| 16 | 1.58% | 4.65% | 2.87% | 1.95% | 0.51% | 0.17% |
| 17 | 3.80% | 4.42% | 5.15% | 7.38% | 2.59% | 0.65% |
| 18 | 2.10% | 1.77% | 3.15% | 3.22% | 1.40% | 0.44% |
| 19 | 1.42% | 0.48% | 1.15% | 1.59% | 0.65% | 0.34% |
| 20 | 0.61% | 0.25% | 0.53% | 0.88% | 0.53% | 0.27% |
| 21 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 22 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

Shows two problems
1. num1 1 digit 
2. 5-7 digit numbers have a different problem.

#### Analysis of num1=1failures for RDT_64k.
Compare lengthof num1 and num2

Total Failures Analyzed: 2928

| num1\num2 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 455 | 238 | 154 | 114 | 63 | 50 | 72 | 65 | 62 | 50 | 57 | 52 | 46 | 26 | 16 | 15 | 15 | 1 | . | . | . | . |
| **2** | 24 | 8 | 2 | 5 | . | 2 | . | 2 | . | . | . | . | . | . | . | . | . | . | . | 1 | . | . |
| **3** | 1 | . | 1 | . | . | 2 | 1 | 2 | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| **4** | 1 | . | 1 | 1 | 1 | . | . | . | . | . | . | . | . | . | . | 3 | 1 | . | . | . | . | . |
| **5** | 11 | 18 | 26 | 37 | 13 | 27 | 18 | 20 | 16 | 23 | 24 | 26 | 11 | 11 | 11 | 9 | 10 | 10 | . | . | . | . |
| **6** | 62 | 58 | 70 | 60 | 35 | 26 | 30 | 31 | 37 | 20 | 20 | 38 | 16 | 9 | 5 | 7 | 16 | 10 | . | . | . | . |
| **7** | 49 | 58 | 39 | 46 | 22 | 13 | 10 | 8 | 9 | 11 | 10 | 21 | 5 | 4 | 10 | 14 | 16 | . | 1 | . | . | . |
| **8** | 9 | 5 | 5 | 1 | 1 | . | . | 1 | . | 1 | . | . | . | . | . | 1 | 2 | . | 1 | . | . | . |
| **9** | 1 | . | . | 1 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| **10** | 1 | . | . | . | . | 2 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| **11** | . | . | . | . | . | . | . | . | . | . | 1 | . | . | . | . | . | . | . | . | . | . | . |
| **12** | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 2 | . | . | . | . |
| **13** | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 1 | . | . | . | . |
| **14** | . | . | . | . | . | . | . | . | . | . | . | . | 1 | . | . | . | . | 1 | . | . | . | . |
| **15** | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 1 | 2 | 2 | . | . | . | . |
| **16** | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 1 | 2 | 2 | . | . | . | . | . |
| **17** | . | . | . | . | . | . | 1 | . | . | 1 | . | . | . | 1 | 2 | 3 | 14 | . | . | . | . | . |
| **18** | . | . | . | 1 | . | . | 1 | . | . | 1 | 1 | 2 | 3 | 3 | 2 | . | . | . | . | . | . | . |
| **19** | 1 | . | . | 1 | . | . | 1 | 2 | 5 | . | . | . | . | . | . | . | . | . | . | . | . | . |
| **20** | 4 | 1 | 2 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| **21** | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| **22** | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |


## Data: Bring back brackets with BOS and uniform distribution
- Changed the prompt format to \[BOS\](n1)(n2)op? (no space).
- **Example**
   -  "\[BOS\](123)(456)+?"

### Accuracy
|Models/Steps|16000|32000|48000|64000|
|---|---|---|---|---|
|ROPE|96.56%||99.94%|99.99%|
|RDT|97.64%||99.93%|
|UT|92.58%||99.84%|

### Analysis
- Reversal failed is the only reason for failures even at 16000 steps for all 3 models.
- At 16000 steps, reversal failures are not concentrated on 1 digit numbers. Instead failures are spread out, with some 20% concentration on 17 digits and a little less on than 10% on 16 and 18 digits.
- brackets are able to overcome problem faced with identifying num1. 


## Data: Document Masking + 1-22_uniform_BOS
As brackets were able to train well, thinking is that model training is complicating how model sees num1. This is because in batch training all inputs are 256 length long, so when model is guessing num1 reversal it sees a lot of data before [BOS] from previous examples in batch. So if num1 is small the data before [BOS] is even more. the Attention mechanism is trying to guess num1 reversal based on all the data it sees. To overcome this we document mask the tokens before [BOS]. 

### Accuracy
|Models/Steps|16000|32000|48000|
|---|---|---|---|
|ROPE|98.42%|99.71%|99.94%

### Analysis
- The model with document masking is able to learn as well as bracketed prompts.
- Reversal failures at 16k steps are spread evenly just like bracketed prompts. 
- Next step. Find smallest ROPE model that can solve this problem.