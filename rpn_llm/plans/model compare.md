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

#### Comparative Accuracy by num1 Length

| Num1 Length | ROPE_16k | ROPE_32k | RDT_16k | RDT_32k | RDT_48k | RDT_64k |
|---|---|---|---|---|---|---|
|  1 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
|  2 | 0.00% | 68.12% | 0.00% | 40.29% | 77.66% | 94.38% |
|  3 | 34.81% | 79.23% | 0.00% | 96.08% | 99.87% | 99.09% |
|  4 | 0.00% | 73.74% | 0.00% | 94.90% | 82.75% | 98.95% |
|  5 | 27.10% | 43.30% | 24.48% | 78.70% | 37.55% | 58.06% |
|  6 | 24.75% | 40.69% | 22.13% | 70.60% | 26.44% | 28.14% |
|  7 | 33.24% | 67.08% | 30.89% | 92.29% | 35.33% | 54.80% |
|  8 | 38.59% | 87.07% | 25.40% | 97.26% | 61.33% | 96.60% |
|  9 | 57.67% | 89.16% | 21.74% | 97.78% | 95.82% | 99.74% |
| 10 | 77.14% | 85.11% | 24.09% | 96.73% | 98.30% | 99.61% |
| 11 | 87.07% | 86.02% | 32.72% | 97.78% | 99.35% | 99.87% |
| 12 | 86.54% | 82.36% | 33.89% | 97.39% | 99.09% | 99.74% |
| 13 | 82.62% | 87.33% | 52.57% | 98.17% | 99.09% | 99.87% |
| 14 | 87.33% | 84.98% | 60.94% | 97.39% | 98.95% | 99.87% |
| 15 | 85.89% | 82.49% | 62.24% | 93.86% | 98.56% | 99.22% |
| 16 | 83.28% | 75.96% | 53.10% | 92.16% | 97.26% | 99.35% |
| 17 | 59.76% | 77.14% | 15.99% | 70.34% | 86.02% | 97.52% |
| 18 | 77.79% | 90.85% | 48.52% | 87.07% | 92.42% | 98.30% |
| 19 | 84.98% | 97.52% | 81.19% | 93.60% | 96.47% | 98.69% |
| 20 | 93.60% | 98.69% | 91.38% | 96.47% | 97.13% | 98.95% |
| 21 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| 22 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

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

