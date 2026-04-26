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
3. *** Example **
    -  "[BOS]123 456+?"

### Accuracy
|Models/Steps|16000|32000|48000|64000|80000|96000|100000|
|---|---|---|---|---|---|---|---|
|ROPE|
|UT|
|RDT|26.98%|80.46%|

