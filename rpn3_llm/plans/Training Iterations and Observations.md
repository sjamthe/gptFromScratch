# Training Iterations and Observations for 3num math

## Iteration 1

Model: **ut0.7M_2l_8h_256e_mlp3_phaseMask_True_rpn3_3num_328000.pt**

Hyperparameters: All default except mlp3 (2 layer, 8h heads, 256 embedding size, blocksize 2048)

Training Parameters:
- B = 16
- T = 768
- max_lr = 1e-4
- min_lr = max_lr * 0.1
- warmup_steps = 1000
- lr_decay_steps = 65536

Observations: Models trains well to 60k steps then slows down (based on accuracy observations).
- REV2 is the slowest to train. 
- Most (57%) of the errors are in REV2.
- All of REV2 errors are same length swapped digits or hallucinated digits.

| Data Type | Steps | Total Accuracy% | REV1 Errors | MATH1 Errors | REV2 Errors | MATH2 Errors | ANS Errors |
| --- | --- | --- | --- | --- | --- | --- |
|3-num  | 104000| 65.47%| 33 | 7 | 215 | 4 | 39 |
|3-num  | 176000| 73.12%| 29 | 7 | 149 | 5 | 42 |
|3-num  | 328000| 72.42%| 31 | 7 | 135 | 11 | 54 |

## Iteration 2

Model: **ut0.7M_2l_8h_256e_mlp3_phaseMask_True_tau0.8_rpn3_3num_176000.pt**

Hyperparameters: All same as above except tau=0.8

Training Parameters:
All same as above

Observations: Did worse thn above though tau 0.8 should have increased REV2 accuracy!

|Data Type | Steps | Total Accuracy% | REV1 Errors | MATH1 Errors | REV2 Errors | MATH2 Errors | ANS Errors |
| --- | --- | --- | --- | --- | --- | --- |
|3-num  | 176000| 67.67%| 46 | 12 | 159 | 16 | 46 |

## Iteration 3

Model: **ut0.7M_2l_8h_256e_mlp3_phaseMask_True_rpn3_3num_152000.pt**
Repeat same model 1 with longer lr_decay_steps as we saw REV2 learning  tappered off at 60k steps.

Hyperparameters: Sale as model 1
- max_lr = 1e-4 (same)

Training Parameters: All same as model 1 except:
- lr_decay_steps = 120000

|Data Type | Steps | Total Accuracy% | REV1 Errors | MATH1 Errors | REV2 Errors | MATH2 Errors | ANS Errors |
| --- | --- | --- | --- | --- | --- | --- |
|3-num  | 104000| 72.54%| 22 |  1 | 161 | 5 | 48 |
|3-num  | 156000| 71.49%| 29 | 10 | 149 | 8 | 50 |

## Iteration 4

Model: **ut0.7M_2l_8h_256e_mlp3_phaseMask_True_rpn3_3num_152000.pt**
Repeat retrain model3 from 104000 steps, but higher lr_decay_steps.
Hyperparameters: Sale as model 3

Training Parameters: All same as model 3 except:
- lr_decay_steps = 240000
- Start at 104000 steps

Observations: Accuracy dropped 70 43% from 72% when training was resumed at 104k steps. It didn't recover from that. At 152k accuacy was 62% so killed it.

## Iteration 5

Model: **ut0.7M_2l_8h_256e_mlp3_phaseMask_True_rpn3_3num_152000.pt**
Same model as 1, train from scratch. max_lr increased to 3e-4.

Training Parameters: All same as model 1 except:
- max_lr = *3e-4*
- min_lr = max_lr * 0.1
- warmup_steps = 1000
- lr_decay_steps = 156000

Observations: 
| Data Type | Steps | Total Accuracy% | REV1 Errors | MATH1 Errors | REV2 Errors | MATH2 Errors | ANS Errors |
| --- | --- | --- | --- | --- | --- | --- |
|3-num  | 104000| 64.56%  | 33 | 29 | 169 | 19 | 56 |
|3-num  | 156000| 66.74%  | 41 | 33 | 112 | 32 | 69 |

## Iteration 6

Model parameters
- n_dim = 384

Training Parameters: All same as model 1 except:
- max_lr = *3e-4*
- min_lr = max_lr * 0.1
- warmup_steps = 1000
- lr_decay_steps = 200000

Observations: 
Model peek performance was at 59k then it started declining as it overfitted.
Every error except rev2 increased from 56k to 156k.

|Data Type | Steps | Total Accuracy% | REV1 Errors | MATH1 Errors | REV2 Errors | MATH2 Errors | ANS Errors |
| --- | --- | --- | --- | --- | --- | --- |
|3-num  |  56000| 71.84%  | 34 | 38 | 82 | 28 | 61 |
|3-num  | 156000| 65.24%  | 55 | 49 | 73 | 36 | 87 |

## Iteration 7

Model parameters
- n_dim = 384

Data
- Added [SEP] token before 3rd rev number in REV2.

Training Parameters: All same as model 1 except:
- max_lr = *3e-4*
- min_lr = max_lr * 0.1
- warmup_steps = 1000
- lr_decay_steps = 200000


|Data Type | Steps | Total Accuracy% | REV1 Errors | MATH1 Errors | REV2 Errors | MATH2 Errors | ANS Errors |
| --- | --- | --- | --- | --- | --- | --- |
|3-num  | 156000| 89.26%  | 55 | 50 | 86 | 21 | 718 |

## Iteration 8
 Same model as 7, same training parameters. Just changed the data file to use both 2-num and 3-num datasets.

model: ut1.5M_2l_8h_384e_mlp3_phaseMask_True_rpn3_160000.pt

| Data Type | Steps | Total Accuracy% | REV1 Errors | MATH1 Errors | REV2 Errors | MATH2 Errors | ANS Errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3-num  | 160000| 89.00%  | 15 | 26 | 170 | 33 | 708 |
| 2-num  | 160000| 99.63%  | 9 | 16 | 0 | 0 | 4 |

## Iteration 8
Same model as above. Trained from scratch on fixed data.

We found problems in data generation script. Negative number additions were failing so final ANS was not correct in the data.

| Data Type | Steps | Total Accuracy% | REV1 Errors | MATH1 Errors | REV2 Errors | MATH2 Errors | ANS Errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3-num  | 216000| 97.66%  | 14 | 12 | 43 | 100 | 30 |

## iteration 9
Model 8 did perfect after we fixed the data generation script. We also fixed create_dataset.py to use "[SEP]" between num1 & num2, this model got 100% correcct at 240k snapshot
ut1.5M_2l_8h_384e_mlp3_phaseMask_True_rpn3_240000.pt

| Data Type | Steps | Total Accuracy% | REV1 Errors | MATH1 Errors | REV2 Errors | MATH2 Errors | ANS Errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3-num  | 240000| 97.97%  | 173 | 0 | 0 | 0 | 0 |

3-num: Total: 8521 | Correct: 8348 | Accuracy: 97.97%
  Failures -> Rev: 173, Math: 0 (Math1: 0, Rev2: 0, Math2: 0), Ans: 0
## Closing this work and moving to new work of multiple number operations. 