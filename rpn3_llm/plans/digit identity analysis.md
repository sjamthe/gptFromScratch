# Evaluation Report: 300k-Step RPN Model with Digit-Identity Abstraction

We evaluated the Universal Transformer model trained to 300k steps with **Digit-Identity Abstraction** and **frozen Coordinate Head scale (0.5)**:
`rpn3_llm/models/ut2.1M_2l_8h_384e_mlp4_phaseMask_True_cnt2_crd2_digitAbs_freezeCoordScale_sft_1-6_4num_BOS_300000.pt`

---

## Summary: Digit identity was as failure, it made reversal better but still couldn't rever OOD digits and MATh failed completely.

## 1. High-Level SFT Validation Summary

We performed validation on a 5% subset of the in-distribution validation dataset (`sft_1-6_4num_BOS_val.txt`):

* **Validation Accuracy**: **1.74%** (41 / 2,363 correct)
* **Reversal Failures**: **0.0%** (0 / 2,363 failures) — **100% solved!**
* **Math Failures**: **98.3%** (2,322 / 2,363 failures)

### Progress from 96k Snapshot to 300k Checkpoint

| Metric | 96k Snapshot | 300k Checkpoint | Analysis / Interpretation |
| :--- | :--- | :--- | :--- |
| **Validation Accuracy** | 0.38% (9 / 2363) | **1.74%** (41 / 2363) | 4.5x increase in accuracy; the model is slowly converging on math. |
| **Reversal Failures** | 1.7% (41 / 2363) | **0.0%** (0 / 2363) | **Sequence reversal is fully solved**; copy-routing aliasing is eliminated. |
| **2-Num Equation Acc** | 1.23% (9 / 734) | **5.31%** (39 / 734) | Out-of-distribution math length starting to show signals. |
| **0-Carry Equation Acc** | 1.40% (9 / 642) | **6.07%** (39 / 642) | Arithmetic with no carry-overs is converging first. |

---

## 2. OOD Reversal Validation Results

We evaluated the 300k model on the Out-of-Distribution (OOD) pre-math validation set (`sft_1-14_7num_BOS_pre_math_val.txt`), which contains operand lengths from 1 to 14 digits and up to 7 operands (training was strictly limited to <= 6 digits and <= 4 operands).

### A. Sequence Score Nuance (0.00% Accuracy)
At first glance, the validation script returns **0.00% sequence accuracy** (0 / 1,351 correct). This is a formatting/generation ceiling mismatch:
* The OOD pre-math validation set expects generation to stop exactly at the end of the reversal segment (e.g. `[REV]8[SEP]9-=`).
* Because the model was trained on full equations, it naturally continues generating the subsequent math phase (e.g., predicting `[REV]8[SEP]9-=[MATH]8-9-`). This extra generation is flagged as a sequence failure, even though the reversal segment itself is **100% correct** for all operands with length <= 6.

### B. Digit Truncation ceiling for Lengths > 6
For operands with lengths > 6 digits, the model **fails to reverse them correctly and truncates the output to exactly 6 digits**.

#### Example (7-Digit Input):
* **Prompt**: `[BOS]1234567 9876543+?`
* **Expected Reversal**: `7654321[SEP]3456789+=` (7 digits)
* **Predicted Reversal**: `765431[SEP]345679+=` (6 digits)
* **Diagnosis**: The model skipped `2` in the first operand and `8` in the second operand, truncating both reversed outputs to exactly 6 digits.

#### Root Cause: Coordinate Ceiling under Digit Abstraction
1. Under **Digit-Identity Abstraction**, all digit values are replaced by the same abstract token embedding during reversal self-attention. This means the model **cannot** use digit-value matching (a content-matching shortcut) to guide the reversal alignment. It must rely *entirely* on the relative coordinate distance mapping computed by the Coordinate Head.
2. Since the Coordinate Head was trained only on SFT data where operand length was **strictly <= 6 digits**, the model's coordinate mapping representations have learned to track offsets up to 6 positions.
3. Without digit-value anchoring to correct the alignment over longer distances, the coordinate heads fail to map positions beyond index 6. The coordinate routing simply collapses and loops back or truncates at the 6-digit boundary.

---

## 3. The Math Positional Blur Bottleneck

While sequence reversal is completely solved, **98.3% of failures are math failures**. 

### What is Math Positional Blur?
Because digit value matching is disabled/restricted during training, the model cannot use content-matching shortcuts. Consequently, when executing column math (e.g. adding digits column-by-column), the model must query the digits of the operands purely using **relative coordinates** (index offsets). 

This is a much harder coordinate-lookup task. The model struggles with **positional blur**, which manifests in two ways:

#### A. Sticking on Duplicate Digits
When an operand contains duplicate digits in a row, the query's spatial attention smears across them.
* **Example**: Operand `217770` is reversed to `077712`.
* **Correct Math**: Adds digits column-by-column (`0`, then `7`, then `7`, then `7`, then `1`, then `2`).
* **Model's Prediction**: `7+6+0=7 : 7+6+0=7 : 7+9+0=7 : 7+0+0=7 : 1+0+0=1 : 2+0+0=2`.
* **Diagnosis**: The model completely skipped the first digit `0`, got stuck querying the duplicate `7`s for four steps, and then got back on track for `1` and `2`.

#### B. Querying Across Operands
Because coordinates are relative, queries for operand 1 sometimes blur and locate a digit in operand 3 instead.
* **Example**: Prompt `[BOS]49 50758+ 269772+?` (reversed first operands: `94` and `85705+`).
* **Correct Math**: Adds `9` (from `94`) and `8` (from `85705`).
* **Model's Prediction**: `2+8+0=0 ...`
* **Diagnosis**: The model queried `2` (the first digit of the third operand `269772`) instead of `9` from the first operand.

Coordinate-based column lookup is highly complex for multi-operand equations and carries, leading to a much slower convergence rate compared to baseline models that use shortcut content-matching.

---

## 4. Next Steps & Recommendations

To accelerate math convergence and resolve the 6-digit coordinate ceiling under Digit-Identity Abstraction:
1. **Curriculum Training with Lengths > 6**: To allow coordinate attention to track relative offsets past 6 positions, the training set must contain examples of lengths > 6 (e.g., 7 to 10 digits).
2. **Increase CoordinateHead projection dimensions**: Increasing `n_coord` (e.g. from 2 to 3 or 4) or `n_coord_heads` (e.g. from 4 to 8) can provide more resolution for relative position coordinates.
3. **Training learning rate**: A slower LR decay or higher maximum learning rate during SFT could help escape positional blur local minima.
