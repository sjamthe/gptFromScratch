# Implementation Plan - RPN3 Dataset Creation (Updated Spacing)

This plan outlines the steps to create a new dataset for 3-number RPN arithmetic in a new folder `rpn3_llm`. The dataset will mix 2-number and 3-number operations and use a specific, scalable scratchpad pattern with highly compressed spacing.

## User Review Required

> [!IMPORTANT]
> Please review the updated spacing format based on your feedback:
> - Prompt: `123 456+976-?` (No space before 3rd number `976`)
> - Reversal: `[REV]321 654+679-=` (No space before `revnum3` `679`)
> - Math 1: `[MATH]...=975` (No spaces around `=`)
> - Transition: `:975 679-=` (I kept the space here to separate the intermediate result `975` from the 3rd number `679`. Please let me know if you want this removed too, e.g., `:975679-=`).

## Proposed Changes

### New Component: `rpn3_llm`

I will create a new directory `rpn3_llm` to isolate this new phase of the project.

#### [NEW] [create_dataset.py](file:///Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn3_llm/create_dataset.py)

This script will generate the training and validation datasets.

**Format Specifications:**
*   **3-Number Prompt:** `[BOS]num1 num2<op1>num3<op2>?` -> `123 456+976-?`
*   **3-Number Solution:** `[REV]revnum1 revnum2<op1>revnum3<op2>=[MATH]...=revans1:revans1 revnum3<op2>=[MATH]...=revans2[ANS]ans[EOS]`
*   **2-Number Prompt:** `[BOS]num1 num2+?`
*   **2-Number Solution:** `[REV]revnum1 revnum2+=[MATH]...=revans1[ANS]ans[EOS]`

## Verification Plan

### Automated Tests
- I will write a small validation function in `create_dataset.py` to parse a few generated lines and verify they are mathematically correct and follow the requested format.
- I will check that the generated files contain a balanced mix of 2 and 3 number problems.
