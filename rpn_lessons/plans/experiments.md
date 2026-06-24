# Experiments - Collection of experiments that we want to run

## Ex1: Universal Attention
* **Date**: 2026-06-22
* **Base Experiment**: `lesson1_ut_1.5M_T384_wrappedNum_no_tie` 
* **Base Experiment WanDB link**: https://wandb.ai/sjamthe/rpn-curriculum/runs/9mjr49np/overview?nw=nwusersjamthe

**Training Commands:**
uv run python rpn_lessons/train.py --lesson 1 --run_name_suffix wrappedNum  --use_universal_attn

