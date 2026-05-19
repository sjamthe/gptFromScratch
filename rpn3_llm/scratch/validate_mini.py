import sys
import os

# Ensure working directory is in python path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "rpn3_llm"))

import rpn3_llm.validate_rpn_model as val
val.VALIDATION_SET_RATIO = 1.0  # Override to process 100% of mini_val.txt!

# Now run it
val.validate_model(
    checkpoint_path="rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_theta_sftRecal_sft_1-14_7num_BOS_40000.pt",
    test_file_path="rpn3_llm/data/mini_val.txt",
    output_fail_path="rpn3_llm/results/ut1.8M_recal_mini_val_failures.txt"
)
print("Validation finished successfully!")
