#!/bin/bash

#if [ -z "$1" ]; then
#  echo "Usage: ./run_experiments.sh <PID_OF_ROPE_RUN>"
#  echo "Example: ./run_experiments.sh 12345"
#  exit 1
#fi

#ROPE_PID=$1

#echo "Waiting for ROPE model (PID: $ROPE_PID) to finish..."

# Loop and check if the process is still running every 60 seconds
#while kill -0 $ROPE_PID 2>/dev/null; do
#    sleep 60
#done
.venv/bin/python3 rpn_llm/train_rpn.py 62277 rpn_llm/models/rope25M_1-22_tens_comp_clean_tiered_62227.pt --model rope --max_steps 100000

echo "ROPE model finished! Starting UT model..."
.venv/bin/python3 rpn_llm/train_rpn.py 80000 rpn_llm/models/UT3M_1-22_tens_comp_clean_tiered_80000.pt --model ut --max_steps 100000

echo "UT model finished! Starting RDT model..."
.venv/bin/python3 rpn_llm/train_rpn.py 80000 rpn_llm/models/RDT9M_1-22_tens_comp_clean_tiered_80000.pt --model rdt --max_steps 100000

echo "All experiments completed successfully!"
