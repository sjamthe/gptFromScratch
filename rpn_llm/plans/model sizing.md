# Model sizing 

Try to create smallest model that achieves 99% accuracy on the test set.

# Hyperparameter tuning 

## 1st model (7.1M Parameters) - [SUCCESS]
 - n_layer = 4
 - n_head = 6
 - n_embd = 384
 - block_size = 512 (Achieved 99.74% accuracy at 48k steps)
 - Notes: Increasing T to 512 solved the 17-22 digit bottleneck. Reversal is 99.9% solved.

## 2nd model (3.5M Parameters) - [TARGET]
 - n_layer = 2
 - n_head = 4
 - n_embd = 256
 - block_size = 512
 - Goal: Verify if 2 layers are enough to maintain carry logic across 22 digits.
