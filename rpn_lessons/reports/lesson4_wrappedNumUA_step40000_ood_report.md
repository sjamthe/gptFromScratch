# Curriculum Learning Out-of-Distribution (OOD) Performance Report

This report compiles performance data of the Universal Transformer model (`lesson4_wrappedNumUA_step40000.pt`) across gradual OOD scale shifts. In compliance with variables isolation, we tested only one OOD dimension (Digit Length or Operand Count) at a time in steps of 1 above the training thresholds.

## 1. Lesson 1: Reversal Generalization (Digit-scale)
* **In-Distribution limit**: 22 digits.

| Digit Length | Exact Match Accuracy |
|:---:|:---:|
| 23 | 95.00% |
| 24 | 97.00% |
| 25 | 97.00% |
| 26 | 91.00% |
| 27 | 91.00% |
| 28 | 91.00% |
| 29 | 84.00% |
| 30 | 80.00% |

## 2. Lesson 2: Multi-operand Reversal Generalization (Workspace capacity)
* **In-Distribution limits**: 9 digits, 6 operands.

### Digit Length Scaling (with 2 operands)
| Digit Length | Exact Match Accuracy |
|:---:|:---:|
| 10 | 97.00% |
| 11 | 97.00% |
| 12 | 96.00% |
| 13 | 87.00% |

### Operand Count Scaling (with 4 digits)
| Operand Count | Exact Match Accuracy |
|:---:|:---:|
| 7 | 95.00% |
| 8 | 1.00% |
| 9 | 0.00% |
| 10 | 0.00% |

## 3. Lesson 3: Step-by-Step Math Generalization (Alignment & Carry)
* **In-Distribution limits**: 9 digits, 6 operands.

### Digit Length Scaling (with 2 operands)
| Digit Length | Exact Match Accuracy |
|:---:|:---:|
| 10 | 0.00% |
| 11 | 0.00% |
| 12 | 0.00% |
| 13 | 0.00% |

### Operand Count Scaling (with 4 digits)
| Operand Count | Exact Match Accuracy |
|:---:|:---:|
| 7 | 66.00% |
| 8 | 0.00% |
| 9 | 0.00% |
| 10 | 0.00% |

## 4. Lesson 4: Result Reversal & Phase Transitions
* **In-Distribution limits**: 9 digits, 6 operands.

### Digit Length Scaling (with 2 operands)
| Digit Length | Exact Match Accuracy |
|:---:|:---:|
| 10 | 36.00% |
| 11 | 3.00% |
| 12 | 0.00% |
| 13 | 0.00% |

### Operand Count Scaling (with 4 digits)
| Operand Count | Exact Match Accuracy |
|:---:|:---:|
| 7 | 90.00% |
| 8 | 0.00% |
| 9 | 0.00% |
| 10 | 0.00% |

## 5. End-to-End State Machine Integration
* **In-Distribution limits**: 9 digits, 6 operands.

### Digit Length Scaling (with 3 operands)
| Digit Length | Exact Match Accuracy |
|:---:|:---:|
| 10 | 0.00% |
| 11 | 0.00% |
| 12 | 0.00% |

### Operand Count Scaling (with 4 digits)
| Operand Count | Exact Match Accuracy |
|:---:|:---:|
| 7 | 0.00% |
| 8 | 0.00% |
| 9 | 0.00% |
| 10 | 0.00% |

## 6. Failure Analysis & Interpretations

### A. Reversal Scaling (Lesson 1 & 2)
> [!NOTE]
> **Observation**: Reversal pointer logic uses Coordinate heads to track token spaces. Explain below where performance starts degrading as digits increase from 23 to 30.

### B. Step-by-Step Math Scaling (Lesson 3 & 4)
> [!NOTE]
> **Observation**: Math execution requires both alignment tracking and carry tracking. Increasing operands requires longer tail-copy sequences. Explain below the impact of digit length vs. operand counts on math logic.

### C. End-to-End Generalization
> [!NOTE]
> **Observation**: E2E combines all lessons. Explain if compounding errors or coordinate drift dominates OOD failures.
