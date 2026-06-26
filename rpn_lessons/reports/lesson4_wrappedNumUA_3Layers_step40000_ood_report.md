# Curriculum Learning Out-of-Distribution (OOD) Performance Report

This report compiles performance data of the Universal Transformer model (`lesson4_wrappedNumUA_3Layers_step40000.pt`) across gradual OOD scale shifts. In compliance with variables isolation, we tested only one OOD dimension (Digit Length or Operand Count) at a time in steps of 1 above the training thresholds.

## 1. Lesson 1: Reversal Generalization (Digit-scale)
* **In-Distribution limit**: 22 digits.

| Digit Length | Exact Match Accuracy |
|:---:|:---:|
| 23 | 95.00% |
| 24 | 67.00% |
| 25 | 23.00% |
| 26 | 3.00% |
| 27 | 0.00% |
| 28 | 0.00% |
| 29 | 0.00% |
| 30 | 0.00% |

## 2. Lesson 2: Multi-operand Reversal Generalization (Workspace capacity)
* **In-Distribution limits**: 9 digits, 6 operands.

### Digit Length Scaling (with 2 operands)
| Digit Length | Exact Match Accuracy |
|:---:|:---:|
| 10 | 99.00% |
| 11 | 85.00% |
| 12 | 18.00% |
| 13 | 0.00% |

### Operand Count Scaling (with 4 digits)
| Operand Count | Exact Match Accuracy |
|:---:|:---:|
| 7 | 88.00% |
| 8 | 0.00% |
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
| 7 | 67.00% |
| 8 | 0.00% |
| 9 | 0.00% |
| 10 | 0.00% |

## 4. Lesson 4: Result Reversal & Phase Transitions
* **In-Distribution limits**: 9 digits, 6 operands.

### Digit Length Scaling (with 2 operands)
| Digit Length | Exact Match Accuracy |
|:---:|:---:|
| 10 | 37.00% |
| 11 | 5.00% |
| 12 | 0.00% |
| 13 | 0.00% |

### Operand Count Scaling (with 4 digits)
| Operand Count | Exact Match Accuracy |
|:---:|:---:|
| 7 | 68.00% |
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

### Comparison with previous results
2 Layer has better OOD performance than 3 Layer for lessson 1,2.
 

