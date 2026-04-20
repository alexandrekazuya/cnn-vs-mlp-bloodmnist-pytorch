# CNN Results

## Experimental Protocol
- Early stopping patience: 10 epochs
- Model family: CNN
- Main variants tested:
1. Cross Entropy + Adam
2. NLL Loss + Adam
3. Cross Entropy + SGD
4. NLL Loss + SGD
- Dropout variants tested: none, 25%, 50%

## Consolidated Results

| Version | Loss Function | Optimizer | Dropout | Stopped At Epoch | Best Epoch | Train Loss | Validation Loss | Test Accuracy |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1A | Cross Entropy | Adam | 0% | 78 | 68 | 0.0712 | 0.2931 | 90.47% |
| 1B | Cross Entropy | Adam | 50% | 78 | 68 | 0.1771 | 0.2276 | 92.14% |
| 2A | NLL Loss | Adam | 0% | 51 | 41 | 0.1480 | 0.2404 | 91.76% |
| 2B | NLL Loss | Adam | 50% | 62 | 40 | 0.2210 | 0.2225 | 92.63% |
| 3A | Cross Entropy | SGD | 0% | 50 | 40 | 0.0766 | 0.2647 | 91.93% |
| 3B | Cross Entropy | SGD | 25% | 58 | 40 | 0.1078 | 0.2643 | 90.85% |
| 4A | NLL Loss | SGD | 0% | 44 | 34 | 0.1102 | 0.2498 | 91.67% |
| 4B | NLL Loss | SGD | 25% | 58 | 40 | 0.1190 | 0.2428 | 91.58% |

## Key Takeaways
1. Best test accuracy was Version 2B (NLL Loss + Adam + 50% dropout): 92.63%.
2. Adding Dropout increased accuracy on average by 1,014x on Adam setups.
3. Dropout worsened accuracy for SGD setups.
4. SGD variants were competitive but slightly below the best Adam setup.
5. NLL Loss delivered a slightly higher average test accuracy than Cross Entropy (91.91% vs 91.35%), which is about 1.006x better overall.
6. Adam achieved a slightly higher average test accuracy than SGD (91.75% vs 91.51%), which is about 1.003x better overall.