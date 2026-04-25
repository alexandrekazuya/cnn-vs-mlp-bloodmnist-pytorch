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
| 1A | Cross Entropy | Adam | 0% | 78 | 68 | 0.0712 | 0.2638 | 90.47% |
| 1B | Cross Entropy | Adam | 50% | 78 | 68 | 0.1771 | 0.2048 | 92.14% |
| 2A | NLL Loss | Adam | 0% | 51 | 41 | 0.1480 | 0.2164 | 91.76% |
| 2B | NLL Loss | Adam | 50% | 62 | 40 | 0.2210 | 0.2003 | 92.63% |
| 3A | Cross Entropy | SGD | 0% | 50 | 40 | 0.0766 | 0.2382 | 91.93% |
| 3B | Cross Entropy | SGD | 25% | 58 | 40 | 0.1078 | 0.2379 | 90.85% |
| 4A | NLL Loss | SGD | 0% | 44 | 34 | 0.1102 | 0.2248 | 91.67% |
| 4B | NLL Loss | SGD | 25% | 58 | 40 | 0.1190 | 0.2185 | 91.58% |

## Extra Experiments (SGD Tuning)

Given that dropout reduced test accuracy in the SGD-based runs, and considering that SGD may require longer training to improve validation loss compared to Adam, additional experiments were conducted with:
- Dropout: 0%
- Early stopping patience: 20 epochs
- Different learning rates

| Learning Rate | Best/Stop Epoch | Training Loss | Validation Loss | Test Accuracy |
|---|---:|---:|---:|---:|
| 0.03 | 53 | 0.0588 | 0.4384 | 88.86% |
| 0.10 | 21 | 2.0053 | 2.0043 | 19.47% |
| 0.003 | 91 | 0.1232 | 0.2956 | 89.56% |

Interpretation:
1. SGD showed high sensitivity to the learning rate.
2. A high learning rate (0.10) led to clear optimization failure.
3. Learning rate 0.03 produced low training loss but poor validation/test performance, indicating overfitting.
4. Learning rate 0.003 was more stable, but still did not surpass the best SGD baseline reported earlier (91.93%) nor the best Adam configuration (92.63%).

## Key Takeaways
1. Best test accuracy was Version 2B (NLL Loss + Adam + 50% dropout): 92.63%.
2. Adding Dropout increased accuracy on average by 1,014x on Adam setups.
3. Dropout worsened accuracy for SGD setups.
4. SGD variants were competitive but slightly below the best Adam setup.
5. NLL Loss delivered a slightly higher average test accuracy than Cross Entropy (91.91% vs 91.35%), which is about 1.006x better overall.
6. Adam achieved a slightly higher average test accuracy than SGD (91.75% vs 91.51%), which is about 1.003x better overall.

## MLP Results

### Experimental Protocol
- Loss function: Cross Entropy
- Optimizers tested: Adam and SGD (single confirmation run)
- Early stopping and epoch budget were progressively increased for long-run tuning
- Architecture variants tested: 512/128, 1024/256, and 768/256 hidden sizes

### Consolidated Results

| Run | Hidden Layers | Optimizer | Learning Rate | Dropout | Patience | Delta | Weight Decay | Stopped Epoch | Best Epoch | Train Loss | Validation Loss | Test Accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | 512 / 128 | Adam | 0.0010 | 0.50 | 20 | 0.0100 | 0 | 59 | 39 | 0.8730 | 0.6224 | 73.05% |
| M2 | 512 / 128 | Adam | 0.0003 | 0.25 | 20 | 0.0100 | 0 | 62 | 42 | 0.4612 | 0.4358 | 82.43% |
| M3 | 512 / 128 | Adam | 0.0010 | 0.50 | 30 | 0.0010 | 0 | 64 | 34 | 0.8608 | 0.6728 | 73.57% |
| M4 | 512 / 128 | Adam | 0.0005 | 0.40 | 30 | 0.0010 | 0 | 100 | 92 | 0.6085 | 0.4657 | 82.40% |
| M5 | 512 / 128 | Adam | 0.0010 | 0.30 | 40 | 0.0005 | 0 | 100 | 99 | 0.4240 | 0.4135 | 83.02% |
| M6 | 512 / 128 | Adam | 0.0001 | 0.30 | 40 | 0.0005 | 0 | 250 | 230 | 0.2726 | 0.3278 | 87.23% |
| M7 | 1024 / 256 | Adam | 0.0001 | 0.30 | 50 | 0.0002 | 0 | 298 | 248 | 0.2507 | 0.3436 | 87.43% |
| M8 | 768 / 256 | Adam | 0.0001 | 0.30 | 50 | 0.0002 | 0 | 350 | 341 | 0.2236 | 0.3366 | 87.23% |
| M9 | 512 / 128 | Adam | 0.0001 | 0.30 | 50 | 0.0002 | 5e-5 | 329 | 279 | 0.1752 | 0.3188 | 88.40% |
| M10 | 512 / 128 | Adam | 0.0001 | 0.30 | 50 | 0.0002 | 5e-4 | 330 | 280 | 0.2430 | 0.3307 | 87.43% |
| M11 | 512 / 128 | SGD | 0.0010 | 0.00 | 50 | 0.0002 | 5e-4 | 413 | 363 | 0.1918 | 0.3442 | 86.70% |

### Key Takeaways (MLP)
1. Best MLP test accuracy was 88.40% (M9: Adam, LR 0.0001, dropout 0.3, patience 50, delta 0.0002, weight decay 5e-5).
2. Longer training with lower learning rate was critical: MLP performance improved from 73.05% (M1) to 88.40% (M9).
3. Adam consistently outperformed the single SGD confirmation run (88.40% vs 86.70%).
4. Moderate regularization helped most: weight decay 5e-5 was better than 5e-4 and no weight decay in the strongest long runs.
5. Increasing MLP width alone did not beat the best regularized 512/128 setup in this project.
6. Final comparison supports the assignment goal: CNN remained better than MLP on BloodMNIST (92.63% vs 88.40%, a 4.23-point gap).


## CNN Additional Tuning (Replicated Runs)

### Tested Configurations

| ID | Optimizer | LR | Dropout | Weight Decay | Patience | Delta | Batch Size | Best Epoch Info | Test Accuracy Runs | Mean Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| C1 | Adam | 0.0005 | 0.5 | 1e-4 | 30 | 0.0002 | 128 | best=21, stopped=51, train=0.0926, val=0.1752 | 92.52 | 92.52% |
| C2 | Adam | 0.0003 | 0.5 | 2e-4 | 20 | 0.0005 | 128 | best=34, stopped=54, train=0.0569, val=0.1682 | 94.04, 93.25, 93.48 | 93.59% |
| C3 | Adam | 0.0005 | 0.5 | 3e-4 | 20 | 0.0005 | 128 | best=15, stopped=30 | 92.05 | 92.05% |
| C4 | Adam | 0.0003 | 0.6 | 2e-4 | 15 | 0.0005 | 128 | best=21, stopped=36, train=0.1349, val=0.1727 | 92.72 | 92.72% |
| C5 | Adam | 0.0003 | 0.5 | 2e-4 | 20 | 0.0005 | 64 | best=19, stopped=34, train=0.1046, val=0.1499 | 93.80, 93.28, 91.67 | 92.92% |

### Best Run Summary
1. Best single CNN result observed: 94.04% (Configuration C2).
2. Best replicated mean result: 93.59% (Configuration C2, batch size 128).
3. Configuration C2 also produced a stronger mean than the batch size 64 variant (C5: 92.92%).



## Why (CNN vs MLP)?
CNNs are significantly superior to MLPs for image classification because they preserve spatial structure and use parameter sharing, whereas MLPs flatten images, causing loss of context, higher complexity, and lower accuracy
CNNs excel at feature extraction (edges, patterns) through local connections, while MLPs are prone to overfitting and struggle with spatial variations.

Spatial Information: CNNs maintain the 2D structure of images, recognizing spatial correlations between pixels. MLPs flatten images into a 1D vector, losing this structure.

Parameters: CNNs use fewer parameters due to parameter sharing (kernels/filters), making them efficient. MLPs are fully connected, resulting in too many parameters and, consequently, high redundancy and computational cost.

Translation Invariance: CNNs can identify features regardless of their location in the image. MLPs struggle if an object shifts location (e.g., top-left vs. bottom-right).