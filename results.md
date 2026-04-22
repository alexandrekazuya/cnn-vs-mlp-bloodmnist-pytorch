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

MLP

Epoch 39	Training Loss: 0.8730	Validation Loss: 0.6224
Stopped epoch: 59
Best epoch: 39
Test accuracy: 73.05%

LEARNING_RATE = 0.001  # 0.01 for sgd
MOMENTUM = 0.9
DROPOUT = 0.5
PATIENCE = 20
EARLY_STOPIING_DELTA = 0.01

Epoch 62	Training Loss: 0.4612	Validation Loss: 0.4358
Stopped epoch: 62
Best epoch: 42
Test accuracy: 82.43%

LEARNING_RATE = 0.0003  # 0.01 for sgd
MOMENTUM = 0.9
DROPOUT = 0.25
PATIENCE = 20
EARLY_STOPIING_DELTA = 0.01


Epoch 64	Training Loss: 0.8608	Validation Loss: 0.6728
Stopped epoch: 64
Best epoch: 34
Test accuracy: 73.57%
LEARNING_RATE = 0.001  # 0.01 for sgd
MOMENTUM = 0.9
DROPOUT = 0.5
PATIENCE = 30
EARLY_STOPIING_DELTA = 0.001



Epoch 100	Training Loss: 0.6085	Validation Loss: 0.4657
Stopped epoch: 100
Best epoch: 92
Test accuracy: 82.40%
LEARNING_RATE = 0.0005
DROPOUT = 0.4
patience = 30
delta = 0.001


Epoch 100	Training Loss: 0.4240	Validation Loss: 0.4135
Stopped epoch: 100
Best epoch: 99
Test accuracy: 83.02%
LEARNING_RATE = 0.001
DROPOUT = 0.3
patience = 40
delta = 0.0005


def __init__(self, input_dim=3 * 28 * 28, hidden1=512, hidden2=128, num_classes=8, dropout=0.5):
LEARNING_RATE = 0.0001   # 0.01 for sgd
MOMENTUM = 0.9
DROPOUT = 0.3
PATIENCE = 40
EARLY_STOPIING_DELTA = 0.0005
Epoch 250	Training Loss: 0.2726	Validation Loss: 0.3278
Stopped epoch: 250
Best epoch: 230
Test accuracy: 87.23%


def __init__(self, input_dim=3 * 28 * 28, hidden1=1024, hidden2=256, num_classes=8, dropout=0.5):
LEARNING_RATE = 0.0001   # 0.01 for sgd
MOMENTUM = 0.9
DROPOUT = 0.3
PATIENCE = 50
EARLY_STOPIING_DELTA = 0.0002
epoch 298	Training Loss: 0.2507	Validation Loss: 0.3436
Stopped epoch: 298
Best epoch: 248
Test accuracy: 87.43%

def __init__(self, input_dim=3 * 28 * 28, hidden1=768, hidden2=256, num_classes=8, dropout=0.5):
LEARNING_RATE = 0.0001   # 0.01 for sgd
MOMENTUM = 0.9
DROPOUT = 0.3
PATIENCE = 50
EARLY_STOPIING_DELTA = 0.0002
Epoch 350	Training Loss: 0.2236	Validation Loss: 0.3366
Stopped epoch: 350
Best epoch: 341
Test accuracy: 87.23%