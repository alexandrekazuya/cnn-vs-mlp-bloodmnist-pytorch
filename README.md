# CNN vs MLP on BloodMNIST (PyTorch)

This project compares Convolutional Neural Networks (CNN) and Multi-Layer Perceptrons (MLP) for BloodMNIST image classification (MedMNIST v2) using PyTorch.

## Final Outcome

- Best single CNN run: 94.04%
- Best replicated CNN mean: 93.59%
- Best MLP run: 88.40%
- Final conclusion: CNN outperformed MLP by a clear margin on this dataset.

See full experiment logs, tables, and replicated runs in [results.md](results.md).

## Why CNN Won

CNN performed better because image-specific inductive biases matter:
- It preserves spatial information in 2D images.
- It uses local receptive fields and parameter sharing.
- It is more robust to positional variation and usually generalizes better than a flattened-input MLP.

## Project Structure

- [ml/data.py](ml/data.py): BloodMNIST dataset loading and dataloaders
- [ml/models.py](ml/models.py): CNN and MLP model definitions
- [ml/setup.py](ml/setup.py): device, criterion, and optimizer builders
- [ml/training.py](ml/training.py): training loop, early stopping, evaluation
- [experiments.ipynb](experiments.ipynb): main experiment runner
- [legacynotebook.ipynb](legacynotebook.ipynb): original legacy notebook
- [results.md](results.md): all experimental results and analysis

## Reproduce Experiments

1. Open [experiments.ipynb](experiments.ipynb).
2. Run cells from top to bottom.
3. In the configuration cell, choose model and hyperparameters.
4. Run training and evaluation cells.
5. Record outcomes in [results.md](results.md).

## Best CNN Configuration (Replicated Winner)

- Optimizer: Adam
- Learning rate: 0.0003
- Dropout: 0.5
- Weight decay: 2e-4
- Patience: 20
- Delta: 0.0005
- Batch size: 128

This was the strongest replicated setup in this project.
