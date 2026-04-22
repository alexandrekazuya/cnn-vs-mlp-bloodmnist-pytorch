# cnn-vs-mlp-bloodmnist-pytorch

This project explores supervised image classification on BloodMNIST (MedMNIST v2) using PyTorch.

Current status:
- Implemented a reusable training pipeline with train/validation/test splits.
- Added modular code for data loading, models, setup, and training utilities.
- Added a clean experiment notebook to run CNN or MLP from a single switch.
- Recorded experimental results and comparisons in [results.md](results.md).

## Project structure

- `ml/data.py`: BloodMNIST dataset + dataloaders
- `ml/models.py`: `CNN` and `MLP` model classes
- `ml/setup.py`: device, loss, and optimizer builders
- `ml/training.py`: `EarlyStopping`, training loop, and test evaluation
- `experiments.ipynb`: notebook runner for experiments
- `notebook.ipynb`: original working notebook

## Quick start (MLP or CNN)

1. Open `experiments.ipynb`.
2. Run cells from top to bottom.
3. In the config cell, set:
	- `MODEL_NAME = 'mlp'` for MLP baseline or `MODEL_NAME = 'cnn'` for CNN
	- `OPTIMIZER_NAME = 'adam'` or `'sgd'`
4. Run training and evaluation cells.

This flow runs only the selected model, so you can train MLP without running CNN.
