import torch
import torch.nn as nn
import torch.optim as optim


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_criterion(name="cross_entropy"):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "nll":
        return nn.NLLLoss()
    raise ValueError(f"Unsupported criterion: {name}")


def make_optimizer(model, name="adam", lr=0.001, momentum=0.9):
    params = model.parameters()
    if name == "adam":
        return optim.Adam(params, lr=lr)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {name}")
