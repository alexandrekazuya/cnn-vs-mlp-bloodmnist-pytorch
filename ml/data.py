import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO


def build_bloodmnist_dataloaders(batch_size=128, download=True):
    """Build train/val/test dataloaders and metadata for BloodMNIST."""
    data_flag = "bloodmnist"
    info = INFO[data_flag]
    data_class = getattr(medmnist, info["python_class"])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = data_class(split="train", transform=data_transform, download=download)
    val_dataset = data_class(split="val", transform=data_transform, download=download)
    test_dataset = data_class(split="test", transform=data_transform, download=download)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=2 * batch_size, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=2 * batch_size, shuffle=False)

    return {
        "data_flag": data_flag,
        "n_channels": info["n_channels"],
        "n_classes": len(info["label"]),
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }
