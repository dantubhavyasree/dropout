"""MNIST dataset loading and splitting.

Same setup as the basic Colab implementation:
  * 60,000 training images are randomly split into 50,000 train + 10,000
    validation (fixed seed for reproducibility);
  * the official 10,000-image test set is kept separate for evaluation.
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def mnist_loaders(batch_size: int = 128, data_dir: str = "./data", seed: int = 42):
    """Return (train_loader, test_loader, val_loader)."""
    transform = transforms.ToTensor()

    train_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [50000, 10000], generator=gen)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader


def device():
    """Pick the best available device (GPU if present)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")