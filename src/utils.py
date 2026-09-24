"""Plotting helpers for training curves, confusion matrices and samples."""
import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_histories(no_drop, drop, save_dir):
    """Plot train/test accuracy and loss for the with/without dropout runs.

    Writes training_accuracy.png and training_loss.png into ``save_dir``.
    ``no_drop`` and ``drop`` are dicts with keys:
        train_acc, test_acc, train_loss, test_loss  (lists of length epochs)
    """
    epochs = range(1, len(no_drop["train_acc"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, no_drop["train_acc"], "ro-", ms=3, label="No dropout - train")
    ax.plot(epochs, no_drop["test_acc"], "bo-", ms=3, label="No dropout - test")
    ax.plot(epochs, drop["train_acc"], "r--", ms=3, label="Dropout - train")
    ax.plot(epochs, drop["test_acc"], "b--", ms=3, label="Dropout - test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MNIST MLP: Dropout vs No Dropout (accuracy)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/training_accuracy.png", dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, no_drop["train_loss"], "ro-", ms=3, label="No dropout - train")
    ax.plot(epochs, no_drop["test_loss"], "bo-", ms=3, label="No dropout - test")
    ax.plot(epochs, drop["train_loss"], "r--", ms=3, label="Dropout - train")
    ax.plot(epochs, drop["test_loss"], "b--", ms=3, label="Dropout - test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("MNIST MLP: Dropout vs No Dropout (loss)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/training_loss.png", dpi=150)
    plt.show()


def plot_confusion_matrix(cm, classes=range(10), path="results/confusion_matrix.png"):
    """Plot and save a confusion matrix (cm: 10x10 numpy array)."""
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues")
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks(list(classes))
    ax.set_yticks(list(classes))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (counts)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


@torch.no_grad()
def predict(model, images, device):
    """Softmax probabilities for a batch of images."""
    model.eval()
    logits = model(images.to(device))
    return torch.softmax(logits, dim=1).cpu()


def show_samples(images, labels, preds, path, rows=1):
    """Render a grid of (image, true label, predicted class, confidence)."""
    import numpy as np
    n = images.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.6))
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        ax.imshow(images[k].squeeze(), cmap="gray")
        ax.set_title(f"True: {labels[k]}\nPred: {preds[k][0]} ({preds[k][1]:.2f})", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def evaluate(model, loader, device):
    """Return (accuracy %, confusion matrix) over a DataLoader."""
    model.eval()
    correct = 0
    total = 0
    cm = np.zeros((10, 10), dtype=int)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels.cpu(), predicted.cpu()):
                cm[t, p] += 1
    return 100.0 * correct / total, cm