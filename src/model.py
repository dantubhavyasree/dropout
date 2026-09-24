"""MLP network for the MNIST dropout reproduction.

Follows the feed-forward architecture used in the paper and the basic Colab
implementation: 784 inputs -> 1024 ReLU -> 1024 ReLU -> 10 softmax.

An `nn.Dropout` layer (inverted dropout, the form used in Eq. (5) of
Srivastava et al., 2014) is optionally placed after each hidden ReLU.
`nn.Dropout(p)` drops a unit with probability p and multiplies the
survivors by 1/(1-p), so no explicit scaling is needed at test time.
"""
import torch
import torch.nn as nn


class Net(nn.Module):
    """784 -> 1024 -> 1024 -> 10 ReLU MLP with optional dropout."""

    def __init__(self, dropout: bool = False, keep_prob: float = 0.5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(1.0 - keep_prob) if dropout else nn.Identity()

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)