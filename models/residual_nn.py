import torch
import torch.nn as nn


class ResidualNN(nn.Module):
    """
    A lightweight 3-layer Multi-Layer Perceptron (MLP) designed to
    capture non-linear residuals left over by an interpretable base model.
    """

    def __init__(self, input_dim: int = 1, hidden_dim1: int = 32, hidden_dim2: int = 16):
        super(ResidualNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
