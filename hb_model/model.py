"""
MLP regression model for hemoglobin estimation.

HbMLP predicts hemoglobin concentration from a low-dimensional feature
vector extracted from facial signals.

Unlike the HR model, which operates directly on temporal signals,
the Hb model uses engineered amplitude and colour features. A small
multi-layer perceptron is used to reduce the risk of memorizing
subject-specific characteristics because hemoglobin is a subject-level
target.

The target can be standardized during training. Target normalization and
de-normalization are handled by the training pipeline rather than by this
model.
"""

import torch.nn as nn
from torch import Tensor


class HbMLP(nn.Module):
    """
    Small multilayer perceptron for hemoglobin regression.

    Args:
        n_in: Number of input features.
        width: Number of hidden units in the first layer.
        dropout: Dropout probability used after each hidden activation.

    Returns:
        A scalar hemoglobin prediction for each input sample.
    """

    def __init__(self, n_in: int, width: int = 512, dropout: float = 0.3) -> None:
        """
        Initialize the feed-forward regression network.

        Args:
            n_in: Number of input features.
            width: Number of units in the first hidden layer.
            dropout: Dropout probability applied after each hidden activation.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_in, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width // 2),
            nn.BatchNorm1d(width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, width // 4),
            nn.BatchNorm1d(width // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width // 4, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Predict hemoglobin for a batch of feature vectors.

        Args:
            x: Input feature tensor with shape (B, n_in).

        Returns:
            Predicted hemoglobin values with shape (B,).
        """
        return self.net(x).squeeze(-1)
