"""Neural network models for DQN."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """1D CNN that maps an sEMG window to Q-values for each gesture class."""

    def __init__(
        self,
        num_channels: int,
        num_actions: int,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(conv2_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, window_size, channels). Conv1d expects channels first.
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0]
        x = F.relu(self.fc1(x))
        return self.fc2(x)
