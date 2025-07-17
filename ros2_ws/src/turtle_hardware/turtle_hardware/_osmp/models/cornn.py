__all__ = ["coRNN", "coRNNCell"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


class coRNN(nn.Module):
    """
    Important: assumes batch_first=True
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dt: float,
        gamma: float,
        epsilon: float,
        batch_first=True,
    ):
        super(coRNN, self).__init__()
        assert batch_first, "batch_first=True is required"

        self.hidden_dim = hidden_dim
        self.cell = coRNNCell(input_dim, hidden_dim, dt, gamma, epsilon)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # initialize hidden states as zeros of shape (batch_size, hidden_dim)
        hy = torch.zeros((x.size(0), self.hidden_dim), device=x.device)
        hz = torch.zeros((x.size(0), self.hidden_dim), device=x.device)

        # iterate over the sequence
        for t in range(x.size(1)):
            hy, hz = self.cell(x[:, t], hy, hz)
        output = self.readout(hy)

        return output


class coRNNCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dt: float,
        gamma: float = 1.0,
        epsilon: float = 1.0,
    ):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(input_dim + hidden_dim + hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, hy: torch.Tensor, hz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hz = hz + self.dt * (
            torch.tanh(self.i2h(torch.cat((x, hz, hy), 1)))
            - self.gamma * hy
            - self.epsilon * hz
        )
        hy = hy + self.dt * hz

        return hy, hz
