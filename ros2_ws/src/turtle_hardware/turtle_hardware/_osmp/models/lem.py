__all__ = ["LEM", "LEMCell"]
import math
import torch
import torch.nn as nn
from typing import Tuple


class LEMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dt: float = 1.0):
        super(LEMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.inp2hid = nn.Linear(input_dim, 4 * hidden_dim)
        self.hid2hid = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.transform_z = nn.Linear(hidden_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        transformed_inp = self.inp2hid(x)
        transformed_hid = self.hid2hid(y)
        i_dt1, i_dt2, i_z, i_y = transformed_inp.chunk(4, 1)
        h_dt1, h_dt2, h_y = transformed_hid.chunk(3, 1)

        ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1)
        ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2)

        z = (1.0 - ms_dt) * z + ms_dt * torch.tanh(i_y + h_y)
        y = (1.0 - ms_dt_bar) * y + ms_dt_bar * torch.tanh(self.transform_z(z) + i_z)

        return y, z


class LEM(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dt: float = 1.0
    ):
        super(LEM, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = LEMCell(input_dim, hidden_dim, dt)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "classifier" in name and "weight" in name:
                nn.init.kaiming_normal_(param.data)

    def forward(self, input):
        ## initialize hidden states
        y = input.data.new(input.size(1), self.hidden_dim).zero_()
        z = input.data.new(input.size(1), self.hidden_dim).zero_()
        for x in input:
            y, z = self.cell(x, y, z)
        out = self.classifier(y)
        return out
