__all__ = ["LinearClamped", "Cos", "NormFlowWrapper", "GaussianFourierProjection"]
import normflows as nf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union

from .autograd_jacobian import get_jacobian


class LinearClamped(nn.Module):
    """
    Linear layer with user-specified parameters (not to be learrned!)
    """

    __constants__ = ["bias", "in_features", "out_features"]

    def __init__(self, in_features, out_features, weights, bias_values, bias=True):
        super(LinearClamped, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("weight", torch.Tensor(weights))
        if bias:
            self.register_buffer("bias", torch.Tensor(bias_values))

    def forward(self, input):
        if input.dim() == 1:
            return F.linear(input.view(1, -1), self.weight, self.bias)
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class Cos(nn.Module):
    """
    Applies the cosine element-wise function
    """

    def forward(self, inputs):
        return torch.cos(inputs)


class NormFlowWrapper(nn.Module):
    """
    A wrapper class for modules of the normflows library.
    """

    def __init__(self, flow: nn.Module):
        super(NormFlowWrapper, self).__init__()
        self.flow = flow

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z, log_det = self.flow.forward(z)
        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        z, log_det = self.flow.inverse(z)
        return z

    def jacobian(self, z: torch.Tensor) -> torch.Tensor:
        J = get_jacobian(self, z, z.size(-1), create_graph=self.training)
        # J = get_jacobian_vmap(self, z)
        return J


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size: int = 256, scale: float = 1.0):
        super(GaussianFourierProjection, self).__init__()
        self.embedding_size = embedding_size
        self.scale = scale

        self.W = nn.Parameter(
            torch.randn(self.embedding_size) * self.scale, requires_grad=False
        )

    def forward(self, x: torch.Tensor):
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        x_proj = x_proj.squeeze(dim=-2)
        return torch.concatenate([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def normalize_polar_angles(phi: torch.Tensor) -> torch.Tensor:
    """
    Normalize the polar angles to the interval [-pi, pi].
    """
    return torch.remainder(phi + torch.pi, 2 * torch.pi) - torch.pi
