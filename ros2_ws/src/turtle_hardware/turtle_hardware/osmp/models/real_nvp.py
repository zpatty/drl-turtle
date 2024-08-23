__all__ = ["RealNVPCouplingLayer", "RFFN"]
from functools import partial
import normflows as nf
import numpy as np
import torch
from torch.func import jacfwd, jacrev, vjp, vmap
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
from typing import Optional, Tuple, Union


from .autograd_jacobian import get_jacobian
from .mlp import MLP
from .utils import LinearClamped, Cos, GaussianFourierProjection


class RealNVPCouplingLayer(nn.Module):
    """An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(
        self,
        num_inputs: int,
        num_hidden: int,
        mask: torch.Tensor,
        base_network_type: str = "rffn",
        s_nonlinearity=nn.ELU,
        t_nonlinearity=nn.ELU,
        sigma: float = 0.45,
        condition_embedding_dim: int = 0,
        **kwargs,
    ):
        super(RealNVPCouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        # register the mask as a buffer
        self.register_buffer("mask", mask)

        if base_network_type == "fcnn":
            self.scale_net = MLP(
                input_dim=num_inputs + condition_embedding_dim,
                output_dim=num_inputs,
                hidden_dim=num_hidden,
                num_layers=3,
                nonlinearity=s_nonlinearity,
            )
            self.translate_net = MLP(
                input_dim=num_inputs + condition_embedding_dim,
                output_dim=num_inputs,
                hidden_dim=num_hidden,
                num_layers=3,
                nonlinearity=t_nonlinearity,
            )
            # print("Using neural network initialized with identity map!")

            nn.init.zeros_(self.translate_net.network[-1].weight.data)
            nn.init.zeros_(self.translate_net.network[-1].bias.data)

            nn.init.zeros_(self.scale_net.network[-1].weight.data)
            nn.init.zeros_(self.scale_net.network[-1].bias.data)

        elif base_network_type == "rffn":
            # print("Using random fourier feature with bandwidth = {}.".format(sigma))
            self.scale_net = RFFN(
                input_dim=num_inputs + condition_embedding_dim,
                output_dim=num_inputs,
                num_features=num_hidden,
                sigma=sigma,
                **kwargs,
            )
            self.translate_net = RFFN(
                input_dim=num_inputs + condition_embedding_dim,
                output_dim=num_inputs,
                num_features=num_hidden,
                sigma=sigma,
                **kwargs,
            )

            # print("Initializing coupling layers as identity!")
            nn.init.zeros_(self.translate_net.network[-1].weight.data)
            nn.init.zeros_(self.scale_net.network[-1].weight.data)
        else:
            raise TypeError("The network type has not been defined")

    def forward(
        self,
        input: torch.Tensor,
        z_embedding: Optional[torch.Tensor] = None,
        mode: int = 1,
    ):
        """
        Forward pass for the coupling layer.
        Arguments:
            input: the input tensor
            z_embedding: optional conditioning
            mode: the mode of operation (1: forward or -1: inverse)
        """
        masked_input = input * self.mask

        if z_embedding is not None:
            masked_input = torch.cat([masked_input, z_embedding], dim=-1)

        log_s = self.scale_net(masked_input) * (1 - self.mask)
        t = self.translate_net(masked_input) * (1 - self.mask)

        if mode == -1:
            s = torch.exp(-log_s)
            return (input - t) * s
        else:
            s = torch.exp(log_s)
            return input * s + t

    def inverse(self, input: torch.Tensor, **kwargs):
        return self(input, mode=-1, **kwargs)

    def jacobian(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_dim = input.size(0)

        J = get_jacobian(
            partial(self, **kwargs), input, input.size(-1), create_graph=self.training
        )

        """
        # this code leverages the structure as layed out in Eq. (6) of
        # Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using real nvp.
        # arXiv preprint arXiv:1605.08803.
        selector = self.mask.type(torch.bool)
        inv_selector = (1 - self.mask).type(torch.bool)

        # compute the masked (i.e., 1:d) identity matrix
        eye = torch.diag(self.mask)[None, ...].repeat(batch_dim, 1, 1)
        # compute the lower left Jacobian in Eq. (6) of
        forward_unbatched = lambda x: self.forward(x.unsqueeze(0)).squeeze(0)
        forward_dD_fn = lambda x: forward_unbatched(x)[inv_selector]
        J_dD_1d_fn = jacfwd(forward_dD_fn)
        J_dD_1d = vmap(J_dD_1d_fn)(input)

        # assemble the Jacobian
        J_man = eye
        J_man[:, inv_selector, :] = J_dD_1d
        """

        # print("J", J)
        # print("J_man\n", J_man)
        # print("J_err", torch.abs(J_man - J))
        # assert torch.allclose(J_man, J, atol=1e-7), "Jacobian computation error!"

        return J


class RFFN(nn.Module):
    """
    Random Fourier features network.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_features: int,
        sigma: float = 10.0,
        **kwargs,
    ):
        super(RFFN, self).__init__()
        self.sigma = np.ones(input_dim) * sigma
        self.coeff = np.random.normal(0.0, 1.0, (num_features, input_dim))
        self.coeff = self.coeff / self.sigma.reshape(1, len(self.sigma))
        self.offset = 2.0 * np.pi * np.random.rand(1, num_features)

        self.network = nn.Sequential(
            LinearClamped(input_dim, num_features, self.coeff, self.offset),
            Cos(),
            nn.Linear(num_features, output_dim, bias=False),
        )

    def forward(self, x):
        return self.network(x)
