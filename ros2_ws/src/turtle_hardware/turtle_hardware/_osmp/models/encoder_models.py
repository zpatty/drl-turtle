__all__ = ["IdentityEncoder", "BijectiveEncoder"]
from copy import copy, deepcopy
from functools import partial
import normflows as nf
import torch
from torch.func import jacfwd, jacrev, vjp, vmap
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from .autograd_jacobian import get_jacobian, get_jacobian_vmap_jacrev
from .mlp import MLP
from .numerical_jacobian import approx_derivative
from .real_nvp import RealNVPCouplingLayer, RFFN
from .utils import GaussianFourierProjection, NormFlowWrapper


class IdentityEncoder(nn.Module):
    """
    An identity encoder that does nothing.
    """

    def __init__(self, num_dims: int):
        super(IdentityEncoder, self).__init__()
        self.num_dims = num_dims

    def forward(
        self, input: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        J = self.jacobian(input)
        return input, J

    def jacobian(self, inputs, **kwargs):
        batch_size = inputs.size(dim=0)
        J = torch.repeat_interleave(
            torch.eye(self.num_dims, device=inputs.device).unsqueeze(dim=0),
            batch_size,
            dim=0,
        )

        return J


class BijectiveEncoder(nn.Module):
    """
    A sequential container of flows based on coupling layers.
    """

    def __init__(
        self,
        num_dims: int,
        num_blocks: int,
        num_hidden: int,
        coupling_network_type: str = "rffn",
        jacobian_computation_method: str = "autograd",
        numerical_jacobian_abs_step: float = 5e-4,
        condition: bool = False,
        condition_embedding_num_layers: int = 2,
        condition_embedding_nonlinearity: Optional[nn.Module] = nn.Softplus,
        **kwargs,
    ):
        """
        Constructor
        Arguments:
            num_dims: the number of dimensions of the input tensor
            num_blocks: the number of coupling layers
            num_hidden: the number of hidden units in the coupling network
            coupling_network_type: the type of the coupling network
            jacobian_computation_method: the method for computing the jacobian.
                Options are "autograd", "vmap_jacrev" or "numerical"
            numerical_jacobian_abs_step: the absolute step size for numerical jacobian computation
            condition: whether the encoder is conditioned
        """
        super(BijectiveEncoder, self).__init__()
        self.num_dims = num_dims
        self.coupling_network_type = coupling_network_type
        self.jacobian_computation_method = jacobian_computation_method
        self.numerical_jacobian_abs_step = numerical_jacobian_abs_step
        self.condition = condition

        mask = None
        b = None
        match coupling_network_type:
            case "rffn" | "fcnn":
                mask = torch.arange(0, num_dims) % 2  # alternating input
                mask = mask.float()
            case "normflows_real_nvp":
                b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(num_dims)])

        if condition:
            fourier_projection_dim = 4 * self.num_dims
            condition_embedding_dim = self.num_dims
            self.z_embedding_net = nn.Sequential(
                GaussianFourierProjection(embedding_size=fourier_projection_dim),
                MLP(
                    input_dim=2 * fourier_projection_dim,
                    output_dim=condition_embedding_dim,
                    hidden_dim=2 * fourier_projection_dim,
                    num_layers=condition_embedding_num_layers,
                    nonlinearity=condition_embedding_nonlinearity,
                ),
                # nn.Tanh(),
            )
        else:
            condition_embedding_dim = 0
            self.z_embedding_net = None

        layers = []
        for i in range(num_blocks):
            match coupling_network_type:
                case "rffn" | "fcnn":
                    layers += [
                        RealNVPCouplingLayer(
                            num_inputs=num_dims,
                            num_hidden=num_hidden,
                            mask=mask,
                            base_network_type=coupling_network_type,
                            condition_embedding_dim=condition_embedding_dim
                            if condition
                            else 0,
                            **kwargs,
                        ),
                    ]
                    mask = 1 - mask  # flipping mask
                case "normflows_real_nvp":
                    s = nf.nets.MLP([num_dims, 2 * num_dims, num_dims], init_zeros=True)
                    t = nf.nets.MLP([num_dims, 2 * num_dims, num_dims], init_zeros=True)

                    if i % 2 == 0:
                        mask = b
                    else:
                        mask = 1 - b

                    layers += [NormFlowWrapper(nf.flows.MaskedAffineFlow(mask, t, s))]

                    layers += [NormFlowWrapper(nf.flows.ActNorm(num_dims))]
                case "normflows_neural_spline_flow":
                    layers += [
                        NormFlowWrapper(
                            nf.flows.CoupledRationalQuadraticSpline(
                                num_dims,
                                num_blocks=2,
                                num_hidden_channels=kwargs.get("num_hidden", 128),
                                reverse_mask=i % 2 == 1,
                                activation=nn.LeakyReLU,
                                init_identity=True,
                            )
                        ),
                        NormFlowWrapper(
                            nf.flows.LULinearPermute(num_dims, identity_init=True)
                        ),
                    ]
                case _:
                    raise NotImplementedError(
                        f"The Bijective encoder network type {coupling_network_type} has not been implemented"
                    )

        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, mode: int = 1, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward or backward pass for flow modules.
        Args:
            x: input tensor of shape (batch_size, num_dims)
            mode: the mode of operation (1: forward or -1: inverse)
        Return:
            output: the output tensor of shape (batch_size, num_dims)
            J: the jacobian matrix of shape (batch_size, num_dims, num_dims)
        """
        output = self.predict(x, mode=mode, **kwargs)
        J = self.jacobian(x, y=output, mode=mode, **kwargs)

        return output, J

    def predict(
        self, x: torch.Tensor, z: Optional[torch.Tensor] = None, mode: int = 1, **kwargs
    ) -> torch.Tensor:
        if mode == -1:
            iterator = reversed(self.layers)
        else:
            iterator = self.layers

        if z is not None:
            z_embedding = self.z_embedding_net(z)
        else:
            z_embedding = None

        for i, layer in enumerate(iterator):
            if mode == -1:
                x = layer.inverse(x, z_embedding=z_embedding, **kwargs)
            else:
                x = layer.forward(x, z_embedding=z_embedding, **kwargs)

        return x

    def jacobian(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, mode: int = 1, **kwargs
    ) -> torch.Tensor:
        """
        Finds the product of all jacobians
        Arguments:
            x: the input tensor of shape (batch_size, num_dims)
            y: the output tensor of shape (batch_size, num_dims)
            mode: the mode of operation (1: forward or -1: inverse)
        Returns:
            J: the jacobian matrix of shape (batch_size, num_dims, num_dims)
        """
        match self.jacobian_computation_method:
            case "autograd":
                J = get_jacobian(
                    partial(self.predict, mode=mode),
                    x,
                    x.size(-1),
                    create_graph=self.training,
                    **kwargs,
                )
            case "vmap_jacrev":
                J = get_jacobian_vmap_jacrev(
                    partial(self.predict, mode=mode), x, **kwargs
                )
            case "numerical" | "numerical_3_point":
                ## iterate over batch_dim
                # J_ls = []
                # for i in range(x.size(0)):
                #     predict_lambda = lambda _x: (
                #         self.predict(_x, mode=mode)
                #     )
                #     Ji = approx_derivative(
                #         fun=predict_lambda,
                #         x0=x[i],
                #         method="3-point"
                #         if self.jacobian_computation_method == "numerical_3_point"
                #         else "2-point",
                #         abs_step=self.numerical_jacobian_abs_step,
                #         f0=y[i],
                #     )
                #     J_ls.append(Ji)
                # J = torch.stack(J_ls, dim=0)
                J = approx_derivative(
                    fun=partial(self.predict, mode=mode, **kwargs),
                    x0=x,
                    method="3-point"
                    if self.jacobian_computation_method == "numerical_3_point"
                    else "2-point",
                    abs_step=self.numerical_jacobian_abs_step,
                    f0=y,
                )
            case _:
                raise NotImplementedError(
                    f"The jacobian computation method {self.jacobian_computation_method} has not been implemented"
                )

        return J
