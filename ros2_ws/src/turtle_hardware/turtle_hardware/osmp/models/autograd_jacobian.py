__all__ = ["get_jacobian", "get_jacobian_vmap_jacrev"]
import torch
from torch.func import jacfwd, jacrev, vjp, vmap
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union


def get_jacobian(
    net: Union[nn.Module, Callable],
    x: torch.Tensor,
    output_dims: Union[int, Tuple],
    create_graph: bool = False,
    **kwargs,
) -> torch.Tensor:
    if x.ndimension() == 1:
        batch_dim = 1
        input_dims = x.size(0)
    else:
        batch_dim = x.size(0)
        input_dims = x.size()[1:]

    x_m = x.repeat(1, output_dims).view(-1, output_dims)
    model_kwargs = dict()
    for k, v in kwargs.items():
        if v is not None:
            model_kwargs[k] = torch.repeat_interleave(v, output_dims, dim=0)
    # x_m.requires_grad_(True)
    y_m = net(x_m, **model_kwargs)
    jac_mask = torch.eye(output_dims).repeat(batch_dim, 1).to(x.device)
    # jac_mask.requires_grad_(True)
    J = autograd.grad(y_m, x_m, jac_mask, create_graph=create_graph, allow_unused=True)[
        0
    ]
    J = J.reshape(batch_dim, output_dims, output_dims)

    return J


def get_jacobian_vmap_jacrev(
    net: Union[nn.Module, Callable], x: torch.Tensor, **kwargs
) -> torch.Tensor:
    def forward_unbatched_fn(_x: torch.Tensor, _kwargs):
        _kwargs_unsqueeze = {k: _v.unsqueeze(0) for k, _v in _kwargs.items()}
        return net(_x.unsqueeze(0), **_kwargs_unsqueeze).squeeze(0)

    J = vmap(jacrev(forward_unbatched_fn))(x, kwargs)

    return J
