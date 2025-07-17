__all__ = ["VelocityPredictor"]
import torch
import torch.nn as nn
from torch import autograd
from typing import Dict, Optional, Tuple, Union

from osmp.models.dynamics_models import RNNDynamics
from osmp.models.mlp import MLP


class VelocityPredictor(nn.Module):
    """
    taskmap_model: a model that provides the map to a latent space
    dynamics_model: a model that provides the dynamics in the (latent) space
    n_dim_x: observed (input) space dimensions
    n_dim_y: latent (output) space dimentions
    scale_vel (optional): if set to true, learns a scalar velocity multiplier
    scale_vel_num_layers (optional): number of hidden layers for the velocity scalar
    scale_vel_hidden_dim (optional): hidden dimension for the velocity scalar
    is_diffeomorphism (optional): if set to True, use the inverse of the jacobian itself rather than pseudo-inverse
    origin (optional): shifted origin of the input space (this is the goal usually)
    x_translation (optional): translation factor for the input space (assumed that this was alread applied to the input)
    x_scaling (optional): scaling factor for the input space (assumed that this was alread applied to the input)
    eps (optional): a small value to avoid numerical instability
    """

    def __init__(
        self,
        taskmap_model,
        dynamics_model,
        n_dim_x: int,
        n_dim_y: int,
        scale_vel: bool = True,
        scale_vel_num_layers: int = 3,
        scale_vel_hidden_dim: int = 128,
        is_diffeomorphism: bool = True,
        origin: Optional[torch.Tensor] = None,
        x_translation: Union[float, torch.Tensor] = 0.0,
        x_scaling: Union[float, torch.Tensor] = 1.0,
        condition_encoding: bool = False,
        eps: float = 1e-12,
    ):
        super(VelocityPredictor, self).__init__()
        self.taskmap_model = taskmap_model
        self.dynamics_model = dynamics_model
        self.n_dim_x = n_dim_x
        self.n_dim_y = n_dim_y
        self.eps = eps
        self.is_diffeomorphism = is_diffeomorphism
        self.scale_vel = scale_vel
        self.condition_encoding = condition_encoding

        # identity matrix
        I = torch.eye(self.n_dim_x, self.n_dim_x).unsqueeze(0)

        if scale_vel:
            # scaling network (only used when scale_vel param is True!)
            self.log_vel_scalar = MLP(
                n_dim_x,
                1,
                hidden_dim=scale_vel_hidden_dim,
                num_layers=scale_vel_num_layers,
                nonlinearity=nn.LeakyReLU,
            )  # a 2-hidden layer network
        else:
            self.log_vel_scalar = lambda x: x

        # register buffers
        self.register_buffer("I", I)
        self.register_buffer("origin", origin)
        self.register_buffer("x_translation", x_translation)
        self.register_buffer("x_scaling", x_scaling)

        if self.is_diffeomorphism:
            assert (
                n_dim_x == n_dim_y
            ), "Input and Output dims need to be same for diffeomorphism!"

        self.device = torch.device("cpu")

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        encoder_kwargs: Optional[Dict] = None,
        dynamics_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Velocity prediction
        Arguments:
            x: input of shape (batch_size, state_dim)
            h: hidden state of shape (batch_size, hidden_dim)
            z: conditioning of shape (batch_size, cond_dim)
        Returns:
            x_d_hat: predicted velocity
            aux: auxiliary outputs
        """
        if x.ndimension() == 1:
            flatten_output = True  # output the same dimensions as input
            x = x.view(1, -1)
        else:
            flatten_output = False

        if encoder_kwargs is None:
            encoder_kwargs = {}
        if dynamics_kwargs is None:
            dynamics_kwargs = {}

        # initialize auxiliary outputs
        aux = dict()

        y_hat, J_hat = self.taskmap_model(x, z=z, **encoder_kwargs)

        if self.origin is not None:
            # encode the origin into latent space
            y_origin, _ = self.taskmap_model(self.origin, z=z, **encoder_kwargs)
            # Shifting the origin
            y_hat = y_hat - y_origin

        # derivative prediction by the latent space model
        if type(self.dynamics_model) == RNNDynamics:
            y_d_hat, h_next = self.dynamics_model(y_hat, h=h, **dynamics_kwargs)
            aux["h_next"] = h_next
        else:
            y_d_hat = self.dynamics_model(y_hat, **dynamics_kwargs)

        # update auxiliary outputs
        aux.update(
            dict(
                y=y_hat,
                y_d=y_d_hat,
            )
        )

        if self.is_diffeomorphism:
            J_hat_inv = torch.inverse(J_hat)
        else:
            I = self.I.repeat(J_hat.shape[0], 1, 1)
            J_hat_T = J_hat.permute(0, 2, 1)
            J_hat_inv = torch.matmul(
                torch.inverse(torch.matmul(J_hat_T, J_hat) + self.eps * I), J_hat_T
            )

        xd_hat = torch.bmm(
            J_hat_inv, y_d_hat.unsqueeze(2)
        ).squeeze()  # natural gradient descent

        if self.scale_vel:
            x_d = (
                self.vel_scalar(x) * xd_hat
            )  # mutiplying with a learned velocity scalar
        else:
            x_d = xd_hat

        if flatten_output:
            x_d = x_d.squeeze()

        return x_d, aux

    def energy(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the energy of the system
        """
        # encode into latent space
        y_hat, _ = self.taskmap_model(x, z=z)

        if self.origin is not None:
            # encode the origin into latent space
            y_origin, _ = self.taskmap_model(self.origin, z=z)
            # Shifting the origin
            y_hat = y_hat - y_origin

        # evaluating the energy
        E = self.dynamics_model.energy(y_hat)

        return E

    def vel_scalar(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_vel:
            return torch.exp(self.log_vel_scalar(x)) + self.eps
        else:
            return x

    def to(self, device):
        self.device = device
        return super(VelocityPredictor, self).to(device)
