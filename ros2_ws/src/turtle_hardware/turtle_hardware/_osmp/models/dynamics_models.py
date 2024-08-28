__all__ = [
    "MLPDynamics",
    "LinearDynamics",
    "HopfBifurcationDynamics",
    "HopfBifurcationWithPhaseSynchronizationDynamics",
    "VanDerPolOscillatorDynamics",
    "RNNDynamics",
]
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union

from .cornn import coRNN, coRNNCell
from .lem import LEMCell
from .mlp import MLP
from .utils import normalize_polar_angles

MLPDynamics = MLP


class LinearDynamics(nn.Module):
    def __init__(
        self, alpha: Union[float, torch.Tensor] = 1.0, normalize_inputs: bool = False
    ):
        super(LinearDynamics, self).__init__()
        self.alpha = alpha
        self.normalize_inputs = normalize_inputs

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.normalize_inputs:
            y_d = -self.alpha * F.normalize(y, dim=-1)  # normalized quadratic potential
        else:
            y_d = -self.alpha * y

        return y_d

    def compute_energy(self, y: torch.Tensor) -> torch.Tensor:
        if self.normalize_inputs:
            E = 0.5 * torch.norm(y, dim=-1) ** 2
        else:
            E = 0.5 * torch.sum(y**2, dim=-1)

        return E


class HopfBifurcationDynamics(nn.Module):
    """
    This system has a circular limit cycle. It has been used in
        Zhi, W., Liu, K., Zhang, T., & Johnson-Roberson, M. (2023).
        Learning Orbitally Stable Systems for Diagrammatically Teaching. arXiv preprint arXiv:2309.10298.
        https://arxiv.org/abs/2309.10298
    and mirrors Hopf bifurcation: https://en.wikipedia.org/wiki/Hopf_bifurcation#Intuition
    If alpha > 0, the system resembles a supercritical Hopf bifurcation.
    If alpha < 0, the system resembles a subcritical Hopf bifurcation
    """

    def __init__(
        self,
        state_dim: int,
        alpha: Union[float, torch.Tensor] = 1.0,
        mu: Union[float, torch.Tensor] = 1.0,
        R: Union[float, torch.Tensor] = 1.0,
        omega: Union[float, torch.Tensor] = 1.0,
        beta: Union[float, torch.Tensor] = 1.0,
        make_params_learnable: bool = False,
        parametric_omega: bool = False,
        eps: float = 1e-6,
    ):
        super(HopfBifurcationDynamics, self).__init__()
        self.state_dim = state_dim
        assert state_dim >= 2, "State dimension must be at least 2"

        self.mu = mu
        self.make_params_learnable = make_params_learnable
        self.parametric_omega = parametric_omega
        self.eps = eps

        if self.make_params_learnable:
            self.alpha_sqrt_param = nn.Parameter(
                torch.tensor(alpha, requires_grad=False)
            )
            self.R_param = nn.Parameter(torch.tensor(R, requires_grad=False))
            self.beta_sqrt_param = nn.Parameter(
                beta * torch.ones(self.state_dim - 2, requires_grad=False)
            )
        else:
            self.alpha = alpha
            self.R = R
            self.beta = beta

        if parametric_omega:
            self.log_omega_net = MLP(
                2, 1, hidden_dim=128, num_layers=5, nonlinearity=nn.LeakyReLU
            )
            # initialize omega as the identity: log_omega = 0 --> exp(log_omega) = 1
            nn.init.zeros_(self.log_omega_net.network[-1].weight.data)
            nn.init.zeros_(self.log_omega_net.network[-1].bias.data)
        else:
            if not make_params_learnable:
                self.omega = omega
            else:
                self.omega_sqrt_param = nn.Parameter(
                    torch.sqrt(torch.tensor(omega, requires_grad=False))
                )

    def get_limit_cycle_radius(self) -> torch.Tensor:
        if self.make_params_learnable:
            R = self.R_param + self.eps
        else:
            R = self.R
        return R

    def get_params(self, y: torch.Tensor) -> Dict[str, Union[float, torch.Tensor]]:
        # extract the state
        y1, y2 = y[..., 0], y[..., 1]

        # get the limit cycle radius
        R = self.get_limit_cycle_radius()

        if self.make_params_learnable:
            alpha = self.alpha_sqrt_param**2 + self.eps
            beta = self.beta_sqrt_param**2 + self.eps
        else:
            alpha, beta = self.alpha, self.beta

        if self.parametric_omega:
            # compute varphi
            varphi = torch.atan2(y2, y1)[:, None]
            log_omega_net_input = torch.cat(
                [torch.sin(varphi), torch.cos(varphi)], dim=-1
            )
            omega = (
                torch.exp(self.log_omega_net(log_omega_net_input).squeeze(dim=-1))
                + self.eps
            )
            # print("min varphi", torch.min(varphi).item(), "max varphi", torch.max(varphi).item(),
            # "min omega", torch.min(omega).item(), "max omega", torch.max(omega).item())
        elif self.make_params_learnable:
            omega = self.omega_sqrt_param**2 + self.eps
        else:
            omega = self.omega

        params = dict(alpha=alpha, R=R, omega=omega, beta=beta)
        return params

    def forward(
        self,
        y: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        omega: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        params = self.get_params(y)
        R = params["R"]
        if alpha is None:
            alpha = params["alpha"]
        if omega is None:
            omega = params["omega"]
        if beta is None:
            beta = params["beta"]

        # extract the state
        y1, y2 = y[..., 0], y[..., 1]

        y_d = torch.stack(
            [
                -omega * y2 + alpha * (self.mu - (y1**2 + y2**2) / R**2) * y1,
                omega * y1 + alpha * (self.mu - (y1**2 + y2**2) / R**2) * y2,
            ],
            dim=-1,
        )

        if self.state_dim > 2:
            for i in range(2, self.state_dim):
                # linear, attractive dynamics in the remaining dimensions
                y_d = torch.cat([y_d, -beta * y[..., i : i + 1]], dim=-1)

        return y_d

    def limit_cycle_matching_loss(self, y: torch.Tensor) -> torch.Tensor:
        params = self.get_params(y)
        alpha, R, omega, beta = (
            params["alpha"],
            params["R"],
            params["omega"],
            params["beta"],
        )

        # compute the loss between y and the closest point on the limit cycle
        y1, y2 = y[..., 0], y[..., 1]

        # compute the polar coordinates
        r = torch.sqrt(y1**2 + y2**2)
        varphi = torch.atan2(y2, y1)

        # compute the closest point on the limit cycle
        r_closest, varphi_closest = R, varphi

        # compute the MSE between the radius and the closest radius on the limit cycle
        loss_y = torch.mean((r_closest - r) ** 2)

        # compute the MSE between predictions and the origin for the other dimensions
        if self.state_dim > 2:
            y3n = y[..., 2:]
            loss_y = loss_y + torch.mean(y3n**2)

        return loss_y

    def energy(self, y: torch.Tensor) -> torch.Tensor:
        params = self.get_params(y)
        alpha, R, omega, beta = (
            params["alpha"],
            params["R"],
            params["omega"],
            params["beta"],
        )

        # extract the state
        state_dim = y.size(-1)
        y1, y2 = y[..., 0], y[..., 1]

        E = -torch.atan2(y2, y1) - 0.5 * self.mu * (
            1 - (y1**2 + y2**2) / (2 * R**2)
        ) * (y1**2 + y2**2)

        if state_dim > 2:
            E = E + torch.sum(0.5 * alpha * y[..., 2:] ** 2, dim=-1)

        return E

    def limit_cycle(self, varphi: torch.Tensor) -> torch.Tensor:
        """
        Compute the limit cycle of the system
        Arguments:
            varphi: the polar angle of size (batch_size, )
        Returns:
            y: the Cartesian coordinate on the limit cycle of size (batch_size, state_dim)
        """
        y = torch.zeros(varphi.size(0), self.state_dim, device=varphi.device)
        y[:, 0] = self.R * torch.cos(varphi)
        y[:, 1] = self.R * torch.sin(varphi)
        params = self.get_params(y)
        alpha, R, omega, beta = (
            params["alpha"],
            params["R"],
            params["omega"],
            params["beta"],
        )

        # map the polar angle to the Cartesian coordinates
        y = torch.concatenate(
            [
                R * torch.cos(varphi)[:, None],
                R * torch.sin(varphi)[:, None],
                torch.zeros(varphi.size(0), self.state_dim - 2, device=varphi.device),
            ],
            dim=-1,
        )
        return y


class HopfBifurcationWithPhaseSynchronizationDynamics(HopfBifurcationDynamics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, y: torch.Tensor, phase_sync_kp: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # evaluate the Hopf bifurcation dynamics
        y_d = super().forward(y, **kwargs)

        num_systems = y.size(0)
        # extract the state
        y1, y2 = y[..., 0], y[..., 1]

        params = self.get_params(y)
        R = params["R"]

        # compute the polar angle
        varphi = torch.atan2(y2, y1)

        # # compute the desired polar angle as the mean of the polar angle over the batch
        # varphi_des = torch.mean(varphi, dim=0, keepdim=True)

        if num_systems == 1:
            return y_d

        if num_systems == 2:
            varphi_des = normalize_polar_angles(
                varphi[0] + normalize_polar_angles(varphi[1] - varphi[0]) / 2
            )[None]
        elif num_systems > 2:
            # take the first polar angle as the desired polar angle
            varphi_des = varphi[0:1]
        else:
            raise ValueError(
                f"Number of systems must be larger than 0, got {num_systems}"
            )

        # compute the polar angle difference
        varphi_diff = normalize_polar_angles(varphi_des - varphi)

        # compute a proportional controller on the polar angle
        varphi_d = phase_sync_kp * varphi_diff

        # map from polar to Cartesian coordinates
        y1_d = -R * varphi_d * torch.sin(varphi)
        y2_d = R * varphi_d * torch.cos(varphi)
        y_d_pc = torch.concat(
            [
                y1_d[..., None],
                y2_d[..., None],
                torch.zeros(
                    size=y_d.size()[:-1] + (y_d.size(-1) - 2,), device=y.device
                ),
            ],
            dim=-1,
        )

        # add the phase synchronization term on the normal dynamics
        y_d = y_d + y_d_pc

        return y_d


class VanDerPolOscillatorDynamics(nn.Module):
    def __init__(
        self,
        m: Union[float, torch.Tensor] = 1.0,
        k: Union[float, torch.Tensor] = 1.0,
        mu: Union[float, torch.Tensor] = 1.0,
    ):
        super(VanDerPolOscillatorDynamics, self).__init__()
        self.m = m
        self.k = k
        self.mu = mu

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # extract the state
        state_dim = y.size(-1)
        x, x_d = torch.split(y, 1, dim=-1)

        y_d = torch.stack(
            [
                x_d,
                1 / self.m * (self.mu * (1 - x**2) * x_d - self.k * x),
            ],
            dim=-1,
        )

        if state_dim > 2:
            for i in range(2, state_dim):
                # linear, attractive dynamics in the remaining dimensions
                y_d = torch.cat([y_d, -self.alpha * y[..., i : i + 1]], dim=-1)

        return y_d

    def compute_energy(self, y: torch.Tensor) -> torch.Tensor:
        # extract the state
        state_dim = y.size(-1)
        x, x_d = torch.split(y, 1, dim=-1)

        # sum the kinetic and potential energy of the Van der Pol oscillator
        E = 0.5 * self.m * x_d**2 + 0.5 * self.k * x**2

        if state_dim > 2:
            E = E + torch.sum(0.5 * self.alpha * y[..., 2:] ** 2, dim=-1)

        return E


class RNNDynamics(nn.Module):
    def __init__(
        self,
        rnn_type: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        **kwargs,
    ):
        super(RNNDynamics, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnns = nn.ModuleList()
        match rnn_type:
            case "rnn":
                self.rnns.append(nn.RNNCell(input_dim, hidden_dim, **kwargs))
                for _ in range(num_layers - 1):
                    self.rnns.append(nn.RNNCell(hidden_dim, hidden_dim, **kwargs))
                self.total_hidden_state_dim = num_layers * hidden_dim
            case "gru":
                self.rnns.append(nn.GRUCell(input_dim, hidden_dim, **kwargs))
                for _ in range(num_layers - 1):
                    self.rnns.append(nn.GRUCell(hidden_dim, hidden_dim, **kwargs))
                self.total_hidden_state_dim = hidden_dim
            case "lstm":
                self.rnns.append(nn.LSTMCell(input_dim, hidden_dim, **kwargs))
                for _ in range(num_layers - 1):
                    self.rnns.append(nn.LSTMCell(hidden_dim, hidden_dim, **kwargs))
                self.total_hidden_state_dim = 2 * num_layers * hidden_dim
            case "cornn":
                self.rnns.append(coRNNCell(input_dim, hidden_dim, **kwargs))
                for _ in range(num_layers - 1):
                    self.rnns.append(coRNNCell(hidden_dim, hidden_dim, **kwargs))
                self.total_hidden_state_dim = 2 * num_layers * hidden_dim
            case "lem":
                self.rnns.append(LEMCell(input_dim, hidden_dim, **kwargs))
                for _ in range(num_layers - 1):
                    self.rnns.append(LEMCell(hidden_dim, hidden_dim, **kwargs))
                self.total_hidden_state_dim = 2 * num_layers * hidden_dim
            case _:
                raise ValueError(f"Unknown RNN type: {rnn_type}")

        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, y: torch.Tensor, h: Optional[torch.Tensor] = None):
        batch_size = y.size(0)
        if h is None:
            h = torch.zeros(batch_size, self.total_hidden_state_dim, device=y.device)

        match self.rnn_type:
            case "cornn":
                x = y
                h_next = []
                for i, rnn in enumerate(self.rnns):
                    hyi = h[:, 2 * i * self.hidden_dim : (2 * i + 1) * self.hidden_dim]
                    hzi = h[
                        :, (2 * i + 1) * self.hidden_dim : 2 * (i + 1) * self.hidden_dim
                    ]
                    hyi_next, hzi_next = rnn(x, hyi, hzi)
                    h_next.append(torch.cat([hyi_next, hzi_next], dim=-1))
                    x = hyi_next
                h_next = torch.cat(h_next, dim=-1)
                y_d = self.readout(x)
            case "lstm":
                x = y
                h_next = []
                for i, rnn in enumerate(self.rnns):
                    bi = h[:, 2 * i * self.hidden_dim : (2 * i + 1) * self.hidden_dim]
                    ci = h[
                        :, (2 * i + 1) * self.hidden_dim : 2 * (i + 1) * self.hidden_dim
                    ]
                    bi_next, ci_next = rnn(x, (bi, ci))
                    h_next.append(torch.cat([bi_next, ci_next], dim=-1))
                    x = bi_next
                h_next = torch.cat(h_next, dim=-1)
                y_d = self.readout(x)
            case "lem":
                x = y
                h_next = []
                for i, rnn in enumerate(self.rnns):
                    hyi = h[:, 2 * i * self.hidden_dim : (2 * i + 1) * self.hidden_dim]
                    hzi = h[
                        :, (2 * i + 1) * self.hidden_dim : 2 * (i + 1) * self.hidden_dim
                    ]
                    hyi_next, hzi_next = rnn(x, hyi, hzi)
                    h_next.append(torch.cat([hyi_next, hzi_next], dim=-1))
                    x = hyi_next
                h_next = torch.cat(h_next, dim=-1)
                y_d = self.readout(x)
            case _:
                x = y
                h_next = []
                for i, rnn in enumerate(self.rnns):
                    hi = h[:, i * self.hidden_dim : (i + 1) * self.hidden_dim]
                    hi_next = rnn(x, hi)
                    x = hi_next
                    h_next.append(hi_next)
                h_next = torch.cat(h_next, dim=-1)
                y_d = self.readout(x)

        return y_d, h_next
