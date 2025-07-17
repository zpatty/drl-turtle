__all__ = ["rollout_trajectories_euler", "rollout_trajectories_odeint"]
import torch
import numpy as np
from torchdiffeq import odeint
from typing import Dict, Optional, Tuple


def rollout_trajectories_euler(
    model,
    y0: torch.Tensor,
    ts: torch.Tensor,
    time_dependent=False,
    h0: Optional[torch.Tensor] = None,
    z_ts: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate rollouts of the model using the Euler method
    Arguments:
        model: dynamical model (torch.nn.Module)
        y0: initial state (torch.Tensor) of shape (batch_size, num_dims)
        ts: time points for evaluation of the trajectory (torch.Tensor) of shape (batch_size, num_time_steps,)
        time_dependent: whether the model is time-dependent
        h0: optionally: initial hidden state (torch.Tensor) of shape (batch_size, hidden_dim)
        z_ts: optionally, the conditioning for the encoding (torch.Tensor) of shape (batch_size, num_time_steps, condition_dim)
    Returns:
        sim_ts: dictionary containing the time-series data
    """

    # the ode function of a first order system
    def ode_fn(
        t: torch.Tensor,
        y: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # compute the velocity under the given model
        args = ()
        kwargs = {}
        if time_dependent:
            args = args + (t,)
        args = args + (y,)
        if h is not None:
            args = args + (h,)
        if z is not None:
            kwargs["z"] = z

        y_d, aux = model(*args, **kwargs)

        return y_d, aux

    # numerical integration
    y, h = y0, h0
    y_ts = torch.zeros(ts.size() + (y0.size(-1),), dtype=y0.dtype, device=y0.device)
    y_d_ts = torch.zeros_like(y_ts)
    y_ts[..., 0, :] = y0
    if h0 is not None:
        h_ts = torch.zeros(ts.size() + (h0.size(-1),), dtype=h0.dtype, device=y0.device)
    else:
        h_ts = None
    for time_idx in range(1, ts.size(-1)):
        t = ts[..., time_idx]
        dt = t - ts[..., time_idx - 1]  # time step

        # extract the conditioning
        if z_ts is not None:
            z = z_ts[..., time_idx, :]
        else:
            z = None

        # evaluate the time derivative at the current time step
        y_d, aux = ode_fn(t, y, h=h, z=z)

        if h is not None:
            # insert the hidden state
            h = aux["h_next"]
            h_ts[..., time_idx, :] = h

        # update the state
        y = y + dt[..., None] * y_d
        y_ts[..., time_idx, :] = y
        y_d_ts[..., time_idx, :] = y_d

    sim_ts = dict(
        ts=ts,
        y_ts=y_ts,
        y_d_ts=y_d_ts,
    )

    if h_ts is not None:
        sim_ts["h_ts"] = h_ts

    return sim_ts


def rollout_trajectories_odeint(
    model,
    y0: torch.Tensor,
    ts: torch.Tensor,
    method: str = "euler",
    time_dependent=False,
) -> Dict[str, torch.Tensor]:
    """
    Generate rollouts of the model using the odeint solver provided by torchdiffeq
    Arguments:
        model: dynamical model (torch.nn.Module)
        y0: initial state (torch.Tensor)
        ts: time points for evaluation of the trajectory (torch.Tensor)
        method: solver for integration (euler, rk4, dopri5, etc.) For a list of solvers, see torchdiffeq
        time_dependent: whether the model is time-dependent
    Returns:
        sim_ts: dictionary containing the time-series data
    """
    assert (
        torch.sum(torch.diff(ts, dim=0)) == 0
    ), "The time points should be equally spaced!"

    y0 = y0.type(dtype=torch.float64).to(model.device)
    ts = ts.type(dtype=torch.float64).to(model.device)

    # the ode function of a first order system
    def ode_fn(t: float, y: torch.Tensor):
        # cast to float32
        y_float = y.type(torch.float32)

        # compute the velocity under the given model
        if time_dependent:
            y_d, _ = model(t, y_float)
        else:
            y_d, _ = model(y_float)

        return y_d

    # numerical integration
    y_ts = odeint(ode_fn, y0, ts[0], method=method)
    # transpose to have the batch dimension first
    y_ts = torch.transpose(y_ts, 1, 0)

    # evaluate the time derivative at the computed trajectory
    y_ts_float = y_ts.type(torch.float32)
    y_ts_flat = y_ts_float.reshape(-1, y0.size(-1))
    if time_dependent:
        y_d_ts_flat, _ = model(ts.flatten(), y_ts_flat)
    else:
        y_d_ts_flat, _ = model(y_ts_flat)
    # reshape the time derivative
    y_d_ts = y_d_ts_flat.reshape(y_ts.size())

    sim_ts = dict(
        ts=ts,
        y_ts=y_ts,
        y_d_ts=y_d_ts,
    )

    return sim_ts
