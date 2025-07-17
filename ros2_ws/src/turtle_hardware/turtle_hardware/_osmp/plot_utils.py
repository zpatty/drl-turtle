__all__ = ["visualize_field", "visualize_vel", "add_arrow_to_line2d"]
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from torchdiffeq import odeint
from typing import Dict, Optional


def visualize_field(
    X: np.ndarray, Z: np.ndarray, type="stream", cmap="coolwarm", color=None
):
    """
    plot the streamplot or quiver plot for 2-dim vector fields
    see matplotlib.pyplot.streamplot or matplotlib.pyplot.quiver for more information
    :param X: coordinates of the grid
    :param Z: the vector field to be visualized
    :param type ('stream or 'quiver'): type of plot, streamplot or quiverplot
    :param cmap (colormap): colormap of the plot
    :return:
    """
    n_dims = Z.shape[-1]

    # make sure that z has at least 2 dims to visualize
    assert n_dims >= 2

    if color is None:
        color = np.linalg.norm(Z, axis=-1)

    if type == "stream":
        if n_dims == 3:
            x3level = 0
            X_2D = X[:, :, x3level, :2]
            Z_2D = Z[:, :, x3level, :2]
        else:
            X_2D, Z_2D = X, Z

        # create streamplot
        return plt.streamplot(
            *[X_2D[..., i].T for i in range(X_2D.shape[-1])],
            *[Z_2D[..., i].T for i in range(X_2D.shape[-1])],
            color=color,
            cmap=cmap,
        )
    elif type == "quiver":
        # create quiverplot
        return plt.quiver(
            *[X[..., i].T for i in range(X.shape[-1])],
            *[Z[..., i].T for i in range(Z.shape[-1])],
            color=color,
            # units="width",
            cmap=cmap,
            # normalize=True,
            length=0.02,
        )


def visualize_vel(
    model,
    z: Optional[torch.Tensor] = None,
    type="stream",
    num_dims: Optional[int] = None,
    x_lim=None,
    delta=0.05,
    cmap=None,
    color=None,
):
    """
    visualize the velocity model (first order)
    similar to visualize_accel
    :param model (torch.nn.Module): the velocity model to be visualized
    :param z (torch.Tensor): the conditioning variable of shape (n_z,)
    :param type ('stream or 'quiver'): type of plot, streamplot or quiverplot
    :param num_dims (int): the number of dimensions to be visualized
    :param x_lim (array-like): the range of the state-space (positions) to be sampled over
    :param delta (float): the step size for the sampled positions
    :param cmap (colormap): colormap of the plot
    :param color (array-like): the color of the streamplot or quiverplot
    :return: None
    """
    if num_dims is None:
        num_dims = model.n_dim_x

    if x_lim is None:
        x_lim = np.repeat(np.array([[-1.0, 1.0]]), num_dims, axis=0)

    if hasattr(model, "device"):
        device = model.device
    else:
        device = "cpu"

    xi_pts_ls = []
    for i in range(num_dims):
        xi_pts_ls.append(
            torch.linspace(
                x_lim[i, 0],
                x_lim[i, 1],
                np.rint((x_lim[i, 1] - x_lim[i, 0]) / delta).astype(int).item() + 1,
            )
        )

    X_tpl = torch.meshgrid(xi_pts_ls, indexing="ij")

    # generate a flat version of the coordinates for forward pass
    x_test = torch.stack(X_tpl, dim=-1)
    x_test = x_test.reshape(-1, x_test.shape[-1]).to(device)
    x_test.requires_grad_(True)

    # forward pass
    z_test = (
        None
        if z is None
        else torch.ones(
            (x_test.shape[0], z.size(0)), dtype=torch.float32, device=device
        )
        * z[None, ...]
    )
    x_d_pred, _ = model(x_test, z=z_test)

    # detach
    x_d_pred = x_d_pred.detach().cpu()
    if hasattr(model, "x_translation") and hasattr(model, "x_scaling"):
        translation, scaling = (
            model.x_translation.detach().cpu(),
            model.x_scaling.detach().cpu(),
        )
        # denormalize
        X_denorm_tpl = [
            (Xi - translation[i]) / scaling[i] for i, Xi in enumerate(X_tpl)
        ]
        X_denorm = torch.stack(X_denorm_tpl, dim=-1)
        x_d_pred_denorm = x_d_pred / scaling
    else:
        X_denorm = torch.stack(X_tpl, dim=-1)
        x_d_pred_denorm = x_d_pred

    # reshape the velocity to the grid shape
    Z_denorm = x_d_pred_denorm.reshape(X_denorm.shape)

    # visualize the velocity as a vector field (equivalent to the warped potential)
    return visualize_field(
        X_denorm.numpy(), Z_denorm.numpy(), type=type, cmap=cmap, color=color
    )


def add_arrow_to_line2d(
    axes,
    line,
    arrow_locs=None,
    arrowstyle="-|>",
    arrowsize=1,
    transform=None,
):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes:
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    if arrow_locs is None:
        arrow_locs = [0.2, 0.4, 0.6, 0.8]

    if not isinstance(line, mlines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()

    arrow_kw = {
        "arrowstyle": arrowstyle,
        "mutation_scale": 10 * arrowsize,
    }

    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw["color"] = color

    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw["linewidth"] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n : n + 2]), np.mean(y[n : n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform, **arrow_kw
        )
        axes.add_patch(p)
        arrows.append(p)
    return arrows
