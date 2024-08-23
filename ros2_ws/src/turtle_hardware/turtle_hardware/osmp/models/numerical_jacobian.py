__all__ = ["approx_derivative"]
from functools import partial
import torch
from typing import Callable


def approx_derivative(
    fun, x0, method="3-point", rel_step=None, abs_step=None, f0=None, args=(), kwargs={}
):
    """Compute finite difference approximation of the derivatives of a
    vector-valued function.

    If a function maps from R^n to R^m, its derivatives form m-by-n matrix
    called the Jacobian, where an element (i, j) is a partial derivative of
    f[i] with respect to x[j].

    Parameters
    ----------
    fun : callable
        Function of which to estimate the derivatives. The argument x
        passed to this function is ndarray of shape (n,) (never a scalar
        even if n=1). It must return 1-D array_like of shape (m,) or a scalar.
    x0 : array_like of shape (n,) or float
        Point at which to estimate the derivatives. Float will be converted
        to a 1-D array.
    method : {'3-point', '2-point'}, optional
        Finite difference method to use:
            - '2-point' - use the first order accuracy forward or backward
                          difference.
            - '3-point' - use central difference in interior points and the
                          second order accuracy forward or backward difference
                          near the boundary.
    rel_step : None or array_like, optional
        Relative step size to use. If None (default) the absolute step size is
        computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``, with
        `rel_step` being selected automatically, see Notes. Otherwise
        ``h = rel_step * sign(x0) * abs(x0)``. For ``method='3-point'`` the
        sign of `h` is ignored.
    abs_step : array_like, optional
        For ``method='3-point'`` the sign of `abs_step` is ignored. By default
        relative steps are used, only if ``abs_step is not None`` are absolute
        steps used.
    f0 : None or array_like, optional
        If not None it is assumed to be equal to ``fun(x0)``, in this case
        the ``fun(x0)`` is not called. Default is None.
    args, kwargs : tuple and dict, optional
        Additional arguments passed to `fun`. Both empty by default.
        The calling signature is ``fun(x, *args, **kwargs)``.

    Returns
    -------
    J : {ndarray, sparse matrix, LinearOperator}
        Finite difference approximation of the Jacobian matrix.
        Returns a ndarray with shape (m, n).

    See Also
    --------
    check_derivative : Check correctness of a function computing derivatives.

    Notes
    -----
    If `rel_step` is not provided, it assigned as ``EPS**(1/s)``, where EPS is
    determined from the smallest floating point dtype of `x0` or `fun(x0)`,
    ``np.finfo(x0.dtype).eps``, s=2 for '2-point' method and
    s=3 for '3-point' method. Such relative step approximately minimizes a sum
    of truncation and round-off errors, see [1]_. Relative steps are used by
    default. However, absolute steps are used when ``abs_step is not None``.
    If any of the absolute or relative steps produces an indistinguishable
    difference from the original `x0`, ``(x0 + dx) - x0 == 0``, then a
    automatic step size is substituted for that particular entry.

    A finite difference scheme for '3-point' method is selected automatically.
    The well-known central difference scheme is used for points sufficiently
    far from the boundary, and 3-point forward or backward scheme is used for
    points near the boundary. Both schemes have the second-order accuracy in
    terms of Taylor expansion. Refer to [2]_ for the formulas of 3-point
    forward and backward difference schemes.

    For dense differencing when m=1 Jacobian is returned with a shape (n,),
    on the other hand when n=1 Jacobian is returned with a shape (m, 1).
    Our motivation is the following: a) It handles a case of gradient
    computation (m=1) in a conventional way. b) It clearly separates these two
    different cases. b) In all cases np.atleast_2d can be called to get 2-D
    Jacobian with correct dimensions.

    References
    ----------
    .. [1] W. H. Press et. al. "Numerical Recipes. The Art of Scientific
           Computing. 3rd edition", sec. 5.7.

    .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13 (1974), pp. 117-120.

    .. [3] B. Fornberg, "Generation of Finite Difference Formulas on
           Arbitrarily Spaced Grids", Mathematics of Computation 51, 1988.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize._numdiff import approx_derivative
    >>>
    >>> def f(x, c1, c2):
    ...     return np.array([x[0] * torch.sin(c1 * x[1]),
    ...                      x[0] * torch.cos(c2 * x[1])])
    ...
    >>> x0 = torch.tensor([1.0, 0.5 * torch.pi])
    >>> approx_derivative(f, x0, args=(1, 2))
    array([[ 1.,  0.],
           [-1.,  0.]])
    """
    if method not in ["2-point", "3-point"]:
        raise ValueError("Unknown method '%s'. " % method)

    def fun_wrapped(x):
        return fun(x, *args, **kwargs)

    if f0 is None:
        f0 = fun_wrapped(x0)
    else:
        f0 = torch.atleast_1d(f0)

    # by default we use rel_step
    if abs_step is None:
        h = _compute_absolute_step(rel_step, x0, f0, method)
    else:
        # user specifies an absolute step
        h = abs_step * torch.ones_like(x0)

    return _dense_difference(fun_wrapped, x0, f0, h, method=method)


def _compute_absolute_step(rel_step, x0: torch.Tensor, f0: torch.Tensor, method: str):
    """
    Computes an absolute step from a relative step for finite difference
    calculation.

    Parameters
    ----------
    rel_step: None or array-like
        Relative step for the finite difference calculation
    x0 : torch.Tensor
        Parameter vector
    f0 : torch.Tensor or scalar
    method : {'2-point', '3-point'}

    Returns
    -------
    h : float
        The absolute step size

    Notes
    -----
    `h` will always be np.float64. However, if `x0` or `f0` are
    smaller floating point dtypes (e.g. np.float32), then the absolute
    step size will be calculated from the smallest floating point size.
    """
    # this is used instead of np.sign(x0) because we need
    # sign_x0 to be 1 when x0 == 0.
    sign_x0 = (x0 >= 0).type(torch.float64) * 2 - 1

    # User has requested specific relative steps.
    # Don't multiply by max(1, abs(x0) because if x0 < 1 then their
    # requested step is not used.
    abs_step = rel_step * sign_x0 * torch.abs(x0)

    # however we don't want an abs_step of 0, which can happen if
    # rel_step is 0, or x0 is 0. Instead, substitute a realistic step
    rstep = 1e-4
    dx = (x0 + abs_step) - x0
    abs_step = torch.where(
        dx == 0,
        rstep
        * sign_x0
        * torch.maximum(
            torch.tensor(1.0, dtype=x0.dtype, device=x0.device), torch.abs(x0)
        ),
        abs_step,
    )

    return abs_step


def _dense_difference(
    fun: Callable, x0: torch.Tensor, f0: torch.Tensor, h: torch.Tensor, method: str
):
    """
    Compute the finite difference approximation of the Jacobian matrix
    Arguments:
        fun: the function to differentiate. The function has the interface y = fun(x), where
            x is a torch.Tensor of shape (batch_dim, n,) and y is a torch.Tensor of shape (batch_dim, m,)
        x0: the point at which to differentiate. The tensor has shape (batch_dim, n,)
        f0: optionally the function value at x0. The tensor has shape (batch_dim, m,)
        h: the step size
        method: the finite difference method. Either '2-point' or '3-point'
    """
    batch_dim = x0.size(0)
    n = x0.size(-1)
    m = f0.size(-1)
    J_transposed = torch.empty((batch_dim, n, m), device=x0.device)

    for i in range(n):
        if method == "2-point":
            x1 = x0 + torch.concat(
                [
                    torch.zeros((batch_dim, i), device=x0.device),
                    h[:, i : i + 1],
                    torch.zeros((batch_dim, n - i - 1), device=x0.device),
                ],
                dim=-1,
            )
            dx = h[:, i]
            df = fun(x1) - f0
        elif method == "3-point":
            dx02 = torch.concat(
                [
                    torch.zeros((batch_dim, i), device=x0.device),
                    h[i : i + 1],
                    torch.zeros((batch_dim, n - i - 1), device=x0.device),
                ],
                dim=-1,
            )
            x1 = x0 - dx02
            x2 = x0 + dx02
            dx = 2 * h[:, i]
            f1 = fun(x1)
            f2 = fun(x2)
            df = f2 - f1
        else:
            raise ValueError("Unknown method '%s'." % method)

        Ji = df / dx[:, None]
        J_transposed[:, i, :] = Ji

    J = J_transposed.permute(0, 2, 1)

    return J
