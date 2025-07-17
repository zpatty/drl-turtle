import numpy as np


def damped_pinv(A: np.ndarray, damping: float = 0.0) -> np.ndarray:
    """
    Computes the damped pseudo-inverse of the matrix A.
    Args:
        A: The matrix to be pseudo-inverted of shape (m, n).
        damping: The damping factor lambda.
    Returns:
        A_pinv: The pseudo-inverse of A of shape (n, m).
    """
    if A.shape[0] >= A.shape[1]:
        A_pinv = (
            np.linalg.inv(A.T @ A + damping**2 * np.eye(A.shape[1])) @ A.T
        )  # left pseudo-inverse
    else:
        A_pinv = A.T @ np.linalg.inv(A @ A.T + damping**2 * np.eye(A.shape[0]))

    return A_pinv
