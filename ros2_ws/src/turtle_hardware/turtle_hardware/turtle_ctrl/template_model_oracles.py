__all__ = ["reverse_stroke_joint_oracle_factory"]
import numpy as np
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union


def reverse_stroke_joint_oracle_factory(
    s: int = 1, sf: float = 1.0, sw: float = 1.0, q_off: np.ndarray = None
) -> Tuple[Callable, Callable, Callable]:
    """
    The coordinate system of the origin of the arm has the following convention: x: right, y: forward, z: up
    The side_sign for the right flipper is +1, the sign of the left flipper is -1
    The sign of q is according to the convention in MuJoCo (in this case for the right flipper arm):
        q1: rotation around the negative y-axis => positive roll
        q2: rotation around the positive z-axis => positive yaw
        q3: rotation around the negative x-axis => negative pitch
    Arguments:
        s: int: side sign. +1 for the right flipper, -1 for the left flipper
        sf: float: Scaling factor for the joint angles (as a delta from q_off) as scalar float
        sw: float: Time scaling factor (i.e., speed-up factor)
        q_off: np.ndarray: joint space offset
    Returns:
        q_fn: Callable: joint space position trajectory function
        q_d_fn: Callable: joint space velocity trajectory function
        q_dd_fn: Callable: joint space acceleration trajectory function
    """
    if q_off is None:
        q_off = np.array([0.0, 0.0, 30 / 180 * np.pi])

    omega = 0.25 * 2 * np.pi * sw

    # roll / yaw coefficients
    roll_yaw_a0 = 1.653
    roll_yaw_a1 = -36.3
    roll_yaw_b1 = -28.71
    roll_yaw_a2 = -6.255
    roll_yaw_b2 = 8.005
    roll_yaw_a3 = 0.002799
    roll_yaw_b3 = 2.446
    roll_yaw_a4 = 0.5802
    roll_yaw_b4 = 0.531
    # pitch coefficients
    pitch_a0 = 28.92
    pitch_a1 = 39.83
    pitch_b1 = -42.8
    pitch_a2 = -1.354
    pitch_b2 = -18.68
    pitch_a3 = 6.398
    pitch_b3 = 5.168
    pitch_a4 = 10.39
    pitch_b4 = -1.523
    pitch_a5 = 1.066
    pitch_b5 = -1.657
    pitch_a6 = 0.4455
    pitch_b6 = 2.365
    pitch_a7 = 1.597
    pitch_b7 = 0.9559


    def q_fn(t: np.ndarray) -> np.ndarray:
        roll_yaw = (
            + roll_yaw_a0
            + roll_yaw_a1 * np.cos(t * omega)
            + roll_yaw_b1 * np.sin(t * omega)
            + roll_yaw_a2 * np.cos(2 * t * omega)
            + roll_yaw_b2 * np.sin(2 * t * omega)
            + roll_yaw_a3 * np.cos(3 * t * omega)
            + roll_yaw_b3 * np.sin(3 * t * omega)
            + roll_yaw_a4 * np.cos(4 * t * omega)
            + roll_yaw_b4 * np.sin(4 * t * omega)
        ) * np.pi / 180

        pitch = (
            + pitch_a0
            + pitch_a1 * np.cos(t * omega)
            + pitch_b1 * np.sin(t * omega)
            + pitch_a2 * np.cos(2 * t * omega)
            + pitch_b2 * np.sin(2 * t * omega)
            + pitch_a3 * np.cos(3 * t * omega)
            + pitch_b3 * np.sin(3 * t * omega)
            + pitch_a4 * np.cos(4 * t * omega)
            + pitch_b4 * np.sin(4 * t * omega)
            + pitch_a5 * np.cos(5 * t * omega)
            + pitch_b5 * np.sin(5 * t * omega)
            + pitch_a6 * np.cos(6 * t * omega)
            + pitch_b6 * np.sin(6 * t * omega)
            + pitch_a7 * np.cos(7 * t * omega)
            + pitch_b7 * np.sin(7 * t * omega)
        ) * np.pi / 180

        q = np.array([roll_yaw, -s * roll_yaw, -s * pitch]) * sf

        # apply the offset
        q = q + q_off * np.array([1, s, s])

        return q

    def q_d_fn(t: np.ndarray) -> np.ndarray:
        roll_yaw_d = omega * (
            - 1 * roll_yaw_a1 * np.sin(t * omega)
            + 1 * roll_yaw_b1 * np.cos(t * omega)
            - 2 * roll_yaw_a2 * np.sin(2 * t * omega)
            + 2 * roll_yaw_b2 * np.cos(2 * t * omega)
            - 3 * roll_yaw_a3 * np.sin(3 * t * omega)
            + 3 * roll_yaw_b3 * np.cos(3 * t * omega)
            - 4 * roll_yaw_a4 * np.sin(4 * t * omega)
            + 4 * roll_yaw_b4 * np.cos(4 * t * omega)
        ) * np.pi / 180

        pitch_d = omega * (
            - 1 * pitch_a1 * np.sin(t * omega)
            + 1 * pitch_b1 * np.cos(t * omega)
            - 2 * pitch_a2 * np.sin(2 * t * omega)
            + 2 * pitch_b2 * np.cos(2 * t * omega)
            - 3 * pitch_a3 * np.sin(3 * t * omega)
            + 3 * pitch_b3 * np.cos(3 * t * omega)
            - 4 * pitch_a4 * np.sin(4 * t * omega)
            + 4 * pitch_b4 * np.cos(4 * t * omega)
            - 5 * pitch_a5 * np.sin(5 * t * omega)
            + 5 * pitch_b5 * np.cos(5 * t * omega)
            - 6 * pitch_a6 * np.sin(6 * t * omega)
            + 6 * pitch_b6 * np.cos(6 * t * omega)
            - 7 * pitch_a7 * np.sin(7 * t * omega)
            + 7 * pitch_b7 * np.cos(7 * t * omega)
        ) * np.pi / 180

        q_d = np.array([roll_yaw_d, -s * roll_yaw_d, -s * pitch_d]) * sf

        return q_d

    def q_dd_fn(t: np.ndarray) -> np.ndarray:
        roll_yaw_dd = omega**2 * (
            - 1 * roll_yaw_a1 * np.cos(t * omega)
            - 1 * roll_yaw_b1 * np.sin(t * omega)
            - 4 * roll_yaw_a2 * np.cos(2 * t * omega)
            - 4 * roll_yaw_b2 * np.sin(2 * t * omega)
            - 9 * roll_yaw_a3 * np.cos(3 * t * omega)
            - 9 * roll_yaw_b3 * np.sin(3 * t * omega)
            - 16 * roll_yaw_a4 * np.cos(4 * t * omega)
            - 16 * roll_yaw_b4 * np.sin(4 * t * omega)
        ) * np.pi / 180

        pitch_dd = omega**2 * (
            - 1 * pitch_a1 * np.cos(t * omega)
            - 1 * pitch_b1 * np.sin(t * omega)
            - 4 * pitch_a2 * np.cos(2 * t * omega)
            - 4 * pitch_b2 * np.sin(2 * t * omega)
            - 9 * pitch_a3 * np.cos(3 * t * omega)
            - 9 * pitch_b3 * np.sin(3 * t * omega)
            - 16 * pitch_a4 * np.cos(4 * t * omega)
            - 16 * pitch_b4 * np.sin(4 * t * omega)
            - 25 * pitch_a5 * np.cos(5 * t * omega)
            - 25 * pitch_b5 * np.sin(5 * t * omega)
            - 36 * pitch_a6 * np.cos(6 * t * omega)
            - 36 * pitch_b6 * np.sin(6 * t * omega)
            - 49 * pitch_a7 * np.cos(7 * t * omega)
            - 49 * pitch_b7 * np.sin(7 * t * omega)
        ) * np.pi / 180

        q_dd = np.array([roll_yaw_dd, -s * roll_yaw_dd, -s * pitch_dd]) * sf

        return q_dd

    return q_fn, q_d_fn, q_dd_fn
