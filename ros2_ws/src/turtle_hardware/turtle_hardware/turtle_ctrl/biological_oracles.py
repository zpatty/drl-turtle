__all__ = [
    "cornelia_turtle_robot_joint_space_trajectory_factory",
]
import numpy as np
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union


def cornelia_turtle_robot_joint_space_trajectory_factory(
    s: int = 1, sf: float = 1.0, sw: float = 1.0, q_off: np.ndarray = np.zeros(3)
) -> Tuple[Callable, Callable, Callable]:
    """
    Trajectory is from:
        van der Geest, N., Garcia, L., Borret, F., Nates, R., & Gonzalez, A. (2023).
        Soft-robotic green sea turtle (Chelonia mydas) developed to replace animal experimentation provides new insight
        into their propulsive strategies. Scientific Reports, 13(1), 11983.

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
    # unnormalized trajectory duration
    T = 4.3  # s

    # constant values solved from non-linear least squares for roll axis
    roll_a0r = 3.278
    roll_a1r = -60.12
    roll_b1r = -19.3
    roll_a2r = -5.566
    roll_b2r = 15.64
    roll_a3r = -0.08129
    roll_b3r = 3.339
    roll_a4r = 0.4491
    roll_b4r = -0.3352
    roll_a5r = -1.401
    roll_b5r = -0.1525
    roll_a6r = -1.423
    roll_b6r = -0.6598
    roll_a7r = -0.8562
    roll_b7r = -0.2633
    roll_a8r = -0.5337
    roll_b8r = 0.09667
    roll_wr = 1.461

    # constant values solved from non-linear least squares for yaw axis
    yaw_a0y = 0.8961
    yaw_a1y = -25.16
    yaw_b1y = -22.35
    yaw_a2y = -8.635
    yaw_b2y = -2.142
    yaw_a3y = -2.605
    yaw_b3y = 0.2852
    yaw_a4y = -0.9021
    yaw_b4y = -0.8331
    yaw_a5y = -0.1804
    yaw_b5y = 0.4037
    yaw_a6y = -0.01423
    yaw_b6y = 0.1734
    yaw_a7y = -0.06358
    yaw_b7y = 0.1808
    yaw_a8y = -0.0916
    yaw_b8y = 0.1716
    yaw_wy = 1.461

    def q_fn(t: np.ndarray) -> np.ndarray:
        tn = sw * t  # speed-up the trajectory by a factor of sw
        tn = tn % T  # repeat the trajectory every (normalized) 4.3 seconds

        roll = (
            roll_a0r
            + roll_a1r * np.cos(tn * roll_wr)
            + roll_b1r * np.sin(tn * roll_wr)
            + roll_a2r * np.cos(2 * tn * roll_wr)
            + roll_b2r * np.sin(2 * tn * roll_wr)
            + roll_a3r * np.cos(3 * tn * roll_wr)
            + roll_b3r * np.sin(3 * tn * roll_wr)
            + roll_a4r * np.cos(4 * tn * roll_wr)
            + roll_b4r * np.sin(4 * tn * roll_wr)
            + roll_a5r * np.cos(5 * tn * roll_wr)
            + roll_b5r * np.sin(5 * tn * roll_wr)
            + roll_a6r * np.cos(6 * tn * roll_wr)
            + roll_b6r * np.sin(6 * tn * roll_wr)
            + roll_a7r * np.cos(7 * tn * roll_wr)
            + roll_b7r * np.sin(7 * tn * roll_wr)
            + roll_a8r * np.cos(8 * tn * roll_wr)
            + roll_b8r * np.sin(8 * tn * roll_wr)
        )

        yaw = (
            yaw_a0y
            + yaw_a1y * np.cos(tn * yaw_wy)
            + yaw_b1y * np.sin(tn * yaw_wy)
            + yaw_a2y * np.cos(2 * tn * yaw_wy)
            + yaw_b2y * np.sin(2 * tn * yaw_wy)
            + yaw_a3y * np.cos(3 * tn * yaw_wy)
            + yaw_b3y * np.sin(3 * tn * yaw_wy)
            + yaw_a4y * np.cos(4 * tn * yaw_wy)
            + yaw_b4y * np.sin(4 * tn * yaw_wy)
            + yaw_a5y * np.cos(5 * tn * yaw_wy)
            + yaw_b5y * np.sin(5 * tn * yaw_wy)
            + yaw_a6y * np.cos(6 * tn * yaw_wy)
            + yaw_b6y * np.sin(6 * tn * yaw_wy)
            + yaw_a7y * np.cos(7 * tn * yaw_wy)
            + yaw_b7y * np.sin(7 * tn * yaw_wy)
            + yaw_a8y * np.cos(8 * tn * yaw_wy)
            + yaw_b8y * np.sin(8 * tn * yaw_wy)
        )

        # twist/pitch angle
        if 0 <= tn < 0.398:
            pitch = 17500 / 199 * tn
        elif 0.398 <= tn < 2.309:
            pitch = 35
        elif 2.309 <= tn < 2.707:
            pitch = -(17500 / 199) * tn + (94745 / 398)
        elif 2.707 <= tn < 3.026:
            pitch = -225.705 * tn + 610.984
        elif 3.026 <= tn < 3.663:
            pitch = -72
        elif 3.663 <= tn < 4.3:
            pitch = (72000 / 637) * tn - (309600 / 637)
        else:
            raise ValueError(f"Invalid time: {tn}")

        # convert from degrees to radians
        roll = np.deg2rad(roll)
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)

        q = np.array([roll, s * yaw, -s * pitch]) * sf

        # apply the offset
        q = q + q_off * np.array([1, s, s])

        return q

    def q_d_fn(t: np.ndarray) -> np.ndarray:
        tn = sw * t  # speed-up the trajectory by a factor of sw
        tn = tn % T  # repeat the trajectory every (normalized) 4.3 seconds

        roll_d = (
            -roll_a1r * sw * roll_wr * np.sin(tn * roll_wr)
            + roll_b1r * sw * roll_wr * np.cos(tn * roll_wr)
            - 2 * roll_a2r * sw * roll_wr * np.sin(2 * tn * roll_wr)
            + 2 * roll_b2r * sw * roll_wr * np.cos(2 * tn * roll_wr)
            - 3 * roll_a3r * sw * roll_wr * np.sin(3 * tn * roll_wr)
            + 3 * roll_b3r * sw * roll_wr * np.cos(3 * tn * roll_wr)
            - 4 * roll_a4r * sw * roll_wr * np.sin(4 * tn * roll_wr)
            + 4 * roll_b4r * sw * roll_wr * np.cos(4 * tn * roll_wr)
            - 5 * roll_a5r * sw * roll_wr * np.sin(5 * tn * roll_wr)
            + 5 * roll_b5r * sw * roll_wr * np.cos(5 * tn * roll_wr)
            - 6 * roll_a6r * sw * roll_wr * np.sin(6 * tn * roll_wr)
            + 6 * roll_b6r * sw * roll_wr * np.cos(6 * tn * roll_wr)
            - 7 * roll_a7r * sw * roll_wr * np.sin(7 * tn * roll_wr)
            + 7 * roll_b7r * sw * roll_wr * np.cos(7 * tn * roll_wr)
            - 8 * roll_a8r * sw * roll_wr * np.sin(8 * tn * roll_wr)
            + 8 * roll_b8r * sw * roll_wr * np.cos(8 * tn * roll_wr)
        )

        yaw_d = (
            -yaw_a1y * sw * yaw_wy * np.sin(tn * yaw_wy)
            + yaw_b1y * sw * yaw_wy * np.cos(tn * yaw_wy)
            - 2 * yaw_a2y * sw * yaw_wy * np.sin(2 * tn * yaw_wy)
            + 2 * yaw_b2y * sw * yaw_wy * np.cos(2 * tn * yaw_wy)
            - 3 * yaw_a3y * sw * yaw_wy * np.sin(3 * tn * yaw_wy)
            + 3 * yaw_b3y * sw * yaw_wy * np.cos(3 * tn * yaw_wy)
            - 4 * yaw_a4y * sw * yaw_wy * np.sin(4 * tn * yaw_wy)
            + 4 * yaw_b4y * sw * yaw_wy * np.cos(4 * tn * yaw_wy)
            - 5 * yaw_a5y * sw * yaw_wy * np.sin(5 * tn * yaw_wy)
            + 5 * yaw_b5y * sw * yaw_wy * np.cos(5 * tn * yaw_wy)
            - 6 * yaw_a6y * sw * yaw_wy * np.sin(6 * tn * yaw_wy)
            + 6 * yaw_b6y * sw * yaw_wy * np.cos(6 * tn * yaw_wy)
            - 7 * yaw_a7y * sw * yaw_wy * np.sin(7 * tn * yaw_wy)
            + 7 * yaw_b7y * sw * yaw_wy * np.cos(7 * tn * yaw_wy)
            - 8 * yaw_a8y * sw * yaw_wy * np.sin(8 * tn * yaw_wy)
            + 8 * yaw_b8y * sw * yaw_wy * np.cos(8 * tn * yaw_wy)
        )

        # twist/pitch velocity
        if 0 <= tn < 0.398:
            pitch_d = 17500 / 199 * sw
        elif 0.398 <= tn < 2.309:
            pitch_d = 0.0
        elif 2.309 <= tn < 2.707:
            pitch_d = -(17500 / 199) * sw
        elif 2.707 <= tn < 3.026:
            pitch_d = -225.705 * sw
        elif 3.026 <= tn < 3.663:
            pitch_d = -72
        elif 3.663 <= tn < 4.3:
            pitch_d = (72000 / 637) * sw
        else:
            raise ValueError(f"Invalid time: {tn}")

        # convert from degrees to radians
        roll_d = np.deg2rad(roll_d)
        yaw_d = np.deg2rad(yaw_d)
        pitch_d = np.deg2rad(pitch_d)

        q_d = np.array([roll_d, s * yaw_d, -s * pitch_d]) * sf

        return q_d

    def q_dd_fn(t: np.ndarray) -> np.ndarray:
        tn = sw * t  # speed-up the trajectory by a factor of sw
        tn = tn % T  # repeat the trajectory every (normalized) 4.3 seconds

        # roll acceleration
        roll_dd = (
            -roll_a1r * sw**2 * roll_wr**2 * np.cos(tn * roll_wr)
            - roll_b1r * sw**2 * roll_wr**2 * np.sin(tn * roll_wr)
            - 4 * roll_a2r * sw**2 * roll_wr**2 * np.cos(2 * tn * roll_wr)
            - 4 * roll_b2r * sw**2 * roll_wr**2 * np.sin(2 * tn * roll_wr)
            - 9 * roll_a3r * sw**2 * roll_wr**2 * np.cos(3 * tn * roll_wr)
            - 9 * roll_b3r * sw**2 * roll_wr**2 * np.sin(3 * tn * roll_wr)
            - 16 * roll_a4r * sw**2 * roll_wr**2 * np.cos(4 * tn * roll_wr)
            - 16 * roll_b4r * sw**2 * roll_wr**2 * np.sin(4 * tn * roll_wr)
            - 25 * roll_a5r * sw**2 * roll_wr**2 * np.cos(5 * tn * roll_wr)
            - 25 * roll_b5r * sw**2 * roll_wr**2 * np.sin(5 * tn * roll_wr)
            - 36 * roll_a6r * sw**2 * roll_wr**2 * np.cos(6 * tn * roll_wr)
            - 36 * roll_b6r * sw**2 * roll_wr**2 * np.sin(6 * tn * roll_wr)
            - 49 * roll_a7r * sw**2 * roll_wr**2 * np.cos(7 * tn * roll_wr)
            - 49 * roll_b7r * sw**2 * roll_wr**2 * np.sin(7 * tn * roll_wr)
            - 64 * roll_a8r * sw**2 * roll_wr**2 * np.cos(8 * tn * roll_wr)
            - 64 * roll_b8r * sw**2 * roll_wr**2 * np.sin(8 * tn * roll_wr)
        )

        # yaw acceleration
        yaw_dd = (
            -yaw_a1y * sw**2 * yaw_wy**2 * np.cos(tn * yaw_wy)
            - yaw_b1y * sw**2 * yaw_wy**2 * np.sin(tn * yaw_wy)
            - 4 * yaw_a2y * sw**2 * yaw_wy**2 * np.cos(2 * tn * yaw_wy)
            - 4 * yaw_b2y * sw**2 * yaw_wy**2 * np.sin(2 * tn * yaw_wy)
            - 9 * yaw_a3y * sw**2 * yaw_wy**2 * np.cos(3 * tn * yaw_wy)
            - 9 * yaw_b3y * sw**2 * yaw_wy**2 * np.sin(3 * tn * yaw_wy)
            - 16 * yaw_a4y * sw**2 * yaw_wy**2 * np.cos(4 * tn * yaw_wy)
            - 16 * yaw_b4y * sw**2 * yaw_wy**2 * np.sin(4 * tn * yaw_wy)
            - 25 * yaw_a5y * sw**2 * yaw_wy**2 * np.cos(5 * tn * yaw_wy)
            - 25 * yaw_b5y * sw**2 * yaw_wy**2 * np.sin(5 * tn * yaw_wy)
            - 36 * yaw_a6y * sw**2 * yaw_wy**2 * np.cos(6 * tn * yaw_wy)
            - 36 * yaw_b6y * sw**2 * yaw_wy**2 * np.sin(6 * tn * yaw_wy)
            - 49 * yaw_a7y * sw**2 * yaw_wy**2 * np.cos(7 * tn * yaw_wy)
            - 49 * yaw_b7y * sw**2 * yaw_wy**2 * np.sin(7 * tn * yaw_wy)
            - 64 * yaw_a8y * sw**2 * yaw_wy**2 * np.cos(8 * tn * yaw_wy)
            - 64 * yaw_b8y * sw**2 * yaw_wy**2 * np.sin(8 * tn * yaw_wy)
        )

        # twist/pitch acceleration
        pitch_dd = np.array(0.0)

        # convert from degrees to radians
        roll_dd = np.deg2rad(roll_dd)
        yaw_dd = np.deg2rad(yaw_dd)
        pitch_dd = np.deg2rad(pitch_dd)

        q_dd = np.array([roll_dd, s * yaw_dd, -s * pitch_dd]) * sf

        return q_dd

    return q_fn, q_d_fn, q_dd_fn
