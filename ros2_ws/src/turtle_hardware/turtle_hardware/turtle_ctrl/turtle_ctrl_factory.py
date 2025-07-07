__all__ = ["cornelia_joint_space_trajectory_tracking_control_factory"]
from functools import partial
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union

try:
    from biological_oracles import cornelia_turtle_robot_joint_space_trajectory_factory
except:
    from turtle_ctrl.biological_oracles import cornelia_turtle_robot_joint_space_trajectory_factory


def normalize_joint_angles(q: np.ndarray) -> np.ndarray:
    """
    Normalize the joint angles to the interval [-pi, pi].
    """
    return np.remainder(q + np.pi, 2 * np.pi) - np.pi


def cornelia_joint_space_trajectory_tracking_control_factory(
    kp: np.ndarray,
    sf: float = 1.0,
    sw: float = 1.0,
    q_off: Optional[np.ndarray] = None,
) -> Callable:
    """
    Joint space control function for the Cornelia turtle robot.
    Arguments:
        kp: Proportional control gains as array of shape (6,)
        sf: Scaling factor for the joint angles (as a delta from q_off) as scalar float
        sw: Time scaling factor of oracle as scalar float
        q_off: Offset for the joint space trajectory as array of shape (6,)
    """
    if q_off is None:
        q_off = np.zeros((6,))

    kp = np.diag(kp)
    # joint space trajectory functions for the right flipper arm
    q_des_ar_fn, q_d_des_ar_fn, q_dd_des_ar_fn = (
        cornelia_turtle_robot_joint_space_trajectory_factory(s=1, sf=sf, sw=sw, q_off=q_off[:3])
    )
    # joint space trajectory functions for the left flipper arm
    q_des_al_fn, q_d_des_al_fn, q_dd_des_al_fn = (
        cornelia_turtle_robot_joint_space_trajectory_factory(s=-1, sf=sf, sw=sw, q_off=q_off[3:6])
    )

    # control function
    def joint_space_control_fn(t: float, q: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Joint space control function for both flipper arms.
        Arguments:
            t: time in seconds
            q: joint positions of the flipper arms. Assumes the following order:
                [q1, q2, q3, q4, q5, q6], where (q1, q2, q3) are the joint positions of the right flipper arm
                and (q4, q5, q6) are the joint positions of the left flipper arm.
        Returns:
            u: control signal as joint velocities for the flipper arms. Assumes the following order:
                [u1, u2, u3, u4, u5, u6], where (u1, u2, u3) are the joint velocities of the right flipper arm
                and (u4, u5, u6) are the joint velocities of the left flipper arm.
            aux: dictionary of auxiliary outputs
        """
        # normalize the joint angles to the interval [-pi, pi]
        q = normalize_joint_angles(q)

        # extract the desired joint positions and velocities
        q_des_ar = q_des_ar_fn(t)
        q_des_al = q_des_al_fn(t)
        q_d_des_ar = q_d_des_ar_fn(t)
        q_d_des_al = q_d_des_al_fn(t)

        # desired joint positions
        q_des = np.concatenate([q_des_ar, q_des_al], axis=-1)
        # normalize to the range [-pi, pi]
        q_des = normalize_joint_angles(q_des)
        # desired joint velocities
        q_d_des = np.concatenate([q_d_des_ar, q_d_des_al], axis=-1)

        # compute control signal
        u = q_d_des + kp @ normalize_joint_angles(q_des - q)

        aux = dict(
            q=q,
            q_d_des=q_d_des,
            q_des=q_des,
        )

        return u, aux

    return joint_space_control_fn
