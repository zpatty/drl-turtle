__all__ = ["cornelia_joint_space_control_factory"]
import numpy as np
from typing import Callable, Dict, Optional, Tuple

from mjc_turtle_robot.biological_oracles import (
    cornelia_turtle_robot_joint_space_trajectory_factory,
)


def cornelia_joint_space_control_factory(
    kp: np.ndarray,
    q_off: Optional[np.ndarray] = None,
    sf: float = 1.0,
    sw: float = 1.0,
) -> Callable:
    """
    Joint space control function for the Cornelia turtle robot.
    Arguments:
        kp: Proportional control gains as array of shape (6,)
        q_off: Offset for the joint space trajectory as array of shape (6,)
        sf: Scaling factor for the joint angles (as a delta from q_off) as scalar float
        sw: Time scaling factor of oracle as scalar float
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
    def joint_space_control_fn(t: float, q: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
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
        q_des_ar = q_des_ar_fn(t)
        q_des_al = q_des_al_fn(t)
        q_d_des_ar = q_d_des_ar_fn(t)
        q_d_des_al = q_d_des_al_fn(t)

        # desired joint positions
        q_des = np.concatenate([q_des_ar, q_des_al], axis=-1)
        # desired joint velocities
        q_d_des = np.concatenate([q_d_des_ar, q_d_des_al], axis=-1)

        # compute control signal
        u = q_d_des + kp @ (q_des - q)

        aux = dict(
            q=q,
            q_d_des=q_d_des,
            q_des=q_des,
        )

        return u, aux

    return joint_space_control_fn

def cornelia_joint_space_motion_primitive_control_factory(
    q_off: Optional[np.ndarray] = None,
    sf: float = 1.0,
    sw: float = 1.0,
    limit_cycle_kp: float = 1.0,
    phase_sync_kp: float = 0.0,
    encoder_conditioning_mode: Optional[str] = None,
    verbose: bool = True,
) -> Callable:
    """
    Joint space control function for the Cornelia turtle robot.
    Arguments:
        q_off: Offset for the joint space trajectory as array of shape (6,)
        sf: Scaling factor for the joint angles (as a delta from q_off) as scalar float
        sw: Time scaling factor of oracle as scalar float
        limit_cycle_kp: proportional control gain to converge onto the limit cycle
        phase_sync_kp: proportional control gain for phase synchronization
        encoder_conditioning_mode: Mode for encoder conditioning. Options are: None, "pitch_down_test"
        verbose: Boolean flag to print the control function
    """
    if q_off is None:
        q_off = np.zeros((6,))

    match encoder_conditioning_mode:
        case None:
            oracle_name = "CorneliaTurtleRobotJointSpace"
        case "pitch_down_test":
            oracle_name = "CorneliaTurtleRobotJointSpacePitchedDown"
        case _:
            raise ValueError(f"Unknown encoder conditioning mode: {encoder_conditioning_mode}")
    print(f"Using model trained on oracle {oracle_name}")

    from osmp.motion_primitive_factories import orbitally_stable_motion_primitive_factory
    motion_primitive_fn, _ = orbitally_stable_motion_primitive_factory(
        oracle_name=oracle_name,
        num_systems=2,
        model_compilation_mode="aot.compile",
        saved_model_compilation_mode="aot.compile",
        sf=sf,
        sw=sw,
        limit_cycle_kp=limit_cycle_kp,
        phase_sync_kp=phase_sync_kp
    )

    # rotate the control signal according the flipper arm signs
    sign_array = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0]])

    # control function
    def joint_space_control_fn(
            t: float, q: np.ndarray, z: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Joint space control function for both flipper arms.
        Arguments:
            t: time in seconds
            q: joint positions of the flipper arms. Assumes the following order:
                [q1, q2, q3, q4, q5, q6], where (q1, q2, q3) are the joint positions of the right flipper arm
                and (q4, q5, q6) are the joint positions of the left flipper arm.
            z: encoder conditioning signal as array of shape (2, 1)
        Returns:
            u: control signal as joint velocities for the flipper arms. Assumes the following order:
                [u1, u2, u3, u4, u5, u6], where (u1, u2, u3) are the joint velocities of the right flipper arm
                and (u4, u5, u6) are the joint velocities of the left flipper arm.
            aux: dictionary of auxiliary outputs
        """
        # apply the joint angle offset
        q_norm = q - q_off * sign_array.flatten()

        # reshape joint positions as (2, 3)
        q_norm = q_norm.reshape(2, -1)

        # rotate into the right turtle arm frame
        q_norm = q_norm * sign_array

        # compute control signal
        u, aux = motion_primitive_fn(t, q_norm, z=z)

        # rotate the control signal back
        u = u * sign_array

        # reshape control signal as (6,)
        u = u.reshape(-1)

        return u, aux

    return joint_space_control_fn
