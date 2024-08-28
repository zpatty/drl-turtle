__all__ = ["green_sea_turtle_task_space_control_factory"]
import dill
from functools import partial
import numpy as np
from pathlib import Path
from scipy.linalg import block_diag
import sympy as sp
from typing import Callable, Dict, Optional, Tuple, Union
import yaml

import mjc_turtle_robot
from mjc_turtle_robot.biological_oracles import (
    green_sea_turtle_task_space_trajectory_factory,
)
from mjc_turtle_robot.kinematics import damped_pinv


def green_sea_turtle_task_space_control_factory(
    x_off: Optional[np.ndarray] = None,
    sf: Union[float, np.ndarray] = 1.0,
    sw: float = 1.0,
    kp: Optional[np.ndarray] = None,
    kp_th: Union[float, np.ndarray] = 0.0,
    w_th: Optional[Union[float, np.ndarray]] = None,
    pinv_damping: float = 2e-3,
    use_learned_motion_primitive: bool = False,
    use_straight_flipper: bool = False,
    limit_cycle_kp: float = 1.0,
    phase_sync_kp: float = 0.0,
    verbose: bool = True,
) -> Callable:
    """
    Task space control function for the green sea turtle robot.
    Arguments:
        x_off: Offset for the task space trajectory as array of shape (3,)
        sf: Spatial scaling factor as scalar float or array of shape (3,)
        sw: Temporal scaling factor of oracle as scalar float
        kp: Proportional control gains as array of shape (6,)
        kp_th: Proportional control gains for the twist as an array of shape (2,)
        w_th: Weighting factors for the twist tracking in the Jacobian as an array of shape (2,)
        pinv_damping: Damping factor for the pseudo-inverse as scalar float
        use_learned_motion_primitive: Boolean flag to use the learned motion primitive
        use_straight_flipper: Boolean flag to use the straight flipper arm
        limit_cycle_kp: proportional control gain to converge onto the limit cycle
        phase_sync_kp: proportional control gain for phase synchronization
        verbose: Boolean flag to print the control function
    """
    straight_flipper_str = "_straight_flipper" if use_straight_flipper else ""
    straight_flippers_str = "_straight_flippers" if use_straight_flipper else ""

    if type(sf) == float:
        sf = np.array([sf, sf, sf, sf])
    if type(kp_th) == float:
        kp_th = np.array([kp_th, kp_th])
    if type(w_th) == float:
        w_th = np.array([w_th, w_th])

    track_twist = True if w_th is not None and (w_th > 0.0).any() else False

    if use_learned_motion_primitive:
        from osmp.motion_primitive_factories import orbitally_stable_motion_primitive_factory
        motion_primitive_fn, _ = orbitally_stable_motion_primitive_factory(
            oracle_name="GreenSeaTurtleSwimmingWithTwist" if track_twist else "GreenSeaTurtleSwimming",
            num_systems=2,
            model_compilation_mode="aot.compile",
            saved_model_compilation_mode="aot.compile",
            sf=sf if track_twist else sf[:3],
            sw=sw,
            phase_sync_kp=phase_sync_kp
        )

        if kp is not None:
            # make the gains diagonal
            kp = np.stack([np.diag(kp[:3]), np.diag(kp[3:6])], axis=0)

        def x_d_fn(
            t: float, x: np.ndarray, s: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
            """
            Task space velocity function for the green sea turtle robot based on the learned motion primitive.
            Arguments:
                t: time in seconds
                x: current task space position of shape (num_flippers, 3)
                s: flipper arm signs as array of shape (num_flippers,)
            Returns:
                x_d: desired task space velocity of shape (num_flippers, 3)
                aux: dictionary of auxiliary outputs
            """
            num_flippers = x.shape[0]
            if s is None:
                s = np.ones((num_flippers, ))

            if track_twist:
                sign_array = np.concatenate([s[:, None], np.ones((num_flippers, 2)), s[:, None]], axis=1)
            else:
                sign_array = np.concatenate([s[:, None], np.ones((num_flippers, 2))], axis=1)

            aux = dict(x_scaled=x)

            # descale the end-effector position
            x[:, :3] = x[:, :3] - x_off * sign_array[:, :3]

            # rotate into the right turtle arm frame
            x = x * sign_array

            # if kp is not None and np.linalg.norm(x) > 0.50:
            #     # use proportional control to drive us into the workspace
            #     x_d = -np.einsum('nij,nj->ni', kp, x)
            #     aux_mp = dict(
            #         x_unnorm=x,
            #         x_norm=x,
            #         x_d_norm=x_d,
            #         x_d_unnorm=x_d,
            #         y=np.zeros_like(x),
            #         y_d=np.zeros_like(x_d),
            #     )
            #
            #     if verbose:
            #         print(
            #             f"Using proportional control as x norm is {np.linalg.norm(x)}"
            #         )
            # else:
            #     x_d, aux_mp = motion_primitive_fn(t, x)

            x_d, aux_mp = motion_primitive_fn(t, x)
            aux = aux | aux_mp

            # rotate into the turtle arm frame
            x_d = x_d * sign_array
            aux["x_d_scaled"] = x_d

            return x_d, aux

        x_d_fn = partial(x_d_fn, s=np.array([1.0, -1.0]))
    else:
        # task space trajectory functions for the right flipper arm
        x_ra_fn, x_d_ra_fn, x_dd_ra_fn, th_ra_fn, th_d_ra_fn, th_dd_ra_fn = (
            green_sea_turtle_task_space_trajectory_factory(
                s=1, sf=sf, sw=sw, x_off=x_off
            )
        )
        # task space trajectory functions for the left flipper arm
        x_la_fn, x_d_la_fn, x_dd_la_fn, th_la_fn, th_d_la_fn, th_dd_la_fn = (
            green_sea_turtle_task_space_trajectory_factory(
                s=-1, sf=sf, sw=sw, x_off=x_off
            )
        )

        if kp is None:
            kp = 2e0 * np.ones(6)
        # make the gains diagonal
        kp = np.diag(kp)

    # load the symbolic expressions
    symbolic_model_dir = (
        Path(mjc_turtle_robot.__file__).parent.parent.parent / "symbolic_model"
    )
    with open(symbolic_model_dir / "arm_right_dynamics.dill", "rb") as f:
        ar_symbolic_expressions = dill.load(f)
    with open(symbolic_model_dir / "arm_left_dynamics.dill", "rb") as f:
        al_symbolic_expressions = dill.load(f)

    # load the turtle parameters from the mjc_turtle_robot/coefficients/turtle_arm_params.yaml file
    with open(
        str(
            Path(mjc_turtle_robot.__file__).parent
            / "coefficients"
            / f"turtle_arm{straight_flipper_str}_params.yaml"
        ),
        "r",
    ) as f:
        turtle_params = yaml.safe_load(f)
    kinematic_params = [
        value for key, value in turtle_params["kinematic_params"].items()
    ]
    # simplify functions using partial:
    gee_ra_fn = partial(ar_symbolic_expressions["lambda_fns"]["gee"], *kinematic_params)
    gee_la_fn = partial(al_symbolic_expressions["lambda_fns"]["gee"], *kinematic_params)
    Jee_ra_fn = partial(ar_symbolic_expressions["lambda_fns"]["Jee"], *kinematic_params)
    Jee_la_fn = partial(al_symbolic_expressions["lambda_fns"]["Jee"], *kinematic_params)

    # control function
    def task_space_control_fn(
        t: float, q: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Joint space control function for both flipper arms.
        Arguments:
            t: time in seconds
            q: joint positions of the flipper arms. Assumes the following order:
                [q1, q2, q3, q4, q5, q6], where (q1, q2, q3) are the joint positions of the right flipper arm
                and (q4, q5, q6) are the joint positions of the left flipper arm.
        Returns:
            q_d_des: control signal as joint velocities for the flipper arms. Assumes the following order:
                [u1, u2, u3, u4, u5, u6], where (u1, u2, u3) are the joint velocities of the right flipper arm
                and (u4, u5, u6) are the joint velocities of the left flipper arm.
            aux: dictionary of auxiliary outputs
        """
        # evaluate the current task-space positions
        x_ar, x_al = gee_ra_fn(*q[:3])[:3, 3], gee_la_fn(*q[3:6])[:3, 3]
        x = np.concatenate([x_ar[:3], x_al[:3]], axis=-1)
        th = np.concatenate([q[2:3], q[5:6]], axis=-1)

        # initialize auxiliary outputs
        aux = dict(q=q, x=x)

        if use_learned_motion_primitive:
            x_des = None
            # construct input to the motion primitive
            x_mp = x.reshape(2, -1)
            if track_twist:
                x_mp = np.concatenate([x_mp, th.reshape(-1, 1)], axis=1)
            # desired task space velocities
            x_d_des_mp, aux_out = x_d_fn(t, x_mp)
            if track_twist:
                x_d_des = x_d_des_mp[:, :3].flatten()
                th_d_des = x_d_des_mp[:, 3]
            else:
                x_d_des = x_d_des_mp.flatten()
            for key in aux_out:
                aux[key] = aux_out[key].flatten()
        else:
            # desired task space positions
            x_des_ra, x_des_la = x_ra_fn(t), x_la_fn(t)
            x_des = np.concatenate([x_des_ra[:3], x_des_la[:3]], axis=-1)
            # desired task space velocities
            x_d_des_ra, x_d_des_la = x_d_ra_fn(t), x_d_la_fn(t)
            x_d_des = np.concatenate([x_d_des_ra[:3], x_d_des_la[:3]], axis=-1)

            if track_twist:
                # twist angles
                th_des_ra, th_des_la = th_ra_fn(t), th_la_fn(t)
                # desired twist angular velocities
                th_d_des_ra, th_d_des_la = th_d_ra_fn(t), th_d_la_fn(t)
                th_des = np.stack([th_des_ra, th_des_la], axis=-1)
                th_d_des = np.stack([th_d_des_ra, th_d_des_la], axis=-1)
                aux["th_des"], aux["th_d_des"] = th_des, th_d_des

        aux["x_d_des"] = x_d_des

        # compute positional task-space Jacobians
        J_ar, J_al = Jee_ra_fn(*q[:3])[:3], Jee_la_fn(*q[3:6])[:3]

        if track_twist:
            J_ar = np.concatenate([J_ar, np.array([0.0, 0.0, w_th[0]])[None]], axis=0)
            J_al = np.concatenate([J_al, np.array([0.0, 0.0, w_th[1]])[None]], axis=0)

        # stacked positional Jacobian
        Jp = block_diag(J_ar, J_al)
        # inverse of the Jacobian
        Jp_inv = damped_pinv(Jp, damping=pinv_damping)

        if use_learned_motion_primitive:
            u = x_d_des
            u_th = th_d_des
        else:
            # proportional controller + feedforward on the end-effector velocity
            u = x_d_des + kp @ (x_des - x)

            if track_twist:
                # proportional controller + feedforward on the twist velocity
                u_th = th_d_des + kp_th * (th_des - th)

        if track_twist:
            u = np.concatenate([
                u[:3],
                u_th[:1],
                u[3:6],
                u_th[1:],
            ], axis=0)

        aux["u"] = u

        # map the velocity from task to joint space
        q_d_des = Jp_inv @ u
        aux["q_d_des"] = q_d_des

        return q_d_des, aux

    return task_space_control_fn
