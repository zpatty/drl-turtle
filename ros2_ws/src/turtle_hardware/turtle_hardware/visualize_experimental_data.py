import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from osmp.datasets import *

experiment_date = "/home/zach/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/data/"
experiment_id = "08_28_2024_18_06_13_np_data.npz"
oracle_name = "CorneliaTurtleRobotJointSpace"

match experiment_date:
    case "2024-08-16":
        q_off = np.array([0.0, 30.0 / 180 * np.pi, -22.0 / 180 * np.pi])
    case _:
        q_off = np.zeros((3,))

# plotting settings
outputs_dir = Path(__file__).parent / "outputs" / experiment_date
outputs_dir.mkdir(exist_ok=True, parents=True)
figsize = (6.4, 4.8)
dpi = 200
# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Computer Modern Roman"],
#     }
# )


def convert_motors_to_q(q: np.ndarray, qd: np.ndarray, u: np.ndarray):
    q = q - np.pi
    q_new = np.concatenate([-q[..., 3:6], q[..., 0:3], q[..., 6:]], axis=-1)
    qd_new = np.concatenate([qd[..., 3:6], qd[..., 0:3], qd[..., 6:]], axis=-1)
    u_new = np.concatenate([u[..., 3:6], -u[..., 0:3], u[..., 6:]], axis=-1)
    return q_new, qd_new, u_new


def main():
    experimental_data_folder = (
        Path(__file__).parent / "experimental_data" / experiment_date
    )
    print(experimental_data_folder)
    # load the experimental data
    data = np.load(experiment_date + experiment_id)
    ts = data["t"]
    q_ts = data["q"]
    q_d_ts = data["dq"]
    u_ts = data["u"]

    # convert from the motor to the motion primitive convention
    if experiment_date in ["2024-08-16", "2024-08-20"]:
        q_ts, q_d_ts, u_ts = convert_motors_to_q(q_ts, q_d_ts, u_ts)
    else:
        # flip the sign of the left flipper
        q_ts[:, 4:6] = -q_ts[:, 4:6]
        q_d_ts[:, 4:6] = -q_d_ts[:, 4:6]
        u_ts[:, 4:6] = -u_ts[:, 4:6]

    # extract the right and left flipper joint angles
    q_ar_ts = q_ts[:, 0:3]
    q_al_ts = q_ts[:, 3:6]
    q_d_ar_ts = q_d_ts[:, 0:3]
    q_d_al_ts = q_d_ts[:, 3:6]
    u_ar_ts = u_ts[:, 0:3]
    u_al_ts = u_ts[:, 3:6]

    q_off_ar = q_off
    match oracle_name:
        case "GreenSeaTurtleSwimming":
            dataset = GreenSeaTurtleSwimmingKinematics(normalize=True)
        case "GreenSeaTurtleSwimming" | "GreenSeaTurtleSwimmingWithTwist":
            dataset = GreenSeaTurtleSwimmingKinematics(
                normalize=True,
                include_twist_trajectory=True
                if oracle_name == "GreenSeaTurtleSwimmingWithTwist"
                else False,
            )
            if oracle_name == "GreenSeaTurtleSwimmingWithTwist":
                q_off_ar = np.concatenate([q_off, np.zeros((1,))])
        case "CorneliaTurtleRobotJointSpace":
            dataset = CorneliaTurtleRobotJointSpaceTrajectory(normalize=True)
        case _:
            raise ValueError(f"Unknown oracle name: {oracle_name}")
    # get the oracles
    oracles_ts_ors = dataset.get_oracles()
    oracle_ts = oracles_ts_ors[0]
    # denormalize
    oracle_ts["x_ts"] = dataset.denormalize_positions(oracle_ts["x_ts"])
    oracle_ts["x_d_ts"] = dataset.denormalize_velocities(oracle_ts["x_d_ts"])
    # add the joint offset
    oracle_ts["x_ts"] = q_off_ar + oracle_ts["x_ts"]

    # plot the joint angles
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    for i in range(q_ts.shape[1]):
        ax.plot(ts, q_ts[:, i], label=r"$q_{" + str(i) + "}$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint angles [rad]")
    ax.legend(ncol=2)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(outputs_dir / f"{experiment_id}_all_joint_angles.pdf")
    plt.show()

    # plot the control signal and the joint velocities
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    for i in range(u_ts.shape[1]):
        ax.plot(
            ts, u_ts[:, i], linestyle=":", linewidth=2.5, label=r"$u_{" + str(i) + "}$"
        )
    # reset the color cycle
    plt.gca().set_prop_cycle(None)
    for i in range(q_d_ts.shape[1]):
        ax.plot(ts, q_d_ts[:, i], linewidth=2.0, label=r"$\dot{q}_{" + str(i) + "}$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint velocities [rad/s]")
    ax.set_ylim(-10.0, 10.0)
    ax.legend(ncol=2)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(outputs_dir / f"{experiment_id}_joint_velocities.pdf")
    plt.show()

    # plot the joint angles in 3D
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    # plot the oracle
    ax.plot(
        oracle_ts["x_ts"][:, 0],
        oracle_ts["x_ts"][:, 1],
        oracle_ts["x_ts"][:, 2],
        linestyle=":",
        linewidth=2.5,
        label="Oracle",
    )
    ax.plot(q_ar_ts[:, 0], q_ar_ts[:, 1], q_ar_ts[:, 2], label="Right flipper")
    ax.plot(q_al_ts[:, 0], q_al_ts[:, 1], q_al_ts[:, 2], label="Left flipper")
    ax.set_xlabel(r"$q_1$ [rad]")
    ax.set_ylabel(r"$q_2$ [rad]")
    ax.set_zlabel(r"$q_3$ [rad]")
    ax.legend()
    plt.tight_layout()
    fig.savefig(outputs_dir / f"{experiment_id}_joint_angles_3d.pdf")
    plt.show()

    # visualize the phase synchronization


if __name__ == "__main__":
    main()
