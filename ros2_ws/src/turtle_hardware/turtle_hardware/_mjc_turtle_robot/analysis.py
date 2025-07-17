import matplotlib.pyplot as plt
import numpy as np
from os import PathLike
from typing import Dict, Optional


def compute_cost_of_transport(
    ts: np.ndarray,
    base_pos_ts: np.ndarray,
    q_d_ts: np.ndarray,
    tau_ts: np.ndarray,
    m_turtle: float = 6.0,
    g: float = 9.81,
    verbose: bool = False,
) -> float:
    """Compute the cost of transport for a trajectory"""
    # compute the total distance traveled (along y-axis)
    dist = base_pos_ts[-1, 1] - base_pos_ts[0, 1]
    # time elapsed
    duration = ts[-1] - ts[0]
    # average velocity [m/s]
    vel = dist / duration
    # compute the mean power applied by the joint motors
    # important: this assumes that the last 10 DoFs are from the rear legs and front arm joints, respectively
    power_in = np.mean(np.sum(q_d_ts[:, -10:] * tau_ts[:, -10:], axis=-1), axis=0)
    # compute the cost of transport
    cot = power_in / (m_turtle * g * vel)

    if verbose:
        print(
            f"Total distance travelled: {dist} m, "
            f"total duration: {duration} s, "
            f"average velocity: {vel} m/s, "
            f"mean power in: {power_in} W, cot: {cot}"
        )

    return cot


def generate_control_plots(
    sim_ts: Dict[str, np.ndarray],
    dpi: int = 200,
    output_dir: Optional[PathLike] = None,
    show_plots: bool = False,
):
    # Set up the plotting parameters
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Romand"],
        }
    )
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot the base position
    plt.figure(dpi=dpi)
    plt.plot(sim_ts["ts"], sim_ts["base_pos"][..., 0], label=r"$x$")
    plt.plot(sim_ts["ts"], sim_ts["base_pos"][..., 1], label=r"$y$")
    plt.plot(sim_ts["ts"], sim_ts["base_pos"][..., 2], label=r"$z$")
    plt.grid(True)
    plt.box(True)
    plt.xlabel("Time [s]")
    plt.ylabel(r"Position [m]")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / "base_pos.pdf")
        plt.savefig(output_dir / "base_pos.png")
    if show_plots:
        plt.show()

    # Plot the base velocity
    plt.figure(dpi=dpi)
    plt.plot(sim_ts["ts"], sim_ts["base_vel"][..., 0], label=r"$\dot{x}$")
    plt.plot(sim_ts["ts"], sim_ts["base_vel"][..., 1], label=r"$\dot{y}$")
    plt.plot(sim_ts["ts"], sim_ts["base_vel"][..., 2], label=r"$\dot{z}$")
    plt.grid(True)
    plt.box(True)
    plt.xlabel("Time [s]")
    plt.ylabel(r"Velocity [m/s]")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / "base_vel.pdf")
        plt.savefig(output_dir / "base_vel.png")
    if show_plots:
        plt.show()

    # Plot the base acceleration
    plt.figure(dpi=dpi)
    plt.plot(sim_ts["ts"], sim_ts["base_acc"][..., 0], label=r"$\ddot{x}$")
    plt.plot(sim_ts["ts"], sim_ts["base_acc"][..., 1], label=r"$\ddot{y}$")
    plt.plot(sim_ts["ts"], sim_ts["base_acc"][..., 2], label=r"$\ddot{z}$")
    plt.grid(True)
    plt.box(True)
    plt.xlabel("Time [s]")
    plt.ylabel(r"Acceleration [m/$s^2$]")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / "base_acc.pdf")
        plt.savefig(output_dir / "base_acc.png")
    if show_plots:
        plt.show()

    # Plot the base force
    plt.figure(dpi=dpi)
    plt.plot(sim_ts["ts"], sim_ts["base_force"][..., 0], label=r"$f_x$")
    plt.plot(sim_ts["ts"], sim_ts["base_force"][..., 1], label=r"$f_y$")
    plt.plot(sim_ts["ts"], sim_ts["base_force"][..., 2], label=r"$f_z$")
    plt.grid(True)
    plt.box(True)
    plt.xlabel("Time [s]")
    plt.ylabel(r"Force [N]")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / "base_force.pdf")
        plt.savefig(output_dir / "base_force.png")
    if show_plots:
        plt.show()

    print("Average base force:", np.mean(sim_ts["base_force"], axis=0))

    # Plot the base torque
    plt.figure(dpi=dpi)
    plt.plot(sim_ts["ts"], sim_ts["base_torque"][..., 0], label=r"$\tau_x$")
    plt.plot(sim_ts["ts"], sim_ts["base_torque"][..., 1], label=r"$\tau_y$")
    plt.plot(sim_ts["ts"], sim_ts["base_torque"][..., 2], label=r"$\tau_z$")
    plt.grid(True)
    plt.box(True)
    plt.xlabel("Time [s]")
    plt.ylabel(r"Torque [Nm]")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / "base_torque.pdf")
        plt.savefig(output_dir / "base_torque.png")
    if show_plots:
        plt.show()

    # plot the joint positions
    plt.figure(dpi=dpi)
    for i in range(sim_ts["q"].shape[1]):
        plt.plot(sim_ts["ts"], sim_ts["q"][:, i], label=r"$q_{" + str(i) + "}$")
    plt.grid(True)
    plt.box(True)
    plt.xlabel("Time [s]")
    plt.ylabel(r"Joint position")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / "q.pdf")
        plt.savefig(output_dir / "q.png")
    if show_plots:
        plt.show()

    # plot the joint velocities
    plt.figure(dpi=dpi)
    for i in range(sim_ts["qvel"].shape[1]):
        plt.plot(
            sim_ts["ts"], sim_ts["qvel"][:, i], label=r"$\dot{q}_{" + str(i) + "}$"
        )
    plt.grid(True)
    plt.box(True)
    plt.xlabel("Time [s]")
    plt.ylabel(r"Joint velocity [rad/s]")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / "qvel.pdf")
        plt.savefig(output_dir / "qvel.png")
    if show_plots:
        plt.show()

    # Plot the control input
    plt.figure(dpi=dpi)
    for i in range(sim_ts["ctrl"].shape[1]):
        plt.plot(sim_ts["ts"], sim_ts["ctrl"][:, i], label=r"$u_{" + str(i) + "}$")
    plt.grid(True)
    plt.box(True)
    plt.xlabel("Time [s]")
    plt.ylabel(r"Control input [m/s]")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / "ctrl.pdf")
        plt.savefig(output_dir / "ctrl.png")
    if show_plots:
        plt.show()

    # Plot the applied control torque
    plt.figure(dpi=dpi)
    for i in range(sim_ts["actuator_force"].shape[1]):
        plt.plot(
            sim_ts["ts"],
            sim_ts["actuator_force"][:, i],
            label=r"$\tau_{" + str(i) + "}$",
        )
    plt.grid(True)
    plt.box(True)
    plt.xlabel("Time [s]")
    plt.ylabel(r"Control torque [Nm]")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / "actuator_force.pdf")
        plt.savefig(output_dir / "actuator_force.png")
    if show_plots:
        plt.show()

    # Plot the fluid forces
    plt.figure(dpi=dpi)
    for i in range(sim_ts["qfrc_fluid"].shape[1]):
        plt.plot(
            sim_ts["ts"], sim_ts["qfrc_fluid"][:, i], label=r"$f_{" + str(i) + "}$"
        )
    plt.grid(True)
    plt.box(True)
    plt.xlabel("Time [s]")
    plt.ylabel(r"Fluid forces")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / "fluid_forces.pdf")
        plt.savefig(output_dir / "fluid_forces.png")
    if show_plots:
        plt.show()
