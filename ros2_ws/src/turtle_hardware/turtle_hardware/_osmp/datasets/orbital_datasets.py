__all__ = [
    "Ellipse",
    "ImageContour",
    "GreenSeaTurtleSwimmingKinematics",
    "CorneliaTurtleRobotJointSpaceTrajectory",
]
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
import scipy.io as sio
from scipy.signal import savgol_filter
import os
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, Tuple, Union

from .base_dataset import BaseDataset
from osmp.plot_utils import add_arrow_to_line2d


class Ellipse(BaseDataset):
    """'
    Loads a perfect ellipse dataset
    NOTE: The data can be optionally normalized to stay within [-0.5, 0.5]
    """

    def __init__(self, *args, **kwargs):
        super(Ellipse, self).__init__()
        [
            self.ts,
            self.x,
            self.x_d,
            self.oracle_indices,
            self.oracle_ids,
            self.dt,
            self.scaling,
            self.translation,
        ] = self.load_data(*args, **kwargs)

        self.goal = self.x[-2:-1]

        self.n_dims = self.x.shape[1]
        self.n_pts = self.x.shape[0]

    def plot_data(self):
        # denormalize
        x = (self.x - self.translation) / self.scaling
        x_d = self.x_d / self.scaling

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig = plt.figure()
        ax = fig.gca()
        for i in range(len(self.oracle_indices) - 1):
            ax.scatter(
                x[self.oracle_indices[i] : self.oracle_indices[i + 1], 0],
                x[self.oracle_indices[i] : self.oracle_indices[i + 1], 1],
                color=color_cycle[i],
                label="Oracle " + str(i + 1),
            )
        ax.set_xlabel("$x_1$ [m]")
        ax.set_ylabel("$x_2$ [m]")
        ax.grid(True)
        ax.set_aspect("equal")
        ax.legend()
        plt.show()

        fig = plt.figure()
        ax = fig.gca()
        for i in range(len(self.oracle_indices) - 1):
            ax.plot(
                self.ts[self.oracle_indices[i] : self.oracle_indices[i + 1]],
                x[self.oracle_indices[i] : self.oracle_indices[i + 1], 0],
                linestyle=":",
                color=color_cycle[i],
                label=r"Oracle " + str(i + 1) + " $x_1$ [m]",
            )
            ax.plot(
                self.ts[self.oracle_indices[i] : self.oracle_indices[i + 1]],
                x[self.oracle_indices[i] : self.oracle_indices[i + 1], 1],
                linestyle="--",
                color=color_cycle[i],
                label=r"Oracle " + str(i + 1) + " $x_2$ [m]",
            )
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")
        ax.grid(True)
        plt.show()

        fig = plt.figure()
        ax = fig.gca()
        for i in range(len(self.oracle_indices) - 1):
            ax.plot(
                self.ts[self.oracle_indices[i] : self.oracle_indices[i + 1]],
                x_d[self.oracle_indices[i] : self.oracle_indices[i + 1], 0],
                linestyle=":",
                color=color_cycle[i],
                label=r"Oracle " + str(i + 1) + r" $\dot{x}_1$ [m]",
            )
            ax.plot(
                self.ts[self.oracle_indices[i] : self.oracle_indices[i + 1]],
                x_d[self.oracle_indices[i] : self.oracle_indices[i + 1], 1],
                linestyle="--",
                color=color_cycle[i],
                label=r"Oracle " + str(i + 1) + r" $\dot{x}_2$ [m]",
            )
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Velocity [m/s]")
        ax.grid(True)
        plt.show()

    def load_data(
        self,
        num_dims: int = 2,
        a: Union[float, np.ndarray] = 1.0,
        b: Union[float, np.ndarray] = 1.0,
        max_polar_angle: float = 2 * np.pi,
        normalize: bool = False,
    ):
        assert num_dims >= 2, "The number of dimensions must be equal or greater than 2"

        if type(a) in [int, float]:
            num_oracles = 1
            a = np.array([a])
            b = np.array([b])
        else:
            num_oracles = a.shape[0]
            assert b.shape[0] == num_oracles, "Shapes of a and b must be equal"

        dt = 2e-2
        omega = 0.05 * 2 * np.pi  # 20 seconds for one orbit

        duration = max_polar_angle / omega
        tsi = np.arange(0.0, duration, dt)

        oracle_indices = [0]
        oracle_ids = []
        ts = []
        x = []
        x_d = []
        for i, (ai, bi) in enumerate(zip(a, b)):
            xi = np.stack(
                [
                    ai * np.cos(omega * tsi),
                    bi * np.sin(omega * tsi),
                ],
                axis=1,
            )
            x_di = np.stack(
                [
                    -ai * omega * np.sin(omega * tsi),
                    bi * omega * np.cos(omega * tsi),
                ],
                axis=1,
            )

            # add additional dimensions
            if num_dims > 2:
                xi = np.concatenate(
                    [
                        xi,
                        np.repeat(
                            np.sin(omega * tsi)[:, None], repeats=num_dims - 2, axis=1
                        ),
                    ],
                    axis=1,
                )
                x_di = np.concatenate(
                    [
                        x_di,
                        np.repeat(
                            omega * np.cos(omega * tsi)[:, None],
                            repeats=num_dims - 2,
                            axis=1,
                        ),
                    ],
                    axis=1,
                )

            oracle_indices.append(oracle_indices[-1] + xi.shape[0])
            oracle_ids.append(i * np.ones_like(tsi))
            ts.append(tsi)
            x.append(xi)
            x_d.append(x_di)

        oracle_indices = np.array(oracle_indices)
        oracle_ids = np.concatenate(oracle_ids, axis=0)
        ts = np.concatenate(ts, axis=0)
        x = np.concatenate(x, axis=0)
        x_d = np.concatenate(x_d, axis=0)

        if normalize:
            minx = np.min(x, axis=0)
            maxx = np.max(x, axis=0)

            scaling = 1.0 / (maxx - minx)
            translation = -minx / (maxx - minx) - 0.5
        else:
            scaling = np.ones((num_dims,))
            translation = np.zeros((num_dims,))

        x = x * scaling + translation
        x_d = x_d * scaling

        return [
            ts.astype(np.float32),
            x.astype(np.float32),
            x_d.astype(np.float32),
            oracle_indices,
            oracle_ids.astype(np.float32),
            dt,
            scaling,
            translation,
        ]


class ImageContour(BaseDataset):
    """'
    Loads a contour from an image
    NOTE: The data can be optionally normalized to stay within [-0.5, 0.5]
    """

    def __init__(self, *args, **kwargs):
        super(ImageContour, self).__init__()
        [
            self.ts,
            self.x,
            self.x_d,
            self.dt,
            self.scaling,
            self.translation,
        ] = self.load_data(*args, **kwargs)

        # for compatibility with LASA dataset
        self.oracle_indices = np.array([0, self.x.shape[0]])
        self.goal = self.x[-2:-1]

        self.n_dims = self.x.shape[1]
        self.n_pts = self.x.shape[0]

    def plot_data(self):
        # denormalize
        x = (self.x - self.translation) / self.scaling
        x_d = self.x_d / self.scaling
        x_d_norm = np.linalg.norm(x_d, axis=1)

        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(x[:, 0], x[:, 1], color="k")
        ax.set_xlabel(r"$x_1$ [m]")
        ax.set_ylabel(r"$x_2$ [m]")
        ax.grid(True)
        ax.set_aspect("equal")
        plt.show()

        # plot the position in 3D space with the velocity magnitude as color
        fig = plt.figure()
        ax = fig.gca()
        sc = ax.scatter(x[:, 0], x[:, 1], c=x_d_norm, cmap="coolwarm")
        fig.colorbar(sc, label=r"$|\dot{x}|_2$")
        ax.set_xlabel(r"$x_1$ [m]")
        ax.set_ylabel(r"$x_2$ [m]")
        ax.grid(True)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(self.ts, x[:, 0], label=r"$x_1$ [m]")
        ax.plot(self.ts, x[:, 1], label=r"$x_2$ [m]")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")
        ax.grid(True)
        plt.show()

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(self.ts, x_d[:, 0], label=r"$\dot{x}_1$ [m]")
        ax.plot(self.ts, x_d[:, 1], label=r"$\dot{x}_2$ [m]")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Velocity [m/s]")
        ax.grid(True)
        plt.show()

    def load_data(
        self,
        data_name: str,
        num_dims: int = 2,
        image_path: PathLike = None,
        trajectory_duration: float = 20.0,
        max_radius: np.ndarray = np.array(0.5),
        verbose: bool = False,
        normalize: bool = False,
    ):
        assert num_dims >= 2, "The number of dimensions must be equal or greater than 2"

        if image_path is None:
            image_path = (
                Path(__file__).parent.parent.parent.parent
                / "assets"
                / str(data_name.replace("-", "_") + ".png")
            ).resolve()

        img = cv2.imread(str(image_path))

        # set the threshold for the binary image
        threshold = 140
        threshold_mode = cv2.THRESH_BINARY_INV

        if data_name == "square":
            pass
        elif data_name == "star":
            # perform cropping of the width to the height
            w = img.shape[0]
            center = img.shape
            x = center[1] / 2 - w / 2
            img = img[:, int(x) : int(x + w)]

            # set the threshold for the binary image
            threshold = 128
            threshold_mode = cv2.THRESH_BINARY
        elif data_name == "tud-flame":
            # set the threshold for the binary image
            pass
        elif data_name == "mit-csail":
            # set the threshold for the binary image
            pass
        elif data_name == "manta-ray":
            # https://www.flaticon.com/free-icon/manta-ray-shape_47440
            # set the threshold for the binary image
            pass
        elif data_name == "bat":
            # https://www.freepik.com/icon/flying-bat_12311
            pass
        elif data_name == "dolphin":
            pass
        elif data_name == "doge":
            # https://www.shutterstock.com/image-vector/meme-dog-dogecoin-doge-cryptocurrency-260nw-1971113219.jpg
            threshold = 240
        else:
            raise ValueError(f"Unknown image type: {data_name}")

        if verbose:
            print("Image size: ", img.shape)

            # convert the input image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # apply thresholding to convert grayscale to binary image
        ret, img_thresh = cv2.threshold(img_gray, threshold, 255, threshold_mode)

        if verbose:
            # Display the Grayscale Image
            cv2.imshow("Gray Image", img_gray)
            cv2.waitKey(0)

            # Display the Binary Image
            cv2.imshow("Binary Image", img_thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(
            image=img_thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
        )
        contour = contours[0]
        if verbose:
            print("contour shape", contour.shape)

        if verbose:
            # draw contours on the original image
            img_copy = img.copy()
            cv2.drawContours(
                image=img_copy,
                contours=contours,
                contourIdx=-1,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            # see the results
            cv2.imshow("None approximation", img_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # find centroid of the contour
        M = cv2.moments(contour)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid = np.array((cX, cY))
        if verbose:
            # put text and highlight the center
            img_copy = img.copy()
            cv2.circle(img_copy, (cX, cY), 5, (0, 0, 0), -1)
            cv2.putText(
                img_copy,
                "centroid",
                (cX - 25, cY - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

            # display the image
            cv2.imshow("Image with centroid of contour", img_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # subtract the centroid from the contour
        contour = contour - centroid
        # normalize to range [-1, 1]
        pee_sps_norm = (contour / np.max(np.abs(contour)))[:, 0, :]
        # flip the y-axis
        pee_sps_norm[:, 1] = -pee_sps_norm[:, 1]
        if verbose:
            plt.figure(num="Normalized contour")
            plt.plot(pee_sps_norm[:, 0], pee_sps_norm[:, 1])
            plt.axis("equal")
            plt.grid(True)
            plt.box(True)
            plt.show()

        # reduce the number of points
        if data_name == "star":
            sample_step = 2  # only take every 2nd point
            pee_sps_norm = pee_sps_norm[::sample_step, :]
        elif data_name == "tud-flame":
            sample_step = 2  # only take every 2nd point
            pee_sps_norm = pee_sps_norm[::sample_step, :]
        elif data_name == "mit-csail":
            sample_step = 2  # only take every 2nd point
            pee_sps_norm = pee_sps_norm[::sample_step, :]
        elif data_name == "bat":
            sample_step = 1  # take every sample
            pee_sps_norm = pee_sps_norm[::sample_step, :]

        # rescale the contour to the maximum radius
        pee_sps = pee_sps_norm * max_radius

        # filter the trajectory using Savitzky-Golay filter
        window_size = 25
        order = 3
        pee_sps_smoothed = savgol_filter(pee_sps, window_size, order, axis=0)

        if verbose:
            plt.figure(num="Final trajectory")
            ax = plt.gca()
            arrow_locs = np.linspace(0.0, 1.0, 50)
            (traj_line,) = plt.plot(
                pee_sps[:, 0],
                pee_sps[:, 1],
                label="Trajectory",
            )
            add_arrow_to_line2d(ax, traj_line, arrow_locs=arrow_locs, arrowstyle="->")
            # smoothed trajectory
            (traj_smoothed_line,) = plt.plot(
                pee_sps_smoothed[:, 0],
                pee_sps_smoothed[:, 1],
                label="Smoothed trajectory",
            )
            add_arrow_to_line2d(
                ax, traj_smoothed_line, arrow_locs=arrow_locs, arrowstyle="->"
            )
            # plot the start point
            plt.scatter(pee_sps[0, 0], pee_sps[0, 1], color="r", label="Start")
            plt.axis("equal")
            plt.xlabel(r"$x_1$ [m]")
            plt.ylabel(r"$x_2$ [m]")
            plt.grid(True)
            plt.box(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        num_samples = pee_sps.shape[0] - 1
        ts = np.linspace(0.0, trajectory_duration, num_samples)
        dt = trajectory_duration / num_samples

        # create the dataset
        x = pee_sps_smoothed[:-1]
        x_d = np.diff(pee_sps_smoothed, axis=0) / dt

        # add additional dimensions
        if num_dims > 2:
            x = np.concatenate(
                [
                    x,
                    np.zeros((num_samples, num_dims - 2)),
                ],
                axis=1,
            )
            x_d = np.concatenate(
                [
                    x_d,
                    np.zeros((num_samples, num_dims - 2)),
                ],
                axis=1,
            )

        if normalize:
            minx = np.min(x, axis=0)
            maxx = np.max(x, axis=0)

            scaling = 1.0 / (maxx - minx)
            translation = -minx / (maxx - minx) - 0.5
        else:
            scaling = np.ones((num_dims,))
            translation = np.zeros((num_dims,))

        x = x * scaling + translation
        x_d = x_d * scaling

        return [
            ts.astype(np.float32),
            x.astype(np.float32),
            x_d.astype(np.float32),
            dt,
            scaling,
            translation,
        ]


class GreenSeaTurtleSwimmingKinematics(BaseDataset):
    """'
    Based on the green sea turtle swimming kinematics of
        van der Geest, N., Garcia, L., Nates, R., & Godoy, D. A. (2022).
        New insight into the swimming kinematics of wild Green sea turtles (Chelonia mydas).
        Scientific Reports, 12(1), 18151.

    NOTE: The data can be optionally normalized to stay within [-0.5, 0.5]
    """

    def __init__(self, *args, **kwargs):
        super(GreenSeaTurtleSwimmingKinematics, self).__init__()
        [
            self.ts,
            self.x,
            self.x_d,
            self.dt,
            self.scaling,
            self.translation,
        ] = self.load_data(*args, **kwargs)

        # for compatibility with LASA dataset
        self.oracle_indices = np.array([0, self.x.shape[0]])
        self.goal = self.x[-2:-1]

        self.n_dims = self.x.shape[1]
        self.n_pts = self.x.shape[0]

    def plot_data(self):
        # denormalize
        x = (self.x - self.translation) / self.scaling
        x_d = self.x_d / self.scaling
        x_d_norm = np.linalg.norm(x_d, axis=1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], color="k")
        ax.set_xlabel(r"$x_1$ [m]")
        ax.set_ylabel(r"$x_2$ [m]")
        ax.set_zlabel(r"$x_3$ [m]")
        ax.grid(True)
        ax.set_aspect("equal")
        plt.show()

        # plot the position in 3D space with the velocity magnitude as color
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=x_d_norm, cmap="coolwarm")
        fig.colorbar(sc, shrink=0.7, label=r"$|\dot{x}|_2$")
        ax.set_xlabel(r"$x_1$ [m]")
        ax.set_ylabel(r"$x_2$ [m]")
        ax.set_zlabel(r"$x_3$ [m]")
        ax.grid(True)
        ax.set_aspect("equal")
        # plt.tight_layout()
        plt.show()

        fig = plt.figure()
        ax = fig.gca()
        for i in range(x.shape[-1]):
            ax.plot(self.ts, x[:, i], label=r"$x_{" + str(i) + "}$")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")
        ax.grid(True)
        plt.show()

        fig = plt.figure()
        ax = fig.gca()
        for i in range(x_d.shape[-1]):
            ax.plot(self.ts, x_d[:, i], label=r"$\dot{x}_{" + str(i) + "}$")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Velocity [m/s]")
        ax.grid(True)
        plt.show()

    def load_data(
        self,
        num_samples: int = 1000,
        sf: float = 1.0,
        sw: float = 1.0,
        normalize: bool = False,
        include_twist_trajectory: bool = False,
    ):
        """
        Load the green sea turtle swimming kinematics dataset
        Arguments:
            num_samples (int): number of samples
            sf (float): scaling factor for different turtle sizes
            sw (float): scaling factor for different swimming speeds
            normalize (bool): whether to normalize the data
            include_twist_trajectory (bool): whether to include the twist angle in the dataset
        """
        from mjc_turtle_robot.biological_oracles import (
            green_sea_turtle_task_space_trajectory_factory,
        )

        x_fn, x_d_fn, x_dd_fn, th_fn, th_d_fn, th_dd_fn = (
            green_sea_turtle_task_space_trajectory_factory(s=1, sf=sf, sw=sw)
        )

        trajectory_duration = 4.2
        ts = np.linspace(0.0, trajectory_duration, num_samples)
        dt = trajectory_duration / num_samples

        # evaluate the trajectory
        x = np.stack([x_fn(t) for t in ts], axis=0)
        x_d = np.stack([x_d_fn(t) for t in ts], axis=0)
        x_dd = np.stack([x_dd_fn(t) for t in ts], axis=0)

        if include_twist_trajectory:
            th = np.stack([th_fn(t) for t in ts], axis=0)
            th_d = np.stack([th_d_fn(t) for t in ts], axis=0)
            th_dd = np.stack([th_dd_fn(t) for t in ts], axis=0)
            x = np.concatenate([x, th[:, None]], axis=-1)
            x_d = np.concatenate([x_d, th_d[:, None]], axis=-1)
            x_dd = np.concatenate([x_dd, th_dd[:, None]], axis=-1)

        minx = np.min(x, axis=0)
        maxx = np.max(x, axis=0)

        if normalize:
            scaling = 1.0 / (maxx - minx)
            translation = -minx / (maxx - minx) - 0.5
        else:
            scaling = np.ones((x.shape[-1],))
            translation = -(minx + maxx) / 2

        x = translation + x * scaling
        x_d = x_d * scaling

        return [
            ts.astype(np.float32),
            x.astype(np.float32),
            x_d.astype(np.float32),
            dt,
            scaling,
            translation,
        ]


class CorneliaTurtleRobotJointSpaceTrajectory(BaseDataset):
    """'
    Based on the joint-space trajectory of the Cornelia turtle robot
        van der Geest, N., Garcia, L., Borret, F., Nates, R., & Gonzalez, A. (2023).
        Soft-robotic green sea turtle (Chelonia mydas) developed to replace animal experimentation provides new insight
        into their propulsive strategies. Scientific Reports, 13(1), 11983.

    NOTE: The data can be optionally normalized to stay within [-0.5, 0.5]
    """

    def __init__(self, *args, **kwargs):
        super(CorneliaTurtleRobotJointSpaceTrajectory, self).__init__()
        [
            self.ts,
            self.x,
            self.x_d,
            self.dt,
            self.scaling,
            self.translation,
        ] = self.load_data(*args, **kwargs)

        # for compatibility with LASA dataset
        self.oracle_indices = np.array([0, self.x.shape[0]])
        self.goal = self.x[-2:-1]

        self.n_dims = self.x.shape[1]
        self.n_pts = self.x.shape[0]

    def plot_data(self):
        # denormalize
        x = (self.x - self.translation) / self.scaling
        x_d = self.x_d / self.scaling
        x_d_norm = np.linalg.norm(x_d, axis=1)

        # plot the joint angles in 3D space with the velocity magnitude as color
        fig = plt.figure()
        if self.n_dims > 2:
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=x_d_norm, cmap="coolwarm")
        else:
            ax = fig.add_subplot(111)
            sc = ax.scatter(x[:, 0], x[:, 1], c=x_d_norm, cmap="coolwarm")
        fig.colorbar(sc, shrink=0.7, label=r"$|\dot{x}|_2$ [rad/s]")
        ax.set_xlabel(r"$q_1$ [rad]")
        ax.set_ylabel(r"$q_2$ [rad]")
        if self.n_dims > 2:
            ax.set_zlabel(r"$q_3$ [rad]")
        ax.grid(True)
        ax.set_aspect("equal")
        plt.show()

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(self.ts, x[:, 0], label=r"$q_1$ [rad]")
        ax.plot(self.ts, x[:, 1], label=r"$q_2$ [rad]")
        if self.n_dims > 2:
            ax.plot(self.ts, x[:, 2], label=r"$q_3$ [rad]")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Joint angles [rad]")
        ax.grid(True)
        plt.show()

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(self.ts, x_d[:, 0], label=r"$\dot{q}_1$ [rad/s]")
        ax.plot(self.ts, x_d[:, 1], label=r"$\dot{q}_2$ [rad/s]")
        if self.n_dims > 2:
            ax.plot(self.ts, x_d[:, 2], label=r"$\dot{q}_3$ [rad/s]")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Joint velocity [rad/s]")
        ax.grid(True)
        plt.show()

    def load_data(
        self,
        num_samples: int = 1000,
        sw: float = 1.0,
        normalize: bool = False,
        smooth: bool = True,
        verbose: bool = False,
    ):
        """
        Load the green sea turtle swimming kinematics dataset
        Arguments:
            num_samples (int): number of samples
            sf (float): scaling factor for different turtle sizes
            sw (float): scaling factor for different swimming speeds
            normalize (bool): whether to normalize the data
            smooth (bool): whether to smooth the twist angle trajectory
        """
        from mjc_turtle_robot.biological_oracles import (
            cornelia_turtle_robot_joint_space_trajectory_factory,
        )

        q_fn, q_d_fn, q_dd_fn = cornelia_turtle_robot_joint_space_trajectory_factory(
            s=1, sw=sw
        )

        trajectory_duration = 4.3
        dt = trajectory_duration / num_samples
        ts = np.linspace(0.0, trajectory_duration, num_samples)

        # evaluate the trajectory
        x = np.stack([q_fn(t) for t in ts], axis=0)
        x_d = np.stack([q_d_fn(t) for t in ts], axis=0)
        x_dd = np.stack([q_dd_fn(t) for t in ts], axis=0)

        if smooth:
            ts_smoothing = np.arange(-100 * dt, trajectory_duration + 100 * dt, step=dt)
            ts_smoothing_normalized = ts_smoothing % trajectory_duration
            x_to_smooth = np.stack([q_fn(t) for t in ts_smoothing_normalized], axis=0)
            x_d_to_smooth = np.stack(
                [q_d_fn(t) for t in ts_smoothing_normalized], axis=0
            )
            x_dd_to_smooth = np.stack(
                [q_dd_fn(t) for t in ts_smoothing_normalized], axis=0
            )

            q3_unsmoothed = x_to_smooth[:, 2].copy()
            q3_d_unsmoothed = x_d_to_smooth[:, 2].copy()

            # fit a trajectory over the twist angle
            c_twist = Polynomial.fit(ts_smoothing, q3_unsmoothed, deg=30)

            # evaluate the fitted twist angle
            q3_fitted = c_twist(ts)
            # evaluate the velocity
            q3_d_fitted = c_twist.deriv(1)(ts)
            # evaluate the acceleration
            q3_dd_fitted = c_twist.deriv(2)(ts)

            if verbose:
                # twist angle vs. time
                plt.figure(dpi=200)
                # plot the unsmoothed twist angle
                plt.plot(ts_smoothing, q3_unsmoothed, label="Unsmoothed")
                # plot the fitted twist angle
                plt.plot(ts, q3_fitted, label="Fitted")
                plt.xlabel("Time [s]")
                plt.ylabel("Twist angle $q_3$ [rad]")
                plt.legend()
                plt.grid(True)
                plt.show()

                # twist angle velocity vs. time
                plt.figure(dpi=200)
                # plot the unsmoothed twist angle velocity
                plt.plot(ts_smoothing, q3_d_unsmoothed, label="Unsmoothed")
                # plot the fitted twist angle velocity
                plt.plot(ts, q3_d_fitted, label="Fitted")
                plt.xlabel("Time [s]")
                plt.ylabel(r"Twist angular velocity $\dot{q}_3$ [rad/s]")
                plt.legend()
                plt.grid(True)
                plt.show()

            x[:, 2] = q3_fitted
            x_d[:, 2] = q3_d_fitted
            x_dd[:, 2] = q3_dd_fitted

        minx = np.min(x, axis=0)
        maxx = np.max(x, axis=0)

        if normalize:
            scaling = 1.0 / (maxx - minx)
            translation = -minx / (maxx - minx) - 0.5
        else:
            scaling = np.ones((x.shape[-1],))
            translation = -(minx + maxx) / 2

        x = translation + x * scaling
        x_d = x_d * scaling

        return [
            ts.astype(np.float32),
            x.astype(np.float32),
            x_d.astype(np.float32),
            dt,
            scaling,
            translation,
        ]
