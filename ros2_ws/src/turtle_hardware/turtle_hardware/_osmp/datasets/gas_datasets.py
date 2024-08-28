__all__ = ["LASA"]
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import scipy.io as sio
from scipy.signal import savgol_filter

from .base_dataset import BaseDataset


class LASA(BaseDataset):
    """'
    Loads LASA dataset
    NOTE: The data has been smoothed and normalized to stay within [-0.5, 0.5]
    """

    def __init__(self, data_name):
        super(LASA, self).__init__()
        [
            self.x,
            self.x_d,
            self.oracle_indices,
            self.dt,
            self.goal,
            self.scaling,
            self.translation,
        ] = self.load_data(data_name)

        self.n_dims = self.x.shape[1]
        self.n_pts = self.x.shape[0]

    def plot_data(self):
        fig = plt.figure()
        ax = fig.gca()
        x = (self.x - self.translation) / self.scaling

        ax.scatter(x[:, 0], x[:, 1], color="r")
        plt.show()

    def load_data(self, data_name):
        data_path = (
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "lasa_handwriting_dataset"
            / f"{data_name}.mat"
        ).resolve()

        downsample_rate = 4
        start_cut = 10
        end_cut = 5
        tol_cutting = 1.0

        data = sio.loadmat(data_path)
        dataset = data["demos"]

        num_demos = int(dataset.shape[1])
        d = dataset[0, 0]["pos"][0][0].shape[0]

        dt_old = dataset[0, 0]["t"][0][0][0, 1] - dataset[0, 0]["t"][0][0][0, 0]
        dt = round(dt_old * downsample_rate, 2)

        x = np.empty((d, 0))
        oracle_indices = [0]

        for i in range(num_demos):
            demo = dataset[0, i]["pos"][0][0][:, ::downsample_rate]
            num_pts = demo.shape[1]

            demo_smooth = np.zeros_like(demo)
            window_size = int(
                2 * (25.0 * num_pts / 150 // 2) + 1
            )  # just an arbitrary heuristic (can be changed)
            for j in range(d):
                demo_smooth[j, :] = savgol_filter(demo[j, :], window_size, 3)

            demo_smooth = demo_smooth[:, start_cut:-end_cut]
            demo_pos = demo_smooth
            demo_vel = np.diff(demo_smooth, axis=1) / dt
            demo_vel_norm = np.linalg.norm(demo_vel, axis=0)
            ind = np.where(demo_vel_norm > tol_cutting * 150.0 / num_pts)

            demo_pos = demo_pos[:, np.min(ind) : (np.max(ind) + 2)]
            tmp = demo_pos
            for j in range(d):
                tmp[j, :] = savgol_filter(tmp[j, :], window_size, 3)
            demo_pos = tmp

            demo_pos = demo_pos - demo_pos[:, -1].reshape(-1, 1)
            x = np.concatenate((x, demo_pos), axis=1)
            oracle_indices.append(x.shape[1])

        minx = np.min(x, axis=1)
        maxx = np.max(x, axis=1)

        scaling = 1.0 / (maxx - minx)
        translation = -minx / (maxx - minx) - 0.5

        x = x * scaling[:, None] + translation[:, None]
        x_d = np.empty((d, 0))

        for i in range(num_demos):
            demo = x[:, oracle_indices[i] : oracle_indices[i + 1]]
            demo_vel = np.diff(demo, axis=1) / dt
            demo_vel = np.concatenate((demo_vel, np.zeros((d, 1))), axis=1)

            x_d = np.concatenate((x_d, demo_vel), axis=1)

        x = x.T
        x_d = x_d.T
        goal = x[-1].reshape(1, -1)
        scaling = scaling.T
        translation = translation.T

        return [
            x.astype(np.float32),
            x_d.astype(np.float32),
            np.array(oracle_indices),
            dt,
            goal,
            scaling,
            translation,
        ]
