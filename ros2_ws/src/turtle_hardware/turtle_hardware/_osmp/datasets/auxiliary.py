__all__ = ["ConcatenatedDataset"]
import numpy as np
from typing import List

from .base_dataset import BaseDataset


class ConcatenatedDataset(BaseDataset):
    def __init__(self, datasets: List):
        super(ConcatenatedDataset, self).__init__()
        self.datasets = datasets

        self.n_dims = self.datasets[0].n_dims
        self.scaling = np.ones((self.n_dims,))
        self.translation = np.zeros((self.n_dims,))
        self.goal = np.zeros((self.n_dims,))

        oracle_indices, oracle_ids = [], []
        ts = []
        x, x_d = [], []

        oracle_id = 0
        total_num_samples = 0
        for i, dataset in enumerate(self.datasets):
            assert (
                dataset.n_dims == self.n_dims
            ), "All datasets must have the same number of dimensions."
            ts.append(dataset.ts)
            x.append(dataset.x)
            x_d.append(dataset.x_d)
            for i, oracle_idx in enumerate(dataset.oracle_indices[:-1]):
                oracle_indices.append(total_num_samples)
                num_samples = dataset.oracle_indices[i + 1] - oracle_idx
                total_num_samples += dataset.x.shape[0]
                oracle_ids.append(oracle_id * oracle_id * np.ones((num_samples,)))
                oracle_id += 1

        self.oracle_ids = np.concatenate(oracle_ids, axis=0)
        self.ts = np.concatenate(ts, axis=0)
        self.x = np.concatenate(x, axis=0)
        self.x_d = np.concatenate(x_d, axis=0)

        assert (
            total_num_samples == self.x.shape[0]
        ), "Total number of samples must match the concatenated dataset size."
        self.n_pts = total_num_samples

        oracle_indices.append(self.n_pts)
        self.oracle_indices = np.array(oracle_indices)
