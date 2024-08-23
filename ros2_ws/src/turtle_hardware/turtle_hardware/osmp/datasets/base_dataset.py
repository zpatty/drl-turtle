__all__ = ["BaseDataset"]
import numpy as np
import torch
from typing import Dict, List


class BaseDataset:
    def denormalize_positions(self, x: np.ndarray) -> np.ndarray:
        return (x - self.translation) / self.scaling

    def denormalize_velocities(self, xd: np.ndarray) -> np.ndarray:
        return xd / self.scaling

    def get_oracles(self) -> List[Dict[str, np.ndarray]]:
        oracles_ts_ors = []
        for j in range(len(self.oracle_indices) - 1):
            if hasattr(self, "oracle_ids"):
                oracle_id = np.array(self.oracle_ids[self.oracle_indices[j]])
            else:
                oracle_id = np.array(j)
            ts = self.ts[self.oracle_indices[j] : self.oracle_indices[j + 1]]
            x_ts = self.x[self.oracle_indices[j] : self.oracle_indices[j + 1]]
            x_d_ts = self.x_d[self.oracle_indices[j] : self.oracle_indices[j + 1]]

            # add initial conditions and timings to the list
            x0 = x_ts[0]

            oracle_ts = dict(
                oracle_id=oracle_id,
                ts=ts,
                x0=x0,
                x_ts=x_ts,
                x_d_ts=x_d_ts,
            )
            oracles_ts_ors.append(oracle_ts)

        return oracles_ts_ors

    def get_oracles_as_tensors(
        self, requires_grad: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        oracles_ts_ors = self.get_oracles()

        # cast to tensors
        for oracle_ts in oracles_ts_ors:
            for key, value in oracle_ts.items():
                oracle_ts[key] = torch.from_numpy(value).float()

            # set requires_grad to True
            oracle_ts["x_ts"].requires_grad_(requires_grad)
            oracle_ts["x0"].requires_grad_(requires_grad)

        return oracles_ts_ors
