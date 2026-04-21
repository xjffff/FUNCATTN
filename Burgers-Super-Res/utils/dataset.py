"""
BurgersDataset for zero-shot super-resolution.
Extracted from galerkin-transformer/libs/ft.py (uniform-grid path only).

Train: subsample=4  → 2048 points
Val:   subsample=1  → 8192 points
"""
import gc
import os
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class BurgersDataset(Dataset):
    N_FINE = 8192

    def __init__(self, subsample=4,
                 n_grid_fine=8192,
                 train_data=True,
                 train_portion=0.5,
                 valid_portion=100,
                 data_path=None,
                 return_downsample_grid=True,
                 random_state=1127802):
        self.subsample = subsample
        self.n_grid_fine = n_grid_fine
        self.n_grid = n_grid_fine // subsample
        self.h = 1 / n_grid_fine
        self.train_data = train_data
        self.train_portion = train_portion
        self.valid_portion = valid_portion
        self.data_path = data_path
        self.return_downsample_grid = return_downsample_grid
        self._set_seed(random_state)
        self._initialize()

    def __len__(self):
        return self.n_samples

    def _set_seed(self, s):
        os.environ['PYTHONHASHSEED'] = str(s)
        np.random.seed(s)
        torch.manual_seed(s)

    def _initialize(self):
        data = loadmat(self.data_path)
        x_data = data['a']
        y_data = data['u']
        del data
        gc.collect()

        train_len, valid_len = self._train_test_split(len(x_data))

        if self.train_data:
            x_data, y_data = x_data[:train_len], y_data[:train_len]
        else:
            x_data, y_data = x_data[-valid_len:], y_data[-valid_len:]

        self.n_samples = len(x_data)
        grid, grid_fine, nodes, targets = self._get_uniform_data(x_data, y_data)

        self.node_features = nodes[..., None] if nodes.ndim == 2 else nodes
        self.pos = grid[..., None]
        self.pos_fine = grid_fine[..., None]
        self.target = targets[..., None] if targets.ndim == 2 else targets

    def _get_uniform_data(self, x_data, y_data):
        targets = y_data
        targets_diff = self._central_diff(targets, self.h)

        nodes = x_data[:, ::self.subsample]
        targets = targets[:, ::self.subsample]
        targets_diff = targets_diff[:, ::self.subsample]

        targets = np.stack([targets, targets_diff], axis=2)
        grid = np.linspace(0, 1, self.n_grid)
        grid_fine = np.linspace(0, 1, self.n_grid_fine)

        return grid, grid_fine, nodes, targets

    @staticmethod
    def _central_diff(x, h):
        if x.ndim == 2:
            pad_0, pad_1 = x[:, -2], x[:, 1]
            x = np.c_[pad_0, x, pad_1]
            x_diff = (x[:, 2:] - x[:, :-2]) / 2
        return x_diff / h

    def _train_test_split(self, len_data):
        if self.train_portion <= 1:
            train_len = int(self.train_portion * len_data)
        elif 1 < self.train_portion <= len_data:
            train_len = int(self.train_portion)
        else:
            train_len = int(0.8 * len_data)

        if self.valid_portion <= 1:
            valid_len = int(self.valid_portion * len_data)
        elif 1 < self.valid_portion <= len_data:
            valid_len = int(self.valid_portion)
        else:
            valid_len = int(0.1 * len_data)
        return train_len, valid_len

    def __getitem__(self, index):
        if self.return_downsample_grid:
            pos_fine = self.pos[..., :1]  # use coarse grid for pos_fine too
        pos_fine = torch.from_numpy(self.pos_fine)
        pos = torch.from_numpy(self.pos[:, :1])
        node_features = torch.from_numpy(self.node_features[index])
        target = torch.from_numpy(self.target[index])
        return dict(
            node=node_features.float(),
            pos=pos.float(),
            grid=pos_fine.float(),
            edge=torch.tensor([1.0]),   # placeholder (not used by FuncAttnIrregular)
            target=target.float(),
        )
