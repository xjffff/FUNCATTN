"""
Minimal layers needed for SimpleTransformer on Burgers 1D.
Extracted from galerkin-transformer/libs/layers.py
"""
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from functools import partial
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter


def default(value, d):
    return d if value is None else value


class Identity(nn.Module):
    def __init__(self, in_features=None, out_features=None, *args, **kwargs):
        super().__init__()
        if in_features is not None and out_features is not None:
            self.id = nn.Linear(in_features, out_features)
        else:
            self.id = nn.Identity()

    def forward(self, x, edge=None, grid=None):
        return self.id(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2**13):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(2**13) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, in_dim=256, dim_feedforward=1024, out_dim=None,
                 batch_norm=False, activation='relu', dropout=0.1):
        super().__init__()
        out_dim = default(out_dim, in_dim)
        self.lr1 = nn.Linear(in_dim, dim_feedforward)
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(dim_feedforward)
        self.lr2 = nn.Linear(dim_feedforward, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.activation(self.lr1(x))
        x = self.dropout(x)
        if self.batch_norm:
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.lr2(x)


class SpectralConv1d(nn.Module):
    """
    Modified from Zongyi Li's Spectral1dConv.
    Input/Output: (-1, n_grid, features)
    """
    def __init__(self, in_dim, out_dim, modes, n_grid=None,
                 dropout=0.1, return_freq=False, activation='silu', debug=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.modes = modes
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.fourier_weight = Parameter(torch.FloatTensor(in_dim, out_dim, modes, 2))
        xavier_normal_(self.fourier_weight, gain=1 / (in_dim * out_dim))
        self.dropout = nn.Dropout(dropout)
        self.return_freq = return_freq

    @staticmethod
    def complex_matmul_1d(a, b):
        op = partial(torch.einsum, "bix,iox->box")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1]),
        ], dim=-1)

    def forward(self, x):
        seq_len = x.size(1)
        res = self.linear(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x_ft = fft.rfft(x, n=seq_len, norm="ortho")
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        out_ft = self.complex_matmul_1d(x_ft[:, :, :self.modes], self.fourier_weight)
        pad_size = seq_len // 2 + 1 - self.modes
        out_ft = F.pad(out_ft, (0, 0, 0, pad_size), "constant", 0)
        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x = fft.irfft(out_ft, n=seq_len, norm="ortho")
        x = self.activation(x.permute(0, 2, 1) + res)
        if self.return_freq:
            return x, out_ft
        return x
