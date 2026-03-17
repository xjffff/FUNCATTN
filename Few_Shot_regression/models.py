from __future__ import annotations
from typing import Dict, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1])]
            if i < len(sizes) - 2:
                layers += [nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Intention

class IntentionHead(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, latent_dim=1000, ridge=1e-3,
                 shared_encoder=True, init_param=False):
        super().__init__()
        self.ridge = ridge

        def enc_block(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, latent_dim), nn.ReLU(),
                nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                nn.Linear(latent_dim, out_d),
            )

        if shared_encoder:
            self.enc_v = enc_block(x_dim, latent_dim)
            self.enc_k = self.enc_v
            self.enc_q = self.enc_v
        else:
            self.enc_k = enc_block(x_dim, latent_dim)
            self.enc_q = enc_block(x_dim, latent_dim)

        self.y_dim = y_dim

        if init_param:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.0, 0.01)
                    nn.init.zeros_(m.bias)

    def forward(self, xc, yc, xq):
        B, N, _ = xc.shape
        K = self.enc_k(xc)
        Q = self.enc_k(xq)
        V = yc

        Kt = K.transpose(1, 2)
        KKT = torch.bmm(K, Kt)
        I_N = torch.eye(N, device=K.device).unsqueeze(0).expand(B, N, N)
        reg = KKT + (self.ridge + 1e-5) * I_N
        alpha = torch.linalg.solve(reg, V)
        QKt = torch.bmm(Q, Kt)
        y_hat = torch.bmm(QKt, alpha)
        return y_hat


# Attention: standard multi-head cross-attention

class AttentionHead(nn.Module):
    def __init__(self, d_model=128, num_heads=8, init_param=False):
        super().__init__()
        self.enc_k = MLP([1, d_model, d_model, d_model])
        self.enc_v = MLP([1, d_model, d_model, d_model])
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.dec = MLP([d_model, d_model, 1])

        if init_param:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.0, 0.02)
                    nn.init.zeros_(m.bias)

    def forward(self, xc, yc, xq):
        Kemb = self.enc_k(xc)
        Qemb = self.enc_k(xq)
        Vemb = self.enc_v(yc)
        H, _ = self.attn(Qemb, Kemb, Vemb)
        return self.dec(H)


# Linear Attention

class LinearAttentionHead(nn.Module):
    def __init__(self, d_model=128, num_heads=8, init_param=False):
        super().__init__()
        self.enc_k = MLP([1, d_model, d_model, d_model])
        self.enc_v = MLP([1, d_model, d_model, d_model])
        self.dec = MLP([d_model, d_model, d_model, 1])

        if init_param:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.0, 0.01)
                    nn.init.zeros_(m.bias)

    def forward(self, xc, yc, xq):
        Kemb = self.enc_k(xc)
        Qemb = self.enc_k(xq)
        Vemb = self.enc_v(yc)
        D = Kemb.shape[-1]
        Kemb = Kemb.softmax(dim=-1) / math.sqrt(D)
        Qemb = Qemb.softmax(dim=-1) / math.sqrt(D)
        KV = torch.einsum('bnd,bnm->bdm', Kemb, Vemb)
        H = torch.einsum('bnd,bdm->bnm', Qemb, KV)
        return self.dec(H)


# FuncAttn

class FuncAttn(nn.Module):
    def __init__(self, latent_dim=256, num_heads=8,
                 num_groups=64, ridge=1e-4, init_param=False):
        super().__init__()
        assert latent_dim % num_heads == 0
        self.dim_head = latent_dim // num_heads
        self.heads = num_heads
        self.temperature = nn.Parameter(torch.ones([1, 1, 1]) * 0.5)
        self.freq_keep = num_groups

        self.in_project_q = nn.Linear(latent_dim, latent_dim)
        self.in_project_kv = self.in_project_q

        self.slice = nn.Linear(latent_dim, self.freq_keep)
        torch.nn.init.orthogonal_(self.slice.weight)

        def make_encoder(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, latent_dim), nn.ReLU(),
                nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                nn.Linear(latent_dim, out_d),
            )

        self.to_q = make_encoder(1, latent_dim)
        self.to_k = self.to_q
        self.to_v = self.to_q

        if init_param:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.0, 0.01)
                    nn.init.zeros_(m.bias)
        self.ridge = ridge

    def _slice(self, x_mid):
        B, N, Dh = x_mid.shape
        temp = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_weights = torch.softmax(self.slice(x_mid) / temp, dim=1)
        slice_norm = slice_weights.sum(1)
        slice_tokens = torch.einsum("bnc,bng->bgc", x_mid, slice_weights)
        slice_tokens = slice_tokens / ((slice_norm + 1e-5)[:, :, None].repeat(1, 1, Dh))
        return slice_weights, slice_tokens

    def forward(self, xc, yc, x_q):
        B, Nq, C = x_q.shape
        x_q = self.to_q(x_q)
        xc = self.to_k(xc)

        slice_w, slice_token = self._slice(xc)

        q = x_q
        k = slice_token
        v = torch.einsum("bnc,bng->bgc", yc, slice_w)

        kH = k.transpose(1, 2)
        kkH = torch.bmm(k, kH)
        I = torch.eye(kkH.shape[1], device=q.device, dtype=q.dtype).unsqueeze(0)
        reg = (1 - self.ridge) * kkH + self.ridge * I

        qkH = torch.bmm(q, kH)
        C_mat = torch.linalg.solve(reg, qkH, left=False)
        out_slice = torch.bmm(C_mat, v)
        return out_slice


# Transolver

class Transolver(nn.Module):
    def __init__(self, latent_dim=256, learnable_ridge=True, num_heads=8,
                 dropout=0., num_groups=64, init_param=False):
        super().__init__()
        assert latent_dim % num_heads == 0
        self.dim_head = latent_dim // num_heads
        self.heads = num_heads
        self.temperature = nn.Parameter(torch.ones([1, 1, 1]) * 0.5)
        self.freq_keep = num_groups

        self.in_project_q = nn.Linear(latent_dim, latent_dim)
        self.in_project_kv = self.in_project_q

        self.slice = nn.Linear(latent_dim, num_groups)
        torch.nn.init.orthogonal_(self.slice.weight)

        def make_encoder(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, latent_dim), nn.ReLU(),
                nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                nn.Linear(latent_dim, out_d),
            )

        self.to_q = make_encoder(1, latent_dim)
        self.to_k = self.to_q
        self.to_v = self.to_q

        if init_param:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.0, 0.01)
                    nn.init.zeros_(m.bias)

    def _slice(self, x_mid):
        B, N, Dh = x_mid.shape
        temp = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_weights = torch.softmax(self.slice(x_mid) / temp, dim=1)
        slice_norm = slice_weights.sum(1)
        slice_tokens = torch.einsum("bnc,bng->bgc", x_mid, slice_weights)
        slice_tokens = slice_tokens / ((slice_norm + 1e-5)[:, :, None].repeat(1, 1, Dh))
        return slice_weights, slice_tokens

    def forward(self, xc, yc, x_q):
        B, Nq, C = x_q.shape
        x_q = self.to_q(x_q)
        xc = self.to_k(xc)

        slice_w, slice_token = self._slice(xc)

        q = x_q
        k = slice_token
        v = torch.einsum("bnc,bng->bgc", yc, slice_w)

        scores = torch.bmm(q, k.transpose(1, 2))
        scores = scores / math.sqrt(self.dim_head)
        attn_weights = F.softmax(scores, dim=-1)
        out_slice = torch.bmm(attn_weights, v)
        return out_slice
