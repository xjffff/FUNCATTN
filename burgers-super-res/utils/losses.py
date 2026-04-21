"""
WeightedL2Loss for 1D Burgers.
Extracted from galerkin-transformer/libs/ft.py
"""
import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss


class WeightedL2Loss(_WeightedLoss):
    def __init__(self,
                 dilation=2,
                 regularizer=False,
                 h=1/512,
                 beta=1.0,
                 gamma=1e-1,
                 alpha=0.0,
                 metric_reduction='L1',
                 periodic=False,
                 return_norm=True,
                 orthogonal_reg=False,
                 orthogonal_mode='global',
                 delta=1e-4,
                 noise=0.0,
                 debug=False):
        super().__init__()
        self.regularizer = regularizer
        self.noise = noise
        assert dilation % 2 == 0
        self.dilation = dilation
        self.h = h
        self.beta = beta
        self.gamma = gamma * h
        self.alpha = alpha * h
        self.delta = delta * h
        self.eps = 1e-8
        self.periodic = periodic
        self.metric_reduction = metric_reduction
        self.return_norm = return_norm
        self.orthogonal_reg = orthogonal_reg
        self.orthogonal_mode = orthogonal_mode
        self.debug = debug

    def central_diff(self, x, h=None):
        h = self.h if h is None else h
        d = self.dilation
        grad = (x[:, d:] - x[:, :-d]) / d
        return grad / h

    def forward(self, preds, targets,
                preds_prime=None, targets_prime=None,
                preds_latent=None, K=None):
        if preds_latent is None:
            preds_latent = []
        h = self.h
        if self.noise > 0:
            with torch.no_grad():
                targets = targets * (1.0 + self.noise * torch.rand_like(targets))

        target_norm = h * targets.pow(2).sum(dim=1)
        targets_prime_norm = h * targets_prime.pow(2).sum(dim=1) if targets_prime is not None else 1

        loss = self.beta * (h * (preds - targets).pow(2)).sum(dim=1) / target_norm

        if preds_prime is not None and self.alpha > 0:
            grad_diff = h * (preds_prime - K * targets_prime).pow(2)
            loss += self.alpha * grad_diff.sum(dim=1) / targets_prime_norm

        if self.metric_reduction == 'L2':
            metric = loss.mean().sqrt().item()
        elif self.metric_reduction == 'L1':
            metric = loss.sqrt().mean().item()
        elif self.metric_reduction == 'Linf':
            metric = loss.sqrt().max().item()

        loss = loss.sqrt().mean() if self.return_norm else loss.mean()

        if self.regularizer and self.gamma > 0 and targets_prime is not None:
            preds_diff = self.central_diff(preds)
            s = self.dilation // 2
            reg = self.gamma * h * (targets_prime[:, s:-s] - preds_diff).pow(2).sum(dim=1) / targets_prime_norm
            regularizer = reg.sqrt().mean() if self.return_norm else reg.mean()
        else:
            regularizer = torch.tensor([0.0], requires_grad=True, device=preds.device)

        orthogonalizer = torch.tensor([0.0], requires_grad=True, device=preds.device)

        return loss, regularizer, orthogonalizer, metric
