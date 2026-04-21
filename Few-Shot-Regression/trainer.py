from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from data import SineCfg, EpisodeSampler
from models import (
    Intention, Attention, LinearAttention,
    FuncAttn, Transolver,
)


@dataclass
class TrainCfg:
    model: str = "intention"
    iters: int = 5000
    batch_size: int = 8
    K: int = 4
    Q: int = 200
    lr: float = 1e-4
    ridge: float = 1e-3
    device: str = "cuda"
    init_param: bool = False


MODEL_BUILDERS = {
    "intention": lambda cfg: Intention(
        latent_dim=1000, ridge=cfg.ridge, init_param=cfg.init_param,
    ),
    "attention": lambda cfg: Attention(
        d_model=256, num_heads=4, init_param=cfg.init_param,
    ),
    "funcattn": lambda cfg: FuncAttn(
        latent_dim=128, ridge=cfg.ridge, num_groups=8, init_param=cfg.init_param,
    ),
    "linear_attention": lambda cfg: LinearAttention(
        d_model=128, num_heads=8, init_param=cfg.init_param,
    ),
    "transolver": lambda cfg: Transolver(
        latent_dim=256, num_groups=8, init_param=cfg.init_param,
    ),
}


class FewShotTrainer:
    def __init__(self, cfg: TrainCfg):
        self.cfg = cfg
        builder = MODEL_BUILDERS.get(cfg.model)
        if builder is None:
            raise ValueError(
                f"Unknown model '{cfg.model}'. "
                f"Choose from: {list(MODEL_BUILDERS.keys())}"
            )
        self.model = builder(cfg).to(cfg.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg.iters,
        )
        self.sampler = EpisodeSampler(SineCfg(), K=cfg.K, Q=cfg.Q)
        self.losses: list[float] = []

    def step(self) -> float:
        batch = self.sampler.sample_batch(self.cfg.batch_size)
        xc = torch.stack([b[0] for b in batch]).to(self.cfg.device)
        yc = torch.stack([b[1] for b in batch]).to(self.cfg.device)
        xq = torch.stack([b[2] for b in batch]).to(self.cfg.device)
        yq = torch.stack([b[3] for b in batch]).to(self.cfg.device)

        yhat = self.model(xc, yc, xq)
        loss = F.mse_loss(yhat, yq)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()

        val = loss.item()
        self.losses.append(val)
        return val

    @torch.no_grad()
    def eval_mse(self, K_list=(5, 10, 20, 40), episodes=200):
        self.model.eval()
        results = {}
        for K in K_list:
            sampler = EpisodeSampler(SineCfg(), K=K, Q=self.cfg.Q)
            mses = []
            for _ in range(episodes):
                xc, yc, xq, yq = sampler.sample_batch(1)[0]
                xc = xc.unsqueeze(0).to(self.cfg.device)
                yc = yc.unsqueeze(0).to(self.cfg.device)
                xq = xq.unsqueeze(0).to(self.cfg.device)
                yq = yq.unsqueeze(0).to(self.cfg.device)
                yhat = self.model(xc, yc, xq)
                mses.append(F.mse_loss(yhat, yq).item())
            results[K] = sum(mses) / len(mses)
        self.model.train()
        return results
