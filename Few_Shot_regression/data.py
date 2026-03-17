from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass
import math, random

import torch


@dataclass
class SineCfg:
    x_range: Tuple[float, float] = (-6.0, 6.0)
    amp_range: Tuple[float, float] = (0.1, 5.0)
    phase_range: Tuple[float, float] = (0.0, math.pi)


class SineTask:
    def __init__(self, cfg: SineCfg):
        self.A = random.uniform(*cfg.amp_range)
        self.phi = random.uniform(*cfg.phase_range)
        self.xlo, self.xhi = cfg.x_range

    def sample(self, n: int):
        x = torch.empty(n, 1).uniform_(self.xlo, self.xhi)
        y = self.A * torch.sin(x - self.phi)
        return x, y


class EpisodeSampler:
    def __init__(self, cfg: SineCfg, K: int, Q: int):
        self.cfg, self.K, self.Q = cfg, K, Q

    def sample_batch(self, B: int):
        batch = []
        for _ in range(B):
            t = SineTask(self.cfg)
            xc, yc = t.sample(self.K)
            xq, yq = t.sample(self.Q)
            batch.append((xc, yc, xq, yq))
        return batch
