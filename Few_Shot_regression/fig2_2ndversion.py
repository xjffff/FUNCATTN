# ===== figure7_mse_vs_k.py =====
"""Plot MSE vs K curves for all trained models."""
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rc

from models import (
    AttentionHead, IntentionHead, FuncAttn, Transolver
)
from data import (
     SineCfg, SineTask,
)

# ============ STYLE ============
#rc('text', usetex=True)
#rc('text.latex', preamble=r'\usepackage{textcomp}')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX', 'STIXGeneral', 'DejaVu Serif', 'Times'],
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'mathtext.fontset': 'stix',
    'lines.linewidth': 1.0,
})

METHOD_COLORS = {
    'Attention': '#1f77b4',
    'Intention': '#ff7f0e',
    'Ours': '#d62728',
    'Transolver': '#9467bd',
}

# ============ UTILITIES ============

def _load_model(path, ctor, device):
    """Load checkpoint into a freshly constructed model."""
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and any(k in state for k in ("model", "state_dict")):
        state = state.get("model", state.get("state_dict"))
    m = ctor().to(device)
    m.load_state_dict(state, strict=False)
    return m.eval()


@torch.no_grad()
def mse_vs_K(model, K_list, *, episodes=1000, M=200, cfg=SineCfg(), device="cpu"):
    """Returns dict mapping K -> mean MSE."""
    results = {}
    for K in K_list:
        total_se, count = 0.0, 0
        for _ in range(episodes):
            task = SineTask(cfg)
            x_all, y_all = task.sample(M)
            idx = torch.randperm(M)[:K]
            xc = x_all[idx].unsqueeze(0).to(device)
            yc = y_all[idx].unsqueeze(0).to(device)
            xq = x_all.unsqueeze(0).to(device)
            yq = y_all.unsqueeze(0)
            yhat = model(xc, yc, xq).cpu()
            total_se += F.mse_loss(yhat, yq, reduction="sum").item()
            count += M
        results[K] = total_se / count
    return results


# ============ PLOTTING ============

def plot_fig7(curves, save_path="figure7.pdf"):
    """curves: list of (label, mse_dict)"""
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    plt.subplots_adjust(left=0.12, right=0.99, top=0.97, bottom=0.35)

    for label, mse_dict in curves:
        Ks = sorted(mse_dict.keys())
        ax.semilogy(
            Ks, [mse_dict[k] for k in Ks],
            linestyle='-', color=METHOD_COLORS.get(label),
            linewidth=2, label=label,
        )

    ax.set_xlabel("Number of observations")
    ax.text(-0.02, 1.02, 'MSE', transform=ax.transAxes,
            fontsize=9, ha='right', va='bottom')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(
        frameon=False, loc='upper center',
        bbox_to_anchor=(0.5, -0.25), ncol=4,
        columnspacing=1.0, handletextpad=0.4,
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                pad_inches=0.02, facecolor='white')
    plt.close(fig)
    print(f"Saved to {save_path}")


# ============ MAIN ============

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); random.seed(42)

    # model name -> (constructor, checkpoint path)
    MODELS = {
        "Attention": (
            lambda: AttentionHead(d_model=256, num_heads=4),
            "checkpoints/attention.pth",
        ),
        "Intention": (
            lambda: IntentionHead(latent_dim=128, ridge=1e-3),
            "checkpoints/intention.pth",
        ),
        "Transolver": (
            lambda: Transolver(latent_dim=256, num_groups=8, init_param=False),
            "checkpoints/transolver.pth",
        ),
        "Ours": (
            lambda: FuncAttn(latent_dim=128, num_groups=8, ridge=1e-4, init_param=True),
            "checkpoints/funcattn.pth",
        ),
    }

    K_list = [5, 10, 15, 20, 25, 30, 35, 40]
    curves = []
    for name, (ctor, ckpt) in MODELS.items():
        model = _load_model(ckpt, ctor, device)
        mse_dict = mse_vs_K(model, K_list, episodes=1000, M=200, device=device)
        curves.append((name, mse_dict))

    plot_fig7(curves, save_path="figure7.pdf")