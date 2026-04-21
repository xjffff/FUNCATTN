"""Plot prediction curves for all methods: (init vs trained, K=4)."""
import random

import matplotlib.pyplot as plt
import torch

from data import SineCfg
from models import Attention, FuncAttn, Intention, Transolver


class SineTask:
    """SineTask with torch.Generator for exact seed reproducibility."""
    def __init__(self, cfg: SineCfg, g: torch.Generator):
        self.A   = torch.empty(()).uniform_(*cfg.amp_range,   generator=g).item()
        self.phi = torch.empty(()).uniform_(*cfg.phase_range, generator=g).item()
        self.xlo, self.xhi = cfg.x_range

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX', 'STIXGeneral', 'DejaVu Serif', 'Times'],
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
    'lines.linewidth': 1.0,
})

METHOD_SPEC = {
    'Attention': dict(
        color='#1f77b4',
        ctor=lambda: Attention(d_model=256, num_heads=4),
        ckpt='checkpoints/attention_k4.pth',
    ),
    'Intention': dict(
        color='#ff7f0e',
        ctor=lambda: Intention(latent_dim=1000,
                                   ridge=1e-4),
        ckpt='checkpoints/intention_k4.pth',
    ),
    'FuncAttn': dict(
        color='#d62728',
        ctor=lambda: FuncAttn(latent_dim=128, num_groups=8,
                              ridge=1e-3, init_param=True),
        ctor_init=lambda: FuncAttn(latent_dim=128, num_groups=8,
                                   ridge=1e-3, init_param=False),
        ckpt='checkpoints/funcattn_k4.pth',
    ),
    'Transolver': dict(
        color='#9467bd',
        ctor=lambda: Transolver(latent_dim=256, num_groups=8, init_param=False),
        ckpt='checkpoints/transolver_k4.pth',
    ),
}


def _load_model(path, ctor, device):
    """Load checkpoint into a freshly constructed model."""
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and any(k in state for k in ('model', 'state_dict')):
        state = state.get('model', state.get('state_dict'))
    m = ctor().to(device)
    m.load_state_dict(state, strict=False)
    return m.eval()


@torch.no_grad()
def _predict(model, xc, yc, xq, device):
    if model is None:
        return None
    return model(
        xc.unsqueeze(0).to(device),
        yc.unsqueeze(0).to(device),
        xq.unsqueeze(0).to(device),
    ).squeeze(0).cpu()


def _draw_panel(ax, xq, y_true, xc, yc, init_models, trained_models, title, device):
    """Draw ground truth, context scatter, and init/trained predictions."""
    xs = xq.squeeze(-1).numpy()

    ax.plot(xs, y_true.squeeze(-1).numpy(),
            color='gray', linestyle=(0, (2, 2)), linewidth=2.0, alpha=0.8, zorder=1)
    ax.scatter(xc.squeeze(-1).numpy(), yc.squeeze(-1).numpy(),
               s=20, color='black', marker='o', edgecolors='white', linewidths=0.5, zorder=5)

    for name, spec in METHOD_SPEC.items():
        color = spec['color']
        if init_models.get(name) is not None:
            y = _predict(init_models[name], xc, yc, xq, device)
            if y is not None:
                ax.plot(xs, y.squeeze(-1).numpy(), '-', color=color, linewidth=1.5, alpha=0.85)
        if trained_models.get(name) is not None:
            y = _predict(trained_models[name], xc, yc, xq, device)
            if y is not None:
                ax.plot(xs, y.squeeze(-1).numpy(), '-', color=color, linewidth=1.5, label=name)

    ax.set_title(title, pad=4, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-0.75, 0.75)
    ax.set_xticks([-5, 0, 5])


def plot_fig1(
    *,
    K=4, M=200,
    seed_task=13, seed_ctx=999,
    save_path='figure1.pdf',
    device=None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    g_task = torch.Generator().manual_seed(seed_task)
    task = SineTask(SineCfg(), g_task)
    xq = torch.linspace(-5, 5, M).unsqueeze(-1)
    y_true = task.A * torch.sin(xq - task.phi)

    init_models    = {n: s.get('ctor_init', s['ctor'])().to(device).eval() for n, s in METHOD_SPEC.items()}
    trained_models = {n: _load_model(s['ckpt'], s['ctor'], device) for n, s in METHOD_SPEC.items()}

    g_ctx = torch.Generator().manual_seed(seed_ctx)
    idx = torch.randperm(M, generator=g_ctx)[:K]
    xc, yc = xq[idx], y_true[idx]

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.12, right=0.995, top=0.92, bottom=0.35, wspace=0.06)

    _draw_panel(axes[0], xq, y_true, xc, yc, init_models,    {},             'init',    device)
    _draw_panel(axes[1], xq, y_true, xc, yc, {},              trained_models, 'trained', device)

    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved to {save_path}')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random.seed(5)
    torch.manual_seed(5)
    plot_fig1(save_path='figure1.pdf', device=device)
