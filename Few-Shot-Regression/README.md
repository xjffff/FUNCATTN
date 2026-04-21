# Few-Shot Sinusoid Regression

Meta-learning benchmark comparing four in-context regression methods on random sine tasks.  
Each episode samples **K context points** and **200 query points** from a sine with random amplitude and phase; models predict at query locations without any gradient update.

---

## Models

| name | method | key hyperparameters |
|------|--------|---------------------|
| `attention` | Cross-attention (softmax) | `d_model=256`, `num_heads=4` |
| `intention` | Ridge regression in feature space | `latent_dim=1000`, `num_heads=4`, `ridge=1e-4` |
| `funcattn` | Ridge regression with soft-sliced tokens | `latent_dim=128`, `num_groups=8`, `ridge=1e-4` |
| `transolver` | Softmax attention with soft-sliced tokens | `latent_dim=128`, `num_groups=8` |

All encoders are 4-layer MLPs mapping `x ∈ ℝ` to the latent space.  
`init_param=True` initialises all linear weights with `N(0, 0.01)`.

---

## Train

```bash
python train.py --models intention funcattn --K 32 --iters 50000
```

Key options:

| flag | default | description |
|------|---------|-------------|
| `--models` | all uncommented in `DEFAULT_CONFIGS` | which models to train |
| `--iters` | 50000 | number of gradient steps |
| `--K` | per-model default | context set size (overrides all models) |
| `--seed` | 0 | global random seed |
| `--save_dir` | `checkpoints/` | checkpoint output directory |

Checkpoints are saved as `checkpoints/{name}_k{K}.pth`.  
Run twice (e.g. `--K 4` and `--K 32`) to produce the checkpoints required by both figures.

---

## Plot

**Figure 1** — prediction curves at init vs. trained (K = 4):

```bash
python plot_fig1.py
```

Loads `checkpoints/{name}_k4.pth` for each method and saves `figure1.pdf`.

**Figure 2** — MSE vs. number of observations (K = 5…40, trained at K = 32):

```bash
python plot_fig2.py
```

Loads `checkpoints/{name}_k32.pth` for each method and saves `figure2.pdf`.
