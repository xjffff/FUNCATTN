# RNA Point Cloud Segmentation

This is adapted from [DiffusionNet](https://github.com/nmwsharp/diffusion-net)'s RNA mesh segmentation benchmark, modified to operate on point clouds rather than meshes.

---

## Data

Clone the dataset into the `data/` subdirectory:

```bash
cd data
git clone https://github.com/nmwsharp/RNA-Surface-Segmentation-Dataset.git
```

Meshes will be stored at `data/RNA-Surface-Segmentation-Dataset/off/`.

---

## Train

```bash
python rna_point_segmentation.py --model funcattn --input_features xyz
```

Supported models: `funcattn`, `diffusionnet`, `transolver`

Input features: `xyz` (raw coordinates) or `hks` (heat kernel signatures)

Key options:

| flag | default | description |
|------|---------|-------------|
| `--model` | `diffusionnet` | model to use |
| `--input_features` | `xyz` | input feature type |
| `--embed_dim` | 128 | FuncAttn embedding dimension |
| `--num_heads` | 8 | number of attention heads |
| `--n_layers` | 4 | number of layers |
| `--num_basis` | 128 | number of bases |
| `--n_epoch` | 200 | training epochs |
| `--lr` | 1e-3 | learning rate |
| `--scheduler` | `onecycle` | LR scheduler (`step`, `cosine`, `onecycle`) |
| `--wandb` | — | enable wandb logging |

Checkpoints are saved to `trained_models/{exp_name}.pth`.

---

## Acknowledgement

We appreciate the following GitHub repo for their valuable code base and datasets:

https://github.com/nmwsharp/diffusion-net
