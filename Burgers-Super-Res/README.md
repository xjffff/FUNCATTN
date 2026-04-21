# Burgers Super-Resolution

Zero-shot super-resolution for the 1D Burgers equation using a SimpleTransformer with functional attention and a spectral (iFFT) decoder.  
Train on a **2048-pt** grid, evaluate on **8192-pt** without retraining.

Based on [Galerkin Transformer (Cao, NeurIPS 2021)](https://arxiv.org/abs/2105.14995).

---

## Data

Download `burgers_data_R10.mat` from [Zongyi Li's Google Drive](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing) and place it in this folder.

---

## Train

```bash
python exp_burgers_super_res.py --data-path ./burgers_data_R10.mat
```

Key options:

| flag | default | description |
|------|---------|-------------|
| `--subsample` | 4 | training grid subsampling (4 → 2048 pts) |
| `--epochs` | 100 | number of training epochs |
| `--lr` | 1e-3 | max learning rate (OneCycleLR) |
| `--gamma` | 0.1 | H¹ gradient regularizer weight |
| `--model-save-path` | `./models` | checkpoint output directory |

Checkpoints are saved to `./models/burgers_super_res_<date>.pt`.

---

## Evaluate

Open and run [`eval_burgers_super_res.ipynb`](./eval_burgers_super_res.ipynb).

It reports the relative L² error at super-resolution (8192 pts), and plots predictions against ground truth.
