# PDE Standard Benchmark

Six PDE operator-learning benchmarks spanning structured grids, structured meshes, and irregular point clouds.

---

## Datasets

Download each dataset and point `--data_path` in the corresponding script to its location.

| Dataset | Task | Geometry | Download |
|---------|------|----------|----------|
| Elasticity | Estimate material inner stress | Point Cloud | [Google Drive](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Plasticity | Estimate material deformation over time | Structured Mesh | [Google Drive](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Navier-Stokes | Predict future fluid velocity | Regular Grid | [Google Drive](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Darcy | Estimate fluid pressure through medium | Regular Grid | [Google Drive](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Airfoil | Estimate airflow velocity around airfoil | Structured Mesh | [Google Drive](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Pipe | Estimate fluid velocity in a pipe | Structured Mesh | [Google Drive](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |

---

## Training

Each benchmark has a dedicated script under `scripts/` with the recommended hyperparameters:

```bash
bash scripts/FuncAttn_Elas.sh        # Elasticity
bash scripts/FuncAttn_Plasticity.sh  # Plasticity
bash scripts/FuncAttn_NS.sh          # Navier-Stokes
bash scripts/FuncAttn_Darcy.sh       # Darcy
bash scripts/FuncAttn_Airfoil.sh     # Airfoil
bash scripts/FuncAttn_Pipe.sh        # Pipe
```

Checkpoints are written to `./checkpoints/{save_name}.pt`. To evaluate a saved checkpoint, pass `--eval 1` with the same `--save_name`. Prediction and error plots are saved to `./results/{save_name}/`.

---

## Results

Relative L² error (%) on all six benchmarks. Our method outperforms the previous best model Transolver across the board.

![main results](../assets/main_results.png)
