# Airfoil Design — AirfRANS

Out-of-distribution generalization benchmark for airfoil design. The task requires the model to estimate surrounding and surface physical quantities of a 2D airfoil under varying Reynolds numbers and angles of attack.

This code is built on top of [AirfRANS](https://github.com/Extrality/AirfRANS).

---

## Data

Download the dataset from [AirfRANS](https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip) (9.3 GB) and place it under a local directory (e.g. `/data/naca/Dataset`).

> Note: [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) is required.

---

## Train

```bash
bash scripts/FuncAttn.sh
```

Update `--my_path` in the script to your dataset path. The benchmark supports four evaluation settings:

| Setting | Argument |
|---------|----------|
| Full data | `-t full` |
| Scarce data | `-t scarce` |
| OOD Reynolds numbers | `-t reynolds` |
| OOD angles of attack | `-t aoa` |

---

## Evaluate

```bash
bash scripts/Evaluation.sh
```

---

## Acknowledgement

We appreciate the following GitHub repo for their valuable code base and datasets:

https://github.com/Extrality/AirfRANS
