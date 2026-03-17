from __future__ import annotations
import argparse, os, random, json

import torch
import numpy as np

from trainer import TrainCfg, FewShotTrainer

DEFAULT_CONFIGS = {
    #"attention": dict(
    #    batch_size=8, K=32, Q=200, lr=1e-3, ridge=1e-3, init_param=False,
    #),
    #"intention": dict(
    #    batch_size=8, K=32, Q=200, lr=3e-4, ridge=1e-3, init_param=False,
    #),
    "funcattn": dict(
        batch_size=8, K=32, Q=200, lr=1e-4, ridge=1e-4, init_param=True,
    ),
    #"transolver": dict(
    #    batch_size=8, K=32, Q=200, lr=1e-4, ridge=1e-3, init_param=False,
    #),
    #"linear_attention": dict(
    #    batch_size=8, K=32, Q=200, lr=1e-4, ridge=1e-3, init_param=False,
    #),
}


def train_one(name: str, iters: int, device: str, save_dir: str, print_every: int):
    kw = DEFAULT_CONFIGS[name]
    cfg = TrainCfg(model=name, iters=iters, device=device, **kw)
    trainer = FewShotTrainer(cfg)

    print(f"\n{'='*50}")
    print(f"Training {name}  (iters={iters}, bs={cfg.batch_size}, K={cfg.K})")
    print(f"{'='*50}")

    for it in range(1, iters + 1):
        loss = trainer.step()
        if it % print_every == 0:
            print(f"  [{name} {it:05d}/{iters}] MSE: {loss:.5f}")

    # --- eval ---
    eval_results = trainer.eval_mse()
    print(f"  Eval MSE: {eval_results}")

    # --- save ---
    ckpt_path = os.path.join(save_dir, f"{name}.pth")
    torch.save(trainer.model.state_dict(), ckpt_path)
    print(f"  Saved -> {ckpt_path}")

    return eval_results, trainer.losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=list(DEFAULT_CONFIGS.keys()),
                        help="Which models to train")
    parser.add_argument("--iters", type=int, default=50000)
    parser.add_argument("--print_every", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    all_results = {}
    for name in args.models:
        eval_res, losses = train_one(
            name, args.iters, device, args.save_dir, args.print_every,
        )
        all_results[name] = eval_res

    # save a summary json alongside checkpoints
    summary_path = os.path.join(args.save_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nEval summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
