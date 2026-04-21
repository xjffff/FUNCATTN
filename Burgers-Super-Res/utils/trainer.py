"""
Training loop for Burgers super-resolution.
Extracted from galerkin-transformer/libs/utils_ft.py
"""
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# terminal colours
# ---------------------------------------------------------------------------
class Colors:
    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    end = "\033[0m"


def color(string: str, c: str = Colors.yellow) -> str:
    return f"{c}{string}{Colors.end}"


# ---------------------------------------------------------------------------
# misc helpers
# ---------------------------------------------------------------------------
EPOCH_SCHEDULERS = [
    'ReduceLROnPlateau', 'StepLR', 'MultiplicativeLR',
    'MultiStepLR', 'ExponentialLR', 'LambdaLR',
]


def save_pickle(var, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(var, f)


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = 0
    for p in model_parameters:
        num_params += p.numel() * (1 + p.is_complex())
    return num_params


# ---------------------------------------------------------------------------
# per-batch train / per-epoch validate
# ---------------------------------------------------------------------------
def train_batch_burgers(model, loss_func, data, optimizer, lr_scheduler, device, grad_clip=0.999):
    optimizer.zero_grad()
    x, edge = data["node"].to(device), data["edge"].to(device)
    pos, grid = data['pos'].to(device), data['grid'].to(device)
    out_ = model(x, edge, pos, grid)

    if isinstance(out_, dict):
        out = out_['preds']
        y_latent = out_['preds_latent']
    elif isinstance(out_, tuple):
        out = out_[0]
        y_latent = None

    target = data["target"].to(device)
    u, up = target[..., 0], target[..., 1]

    if out.size(2) == 2:
        u_pred, up_pred = out[..., 0], out[..., 1]
        loss, reg, ortho, _ = loss_func(u_pred, u, up_pred, up, preds_latent=y_latent)
    elif out.size(2) == 1:
        u_pred = out[..., 0]
        loss, reg, ortho, _ = loss_func(u_pred, u, targets_prime=up, preds_latent=y_latent)

    loss = loss + reg + ortho
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if lr_scheduler:
        lr_scheduler.step()

    try:
        up_pred = out[..., 1]
    except Exception:
        up_pred = u_pred

    return (loss.item(), reg.item(), ortho.item()), u_pred, up_pred


def validate_epoch_burgers(model, metric_func, valid_loader, device):
    model.eval()
    metric_val = []
    for _, data in enumerate(valid_loader):
        with torch.no_grad():
            x, edge = data["node"].to(device), data["edge"].to(device)
            pos, grid = data['pos'].to(device), data['grid'].to(device)
            out_ = model(x, edge, pos, grid)

            if isinstance(out_, dict):
                u_pred = out_['preds'][..., 0]
            elif isinstance(out_, tuple):
                u_pred = out_[0][..., 0]

            target = data["target"].to(device)
            u = target[..., 0]
            _, _, _, metric = metric_func(u_pred, u)
            try:
                metric_val.append(metric.item())
            except Exception:
                metric_val.append(metric)

    return dict(metric=np.mean(metric_val, axis=0))


# ---------------------------------------------------------------------------
# main training loop
# ---------------------------------------------------------------------------
def run_train(model, loss_func, metric_func,
              train_loader, valid_loader,
              optimizer, lr_scheduler,
              train_batch=None,
              validate_epoch=None,
              epochs=10,
              device="cuda",
              mode='min',
              tqdm_mode='batch',
              patience=10,
              grad_clip=0.999,
              start_epoch: int = 0,
              model_save_path='./models',
              save_mode='state_dict',
              model_name='model.pt',
              result_name='result.pkl'):

    loss_train = []
    loss_val = []
    loss_epoch = []
    lr_history = []

    if patience is None or patience == 0:
        patience = epochs
    end_epoch = start_epoch + epochs
    best_val_metric = -np.inf if mode == 'max' else np.inf
    best_val_epoch = None
    stop_counter = 0
    is_epoch_scheduler = any(s in str(lr_scheduler.__class__) for s in EPOCH_SCHEDULERS)
    tqdm_epoch = (tqdm_mode != 'batch')

    os.makedirs(model_save_path, exist_ok=True)

    with tqdm(total=end_epoch - start_epoch, disable=not tqdm_epoch) as pbar_ep:
        for epoch in range(start_epoch, end_epoch):
            model.train()
            with tqdm(total=len(train_loader), disable=tqdm_epoch) as pbar_batch:
                for batch in train_loader:
                    sched = None if is_epoch_scheduler else lr_scheduler
                    loss, _, _ = train_batch(
                        model, loss_func, batch, optimizer, sched, device, grad_clip=grad_clip)
                    loss = np.array(loss)
                    loss_epoch.append(loss)
                    lr = optimizer.param_groups[0]['lr']
                    lr_history.append(lr)

                    desc = f"epoch: [{epoch+1}/{end_epoch}]"
                    if loss.ndim == 0:
                        _loss_mean = np.mean(loss_epoch)
                        desc += f" loss: {_loss_mean:.3e}"
                    else:
                        _loss_mean = np.mean(loss_epoch, axis=0)
                        for j in range(len(_loss_mean)):
                            if _loss_mean[j] > 0:
                                desc += f" | loss {j}: {_loss_mean[j]:.3e}"
                    desc += f" | current lr: {lr:.3e}"
                    pbar_batch.set_description(desc)
                    pbar_batch.update()

            loss_train.append(_loss_mean)
            loss_epoch = []

            val_result = validate_epoch(model, metric_func, valid_loader, device)
            loss_val.append(val_result["metric"])
            val_metric = val_result["metric"].sum() if hasattr(val_result["metric"], 'sum') else val_result["metric"]

            improved = (val_metric > best_val_metric) if mode == 'max' else (val_metric < best_val_metric)
            if improved:
                best_val_epoch = epoch
                best_val_metric = val_metric
                stop_counter = 0
                if save_mode == 'state_dict':
                    torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
                else:
                    torch.save(model, os.path.join(model_save_path, model_name))
            else:
                stop_counter += 1

            if lr_scheduler and is_epoch_scheduler:
                if 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
                    lr_scheduler.step(val_metric)
                else:
                    lr_scheduler.step()

            if stop_counter > patience:
                print(f"Early stop at epoch {epoch}")
                break

            val_metric_scalar = float(np.array(val_result["metric"]).sum())
            desc = color(f"| val metric: {val_metric_scalar:.3e} ", Colors.blue)
            desc += color(f"| best val: {best_val_metric:.3e} at epoch {best_val_epoch+1}", Colors.yellow)
            desc += color(f" | early stop: {stop_counter} ", Colors.red)
            desc += color(f" | current lr: {lr:.3e}", Colors.magenta)

            if not tqdm_epoch:
                tqdm.write("\n" + desc + "\n")
            else:
                desc_ep = color("", Colors.green)
                if np.array(_loss_mean).ndim == 0:
                    desc_ep += color(f"| loss: {_loss_mean:.3e} ", Colors.green)
                else:
                    for j in range(len(_loss_mean)):
                        if _loss_mean[j] > 0:
                            desc_ep += color(f"| loss {j}: {_loss_mean[j]:.3e} ", Colors.green)
                desc_ep += desc
                pbar_ep.set_description(desc_ep)
                pbar_ep.update()

            result = dict(
                best_val_epoch=best_val_epoch,
                best_val_metric=best_val_metric,
                loss_train=np.asarray(loss_train),
                loss_val=np.asarray(loss_val),
                lr_history=np.asarray(lr_history),
                optimizer_state=optimizer.state_dict(),
            )
            save_pickle(result, os.path.join(model_save_path, result_name))

    return result
