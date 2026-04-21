from __future__ import annotations
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import wandb

import src.models as models
from rna_pointcloud_dataset import RNAPointCloudDataset
from visualize_point_cloud_segmentation import log_point_cloud_segmentation_visualizations

N_CLASS = 260


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--input_features', type=str, default='xyz',
                        choices=['xyz', 'hks'])
    parser.add_argument('--model', type=str, default='diffusionnet',
                        choices=['diffusionnet', 'funcattn', 'transolver'])
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--vis_every', type=int, default=10)
    parser.add_argument('--n_vis_samples', type=int, default=3)
    # DiffusionNet
    parser.add_argument('--C_width', type=int, default=128)
    parser.add_argument('--N_block', type=int, default=4)
    # FuncAttn / Transolver
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--mlp_ratio', type=int, default=2)
    parser.add_argument('--num_basis', type=int, default=128)
    # shared
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_eigen', type=int, default=128)
    # training
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decay_every', type=int, default=50)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--scheduler', type=str, default='onecycle',
                        choices=['step', 'cosine', 'onecycle'])
    return parser.parse_args()


def exp_name(args) -> str:
    ts = datetime.now().strftime('%m%d_%H%M')
    if args.model == 'diffusionnet':
        base = f'diffusionnet_{args.input_features}_w{args.C_width}_b{args.N_block}_e{args.num_eigen}_lr{args.lr:.0e}'
    else:
        base = f'{args.model}_{args.input_features}_d{args.embed_dim}_h{args.num_heads}_l{args.n_layers}_b{args.num_basis}_lr{args.lr:.0e}'
    if args.dropout != 0.1:
        base += f'_drop{args.dropout:.2f}'
    return f'{base}_{ts}'


def build_model(args, device):
    last_act = lambda x: F.log_softmax(x, dim=-1)
    c_in = {'xyz': 3, 'hks': 16}[args.input_features]
    if args.model == 'funcattn':
        return models.FuncAttn(
            embed_dim=args.embed_dim, num_heads=args.num_heads, n_layers=args.n_layers,
            dropout=args.dropout, num_basis=args.num_basis, mlp_ratio=args.mlp_ratio,
            act='gelu', input_features=args.input_features, num_freqs=10,
            last_activation=last_act,
        ).to(device)
    if args.model == 'transolver':
        return models.Transolver(
            embed_dim=args.embed_dim, num_heads=args.num_heads, n_layers=args.n_layers,
            dropout=args.dropout, num_basis=args.num_basis, mlp_ratio=args.mlp_ratio,
            act='gelu', input_features=args.input_features, num_freqs=10,
            last_activation=last_act,
        ).to(device)
    return models.layers.DiffusionNet(
        C_in=c_in, C_out=N_CLASS, C_width=args.C_width, N_block=args.N_block,
        last_activation=last_act, outputs_at='vertices', dropout=True,
    ).to(device)


def get_features(input_features, verts, evals, evecs):
    if input_features == 'hks':
        return models.geometry.compute_hks_autoscale(evals, evecs, 16)
    return verts


def run_epoch(model, loader, optimizer, scheduler, args, device, training):
    model.train() if training else model.eval()
    correct, total_num, total_loss = 0, 0, 0.0
    augment = training and args.input_features == 'xyz'

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for data in tqdm(loader):
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data
            verts, faces, frames = verts.to(device), faces.to(device), frames.to(device)
            mass, L = mass.to(device), L.to(device)
            evals, evecs = evals.to(device), evecs.to(device)
            gradX, gradY, labels = gradX.to(device), gradY.to(device), labels.to(device)

            if augment:
                verts = models.utils.random_rotate_points(verts)

            features = get_features(args.input_features, verts, evals, evecs)
            preds = model(features, mass, L=L, evals=evals, evecs=evecs,
                          gradX=gradX, gradY=gradY)
            loss = F.nll_loss(preds, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            correct += preds.argmax(dim=1).eq(labels).sum().item()
            total_num += labels.shape[0]
            total_loss += loss.item() * labels.shape[0]

    if training and scheduler is not None:
        scheduler.step()

    return correct / total_num, total_loss / total_num


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    training = not args.evaluate

    name = exp_name(args)
    print(f'Experiment: {name}')

    base_path = os.path.dirname(__file__)
    dataset_path = os.path.join(base_path, 'data/RNA-Surface-Segmentation-Dataset')
    op_cache_dir = os.path.join(base_path, 'data', 'op_cache')
    save_path = os.path.join(base_path, 'trained_models', f'{name}.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    test_dataset = RNAPointCloudDataset(
        dataset_path, train=False, k_eig=args.num_eigen,
        use_cache=True, op_cache_dir=op_cache_dir,
    )
    test_loader = DataLoader(test_dataset, batch_size=None)

    if training:
        train_dataset = RNAPointCloudDataset(
            dataset_path, train=True, k_eig=args.num_eigen,
            use_cache=True, op_cache_dir=op_cache_dir,
        )
        train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
        print(f'Train: {len(train_dataset)}  Test: {len(test_dataset)}')

    model = build_model(args, device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters())}')

    if not training:
        pretrain_path = os.path.join(
            base_path, f'pretrained_models/rna_mesh_seg_{args.input_features}_4x128.pth'
        )
        model.load_state_dict(torch.load(pretrain_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.n_epoch, eta_min=args.lr * 0.01)
    elif args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=args.n_epoch,
            pct_start=0.3, final_div_factor=1000)
    else:
        scheduler = None

    if args.wandb:
        config = vars(args)
        wandb.init(project='rna-segmentation', name=name, config=config)
        wandb.log({'num_params': sum(p.numel() for p in model.parameters())}, step=0)
        wandb.watch(model, log='all')

    if training:
        for epoch in range(args.n_epoch):
            train_acc, train_loss = run_epoch(
                model, train_loader, optimizer, scheduler, args, device, training=True)
            test_acc, test_loss = run_epoch(
                model, test_loader, optimizer, scheduler, args, device, training=False)

            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:3d}  '
                  f'train {100*train_acc:.2f}% loss {train_loss:.4f}  '
                  f'test {100*test_acc:.2f}% loss {test_loss:.4f}  '
                  f'lr {current_lr:.2e}')

            if args.wandb:
                wandb.log({
                    'epoch': epoch, 'train/accuracy': train_acc, 'test/accuracy': test_acc,
                    'train/loss': train_loss, 'test/loss': test_loss, 'lr': current_lr,
                }, step=epoch)
                if epoch % args.vis_every == 0 or epoch == args.n_epoch - 1:
                    log_point_cloud_segmentation_visualizations(
                        args, model=model, test_loader=test_loader, device=device,
                        input_features=args.input_features, n_samples=args.n_vis_samples,
                        n_classes=N_CLASS, epoch=epoch,
                    )

        torch.save(model.state_dict(), save_path)
        print(f'Saved -> {save_path}')

    test_acc, test_loss = run_epoch(
        model, test_loader, optimizer, None, args, device, training=False)
    print(f'Test accuracy: {100*test_acc:.2f}%  Loss: {test_loss:.4f}')

    if args.wandb:
        log_point_cloud_segmentation_visualizations(
            args, model=model, test_loader=test_loader, device=device,
            input_features=args.input_features, n_samples=args.n_vis_samples,
            n_classes=N_CLASS, epoch=args.n_epoch if training else 0,
        )


if __name__ == '__main__':
    main()
