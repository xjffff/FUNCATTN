import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import wandb
from datetime import datetime
from visualize_point_cloud_segmentation import (
    log_point_cloud_segmentation_visualizations,
)
import src.models as models

from rna_point_segmentation import RNAPointCloudDataset

# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--evaluate", action="store_true", help="evaluate using the pretrained model"
)
parser.add_argument(
    "--input_features",
    type=str,
    help="what features to use as input ('xyz' or 'hks') default: hks",
    default="xyz",
)
parser.add_argument(
    "--model",
    type=str,
    help="which model to use ('diffusionnet' or 'FuncAttn' or 'FuncAttn_learned' or 'pointnet2')",
    default="diffusionnet",
)
parser.add_argument("--wandb", action="store_true", help="use wandb logging")
parser.add_argument(
    "--vis_every", type=int, help="visualize every N epochs (default: 10)", default=10
)
parser.add_argument(
    "--n_vis_samples",
    type=int,
    help="number of samples to visualize (default: 3)",
    default=3,
)

# Model hyperparameters - DiffusionNet
parser.add_argument(
    "--C_width", type=int, help="DiffusionNet width (default: 128)", default=128
)
parser.add_argument(
    "--N_block", type=int, help="DiffusionNet number of blocks (default: 4)", default=4
)

# Model hyperparameters - Functional Attention
parser.add_argument(
    "--embed_dim",
    type=int,
    help="FuncAttn embedding dimension (default: 128)",
    default=128,
)
parser.add_argument(
    "--num_heads",
    type=int,
    help="FuncAttn number of attention heads (default: 8)",
    default=8,
)
parser.add_argument(
    "--n_layers", type=int, help="FuncAttn number of layers (default: 4)", default=4
)
parser.add_argument(
    "--mlp_ratio", type=int, help="FuncAttn MLP ratio (default: 2)", default=2
)
parser.add_argument(
    "--num_basis",
    type=int,
    help="FuncAttn number of basis (default: 128)",
    default=128,
)

# Shared hyperparameters
parser.add_argument(
    "--dropout", type=float, help="Dropout rate (default: 0.1)", default=0.1
)
parser.add_argument(
    "--num_eigen", type=int, help="Number of eigenvectors (default: 128)", default=128
)

# Training hyperparameters
parser.add_argument(
    "--n_epoch", type=int, help="Number of epochs (default: 200)", default=200
)
parser.add_argument(
    "--lr", type=float, help="Learning rate (default: 1e-3)", default=1e-3
)
parser.add_argument(
    "--decay_every", type=int, help="LR decay frequency (default: 50)", default=50
)
parser.add_argument(
    "--decay_rate", type=float, help="LR decay rate (default: 0.5)", default=0.5
)
parser.add_argument(
    "--scheduler",
    type=str,
    help="LR scheduler type ('step', 'cosine', 'onecycle') default: step",
    default="onecycle",
)
args = parser.parse_args()


def generate_exp_name(args):
    """Generate experiment name based on model and hyperparameters."""
    timestamp = datetime.now().strftime("%m%d_%H%M")

    if args.model == "funcattn":
        # FuncAttn_Learned naming: FuncAttn_learned_xyz_d128_h8_l4_b64_lr1e-3_pos_share
        name = "funcattn_{}_d{}_h{}_l{}_b{}_lr{:.0e}_{}_{}".format(
            args.input_features,
            args.embed_dim,
            args.num_heads,
            args.n_layers,
            args.num_basis,
        )
    elif args.model == "diffusionnet":
        # DiffusionNet naming: diffusionnet_xyz_w128_b4_e128_lr1e-3
        name = "diffusionnet_{}_w{}_b{}_e{}_lr{:.0e}".format(
            args.input_features, args.C_width, args.N_block, args.num_eigen, args.lr
        )
    elif args.model == "transolver":
        # Transolver naming: transolver_xyz_d128_h8_l4_b64_lr1e-3_pos_share
        name = "transolver_{}_d{}_h{}_l{}_b{}_lr{:.0e}_{}_{}".format(
            args.input_features,
            args.embed_dim,
            args.num_heads,
            args.n_layers,
            args.num_basis,
            args.lr,
        )
    else:
        # Fallback for unknown models
        name = "{}_{}_lr{:.0e}".format(args.model, args.input_features, args.lr)

    # Add dropout if not default
    if args.dropout != 0.1:
        name += "_drop{:.2f}".format(args.dropout)

    # Add timestamp for uniqueness
    name += "_{}".format(timestamp)

    return name


device = "cuda"  # torch.device('cuda:0')
dtype = torch.float32

# problem/dataset things
n_class = 260

# visualization settings
vis_every = args.vis_every
n_vis_samples = args.n_vis_samples

# model
input_features = args.input_features  # one of ['xyz', 'hks']
k_eig = args.num_eigen

# training settings
train = not args.evaluate
n_epoch = args.n_epoch
lr = args.lr
decay_every = args.decay_every
decay_rate = args.decay_rate
augment_random_rotate = input_features == "xyz"

# Generate experiment name
exp_name = generate_exp_name(args)
print(f"Experiment name: {exp_name}")

if args.wandb:
    if args.model == "diffusionnet":
        wandb.init(
            project="rna-segmentation",
            name=exp_name,
            config={
                "input_features": args.input_features,
                "model": args.model,
                "C_width": args.C_width,
                "N_block": args.N_block,
                "vis_every": vis_every,
                "n_vis_samples": n_vis_samples,
                "n_epoch": n_epoch,
                "lr": lr,
                "decay_every": decay_every,
                "decay_rate": decay_rate,
                "scheduler": args.scheduler,
            },
        )
    elif args.model == "funcattn":
        wandb.init(
            project="rna-segmentation",
            name=exp_name,
            config={
                "input_features": args.input_features,
                "model": args.model,
                # Model-specific hyperparameters
                "embed_dim": args.embed_dim,
                "num_heads": args.num_heads,
                "n_layers": args.n_layers,
                "mlp_ratio": args.mlp_ratio,
                "num_basis": args.num_basis,
                "dropout": args.dropout,
                # Training hyperparameters
                "vis_every": vis_every,
                "n_vis_samples": n_vis_samples,
                "n_epoch": n_epoch,
                "lr": lr,
                "decay_every": decay_every,
                "decay_rate": decay_rate,
                "scheduler": args.scheduler,
            },
        )
    elif args.model == "transolver":
        wandb.init(
            project="rna-segmentation",
            name=exp_name,
            config={
                "input_features": args.input_features,
                "model": args.model,
                # Model-specific hyperparameters
                "embed_dim": args.embed_dim,
                "num_heads": args.num_heads,
                "n_layers": args.n_layers,
                "mlp_ratio": args.mlp_ratio,
                "num_basis": args.num_basis,
                # Shared hyperparameters
                "dropout": args.dropout,
                # Training hyperparameters
                "vis_every": vis_every,
                "n_vis_samples": n_vis_samples,
                "n_epoch": n_epoch,
                "lr": lr,
                "decay_every": decay_every,
                "decay_rate": decay_rate,
                "scheduler": args.scheduler,
            },
        )

# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", "op_cache")
pretrain_path = os.path.join(
    base_path, "pretrained_models/rna_mesh_seg_{}_4x128.pth".format(input_features)
)
# save model path by model type and input features
model_save_dir = os.path.join(base_path, "trained_models")
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, f"{exp_name}.pth")
dataset_path = os.path.join(base_path, "data/RNA-Surface-Segmentation-Dataset")


# === Load datasets

# Load the test dataset
test_dataset = RNAPointCloudDataset(
    dataset_path, train=False, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir
)
# test_dataset = RNAPointCloudDataset(dataset_path, train=False, k_eig=k_eig, model_type=args.model, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

# Load the train dataset
if train:
    train_dataset = RNAPointCloudDataset(
        dataset_path, train=True, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir
    )
    # train_dataset = RNAPointCloudDataset(dataset_path, train=True, k_eig=k_eig, model_type=args.model, op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

print(
    f"Number of training samples: {len(train_dataset) if train else 0}"
    f", Number of test samples: {len(test_dataset)}"
)


# === Create the model

C_in = {"xyz": 3, "hks": 16}[input_features]  # dimension of input features

if args.model == "FuncAttn_learned":
    model = models.FuncAttn(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        num_basis=args.num_basis,
        mlp_ratio=args.mlp_ratio,
        act="gelu",
        input_features=input_features,
        num_freqs=10,
        last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
    )
elif args.model == "transolver":
    model = models.Transolver(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        num_basis=args.num_basis,
        mlp_ratio=args.mlp_ratio,
        act="gelu",
        input_features=input_features,
        num_freqs=10,
        last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
    )
else:  # diffusionnet
    model = models.layers.DiffusionNet(
        C_in=C_in,
        C_out=n_class,
        C_width=args.C_width,
        N_block=args.N_block,
        last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
        outputs_at="vertices",
        dropout=True,
    )


model = model.to(device)

num_params = sum(p.numel() for p in model.parameters())
if args.wandb:
    wandb.log({"num_params": num_params}, step=0)
    wandb.watch(model, log="all")
print(f"The model has {num_params} parameters.")

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))


# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# Learning rate scheduler
if args.scheduler == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epoch, eta_min=lr * 0.01
    )
elif args.scheduler == "onecycle":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=n_epoch, pct_start=0.3, final_div_factor=1000
    )
else:  # 'step'
    scheduler = None


def train_epoch(epoch):

    # Implement lr decay
    if args.scheduler == "step":
        if epoch > 0 and epoch % decay_every == 0:
            global lr
            lr *= decay_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()

    correct = 0
    total_num = 0
    total_loss = 0.0
    for data in tqdm(train_loader):

        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

        # Move to device
        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        labels = labels.to(device)

        # Randomly rotate positions
        if augment_random_rotate:
            verts = models.utils.random_rotate_points(verts)

        # Construct features
        if input_features == "xyz":
            features = verts
        elif input_features == "hks":
            features = models.geometry.compute_hks_autoscale(evals, evecs, 16)

        if args.model == "pointnet2":
            xyz_input = verts.T.unsqueeze(0)  # (1, 3, N)
            pointnet_input = torch.cat(
                [xyz_input, xyz_input, xyz_input], dim=1
            )  # (1, 9, N)
            preds, _ = model(pointnet_input)
            preds = preds.squeeze(0)  # (N, n_class)
        # Apply the model
        elif args.model == "diffusionnet":
            preds = model(
                features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY
            )
        else:
            preds = model(
                features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY
            )

        # Evaluate loss
        loss = torch.nn.functional.nll_loss(preds, labels)
        loss.backward()

        # track accuracy
        pred_labels = torch.max(preds, dim=1).indices
        this_correct = pred_labels.eq(labels).sum().item()
        this_num = labels.shape[0]
        correct += this_correct
        total_num += this_num
        total_loss += loss.item() * this_num

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    train_acc = correct / total_num
    train_loss = total_loss / total_num
    return train_acc, train_loss


# Do an evaluation pass on the test dataset
def test():

    model.eval()

    correct = 0
    total_num = 0
    total_loss = 0.0
    with torch.no_grad():

        for data in tqdm(test_loader):

            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)

            # Construct features
            if input_features == "xyz":
                features = verts
            elif input_features == "hks":
                features = models.geometry.compute_hks_autoscale(
                    evals, evecs, 16
                )

            if args.model == "pointnet2":
                xyz_input = verts.T.unsqueeze(0)  # (1, 3, N)
                pointnet_input = torch.cat(
                    [xyz_input, xyz_input, xyz_input], dim=1
                )  # (1, 9, N)
                preds, _ = model(pointnet_input)
                preds = preds.squeeze(0)  # (N, n_class)
            else:
                # Apply the model
                preds = model(
                    features,
                    mass,
                    L=L,
                    evals=evals,
                    evecs=evecs,
                    gradX=gradX,
                    gradY=gradY,
                )

            loss = torch.nn.functional.nll_loss(preds, labels)
            total_loss += loss.item() * labels.shape[0]
            # track accuracy
            pred_labels = torch.max(preds, dim=1).indices
            this_correct = pred_labels.eq(labels).sum().item()
            this_num = labels.shape[0]
            correct += this_correct
            total_num += this_num

    test_acc = correct / total_num
    test_loss = total_loss / total_num
    return test_acc, test_loss


if train:
    print("Training...")

    for epoch in range(n_epoch):
        train_acc, train_loss = train_epoch(epoch)
        test_acc, test_loss = test()

        # Step scheduler (for cosine and onecycle)
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = lr

        if args.wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                    "learning_rate": current_lr,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                },
                step=epoch,
            )

        print(
            "Epoch {} - Train overall: {:06.3f}% Loss: {:.4f}  Test overall: {:06.3f}% Loss: {:.4f}".format(
                epoch, 100 * train_acc, train_loss, 100 * test_acc, test_loss
            )
        )

        if args.wandb and (epoch % vis_every == 0 or epoch == n_epoch - 1):
            print(f"Generating visualizations for epoch {epoch}...")
            log_point_cloud_segmentation_visualizations(
                args,
                model=model,
                test_loader=test_loader,
                device=device,
                input_features=input_features,
                n_samples=n_vis_samples,
                n_classes=n_class,
                epoch=epoch,
            )

    print(" ==> saving last model to " + model_save_path)
    torch.save(model.state_dict(), model_save_path)


# Test
test_acc, test_loss = test()
print("Overall test accuracy: {:06.3f}% Loss: {:.4f}".format(100 * test_acc, test_loss))

if args.wandb:
    print("Generating final visualizations...")
    log_point_cloud_segmentation_visualizations(
        args,
        model=model,
        test_loader=test_loader,
        device=device,
        input_features=input_features,
        n_samples=n_vis_samples,
        n_classes=n_class,
        epoch=n_epoch if train else 0,
    )
