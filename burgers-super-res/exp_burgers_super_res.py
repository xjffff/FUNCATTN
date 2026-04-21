"""
Zero-shot super-resolution for Burgers equation.
Train on 2048-pt grid (subsample=4), evaluate on 8192-pt grid (subsample=1).
Mirrors galerkin-transformer/examples/ex1_burgers_super_res.py.

python exp_burgers_super_res.py --data-path /path/to/burgers_data_R10.mat
"""
import argparse
import os
from datetime import date

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from model import SimpleTransformer
from utils import (
    BurgersDataset,
    WeightedL2Loss,
    run_train,
    train_batch_burgers,
    validate_epoch_burgers,
    get_num_params,
)

SEED = 1127802


def get_seed(s):
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)



DEFAULT_CONFIG = dict(
    node_feats=1,
    edge_feats=None,
    pos_dim=1,
    n_targets=1,
    n_hidden=92,
    num_feat_layers=0,
    num_encoder_layers=4,
    n_head=1,
    pred_len=0,
    n_freq_targets=0,
    dim_feedforward=200,
    feat_extract_type=None,
    attention_type='functional',
    xavier_init=0.001,
    diagonal_weight=0.01,
    symmetric_init=False,
    layer_norm=False,
    attn_norm=False,
    batch_norm=False,
    spacial_residual=False,
    return_attn_weight=False,
    return_latent=False,
    residual_type='plus',
    seq_len=None,
    bulk_regression=False,
    decoder_type='ifft',
    freq_dim=48,
    num_regressor_layers=2,
    fourier_modes=16,
    spacial_dim=1,
    dropout=0.0,                                                                                                                                                                                            
    encoder_dropout=0.0,
    ffn_dropout=0.0,                                                                                                                                                                                        
    decoder_dropout=0.0,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Burgers zero-shot super-resolution')
    parser.add_argument('--data-path', type=str, required=True,
                        help='path to burgers_data_R10.mat')
    parser.add_argument('--model-save-path', type=str, default='./models',
                        help='directory to save checkpoints (default: ./models)')
    parser.add_argument('--subsample', type=int, default=4,
                        help='train grid subsampling factor (default: 4 → 2048 pts)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='train batch size (default: 8)')
    parser.add_argument('--val-batch-size', type=int, default=4,
                        help='validation batch size (default: 4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='max learning rate for OneCycleLR (default: 1e-3)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='gradient regularizer strength (default: 0.1)')
    parser.add_argument('--layer-norm', action='store_true', default=False,
                        help='use LayerNorm (default: uses instance/attn norm)')
    parser.add_argument('--show-batch', action='store_true', default=False,
                        help='show per-batch progress bar instead of per-epoch')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'random seed (default: {SEED})')
    return parser.parse_args()


def main():
    args = parse_args()
    get_seed(args.seed)

    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    loader_kwargs = {'pin_memory': True} if cuda else {}

    train_dataset = BurgersDataset(
        subsample=args.subsample,
        train_data=True,
        train_portion=0.5,
        data_path=args.data_path,
    )
    valid_dataset = BurgersDataset(
        subsample=1,
        train_data=False,
        valid_portion=100,
        data_path=args.data_path,
    )
    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size,
                              shuffle=False, drop_last=False, **loader_kwargs)

    sample = next(iter(train_loader))
    print('=' * 20, 'Data loader batch', '=' * 20)
    for key, val in sample.items():
        print(key, "\t", val.shape)
    print('=' * 59)

    config = dict(DEFAULT_CONFIG)
    config['attn_norm'] = not args.layer_norm
    config['layer_norm'] = args.layer_norm

    get_seed(args.seed)
    torch.cuda.empty_cache()
    model = SimpleTransformer(**config).to(device)
    print(f"Model: {model.__class__.__name__}  |  Params: {get_num_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        div_factor=1e4,
        final_div_factor=1e4,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
    )

    h_train = (1 / 8192) * args.subsample
    h_eval  = 1 / 8192

    loss_func   = WeightedL2Loss(regularizer=True,  h=h_train, gamma=args.gamma)
    metric_func = WeightedL2Loss(regularizer=False, h=h_eval)

    suffix = str(date.today())
    model_name  = f"burgers_super_res_{suffix}.pt"
    result_name = f"burgers_super_res_{suffix}.pkl"
    tqdm_mode   = 'epoch' if not args.show_batch else 'batch'

    result = run_train(
        model, loss_func, metric_func,
        train_loader, valid_loader,
        optimizer, scheduler,
        train_batch=train_batch_burgers,
        validate_epoch=validate_epoch_burgers,
        epochs=args.epochs,
        patience=None,
        tqdm_mode=tqdm_mode,
        model_save_path=args.model_save_path,
        model_name=model_name,
        result_name=result_name,
        device=device,
    )

    model.load_state_dict(
        torch.load(os.path.join(args.model_save_path, model_name), map_location=device))
    model.eval()
    val_result = validate_epoch_burgers(model, metric_func, valid_loader, device)
    print(f"\nBest model val metric: {val_result['metric']:.4e}")

    return result


if __name__ == '__main__':
    main()
