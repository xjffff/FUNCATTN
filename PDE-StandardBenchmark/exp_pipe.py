import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import *
from utils.testloss import TestLoss
from einops import rearrange
from model_dict import get_model
from utils.normalizer import UnitTransformer
from utils.experiment import (
    seed_everything,
    generate_seed,
    ExperimentLogger,
    TrainingTimer,
    get_memory_usage
)

parser = argparse.ArgumentParser('Training Transformer')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='FuncAttn_2D')
parser.add_argument('--n-hidden', type=int, default=128, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=8, help='layers')
parser.add_argument('--n-heads', type=int, default=8, help='attention heads')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=0.5, help='gradient clipping')
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--basis_num', type=int, default=64, help='number of basis')
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='pipe_FuncAttn')
parser.add_argument('--data_path', type=str, default='/home/stud/xjie/storage/user/Transolver/PDE-Solving-StandardBenchmark/dataset/pipe')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

ntrain = args.ntrain
ntest = 200
eval = args.eval
save_name = args.save_name

# PIPE grid: 129 x 129 uniform grid
S1 = 129
S2 = 129


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    seed = generate_seed()
    seed_everything(seed)

    # ===================== Data Loading =====================
    INPUT_X = os.path.join(args.data_path, 'Pipe_X.npy')
    INPUT_Y = os.path.join(args.data_path, 'Pipe_Y.npy')
    OUTPUT_Q = os.path.join(args.data_path, 'Pipe_Q.npy')

    meshX = np.load(INPUT_X)
    meshY = np.load(INPUT_Y)
    Q = np.load(OUTPUT_Q)[:, 0]

    meshX = torch.tensor(meshX, dtype=torch.float)
    meshY = torch.tensor(meshY, dtype=torch.float)
    output = torch.tensor(Q, dtype=torch.float)

    fx_all = torch.stack([meshX, meshY], dim=-1)

    grid_x = torch.linspace(0, 1, S1)
    grid_y = torch.linspace(0, 1, S2)
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
    pos_single = torch.stack([grid_x, grid_y], dim=-1)  # (129, 129, 2)
    pos_all = pos_single.unsqueeze(0).expand(meshX.shape[0], -1, -1, -1)  # (2310, 129, 129, 2)

    N_total = meshX.shape[0]
    print(f"pos: {pos_all.shape}, fx: {fx_all.shape}, output: {output.shape}")

    # ===================== Train/Test Split =====================
    # Source: train = [:1000], val = [-200:]
    pos_train = pos_all[:ntrain].reshape(ntrain, -1, 2)
    fx_train = fx_all[:ntrain].reshape(ntrain, -1, 2)
    y_train = output[:ntrain].reshape(ntrain, -1)

    pos_test = pos_all[-ntest:].reshape(ntest, -1, 2)
    fx_test = fx_all[-ntest:].reshape(ntest, -1, 2)
    y_test = output[-ntest:].reshape(ntest, -1)

    print(f"Train: pos {pos_train.shape}, fx {fx_train.shape}, y {y_train.shape}")
    print(f"Test:  pos {pos_test.shape}, fx {fx_test.shape}, y {y_test.shape}")

    g = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos_train, fx_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=g
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos_test, fx_test, y_test),
        batch_size=args.batch_size,
        shuffle=False
    )

    print("Dataloading is over.")

    # ===================== Model =====================
    model = get_model(args).Model(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        dropout=args.dropout,
        n_head=args.n_heads,
        Time_Input=False,
        mlp_ratio=args.mlp_ratio,
        fun_dim=2,
        out_dim=1,
        basis_num=args.basis_num,
        ref=args.ref,
        unified_pos=args.unified_pos,
        H=S1,
        W=S2
    ).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    print(args)
    print(model)
    total_params = count_parameters(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )
    myloss = TestLoss(size_average=False)

    run_name = f"{args.save_name}_basis{args.basis_num}_seed{seed}"

    config = {
        'lr': args.lr,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'basis_num': args.basis_num,
        'seed': seed,
        'total_params': total_params,
    }

    logger = ExperimentLogger(
        project_name="Pipe",
        run_name=run_name,
        config=config,
        tags=[args.model],
        enabled=not args.eval
    )

    # ===================== Eval Mode =====================
    if eval:
        print("model evaluation")
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"), strict=False)
        model.eval()

        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        rel_err = 0.0
        showcase = 10
        id = 0

        with torch.no_grad():
            for pos, fx, y in test_loader:
                id += 1
                pos, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                out = model(pos, fx).squeeze(-1)

                tl = myloss(out, y).item()
                rel_err += tl

                if id < showcase:
                    print(id)
                    # Prediction
                    plt.figure()
                    plt.axis('off')
                    plt.pcolormesh(
                        pos[0, :, 0].reshape(S1, S2).detach().cpu().numpy(),
                        pos[0, :, 1].reshape(S1, S2).detach().cpu().numpy(),
                        out[0, :].reshape(S1, S2).detach().cpu().numpy(),
                        shading='auto',
                        cmap='coolwarm'
                    )
                    plt.colorbar()
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_pred.pdf"),
                        bbox_inches='tight', pad_inches=0
                    )
                    plt.close()

                    # Ground truth
                    plt.figure()
                    plt.axis('off')
                    plt.pcolormesh(
                        pos[0, :, 0].reshape(S1, S2).detach().cpu().numpy(),
                        pos[0, :, 1].reshape(S1, S2).detach().cpu().numpy(),
                        y[0, :].reshape(S1, S2).detach().cpu().numpy(),
                        shading='auto',
                        cmap='coolwarm'
                    )
                    plt.colorbar()
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_gt.pdf"),
                        bbox_inches='tight', pad_inches=0
                    )
                    plt.close()

                    # Error
                    plt.figure()
                    plt.axis('off')
                    plt.pcolormesh(
                        pos[0, :, 0].reshape(S1, S2).detach().cpu().numpy(),
                        pos[0, :, 1].reshape(S1, S2).detach().cpu().numpy(),
                        (out[0, :] - y[0, :]).reshape(S1, S2).detach().cpu().numpy(),
                        shading='auto',
                        cmap='coolwarm'
                    )
                    plt.colorbar()
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_error.pdf"),
                        bbox_inches='tight', pad_inches=0
                    )
                    plt.close()

        rel_err /= ntest
        print("rel_err:{}".format(rel_err))

    # ===================== Training Loop =====================
    else:
        timer = TrainingTimer()

        for ep in range(args.epochs):
            timer.start_epoch()
            torch.cuda.reset_peak_memory_stats()

            model.train()
            train_loss = 0

            for pos, fx, y in train_loader:
                pos, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                optimizer.zero_grad()

                out = model(pos, fx).squeeze(-1)

                loss = myloss(out, y)
                loss.backward()

                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                train_loss += loss.item()
                scheduler.step()

            epoch_time = timer.end_epoch()
            max_mem = get_memory_usage()

            train_loss /= ntrain

            model.eval()
            rel_err = 0.0

            with torch.no_grad():
                for pos, fx, y in test_loader:
                    pos, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                    out = model(pos, fx).squeeze(-1)

                    tl = myloss(out, y).item()
                    rel_err += tl

            rel_err /= ntest

            print("Epoch {} , train_loss:{:.5f} , test_loss:{:.5f}".format(ep, train_loss, rel_err))

            logger.log({
                'epoch': ep,
                'train/l2_loss': train_loss,
                'test/rel_err': rel_err,
                'lr': optimizer.param_groups[0]['lr'],
                'epoch_time_sec': epoch_time,
                'max_memory_MB': max_mem
            }, step=ep)

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))

        logger.log_summary({
            'final_test_rel_err': rel_err,
            'total_train_time_sec': timer.total_time,
            'avg_epoch_time_sec': timer.get_avg_epoch_time(args.epochs),
        })

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))

        logger.finish()


if __name__ == "__main__":
    main()