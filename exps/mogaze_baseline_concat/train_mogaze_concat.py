"""
Training Script: Concatenation Fusion Ablation Study

Runs all 6 LOPOCV folds sequentially without intervention.
Each fold trains for 80k iterations with cosine annealing + warmup.
Models saved to: log_concat/snapshot/fold_<participant>/

Usage:
    python train_mogaze_concat.py
    python train_mogaze_concat.py --seed 999
    python train_mogaze_concat.py --folds p1 p2   # run specific folds only
"""

import argparse
import os, sys
import math
import numpy as np
import copy
import time
import gc

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader

from mogaze_config_concat import config
from concat_model import SiMLPeConcatFusion
from mogaze_dataset_concat import MoGazeConcatDataset, ALL_PARTICIPANTS


# ─── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', type=int, default=999, help='random seed')
parser.add_argument('--folds', nargs='+', default=None,
                    help='specific folds to run (e.g., --folds p1 p2). Default: all 6')
parser.add_argument('--num', type=int, default=48, help='number of MLP blocks')
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--iters', type=int, default=80000, help='training iterations per fold')
args = parser.parse_args()


# ─── Utilities ──────────────────────────────────────────────────────────────────
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def update_lr_cosine(nb_iter, total_iter, max_lr, min_lr, optimizer, warmup_iters):
    """Cosine annealing with linear warmup — same schedule as gated model."""
    if nb_iter < warmup_iters:
        lr = min_lr + (max_lr - min_lr) * (nb_iter / warmup_iters)
    else:
        progress = (nb_iter - warmup_iters) / (total_iter - warmup_iters)
        lr = min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return optimizer, lr


# ─── Training step ──────────────────────────────────────────────────────────────
def train_step(motion_input, motion_target, model, optimizer, nb_iter,
               dct_m, idct_m, config_ref):
    """
    Single training step — identical pipeline to gated model training:
      DCT → model → IDCT → residual offset → position loss + velocity loss
    """
    # DCT on input
    if config_ref.deriv_input:
        motion_input_ = motion_input.clone()
        motion_input_ = torch.matmul(
            dct_m[:, :, :config_ref.motion.h36m_input_length],
            motion_input_.cuda()
        )
    else:
        motion_input_ = motion_input.clone().cuda()

    # Forward pass
    motion_pred = model(motion_input_.cuda())

    # IDCT
    motion_pred = torch.matmul(
        idct_m[:, :config_ref.motion.h36m_input_length, :],
        motion_pred
    )

    # Residual offset (pose only — first 63 dims of last input frame)
    if config_ref.deriv_output:
        offset = motion_input[:, -1:, :63].cuda()
        motion_pred = motion_pred[:, :config_ref.motion.h36m_target_length_train] + offset
    else:
        motion_pred = motion_pred[:, :config_ref.motion.h36m_target_length_train]

    # ── Loss computation ──
    b, n, _ = motion_target.shape
    target_pose = motion_target[:, :, :63].cuda()  # pose component of target

    joints = 21
    pred_joints = motion_pred.reshape(b, n, joints, 3).reshape(-1, 3)
    target_joints = target_pose.reshape(b, n, joints, 3).reshape(-1, 3)

    # Position loss (MPJPE)
    pose_loss = torch.mean(torch.norm(pred_joints - target_joints, 2, 1))

    # Velocity loss
    if config_ref.use_relative_loss:
        pred_full = motion_pred.reshape(b, n, joints, 3)
        target_full = target_pose.reshape(b, n, joints, 3)
        dpred = pred_full[:, 1:] - pred_full[:, :-1]
        dtarget = target_full[:, 1:] - target_full[:, :-1]
        vel_loss = torch.mean(torch.norm((dpred - dtarget).reshape(-1, 3), 2, 1))
        total_loss = pose_loss + vel_loss
    else:
        total_loss = pose_loss

    # Backward + optimize
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


# ─── Train one fold ─────────────────────────────────────────────────────────────
def train_fold(test_participant, fold_config, seed):
    """Train a single LOPOCV fold."""

    print(f"\n{'='*80}")
    print(f"  FOLD: hold out {test_participant}  |  seed={seed}")
    print(f"{'='*80}")

    # Deterministic setup
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Snapshot directory for this fold
    fold_snapshot_dir = os.path.join(
        fold_config.log_dir, 'snapshot', f'fold_{test_participant}'
    )
    ensure_dir(fold_snapshot_dir)

    # DCT matrices
    dct_m, idct_m = get_dct_matrix(fold_config.motion.h36m_input_length_dct)
    dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

    # Model
    model = SiMLPeConcatFusion(fold_config)
    model.train()
    model.cuda()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.4f}M")

    # Dataset
    dataset = MoGazeConcatDataset(fold_config, test_participant=test_participant,
                                  data_aug=fold_config.data_aug)
    dataloader = DataLoader(dataset, batch_size=fold_config.batch_size,
                            num_workers=fold_config.num_workers, drop_last=True,
                            shuffle=True, pin_memory=True)

    print(f"Train samples: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=fold_config.cos_lr_min,
                                 weight_decay=fold_config.weight_decay)

    # ── Training loop ──
    total_iters = fold_config.cos_lr_total_iters
    nb_iter = 0
    avg_loss = 0.0
    fold_start_time = time.time()

    print(f"Training for {total_iters} iterations...")
    print(f"LR schedule: warmup {fold_config.cos_lr_warmup_iters} iters, "
          f"then cosine {fold_config.cos_lr_max:.2e} -> {fold_config.cos_lr_min:.2e}")

    while nb_iter < total_iters:
        for motion_input, motion_target in dataloader:

            loss = train_step(motion_input, motion_target, model, optimizer,
                              nb_iter, dct_m, idct_m, fold_config)

            # Update LR
            optimizer, current_lr = update_lr_cosine(
                nb_iter, total_iters, fold_config.cos_lr_max, fold_config.cos_lr_min,
                optimizer, fold_config.cos_lr_warmup_iters
            )

            avg_loss += loss
            nb_iter += 1

            # Print progress
            if nb_iter % fold_config.print_every == 0:
                avg_loss /= fold_config.print_every
                elapsed = time.time() - fold_start_time
                eta = elapsed / nb_iter * (total_iters - nb_iter)

                print(f"  [{test_participant}] Iter {nb_iter}/{total_iters} | "
                      f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                      f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
                avg_loss = 0.0

            # Save checkpoint
            if nb_iter % fold_config.save_every == 0:
                ckpt_path = os.path.join(fold_snapshot_dir, f'model-iter-{nb_iter}.pth')
                torch.save(model.state_dict(), ckpt_path)

            if nb_iter >= total_iters:
                break

    # Save final model
    final_path = os.path.join(fold_snapshot_dir, 'model-final.pth')
    torch.save(model.state_dict(), final_path)

    elapsed = time.time() - fold_start_time
    print(f"  [{test_participant}] Training complete in {elapsed/60:.1f} minutes")
    print(f"  [{test_participant}] Final model: {final_path}")

    # Cleanup GPU memory
    del model, optimizer, dataset, dataloader, dct_m, idct_m
    torch.cuda.empty_cache()
    gc.collect()

    return final_path


# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Determine which folds to run
    if args.folds:
        folds = args.folds
        for f in folds:
            assert f in ALL_PARTICIPANTS, f"Unknown participant: {f}. Valid: {ALL_PARTICIPANTS}"
    else:
        folds = ALL_PARTICIPANTS

    # Apply CLI overrides to config
    config.motion_mlp.num_layers = args.num
    config.batch_size = args.batch_size
    config.cos_lr_total_iters = args.iters
    # Keep these at defaults (spatial FC, with normalization, etc.)
    config.motion_fc_in.temporal_fc = False
    config.motion_fc_out.temporal_fc = False
    config.motion_mlp.with_normalization = True
    config.motion_mlp.norm_axis = 'spatial'
    config.motion_mlp.spatial_fc_only = False

    print("=" * 80)
    print("  CONCATENATION FUSION ABLATION — LOPOCV TRAINING")
    print("=" * 80)
    print(f"Folds to run:      {folds}")
    print(f"Seed:              {args.seed}")
    print(f"MLP blocks:        {config.motion_mlp.num_layers}")
    print(f"Iterations/fold:   {config.cos_lr_total_iters}")
    print(f"Batch size:        {config.batch_size}")
    print(f"LR:                {config.cos_lr_max:.2e} -> {config.cos_lr_min:.2e}")
    print(f"Architecture:      Input[66] -> FC[66->63] -> 48xMLP[63] -> FC[63->63] -> Output[63]")
    print(f"Output dir:        {config.log_dir}")
    print("=" * 80)

    all_start = time.time()
    fold_results = {}

    for fold_idx, test_p in enumerate(folds):
        print(f"\n>>> Starting fold {fold_idx+1}/{len(folds)}: hold out {test_p}")

        final_path = train_fold(test_p, config, args.seed)
        fold_results[test_p] = final_path

    total_time = time.time() - all_start
    print(f"\n{'='*80}")
    print(f"  ALL FOLDS COMPLETE — Total time: {total_time/3600:.1f} hours")
    print(f"{'='*80}")

    for test_p, path in fold_results.items():
        print(f"  Fold {test_p}: {path}")

    print(f"\nNext: Run test_mogaze_concat.py to evaluate all folds and compute weighted MPJPE")


if __name__ == "__main__":
    main()
