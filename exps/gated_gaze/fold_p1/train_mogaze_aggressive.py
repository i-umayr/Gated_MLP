"""Train the gated MLP with the aggressive gate-opening schedule on one MoGaze LOPOCV fold."""
import argparse
import os, sys
import json
import math
import numpy as np
import copy

# Required for deterministic cuBLAS
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sys.path.insert(0, os.path.dirname(__file__))

from mogaze_config_gated import config
from gated_attention_model import SiMLPeWithGatedGaze, create_gated_model
from mogaze_dataset_gaze import MoGazeGazeDataset, MoGazeGazeEval

from aggressive_gate_trainer import AggressiveGateTrainer

# Logging helpers from lib/utils, with a local fallback
try:
    from utils.logger import get_logger, print_and_log_info
    from utils.pyt_utils import link_file, ensure_dir
except ImportError:
    def get_logger(path, name):
        import logging
        logger = logging.getLogger(name)
        handler = logging.FileHandler(path, encoding='utf-8')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def print_and_log_info(logger, message):
        print(message)
        if logger:
            try:
                logger.info(message)
            except UnicodeEncodeError:
                import re
                clean_message = re.sub(r'[^\x00-\x7F]+', '', message)
                logger.info(clean_message)

    def link_file(src, dst): pass
    def ensure_dir(path): os.makedirs(path, exist_ok=True)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default='aggressive_gate_learning.txt', help='experiment name')
parser.add_argument('--seed', type=int, default=999, help='random seed')
parser.add_argument('--temporal-only', action='store_true', help='temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='layernorm axis')
parser.add_argument('--with-normalization', action='store_true', help='use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='use only spatial fc')
parser.add_argument('--num', type=int, default=48, help='num of blocks')
parser.add_argument('--pretrained-baseline', type=str,
                   default=None,
                   help='path to pretrained baseline model')
parser.add_argument('--gate-lr-multiplier', type=float, default=10.0,
                   help='maximum gate learning rate multiplier')
parser.add_argument('--gate-reg-weight', type=float, default=0.1,
                   help='gate regularization weight')

args = parser.parse_args()

torch.use_deterministic_algorithms(True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

writer = SummaryWriter(log_dir=f'runs/aggressive_gate_{args.seed}')

try:
    acc_log = open(args.exp_name, 'a', encoding='utf-8')
except:
    acc_log = open('aggressive_gate_log.txt', 'a', encoding='utf-8')

config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num

config.cos_lr_total_iters = 80000
config.cos_lr_warmup_iters = 2000

acc_log.write('Gated MLP - aggressive gate-opening schedule\n')
acc_log.write(f'Seed: {args.seed}\n')
acc_log.write(f'Max gate LR multiplier: {args.gate_lr_multiplier}, gate regularisation weight: {args.gate_reg_weight}\n')

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

dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def update_lr_cosine_annealing_aggressive(nb_iter, total_iter, max_lr, min_lr, optimizer,
                                          warmup_iters, aggressive_trainer=None):
    """
    Linear warmup from min_lr to max_lr over warmup_iters, then cosine decay to
    min_lr. The gate parameter group gets the base LR times the schedule's gate
    LR multiplier.
    """

    if nb_iter < warmup_iters:
        base_lr = min_lr + (max_lr - min_lr) * (nb_iter / warmup_iters)
    else:
        progress = (nb_iter - warmup_iters) / (total_iter - warmup_iters)
        base_lr = min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    if aggressive_trainer:
        gate_lr_multiplier = aggressive_trainer.get_gate_learning_rate_multiplier()

        for param_group in optimizer.param_groups:
            if 'gaze_gate' in param_group.get('name', ''):
                param_group["lr"] = base_lr * gate_lr_multiplier
            else:
                param_group["lr"] = base_lr
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr

    return optimizer, base_lr

def train_step_aggressive(mogaze_motion_input, mogaze_motion_target, model, optimizer, nb_iter,
                         total_iter, max_lr, min_lr, aggressive_trainer=None):
    """One optimisation step."""

    if config.deriv_input:
        b, n, c = mogaze_motion_input.shape
        mogaze_motion_input_ = mogaze_motion_input.clone()
        mogaze_motion_input_ = torch.matmul(
            dct_m[:, :, :config.motion.h36m_input_length],
            mogaze_motion_input_.cuda()
        )
    else:
        mogaze_motion_input_ = mogaze_motion_input.clone()

    # Attention outputs (for the alignment loss) are needed after the baseline stage
    if aggressive_trainer and aggressive_trainer.current_stage != 'baseline':
        motion_pred, attention_info = model(
            mogaze_motion_input_.cuda(),
            return_attention=True
        )
    else:
        motion_pred = model(mogaze_motion_input_.cuda())
        attention_info = None

    motion_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], motion_pred)

    if config.deriv_output:
        # Residual on the pose part of the last observed frame
        offset = mogaze_motion_input[:, -1:, :63].cuda()
        motion_pred = motion_pred[:, :config.motion.h36m_target_length_train] + offset
    else:
        motion_pred = motion_pred[:, :config.motion.h36m_target_length_train]

    if aggressive_trainer:
        loss, loss_info = aggressive_trainer.compute_aggressive_loss(
            motion_pred, mogaze_motion_target, attention_info
        )
    else:
        # Position + velocity loss without the schedule
        b, n, c_target = mogaze_motion_target.shape
        mogaze_motion_target_pose = mogaze_motion_target[:, :, :63].cuda()

        joints = 21
        motion_pred_joints = motion_pred.reshape(b, n, joints, 3).reshape(-1, 3)
        mogaze_motion_target_joints = mogaze_motion_target_pose.reshape(b, n, joints, 3).reshape(-1, 3)

        pose_loss = torch.mean(torch.norm(motion_pred_joints - mogaze_motion_target_joints, 2, 1))

        if config.use_relative_loss:
            motion_pred_full = motion_pred.reshape(b, n, joints, 3)
            dmotion_pred = motion_pred_full[:, 1:] - motion_pred_full[:, :-1]
            motion_target_full = mogaze_motion_target_pose.reshape(b, n, joints, 3)
            dmotion_target = motion_target_full[:, 1:] - motion_target_full[:, :-1]
            velocity_loss = torch.mean(torch.norm((dmotion_pred - dmotion_target).reshape(-1, 3), 2, 1))
            pose_loss = pose_loss + velocity_loss

        loss = pose_loss
        loss_info = {
            'motion_loss': pose_loss.item(),
            'total_loss': loss.item(),
            'gate_weight': model.get_gate_weight()
        }

    writer.add_scalar('Loss/Total', loss_info['total_loss'], nb_iter)
    writer.add_scalar('Loss/Motion', loss_info['motion_loss'], nb_iter)
    writer.add_scalar('Gate/Weight', loss_info['gate_weight'], nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    optimizer, base_lr = update_lr_cosine_annealing_aggressive(
        nb_iter, total_iter, max_lr, min_lr, optimizer,
        config.cos_lr_warmup_iters, aggressive_trainer
    )

    writer.add_scalar('LR/Base', base_lr, nb_iter)
    if aggressive_trainer:
        gate_lr = base_lr * aggressive_trainer.get_gate_learning_rate_multiplier()
        writer.add_scalar('LR/Gate', gate_lr, nb_iter)

    return loss_info, optimizer, base_lr

def main():
    pretrained_model = None
    if args.pretrained_baseline:
        print(f"Loading pretrained baseline from {args.pretrained_baseline}")
        from model import siMLPe as Model
        pretrained_model = Model(config)
        pretrained_model.load_state_dict(torch.load(args.pretrained_baseline))

    model = create_gated_model(config, pretrained_model)
    model.train()
    model.cuda()

    total_params = sum(p.numel() for p in model.parameters()) / 1000000.0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0

    print(f"Gated MLP: {total_params:.2f}M parameters ({trainable_params:.2f}M trainable)")

    config.motion.h36m_target_length = config.motion.h36m_target_length_train
    dataset = MoGazeGazeDataset(config, 'train', config.data_aug)

    dataloader = DataLoader(dataset, batch_size=config.batch_size,
                            num_workers=config.num_workers, drop_last=True,
                            shuffle=True, pin_memory=True)

    aggressive_trainer = AggressiveGateTrainer(model, config, logger=print)

    # Gate LR multiplier decays from args.gate_lr_multiplier to half of it
    if args.gate_lr_multiplier is not None:
        aggressive_trainer.max_gate_lr_multiplier = args.gate_lr_multiplier
        aggressive_trainer.initial_gate_lr_multiplier = args.gate_lr_multiplier * 0.5
    if args.gate_reg_weight is not None:
        aggressive_trainer.gate_regularization_weight = args.gate_reg_weight

    aggressive_trainer.prepare_model_for_stage()

    for stage_name, stage_info in aggressive_trainer.stages.items():
        print(f"Stage {stage_name} ({stage_info['iterations']} iterations): {stage_info['description']}")

    # The gate gets its own parameter group (separate learning rate)
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'gaze_gate' not in n], 'name': 'main'},
        {'params': [model.gaze_gate], 'name': 'gaze_gate'}
    ]

    optimizer = torch.optim.Adam(param_groups,
                                lr=config.cos_lr_min,  # warmup starts at min_lr
                                weight_decay=config.weight_decay)

    ensure_dir(config.snapshot_dir.replace('log_gated', 'log_aggressive'))
    aggressive_snapshot_dir = config.snapshot_dir.replace('log_gated', 'log_aggressive')
    logger = get_logger(config.log_file.replace('log_gated', 'log_aggressive'), 'aggressive_train')

    print_and_log_info(logger, "Training gated MLP (aggressive schedule)")
    print_and_log_info(logger, f"Parameters: {total_params:.2f}M | seed: {args.seed}")
    print_and_log_info(logger, f"LR: warmup {config.cos_lr_warmup_iters} iterations, cosine decay "
                               f"{config.cos_lr_max:.1e} -> {config.cos_lr_min:.1e}")

    ##### ------ training ------- #####
    nb_iter = 0
    avg_loss = 0.0
    avg_lr = 0.0
    best_gate_weight = 0.0
    best_performance = float('inf')

    print_and_log_info(logger, f"Total iterations: {config.cos_lr_total_iters}")

    # Gate values reached during training (stored in the checkpoints)
    gate_opening_milestones = [0.1, 0.2, 0.3, 0.4, 0.5]
    achieved_milestones = []

    while (nb_iter + 1) < config.cos_lr_total_iters:

        for (mogaze_motion_input, mogaze_motion_target) in dataloader:

            loss_info, optimizer, current_lr = train_step_aggressive(
                mogaze_motion_input, mogaze_motion_target, model, optimizer,
                nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min,
                aggressive_trainer
            )

            avg_loss += loss_info['total_loss']
            avg_lr += current_lr

            current_gate = loss_info['gate_weight']
            for milestone in gate_opening_milestones:
                if milestone not in achieved_milestones and current_gate >= milestone:
                    achieved_milestones.append(milestone)
                    print_and_log_info(logger, f"Gate reached {milestone:.1f} (current: {current_gate:.4f})")

            if (nb_iter + 1) % config.print_every == 0:
                avg_loss = avg_loss / config.print_every
                avg_lr = avg_lr / config.print_every

                aggressive_trainer.log_aggressive_progress(loss_info, writer, nb_iter + 1)

                summary = aggressive_trainer.get_aggressive_summary()

                print_and_log_info(logger, f"Iter {nb_iter + 1} | stage: {summary['current_stage']} | "
                                           f"progress: {summary['total_progress']}")
                print_and_log_info(logger, f"   gate: {summary['current_gate_weight']:.4f} (best: {summary['best_gate_weight']:.4f}) | "
                                           f"gate LR x{summary['gate_learning_rate_multiplier']:.1f}")
                print_and_log_info(logger, f"   avg loss: {avg_loss:.4f} | base LR: {avg_lr:.2e}")

                if nb_iter < config.cos_lr_warmup_iters:
                    phase = "WARMUP"
                    progress_pct = (nb_iter / config.cos_lr_warmup_iters) * 100
                else:
                    phase = "COSINE DECAY"
                    progress_pct = ((nb_iter - config.cos_lr_warmup_iters) /
                              (config.cos_lr_total_iters - config.cos_lr_warmup_iters)) * 100

                print_and_log_info(logger, f"   LR phase: {phase} ({progress_pct:.1f}% complete)")

                if 'gate_regularization_loss' in loss_info and loss_info['gate_regularization_loss'] > 0:
                    print_and_log_info(logger, f"   gate regularisation loss: {loss_info['gate_regularization_loss']:.4f}")

                if summary['gate_stuck_counter'] > 50:
                    print_and_log_info(logger, f"   gate unchanged for {summary['gate_stuck_counter']} iterations")

                avg_loss = 0
                avg_lr = 0

            if (nb_iter + 1) % config.save_every == 0:
                checkpoint_path = aggressive_snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth'

                aggressive_trainer.save_checkpoint(
                    checkpoint_path,
                    optimizer,
                    {
                        'iteration': nb_iter + 1,
                        'gate_weight': model.get_gate_weight(),
                        'best_gate_weight': aggressive_trainer.best_gate_weight,
                        'gate_stuck_counter': aggressive_trainer.gate_stuck_counter,
                        'achieved_milestones': achieved_milestones,
                        'strategy': 'AGGRESSIVE_GATE_LEARNING',
                        'args': vars(args)
                    }
                )

                print_and_log_info(logger, f"Checkpoint saved at iteration {nb_iter + 1}")

                current_gate = model.get_gate_weight()
                if current_gate > best_gate_weight:
                    best_gate_weight = current_gate
                    best_gate_path = aggressive_snapshot_dir + '/model-best-gate-aggressive.pth'
                    torch.save(model.state_dict(), best_gate_path)
                    print_and_log_info(logger, f"Highest gate value so far: {best_gate_weight:.4f}")

                summary = aggressive_trainer.get_aggressive_summary()
                for key, value in summary.items():
                    if key != 'strategy':
                        print_and_log_info(logger, f"   {key}: {value}")

            if aggressive_trainer.is_training_complete():
                print_and_log_info(logger, "All training stages completed")
                break

            if (nb_iter + 1) == config.cos_lr_total_iters:
                break

            aggressive_trainer.step()

            nb_iter += 1

    writer.close()

    final_summary = aggressive_trainer.get_aggressive_summary()
    print_and_log_info(logger, f"Final gate value: {final_summary['current_gate_weight']:.4f} "
                               f"(highest: {final_summary['best_gate_weight']:.4f})")

    final_model_path = aggressive_snapshot_dir + f'/model-final-aggressive.pth'
    torch.save(model.state_dict(), final_model_path)
    print_and_log_info(logger, f"Final model saved: {final_model_path}")

    acc_log.close()

if __name__ == "__main__":
    main()
