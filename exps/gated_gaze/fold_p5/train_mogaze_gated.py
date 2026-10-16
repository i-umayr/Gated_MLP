"""Train the gated MLP on one MoGaze LOPOCV fold (use --progressive for the progressive gate-opening schedule)."""
import argparse
import os, sys
import json
import math
import numpy as np
import copy

# Required for deterministic cuBLAS
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sys.path.insert(0, os.path.dirname(__file__))

try:
    from mogaze_config_gated import config
    from gated_attention_model import SiMLPeWithGatedGaze, create_gated_model
    from progressive_trainer import ProgressiveGazeTrainer, GateMonitor
    from mogaze_dataset_gaze import MoGazeGazeDataset, MoGazeGazeEval
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

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

    def link_file(src, dst):
        pass

    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default='mogaze_gated_attention.txt', help='experiment name')
parser.add_argument('--seed', type=int, default=888, help='random seed')
parser.add_argument('--temporal-only', action='store_true', help='temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='layernorm axis')
parser.add_argument('--with-normalization', action='store_true', help='use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='use only spatial fc')
parser.add_argument('--num', type=int, default=48, help='num of blocks')
parser.add_argument('--weight', type=float, default=1., help='loss weight')
parser.add_argument('--progressive', action='store_true', help='use progressive training')
parser.add_argument('--pretrained-pose', type=str, default=None, help='path to pretrained pose-only model')

args = parser.parse_args()

torch.use_deterministic_algorithms(True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
writer = SummaryWriter(log_dir=f'runs/gated_attention_{args.seed}')

try:
    acc_log = open(args.exp_name, 'a', encoding='utf-8')
except:
    acc_log = open('gated_training_log.txt', 'a', encoding='utf-8')

config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num

config.cos_lr_total_iters = 80000
config.cos_lr_warmup_iters = 2000

acc_log.write('Gated MLP\n')
acc_log.write(f'Seed: {args.seed}\n')
acc_log.write(f'Progressive Training: {args.progressive}\n')

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

def update_lr_cosine_annealing_gated(nb_iter, total_iter, max_lr, min_lr, optimizer,
                                     warmup_iters, progressive_trainer=None):
    """
    Linear warmup from min_lr to max_lr over warmup_iters, then cosine decay to
    min_lr. With the progressive schedule, the gate parameter group gets
    0.1 x base LR x the schedule's gate LR multiplier.
    """

    if nb_iter < warmup_iters:
        base_lr = min_lr + (max_lr - min_lr) * (nb_iter / warmup_iters)
    else:
        progress = (nb_iter - warmup_iters) / (total_iter - warmup_iters)
        base_lr = min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    if progressive_trainer:
        gate_lr_multiplier = progressive_trainer.get_gate_learning_rate_multiplier()

        for param_group in optimizer.param_groups:
            if 'gaze_gate' in param_group.get('name', ''):
                param_group["lr"] = base_lr * gate_lr_multiplier * 0.1
            else:
                param_group["lr"] = base_lr
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr

    return optimizer, base_lr

def train_step_gated(mogaze_motion_input, mogaze_motion_target, model, optimizer, nb_iter,
                    total_iter, max_lr, min_lr, progressive_trainer=None):
    """One optimisation step."""

    if config.deriv_input:
        b, n, c = mogaze_motion_input.shape  # c = 66 (pose + gaze)
        mogaze_motion_input_ = mogaze_motion_input.clone()
        mogaze_motion_input_ = torch.matmul(
            dct_m[:, :, :config.motion.h36m_input_length],
            mogaze_motion_input_.cuda()
        )
    else:
        mogaze_motion_input_ = mogaze_motion_input.clone()

    # Attention outputs (for the alignment loss) are needed after the baseline stage
    if progressive_trainer and progressive_trainer.current_stage != 'baseline':
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

    if progressive_trainer:
        loss, loss_info = progressive_trainer.compute_progressive_loss(
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

    optimizer, base_lr = update_lr_cosine_annealing_gated(
        nb_iter, total_iter, max_lr, min_lr, optimizer,
        config.cos_lr_warmup_iters, progressive_trainer
    )

    writer.add_scalar('LR/Base', base_lr, nb_iter)
    if progressive_trainer:
        gate_lr = base_lr * progressive_trainer.get_gate_learning_rate_multiplier() * 0.1
        writer.add_scalar('LR/Gate', gate_lr, nb_iter)

    return loss_info, optimizer, base_lr

def main():
    pretrained_model = None
    if args.pretrained_pose:
        print(f"Loading pretrained pose model from {args.pretrained_pose}")
        from model import siMLPe as Model
        pretrained_model = Model(config)
        pretrained_model.load_state_dict(torch.load(args.pretrained_pose))

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

    progressive_trainer = None
    gate_monitor = GateMonitor()

    if args.progressive:
        progressive_trainer = ProgressiveGazeTrainer(model, config, logger=None)
        progressive_trainer.prepare_model_for_stage()

        for stage_name, stage_info in progressive_trainer.stages.items():
            print(f"Stage {stage_name} ({stage_info['iterations']} iterations): {stage_info['description']}")

    # The gate gets its own parameter group (separate learning rate)
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'gaze_gate' not in n], 'name': 'main'},
        {'params': [model.gaze_gate], 'name': 'gaze_gate'}
    ]

    optimizer = torch.optim.Adam(param_groups,
                                 lr=config.cos_lr_min,  # warmup starts at min_lr
                                 weight_decay=config.weight_decay)

    ensure_dir(config.snapshot_dir)
    logger = get_logger(config.log_file.replace('log_gaze', 'log_gated'), 'train')

    print_and_log_info(logger, "Training gated MLP")
    print_and_log_info(logger, f"Parameters: {total_params:.2f}M | progressive schedule: {args.progressive} | seed: {args.seed}")
    print_and_log_info(logger, f"LR: warmup {config.cos_lr_warmup_iters} iterations, cosine decay "
                               f"{config.cos_lr_max:.1e} -> {config.cos_lr_min:.1e}")

    ##### ------ training ------- #####
    nb_iter = 0
    avg_loss = 0.
    avg_lr = 0.
    best_gate_weight = 0.0

    if progressive_trainer:
        total_iterations = sum(stage['iterations'] for stage in progressive_trainer.stages.values())
    else:
        total_iterations = config.cos_lr_total_iters

    print_and_log_info(logger, f"Total iterations: {total_iterations}")

    while (nb_iter + 1) < total_iterations:

        for (mogaze_motion_input, mogaze_motion_target) in dataloader:

            loss_info, optimizer, current_lr = train_step_gated(
                mogaze_motion_input, mogaze_motion_target, model, optimizer,
                nb_iter, total_iterations, config.cos_lr_max, config.cos_lr_min,
                progressive_trainer
            )

            gate_monitor.update(loss_info['gate_weight'], loss_info['total_loss'])

            avg_loss += loss_info['total_loss']
            avg_lr += current_lr

            if (nb_iter + 1) % config.print_every == 0:
                avg_loss = avg_loss / config.print_every
                avg_lr = avg_lr / config.print_every

                if progressive_trainer:
                    progressive_trainer.log_progress(loss_info, writer, nb_iter + 1)

                    stage_info = progressive_trainer.get_current_stage_info()
                    progress_pct = stage_info['progress'] * 100

                    print_and_log_info(logger, f"Iter {nb_iter + 1} | stage: {stage_info['current_stage']}")
                    print_and_log_info(logger, f"   stage progress: {progress_pct:.1f}% | gate: {loss_info['gate_weight']:.4f}")
                    print_and_log_info(logger, f"   avg loss: {avg_loss:.4f} | LR: {avg_lr:.2e}")

                    if 'alignment_loss' in loss_info:
                        print_and_log_info(logger, f"   alignment loss: {loss_info['alignment_loss']:.4f}")

                else:
                    print_and_log_info(logger, f"Iter {nb_iter + 1}")
                    print_and_log_info(logger, f"   gate: {loss_info['gate_weight']:.4f}")
                    print_and_log_info(logger, f"   avg loss: {avg_loss:.4f} | LR: {avg_lr:.2e}")

                if nb_iter < config.cos_lr_warmup_iters:
                    phase = "WARMUP"
                    progress_pct = (nb_iter / config.cos_lr_warmup_iters) * 100
                else:
                    phase = "COSINE DECAY"
                    progress_pct = ((nb_iter - config.cos_lr_warmup_iters) /
                              (total_iterations - config.cos_lr_warmup_iters)) * 100

                print_and_log_info(logger, f"   LR phase: {phase} ({progress_pct:.1f}% complete)")

                gate_stats = gate_monitor.get_statistics()
                if gate_stats:
                    print_and_log_info(logger, f"   gate stats: mean={gate_stats['mean_gate']:.4f}, "
                                             f"std={gate_stats['gate_std']:.4f}, "
                                             f"velocity={gate_stats['gate_velocity']:.6f}")

                avg_loss = 0
                avg_lr = 0

            if (nb_iter + 1) % config.save_every == 0:
                checkpoint_path = config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth'

                if progressive_trainer:
                    # Includes optimizer and schedule state
                    progressive_trainer.save_checkpoint(
                        checkpoint_path,
                        optimizer,
                        {
                            'iteration': nb_iter + 1,
                            'gate_weight': model.get_gate_weight(),
                            'gate_stats': gate_monitor.get_statistics()
                        }
                    )

                    training_summary = progressive_trainer.get_training_summary()
                    for key, value in training_summary.items():
                        print_and_log_info(logger, f"   {key}: {value}")

                else:
                    torch.save(model.state_dict(), checkpoint_path)

                print_and_log_info(logger, f"Checkpoint saved at iteration {nb_iter + 1}")

                current_gate = model.get_gate_weight()
                if current_gate > best_gate_weight:
                    best_gate_weight = current_gate
                    print_and_log_info(logger, f"Highest gate value so far: {best_gate_weight:.4f}")

                    best_path = config.snapshot_dir + '/model-best-gate.pth'
                    torch.save(model.state_dict(), best_path)

            if progressive_trainer and progressive_trainer.is_training_complete():
                print_and_log_info(logger, "All training stages completed")
                break

            if (nb_iter + 1) == total_iterations:
                break

            if progressive_trainer:
                progressive_trainer.step()

            nb_iter += 1

    writer.close()
    print_and_log_info(logger, f"Final model: {config.snapshot_dir}/model-iter-{nb_iter + 1}.pth")
    print_and_log_info(logger, f"Final gate value: {model.get_gate_weight():.4f} (highest: {best_gate_weight:.4f})")

    final_gate_stats = gate_monitor.get_statistics()
    if final_gate_stats:
        for key, value in final_gate_stats.items():
            print_and_log_info(logger, f"   {key}: {value}")

    if progressive_trainer:
        final_summary = progressive_trainer.get_training_summary()
        for key, value in final_summary.items():
            print_and_log_info(logger, f"   {key}: {value}")

    acc_log.close()

if __name__ == "__main__":
    main()
