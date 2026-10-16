"""Evaluate a trained siMLPe baseline (pose only) on the held-out participant of this fold."""
import argparse
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import pandas as pd

from mogaze_config_3d import config
from model import siMLPe as Model
from mogaze_dataset_3d import MoGaze3DEval

KEY_FRAMES = [5, 11, 17, 23, 29]  # 0-indexed frames at 30 fps
KEY_TIMES = [200, 400, 600, 800, 1000]  # ms

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

class MPJPEMetrics:
    """Collects per-sample MPJPE and per-frame errors."""

    def __init__(self):
        self.sample_errors = []  # mean error of each sample
        self.temporal_errors = defaultdict(list)  # per frame: mean error of each batch

    def add_batch_results(self, pred_joints, target_joints):
        # pred_joints, target_joints: [batch, 30, 21, 3] in metres
        joint_errors = np.linalg.norm(pred_joints - target_joints, axis=-1) * 1000  # mm

        self.sample_errors.extend(np.mean(joint_errors, axis=(1, 2)))

        for t in range(30):
            self.temporal_errors[t].append(np.mean(joint_errors[:, t, :]))

    def compute(self):
        sample_errors = np.array(self.sample_errors)
        metrics = {
            'mean_mpjpe': np.mean(sample_errors),
            'std_mpjpe': np.std(sample_errors),
            'n_samples': len(sample_errors),
        }
        # Per-horizon errors are averaged over batches
        for frame_idx, time_ms in zip(KEY_FRAMES, KEY_TIMES):
            metrics[f'mpjpe_{time_ms}ms'] = np.mean(self.temporal_errors[frame_idx])
            metrics[f'std_{time_ms}ms'] = np.std(self.temporal_errors[frame_idx])
        return metrics

def print_report(metrics):
    print("=" * 60)
    print("siMLPe (no gaze) - MPJPE on the held-out participant")
    print("=" * 60)
    print(f"Samples: {metrics['n_samples']}")
    print(f"MPJPE:   {metrics['mean_mpjpe']:.2f} ± {metrics['std_mpjpe']:.2f} mm")
    for time_ms in KEY_TIMES:
        print(f"{time_ms:5d} ms: {metrics[f'mpjpe_{time_ms}ms']:7.2f} ± {metrics[f'std_{time_ms}ms']:5.2f} mm")
    print("=" * 60)

def evaluate(config, model, dataloader):
    model.eval()

    dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
    dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

    metrics_collector = MPJPEMetrics()

    with torch.no_grad():
        for batch_idx, (motion_input, motion_target) in enumerate(dataloader):
            if isinstance(motion_input, np.ndarray):
                motion_input = torch.tensor(motion_input).float()
            if isinstance(motion_target, np.ndarray):
                motion_target = torch.tensor(motion_target).float()

            motion_input = motion_input.cuda()
            motion_target = motion_target.cuda()

            b, n, c = motion_input.shape

            # Auto-regressive prediction of 30 frames, 10 frames per step
            outputs = []
            step = config.motion.h36m_target_length_train

            if step == 30:
                num_step = 1
            else:
                num_step = 30 // step + 1

            current_input = motion_input.clone()

            for idx in range(num_step):
                if config.deriv_input:
                    motion_input_ = current_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_)
                else:
                    motion_input_ = current_input.clone()

                output = model(motion_input_)
                output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]

                if config.deriv_output:
                    output = output + current_input[:, -1:, :].repeat(1, step, 1)

                output = output.reshape(-1, 63)
                output = output.reshape(b, step, -1)
                outputs.append(output)

                current_input = torch.cat([current_input[:, step:], output], axis=1)

            motion_pred = torch.cat(outputs, axis=1)[:, :30]  # [batch, 30, 63]

            pred_3d = motion_pred.cpu().numpy().reshape(b, 30, 21, 3)
            target_3d = motion_target.cpu().numpy().reshape(b, 30, 21, 3)

            metrics_collector.add_batch_results(pred_3d, target_3d)

    return metrics_collector.compute()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-pth', type=str,
                       default='log_3d/snapshot/model-iter-80000.pth',
                       help='path to trained 3D position model')
    parser.add_argument('--save-results', type=str, default=None,
                       help='save results to CSV file')
    args = parser.parse_args()

    model = Model(config)

    if not os.path.exists(args.model_pth):
        print(f"Model file not found: {args.model_pth}")
        exit(1)
    state_dict = torch.load(args.model_pth, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded {args.model_pth}")

    model.eval()
    model.cuda()

    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    test_dataset = MoGaze3DEval(config, 'test')

    dataloader = DataLoader(test_dataset, batch_size=16,
                          num_workers=0, drop_last=False,
                          shuffle=False, pin_memory=True)

    metrics = evaluate(config, model, dataloader)
    print_report(metrics)

    if args.save_results:
        pd.DataFrame([metrics]).to_csv(args.save_results, index=False)
        print(f"Results saved to {args.save_results}")

if __name__ == "__main__":
    main()
