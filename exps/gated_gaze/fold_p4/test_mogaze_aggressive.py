"""Evaluate a gated MLP trained with the aggressive schedule on the held-out participant of this fold."""
import argparse
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import pandas as pd
from scipy import stats

from mogaze_config_gated import config
from gated_attention_model import SiMLPeWithGatedGaze, create_gated_model
from mogaze_dataset_gaze import MoGazeGazeEval

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
    """Collects per-sample MPJPE, per-frame errors and the gate value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sample_errors = []  # mean error of each sample
        self.temporal_errors = defaultdict(list)  # per frame: mean error of each batch
        self.gate_weights = []

    def add_batch_results(self, pred_joints, target_joints, gate_weight=None):
        # pred_joints, target_joints: [batch, 30, 21, 3] in metres
        batch_size = pred_joints.shape[0]

        joint_errors = np.linalg.norm(pred_joints - target_joints, axis=-1) * 1000  # mm

        sample_avg_errors = np.mean(joint_errors, axis=(1, 2))
        self.sample_errors.extend(sample_avg_errors)

        for t in range(30):
            frame_error = np.mean(joint_errors[:, t, :])
            self.temporal_errors[t].append(frame_error)

        if gate_weight is not None:
            self.gate_weights.extend([gate_weight] * batch_size)

    def compute_metrics(self):
        """MPJPE with 95% confidence intervals, per-horizon errors and gate statistics."""
        sample_errors = np.array(self.sample_errors)

        mean_mpjpe = np.mean(sample_errors)
        std_mpjpe = np.std(sample_errors, ddof=1)
        n_samples = len(sample_errors)

        confidence_level = 0.95
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df=n_samples-1)
        margin_error = t_critical * (std_mpjpe / np.sqrt(n_samples))
        ci_lower = mean_mpjpe - margin_error
        ci_upper = mean_mpjpe + margin_error

        # 200 / 400 / 600 / 800 / 1000 ms = frames 6 / 12 / 18 / 24 / 30 at 30 fps
        key_frames = [5, 11, 17, 23, 29]  # 0-indexed
        key_times = [200, 400, 600, 800, 1000]

        # Per-horizon errors are averaged over batches
        temporal_results = {}
        for frame_idx, time_ms in zip(key_frames, key_times):
            if frame_idx < 30 and frame_idx in self.temporal_errors:
                t_errors = np.array(self.temporal_errors[frame_idx])
                t_mean = np.mean(t_errors)
                t_std = np.std(t_errors, ddof=1)

                t_n = len(t_errors)
                t_critical_t = stats.t.ppf(1 - alpha/2, df=t_n-1)
                t_margin = t_critical_t * (t_std / np.sqrt(t_n))

                temporal_results[f'mpjpe_{time_ms}ms'] = {
                    'mean': t_mean,
                    'std': t_std,
                    'ci_lower': t_mean - t_margin,
                    'ci_upper': t_mean + t_margin,
                    'n_samples': t_n
                }

        gate_stats = {}
        if self.gate_weights:
            gate_weights = np.array(self.gate_weights)
            gate_stats = {
                'mean_gate_weight': np.mean(gate_weights),
                'std_gate_weight': np.std(gate_weights, ddof=1),
                'gate_utilization_rate': np.mean(gate_weights > 0.1)
            }

        return {
            'overall': {
                'mean_mpjpe': mean_mpjpe,
                'std_mpjpe': std_mpjpe,
                'ci_95_lower': ci_lower,
                'ci_95_upper': ci_upper,
                'n_samples': n_samples
            },
            'temporal': temporal_results,
            'gate': gate_stats
        }

def evaluate(config, model, dataloader):
    """Auto-regressive prediction of 30 frames (10 per step) on the CPU."""
    model.eval()

    device = torch.device('cpu')
    model = model.to(device)

    dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
    dct_m = torch.tensor(dct_m).float().to(device).unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().to(device).unsqueeze(0)

    mpjpe_metrics = MPJPEMetrics()

    with torch.no_grad():
        for batch_idx, (motion_input, motion_target) in enumerate(dataloader):
            motion_input = motion_input.to(device)
            motion_target = motion_target.to(device)

            b, n, c = motion_input.shape

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
                    pose_offset = current_input[:, -1:, :63]
                    output = output + pose_offset

                # The last observed gaze is kept constant for the predicted frames
                batch_size, seq_len, _ = output.shape
                last_gaze = current_input[:, -1:, 63:66]
                extended_gaze = last_gaze.repeat(1, seq_len, 1)

                output_with_gaze = torch.cat([output, extended_gaze], dim=2)
                outputs.append(output)

                current_input = torch.cat([current_input[:, step:], output_with_gaze], axis=1)

            motion_pred = torch.cat(outputs, axis=1)[:, :30]  # [batch, 30, 63]

            motion_pred_np = motion_pred.cpu().numpy()
            motion_target_np = motion_target.cpu().numpy()

            target_pose = motion_target_np[:, :, :63]

            pred_pose_3d = motion_pred_np.reshape(b, 30, 21, 3)
            target_pose_3d = target_pose.reshape(b, 30, 21, 3)

            mpjpe_metrics.add_batch_results(
                pred_pose_3d,
                target_pose_3d,
                gate_weight=model.get_gate_weight()
            )

    return mpjpe_metrics.compute_metrics()

def print_report(metrics, model_name="Gated MLP (aggressive)"):
    print("=" * 60)
    print(f"{model_name} - MPJPE on the held-out participant")
    print("=" * 60)

    overall = metrics['overall']
    print(f"Samples: {overall['n_samples']}")
    print(f"MPJPE:   {overall['mean_mpjpe']:.2f} ± {overall['std_mpjpe']:.2f} mm "
          f"(95% CI [{overall['ci_95_lower']:.2f}, {overall['ci_95_upper']:.2f}])")

    temporal = metrics['temporal']
    for time_ms in [200, 400, 600, 800, 1000]:
        key = f'mpjpe_{time_ms}ms'
        if key in temporal:
            t = temporal[key]
            print(f"{time_ms:5d} ms: {t['mean']:7.2f} ± {t['std']:5.2f} mm")

    if metrics['gate']:
        print(f"Gate value: {metrics['gate']['mean_gate_weight']:.4f}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-pth', type=str,
                       default='log_aggressive/snapshot/model-final-aggressive.pth',
                       help='path to trained model')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='batch size for evaluation')
    parser.add_argument('--save-results', type=str, default=None,
                       help='save results to CSV file')
    args = parser.parse_args()

    model = create_gated_model(config)

    if not os.path.exists(args.model_pth):
        print(f"Model file not found: {args.model_pth}")
        exit(1)
    try:
        state_dict = torch.load(args.model_pth, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    except:
        # Full training checkpoint (model + optimizer + schedule state)
        checkpoint = torch.load(args.model_pth, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
    print(f"Loaded {args.model_pth}")

    model.eval()

    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    test_dataset = MoGazeGazeEval(config, 'test')

    dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                          num_workers=0, drop_last=False,
                          shuffle=False, pin_memory=False)

    metrics = evaluate(config, model, dataloader)

    print_report(metrics)

    if args.save_results:
        results_flat = {}
        results_flat.update(metrics['overall'])
        results_flat.update({f'temporal_{k}': v['mean'] for k, v in metrics['temporal'].items()})
        results_flat.update({f'temporal_{k}_std': v['std'] for k, v in metrics['temporal'].items()})
        results_flat.update(metrics['gate'])

        pd.DataFrame([results_flat]).to_csv(args.save_results, index=False)
        print(f"Results saved to {args.save_results}")

if __name__ == "__main__":
    main()
