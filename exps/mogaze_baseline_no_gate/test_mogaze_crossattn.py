"""
Test Script: Cross-Attention (No Gate) Ablation -- Method 3

Evaluates all 6 LOPOCV folds and computes:
  - Per-fold MPJPE at 200/400/600/800/1000ms
  - Weighted average MPJPE across folds
  - Per-fold mean, standard deviation, minimum and maximum

The autoregressive evaluation is IDENTICAL to the concat / additive ablations:
gaze is carried forward from the last input frame, the model splits pose/gaze
internally, and predictions are residual offsets from the last observed pose.
model.eval() disables cross-attention dropout for deterministic evaluation.

Usage:
    python test_mogaze_crossattn.py
    python test_mogaze_crossattn.py --model-dir log_crossattn/snapshot
    python test_mogaze_crossattn.py --folds p1 p2  # evaluate specific folds
"""

import argparse
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time

sys.path.insert(0, os.path.dirname(__file__))

from mogaze_config_crossattn import config
from crossattn_model import SiMLPeCrossAttnNoGate
from mogaze_dataset_crossattn import MoGazeCrossAttnEval, ALL_PARTICIPANTS


# --- Utilities --------------------------------------------------------------------
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


# --- Evaluate one fold ------------------------------------------------------------
def evaluate_fold(test_participant, model_path, fold_config, device='cuda'):
    print(f"\n--- Evaluating fold: {test_participant} ---")
    print(f"    Model: {model_path}")

    model = SiMLPeCrossAttnNoGate(fold_config)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    dct_m, idct_m = get_dct_matrix(fold_config.motion.h36m_input_length_dct)
    dct_m = torch.tensor(dct_m).float().to(device).unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().to(device).unsqueeze(0)

    eval_dataset = MoGazeCrossAttnEval(fold_config, test_participant=test_participant)
    dataloader = DataLoader(eval_dataset, batch_size=32,
                            num_workers=0, drop_last=False, shuffle=False)

    n_samples = len(eval_dataset)
    print(f"    Eval samples: {n_samples}")

    if n_samples == 0:
        print(f"    WARNING: No eval samples for {test_participant}!")
        return None

    all_frame_errors = []

    step = fold_config.motion.h36m_target_length_train  # 10
    num_step = 30 // step if 30 % step == 0 else 30 // step + 1

    with torch.no_grad():
        for motion_input, motion_target in dataloader:
            motion_input = motion_input.to(device)    # [b, 50, 66]
            motion_target = motion_target.to(device)  # [b, 30, 66]
            b = motion_input.shape[0]

            outputs = []
            current_input = motion_input.clone()

            for _ in range(num_step):
                if fold_config.deriv_input:
                    inp = torch.matmul(
                        dct_m[:, :, :fold_config.motion.h36m_input_length],
                        current_input
                    )
                else:
                    inp = current_input.clone()

                out = model(inp)

                out = torch.matmul(
                    idct_m[:, :fold_config.motion.h36m_input_length, :],
                    out
                )[:, :step, :]

                if fold_config.deriv_output:
                    pose_offset = current_input[:, -1:, :63]
                    out = out + pose_offset

                outputs.append(out)

                last_gaze = current_input[:, -1:, 63:66]
                extended_gaze = last_gaze.repeat(1, step, 1)
                out_with_gaze = torch.cat([out, extended_gaze], dim=2)  # [b, step, 66]
                current_input = torch.cat([current_input[:, step:], out_with_gaze], dim=1)

            motion_pred = torch.cat(outputs, dim=1)[:, :30]  # [b, 30, 63]
            target_pose = motion_target[:, :30, :63]          # [b, 30, 63]

            pred_3d = motion_pred.reshape(b, 30, 21, 3)
            target_3d = target_pose.reshape(b, 30, 21, 3)
            joint_errors = torch.norm(pred_3d - target_3d, dim=-1) * 1000  # mm
            frame_errors = joint_errors.mean(dim=2)  # [b, 30]

            all_frame_errors.append(frame_errors.cpu().numpy())

    all_frame_errors = np.concatenate(all_frame_errors, axis=0)  # [N, 30]
    mean_per_frame = all_frame_errors.mean(axis=0)               # [30]
    overall_mpjpe = mean_per_frame.mean()

    key_frames = {'200ms': 5, '400ms': 11, '600ms': 17, '800ms': 23, '1000ms': 29}
    temporal = {k: mean_per_frame[v] for k, v in key_frames.items()}

    print(f"    MPJPE: {overall_mpjpe:.1f} mm  |  "
          f"200ms: {temporal['200ms']:.1f}  400ms: {temporal['400ms']:.1f}  "
          f"600ms: {temporal['600ms']:.1f}  800ms: {temporal['800ms']:.1f}  "
          f"1000ms: {temporal['1000ms']:.1f}")

    return {
        'participant': test_participant,
        'n_samples': n_samples,
        'overall_mpjpe': overall_mpjpe,
        'temporal': temporal,
        'per_frame': mean_per_frame,
    }


# --- Main -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='log_crossattn/snapshot',
                        help='directory containing fold_<participant>/ subdirectories')
    parser.add_argument('--model-name', type=str, default='model-final.pth',
                        help='model filename inside each fold directory')
    parser.add_argument('--folds', nargs='+', default=None,
                        help='specific folds to evaluate (default: all 6)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device for evaluation (cuda or cpu)')
    args = parser.parse_args()

    folds = args.folds if args.folds else ALL_PARTICIPANTS

    config.motion_fc_in.temporal_fc = False
    config.motion_fc_out.temporal_fc = False
    config.motion_mlp.with_normalization = True
    config.motion_mlp.norm_axis = 'spatial'
    config.motion_mlp.spatial_fc_only = False

    print("=" * 80)
    print("  CROSS-ATTENTION (NO GATE) ABLATION -- LOPOCV EVALUATION")
    print("=" * 80)
    print(f"Model dir:   {args.model_dir}")
    print(f"Model file:  {args.model_name}")
    print(f"Folds:       {folds}")
    print(f"Device:      {args.device}")
    print("=" * 80)

    fold_results = []

    for test_p in folds:
        model_path = os.path.join(args.model_dir, f'fold_{test_p}', args.model_name)

        if not os.path.exists(model_path):
            print(f"\nWARNING: Model not found for fold {test_p}: {model_path}")
            continue

        result = evaluate_fold(test_p, model_path, config, device=args.device)
        if result is not None:
            fold_results.append(result)

    if not fold_results:
        print("\nNo folds evaluated! Check model paths.")
        return

    total_samples = sum(r['n_samples'] for r in fold_results)
    weighted_mpjpe = sum(r['n_samples'] * r['overall_mpjpe'] for r in fold_results) / total_samples

    temporal_keys = ['200ms', '400ms', '600ms', '800ms', '1000ms']
    weighted_temporal = {}
    for k in temporal_keys:
        weighted_temporal[k] = sum(
            r['n_samples'] * r['temporal'][k] for r in fold_results
        ) / total_samples

    print(f"\n{'='*80}")
    print(f"  RESULTS: CROSS-ATTENTION (NO GATE) (LOPOCV)")
    print(f"{'='*80}")

    print(f"\nPer-fold MPJPE (mm):")
    print(f"{'Fold':<8} {'Samples':>8} {'200ms':>8} {'400ms':>8} {'600ms':>8} {'800ms':>8} {'1000ms':>8} {'Avg':>8}")
    print("-" * 72)
    for r in fold_results:
        t = r['temporal']
        print(f"{r['participant']:<8} {r['n_samples']:>8} "
              f"{t['200ms']:>8.1f} {t['400ms']:>8.1f} {t['600ms']:>8.1f} "
              f"{t['800ms']:>8.1f} {t['1000ms']:>8.1f} {r['overall_mpjpe']:>8.1f}")

    print("-" * 72)
    print(f"{'Weighted':<8} {total_samples:>8} "
          f"{weighted_temporal['200ms']:>8.1f} {weighted_temporal['400ms']:>8.1f} "
          f"{weighted_temporal['600ms']:>8.1f} {weighted_temporal['800ms']:>8.1f} "
          f"{weighted_temporal['1000ms']:>8.1f} {weighted_mpjpe:>8.1f}")

    fold_mpjpes = [r['overall_mpjpe'] for r in fold_results]
    print(f"\nStatistical Summary:")
    print(f"  Folds evaluated:    {len(fold_results)}")
    print(f"  Total samples:      {total_samples}")
    print(f"  Weighted avg MPJPE: {weighted_mpjpe:.1f} mm")
    print(f"  Per-fold mean:      {np.mean(fold_mpjpes):.1f} mm")
    print(f"  Per-fold std:       {np.std(fold_mpjpes, ddof=1):.1f} mm")
    print(f"  Per-fold min/max:   {np.min(fold_mpjpes):.1f} / {np.max(fold_mpjpes):.1f} mm")

    model = SiMLPeCrossAttnNoGate(config)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters:   {total_params:.4f}M")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
