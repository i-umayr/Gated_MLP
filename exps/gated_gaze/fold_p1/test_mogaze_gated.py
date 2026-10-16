"""Evaluate a gated MLP trained with the progressive schedule on the held-out participant of this fold."""
import argparse
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

from mogaze_config_gated import config
from gated_attention_model import SiMLPeWithGatedGaze
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

def evaluate(config, model, dataloader):
    """
    Auto-regressive prediction of 30 frames (10 per step).
    Returns the MPJPE over all samples and the per-frame error averaged over batches (mm).
    """
    model.eval()

    dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
    dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

    total_error = 0.0
    num_samples = 0
    errors_per_frame = []

    with torch.no_grad():
        for batch_idx, (motion_input, motion_target) in enumerate(dataloader):
            if isinstance(motion_input, np.ndarray):
                motion_input = torch.tensor(motion_input).float()
            if isinstance(motion_target, np.ndarray):
                motion_target = torch.tensor(motion_target).float()

            motion_input = motion_input.cuda()  # [batch, 50, 66]
            motion_target = motion_target.cuda()  # [batch, 30, 66]

            b, n, c = motion_input.shape
            num_samples += b

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

            pose_error_mm = np.linalg.norm(pred_pose_3d - target_pose_3d, axis=3) * 1000

            frame_errors = np.mean(pose_error_mm, axis=(0, 2))  # [30]
            total_error += np.sum(pose_error_mm)

            if len(errors_per_frame) == 0:
                errors_per_frame = frame_errors
            else:
                errors_per_frame += frame_errors

    avg_pose_error = total_error / (num_samples * 30 * 21)
    errors_per_frame = errors_per_frame / len(dataloader)

    return avg_pose_error, errors_per_frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-pth', type=str,
                       default='log_gated/snapshot/model-iter-80000.pth',
                       help='path to trained gated model')
    parser.add_argument('--save-results', type=str, default=None,
                       help='save results to CSV file')
    args = parser.parse_args()

    model = SiMLPeWithGatedGaze(config)

    if not os.path.exists(args.model_pth):
        print(f"Model file not found: {args.model_pth}")
        exit(1)
    try:
        state_dict = torch.load(args.model_pth, weights_only=True)
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
    model.cuda()

    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    test_dataset = MoGazeGazeEval(config, 'test')

    dataloader = DataLoader(test_dataset, batch_size=16,
                          num_workers=0, drop_last=False,
                          shuffle=False, pin_memory=True)

    avg_pose_error, errors_per_frame = evaluate(config, model, dataloader)

    # 200 / 400 / 600 / 800 / 1000 ms = frames 6 / 12 / 18 / 24 / 30 at 30 fps
    target_frames = [6, 12, 18, 24, 30]
    print("=" * 60)
    print("Gated MLP (progressive) - MPJPE on the held-out participant")
    print("=" * 60)
    print(f"Samples: {len(test_dataset)}")
    print(f"MPJPE:   {avg_pose_error:.2f} mm")
    for frame in target_frames:
        ms = int(frame * 1000 / 30)
        print(f"{ms:5d} ms: {errors_per_frame[frame-1]:7.2f} mm")
    print(f"Gate value: {model.get_gate_weight():.4f}")
    print("=" * 60)

    if args.save_results:
        results = {'mean_mpjpe': avg_pose_error}
        results.update({f'mpjpe_{int(f * 1000 / 30)}ms': errors_per_frame[f-1] for f in target_frames})
        pd.DataFrame([results]).to_csv(args.save_results, index=False)
        print(f"Results saved to {args.save_results}")

if __name__ == "__main__":
    main()
