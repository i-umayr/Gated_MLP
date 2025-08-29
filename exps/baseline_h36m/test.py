import argparse
import os, sys
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from scipy import stats

from config import config
from model import siMLPe as Model
from datasets.h36m_eval import H36MEval
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

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

class ComprehensiveMetricsH36M:
    """Comprehensive metrics collection for H36M motion prediction evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.frame_errors = []  # [num_samples, 25, 22] - per frame, per joint errors (H36M uses 22 joints)
        self.sample_errors = []  # [num_samples] - per sample average errors
        self.joint_errors = defaultdict(list)  # per joint type errors
        self.temporal_errors = defaultdict(list)  # per time step errors
        
    def add_batch_results(self, pred_joints, target_joints):
        """Add results from a batch for comprehensive analysis"""
        # pred_joints: [batch, 25, 22, 3] - H36M uses 25 frames, 22 joints
        # target_joints: [batch, 25, 22, 3]
        
        batch_size = pred_joints.shape[0]
        
        # Calculate per-joint errors (in mm)
        joint_errors = np.linalg.norm(pred_joints - target_joints, axis=-1)  # Already in mm from H36M
        
        # Store frame-wise errors
        self.frame_errors.extend(joint_errors)
        
        # Store sample-wise average errors
        sample_avg_errors = np.mean(joint_errors, axis=(1, 2))
        self.sample_errors.extend(sample_avg_errors)
        
        # Store temporal errors (average over joints and batch)
        for t in range(25):
            frame_error = np.mean(joint_errors[:, t, :])
            self.temporal_errors[t].append(frame_error)
            
        # Store per-joint errors (average over time and batch)
        for j in range(22):
            joint_error = np.mean(joint_errors[:, :, j])
            self.joint_errors[j].extend([joint_error] * batch_size)
    
    def compute_comprehensive_metrics(self):
        """Compute all comprehensive metrics"""
        frame_errors = np.array(self.frame_errors)  # [num_samples, 25, 22]
        sample_errors = np.array(self.sample_errors)
        
        metrics = {}
        
        # === BASIC STATISTICS ===
        metrics['mean_mpjpe'] = np.mean(sample_errors)
        metrics['std_mpjpe'] = np.std(sample_errors)
        metrics['median_mpjpe'] = np.median(sample_errors)
        metrics['p95_mpjpe'] = np.percentile(sample_errors, 95)
        metrics['p5_mpjpe'] = np.percentile(sample_errors, 5)
        metrics['min_mpjpe'] = np.min(sample_errors)
        metrics['max_mpjpe'] = np.max(sample_errors)
        
        # === TEMPORAL ANALYSIS ===
        temporal_means = []
        temporal_stds = []
        for t in range(25):
            if t in self.temporal_errors:
                t_errors = self.temporal_errors[t]
                temporal_means.append(np.mean(t_errors))
                temporal_stds.append(np.std(t_errors))
            else:
                temporal_means.append(0)
                temporal_stds.append(0)
        
        metrics['temporal_means'] = np.array(temporal_means)
        metrics['temporal_stds'] = np.array(temporal_stds)
        
        # H36M standard time horizons (80ms, 160ms, 320ms, 400ms, 560ms, 720ms, 880ms, 1000ms)
        # At 25fps: 80ms=2frames, 160ms=4frames, 320ms=8frames, 400ms=10frames, 
        # 560ms=14frames, 720ms=18frames, 880ms=22frames, 1000ms=25frames
        h36m_frames = [1, 3, 7, 9, 13, 17, 21, 24]  # 0-indexed
        h36m_times = [80, 160, 320, 400, 560, 720, 880, 1000]
        
        for i, (frame_idx, time_ms) in enumerate(zip(h36m_frames, h36m_times)):
            if frame_idx < len(temporal_means):
                metrics[f'mpjpe_{time_ms}ms'] = temporal_means[frame_idx]
                metrics[f'std_{time_ms}ms'] = temporal_stds[frame_idx]
        
        # Standard results_keys format (#2, #4, #8, #10, #14, #18, #22, #25)
        standard_frames = [1, 3, 7, 9, 13, 17, 21, 24]  # 0-indexed for frames 2,4,8,10,14,18,22,25
        for i, frame_idx in enumerate(standard_frames):
            frame_num = frame_idx + 1  # Convert to 1-indexed
            if frame_idx < len(temporal_means):
                metrics[f'frame_{frame_num}'] = temporal_means[frame_idx]
        
        # === PER-JOINT ANALYSIS ===
        # H36M joint names (22 joints used for evaluation)
        h36m_joint_names = [
            'hip', 'right_hip', 'right_knee', 'right_foot', 'left_hip', 'left_knee', 'left_foot',
            'spine', 'thorax', 'neck', 'head', 'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist', 'left_hip_extra', 'left_knee_extra',
            'left_foot_extra', 'right_hip_extra', 'right_knee_extra'
        ]
        
        joint_stats = {}
        for j in range(min(22, len(h36m_joint_names))):
            if j in self.joint_errors:
                j_errors = self.joint_errors[j]
                joint_stats[h36m_joint_names[j]] = {
                    'mean': np.mean(j_errors),
                    'std': np.std(j_errors),
                    'median': np.median(j_errors)
                }
        
        metrics['joint_stats'] = joint_stats
        
        # === CONSISTENCY METRICS ===
        # Coefficient of variation (std/mean) - lower is more consistent
        metrics['cv_mpjpe'] = metrics['std_mpjpe'] / metrics['mean_mpjpe']
        
        # Interquartile range
        metrics['iqr_mpjpe'] = np.percentile(sample_errors, 75) - np.percentile(sample_errors, 25)
        
        # === PREDICTION QUALITY METRICS ===
        # Temporal degradation rate (how much error increases over time)
        if len(temporal_means) >= 2:
            # Linear regression to find degradation rate
            time_points = np.arange(len(temporal_means))
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, temporal_means)
            metrics['degradation_rate'] = slope  # mm per frame
            metrics['degradation_r2'] = r_value ** 2
        
        return metrics
    
    def print_comprehensive_report(self, metrics, model_name="siMLPe H36M Model"):
        """Print a comprehensive evaluation report"""
        print("=" * 100)
        print(f"COMPREHENSIVE EVALUATION REPORT - {model_name}")
        print("=" * 100)
        
        # === HEADLINE METRICS ===
        print(f"\n🎯 HEADLINE RESULTS:")
        print(f"   Average MPJPE: {metrics['mean_mpjpe']:.1f} ± {metrics['std_mpjpe']:.1f} mm")
        print(f"   Median MPJPE:  {metrics['median_mpjpe']:.1f} mm")
        print(f"   95th Percentile: {metrics['p95_mpjpe']:.1f} mm (worst-case performance)")
        print(f"   Consistency (CV): {metrics['cv_mpjpe']:.3f} (lower = more consistent)")
        
        # === TEMPORAL BREAKDOWN ===
        print(f"\n⏱️  TEMPORAL PERFORMANCE (H36M Standard):")
        h36m_times = [80, 160, 320, 400, 560, 720, 880, 1000]
        for time_ms in h36m_times:
            if f'mpjpe_{time_ms}ms' in metrics:
                mean_key = f'mpjpe_{time_ms}ms'
                std_key = f'std_{time_ms}ms'
                print(f"   {time_ms:4d}ms: {metrics[mean_key]:5.1f} ± {metrics[std_key]:4.1f} mm")
        
        if 'degradation_rate' in metrics:
            print(f"   Degradation Rate: {metrics['degradation_rate']:.2f} mm/frame (R²={metrics['degradation_r2']:.3f})")
        
        # === STANDARD H36M RESULTS FORMAT ===
        print(f"\n📊 STANDARD H36M RESULTS (frames #2, #4, #8, #10, #14, #18, #22, #25):")
        standard_results = []
        for frame_num in [2, 4, 8, 10, 14, 18, 22, 25]:
            if f'frame_{frame_num}' in metrics:
                result = metrics[f'frame_{frame_num}']
                standard_results.append(result)
                print(f"   #{frame_num:2d}: {result:5.1f} mm")
        
        # === COMPARISON WITH SIMLPE PAPER RESULTS ===
        print(f"\n📚 SIMLPE PAPER COMPARISON:")
        # siMLPe paper results from Human3.6M (Table 1)
        simlpe_paper_results = {
            'name': 'siMLPe Paper Results',
            'results': [9.6, 21.7, 46.3, 57.3, 75.7, 90.1, 101.8, 109.4],  # 80ms to 1000ms
            'average': 109.4  # 1000ms result
        }
        
        print(f"   siMLPe Paper (1000ms): {simlpe_paper_results['average']:.1f} mm")
        
        # Compare with our results
        if 'mpjpe_1000ms' in metrics:
            our_1000ms = metrics['mpjpe_1000ms']
            if our_1000ms < simlpe_paper_results['average']:
                improvement = ((simlpe_paper_results['average'] - our_1000ms) / simlpe_paper_results['average']) * 100
                print(f"   ✅ vs siMLPe Paper: {improvement:.1f}% BETTER! ({our_1000ms:.1f} vs {simlpe_paper_results['average']:.1f})")
            else:
                diff = our_1000ms - simlpe_paper_results['average']
                print(f"   📈 vs siMLPe Paper: {diff:.1f}mm higher ({our_1000ms:.1f} vs {simlpe_paper_results['average']:.1f})")
        
        # === TOP/BOTTOM PERFORMING JOINTS ===
        if 'joint_stats' in metrics:
            joint_stats = metrics['joint_stats']
            joint_means = {name: stats['mean'] for name, stats in joint_stats.items()}
            
            # Best and worst joints
            best_joints = sorted(joint_means.items(), key=lambda x: x[1])[:3]
            worst_joints = sorted(joint_means.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"\n🦴 JOINT-WISE PERFORMANCE:")
            print(f"   Best Joints:")
            for joint, error in best_joints:
                print(f"     {joint:15s}: {error:5.1f} mm")
            print(f"   Challenging Joints:")
            for joint, error in worst_joints:
                print(f"     {joint:15s}: {error:5.1f} mm")
        
        # === STATISTICAL SUMMARY ===
        print(f"\n📊 STATISTICAL SUMMARY:")
        print(f"   Range: [{metrics['min_mpjpe']:.1f}, {metrics['max_mpjpe']:.1f}] mm")
        print(f"   IQR: {metrics['iqr_mpjpe']:.1f} mm")
        print(f"   5th-95th Percentile: [{metrics['p5_mpjpe']:.1f}, {metrics['p95_mpjpe']:.1f}] mm")
        
        # === TRAINING INSIGHTS ===
        print(f"\n🔬 TRAINING INSIGHTS:")
        print(f"   Dataset: Human3.6M (standard benchmark)")
        print(f"   Architecture: siMLPe MLP with DCT encoding")
        print(f"   Joints: 22 joints used for evaluation")
        print(f"   Prediction: 25 frames (1 second at 25fps)")
        print(f"   Evaluation: Standard H36M MPJPE protocol")
        
        print("=" * 100)

def regress_pred_comprehensive(model, pbar, num_samples, joint_used_xyz, m_p3d_h36, metrics_collector):
    """Enhanced prediction with comprehensive metrics collection"""
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)
    
    dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
    dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

    for batch_idx, (motion_input, motion_target) in enumerate(pbar):
        motion_input = motion_input.cuda()
        b, n, c, _ = motion_input.shape
        num_samples += b

        motion_input = motion_input.reshape(b, n, 32, 3)
        motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)
        outputs = []
        step = config.motion.h36m_target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
            
        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_.cuda())
                else:
                    motion_input_ = motion_input.clone()
                output = model(motion_input_)
                output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1, step, 1)

            output = output.reshape(-1, 22*3)
            output = output.reshape(b, step, -1)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
            
        motion_pred = torch.cat(outputs, axis=1)[:, :25]

        motion_target = motion_target.detach()
        b, n, c, _ = motion_target.shape

        motion_gt = motion_target.clone()

        motion_pred_cpu = motion_pred.detach().cpu()
        pred_rot = motion_pred_cpu.clone().reshape(b, n, 22, 3)
        motion_pred_full = motion_target.clone().reshape(b, n, 32, 3)
        motion_pred_full[:, :, joint_used_xyz] = pred_rot

        tmp = motion_gt.clone()
        tmp[:, :, joint_used_xyz] = motion_pred_full[:, :, joint_used_xyz]
        motion_pred_final = tmp
        motion_pred_final[:, :, joint_to_ignore] = motion_pred_final[:, :, joint_equal]

        # Calculate MPJPE for standard evaluation
        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred_final*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
        
        # Add to comprehensive metrics
        # Convert to numpy for metrics collection
        pred_joints_np = motion_pred_final.cpu().numpy() * 1000  # Convert to mm
        target_joints_np = motion_gt.cpu().numpy() * 1000  # Convert to mm
        
        # Extract only the 22 evaluated joints and reshape
        pred_joints_eval = pred_joints_np[:, :, joint_used_xyz].reshape(b, 25, 22, 3)
        target_joints_eval = target_joints_np[:, :, joint_used_xyz].reshape(b, 25, 22, 3)
        
        metrics_collector.add_batch_results(pred_joints_eval, target_joints_eval)
        
        # Progress indicator
        if batch_idx % 20 == 0:
            print(f"  Processed {batch_idx + 1}/{len(pbar)} batches...")
    
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36

def test_comprehensive_h36m(config, model, dataloader):
    """Test with comprehensive metrics collection"""
    model.eval()
    
    # Initialize comprehensive metrics
    metrics_collector = ComprehensiveMetricsH36M()
    
    print(f"\n🚀 STARTING COMPREHENSIVE H36M EVALUATION")
    print(f"📊 Dataset size: {len(dataloader.dataset)} samples")
    print(f"📐 Data format: H36M standard (32 joints, 22 used for evaluation)")
    print(f"🔄 Prediction horizon: 25 frames (1 second at 25fps)")
    
    m_p3d_h36 = np.zeros([config.motion.h36m_target_length])
    titles = np.array(range(config.motion.h36m_target_length)) + 1
    joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
    num_samples = 0

    pbar = dataloader
    m_p3d_h36 = regress_pred_comprehensive(model, pbar, num_samples, joint_used_xyz, m_p3d_h36, metrics_collector)

    # Standard H36M results
    ret = {}
    for j in range(config.motion.h36m_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    standard_results = [round(ret[key][0], 1) for key in results_keys]
    
    # Compute comprehensive metrics
    metrics = metrics_collector.compute_comprehensive_metrics()
    
    # Print comprehensive report
    metrics_collector.print_comprehensive_report(metrics, "siMLPe H36M Model")
    
    # Print standard results for comparison
    print(f"\n📋 STANDARD H36M RESULTS (for paper comparison):")
    print(f"   {standard_results}")
    
    return metrics, standard_results

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-pth', type=str, default=None, help='path to trained model')
    parser.add_argument('--save-results', type=str, default=None, help='save results to CSV file')
    args = parser.parse_args()

    print("=" * 100)
    print("🎯 COMPREHENSIVE H36M EVALUATION")
    print("   siMLPe Model on Human3.6M Dataset")
    print("=" * 100)

    # Initialize model
    model = Model(config)

    # Load trained weights
    if args.model_pth and os.path.exists(args.model_pth):
        state_dict = torch.load(args.model_pth)
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Loaded model from {args.model_pth}")
    else:
        print(f"❌ Model file not found or not specified: {args.model_pth}")
        exit(1)

    model.eval()
    model.cuda()
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters()) / 1000000.0
    print(f"📊 Model parameters: {total_params:.2f}M")
    print(f"🏗️  Architecture: siMLPe (MLP-based motion prediction)")

    # Setup dataset
    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    dataset = H36MEval(config, 'test')

    dataloader = DataLoader(dataset, batch_size=128,
                          num_workers=1, drop_last=False,
                          shuffle=False, pin_memory=True)

    print(f"📊 Test dataset: Human3.6M evaluation set")
    print(f"🔄 Batch size: 128")
    
    # Run comprehensive evaluation
    metrics, standard_results = test_comprehensive_h36m(config, model, dataloader)
    
    # Save results if requested
    if args.save_results:
        results_df = pd.DataFrame([metrics])
        results_df.to_csv(args.save_results, index=False)
        print(f"💾 Results saved to {args.save_results}")
    
    print(f"\n✅ COMPREHENSIVE H36M EVALUATION COMPLETED!")
    
    # Print key takeaway
    print(f"\n🔑 KEY TAKEAWAY:")
    print(f"   H36M MPJPE: {metrics['mean_mpjpe']:.1f} ± {metrics['std_mpjpe']:.1f} mm")
    print(f"   Standard Results: {standard_results}")
    print(f"   This demonstrates siMLPe performance on the standard H36M benchmark!")

if __name__ == "__main__":
    main()