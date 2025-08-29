import argparse
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from scipy import stats

from mogaze_config_3d import config
from model import siMLPe as Model
from mogaze_dataset_3d import MoGaze3DEval

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

class ComprehensiveMetrics3D:
    """Comprehensive metrics collection for 3D position motion prediction evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.frame_errors = []  # [num_samples, 30, 21] - per frame, per joint errors
        self.sample_errors = []  # [num_samples] - per sample average errors
        self.joint_errors = defaultdict(list)  # per joint type errors
        self.temporal_errors = defaultdict(list)  # per time step errors
        
    def add_batch_results(self, pred_joints, target_joints):
        """Add results from a batch for comprehensive analysis"""
        # pred_joints: [batch, 30, 21, 3]
        # target_joints: [batch, 30, 21, 3]
        
        batch_size = pred_joints.shape[0]
        
        # Calculate per-joint errors (in mm)
        joint_errors = np.linalg.norm(pred_joints - target_joints, axis=-1) * 1000
        
        # Store frame-wise errors
        self.frame_errors.extend(joint_errors)
        
        # Store sample-wise average errors
        sample_avg_errors = np.mean(joint_errors, axis=(1, 2))
        self.sample_errors.extend(sample_avg_errors)
        
        # Store temporal errors (average over joints and batch)
        for t in range(30):
            frame_error = np.mean(joint_errors[:, t, :])
            self.temporal_errors[t].append(frame_error)
            
        # Store per-joint errors (average over time and batch)
        for j in range(21):
            joint_error = np.mean(joint_errors[:, :, j])
            self.joint_errors[j].extend([joint_error] * batch_size)
    
    def compute_comprehensive_metrics(self):
        """Compute all comprehensive metrics"""
        frame_errors = np.array(self.frame_errors)  # [num_samples, 30, 21]
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
        for t in range(30):
            if t in self.temporal_errors:
                t_errors = self.temporal_errors[t]
                temporal_means.append(np.mean(t_errors))
                temporal_stds.append(np.std(t_errors))
            else:
                temporal_means.append(0)
                temporal_stds.append(0)
        
        metrics['temporal_means'] = np.array(temporal_means)
        metrics['temporal_stds'] = np.array(temporal_stds)
        
        # Key time horizons (MoGaze standard: 200ms, 400ms, 600ms, 800ms, 1000ms)
        # At 30fps: 200ms=6frames, 400ms=12frames, 600ms=18frames, 800ms=24frames, 1000ms=30frames
        key_frames = [5, 11, 17, 23, 29]  # 0-indexed
        key_times = [200, 400, 600, 800, 1000]
        
        for i, (frame_idx, time_ms) in enumerate(zip(key_frames, key_times)):
            if frame_idx < len(temporal_means):
                metrics[f'mpjpe_{time_ms}ms'] = temporal_means[frame_idx]
                metrics[f'std_{time_ms}ms'] = temporal_stds[frame_idx]
        
        # === PER-JOINT ANALYSIS ===
        joint_names = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
        ]
        
        joint_stats = {}
        for j in range(min(21, len(joint_names))):
            if j in self.joint_errors:
                j_errors = self.joint_errors[j]
                joint_stats[joint_names[j]] = {
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
    
    def print_comprehensive_report(self, metrics, model_name="3D Position Model"):
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
        print(f"\n⏱️  TEMPORAL PERFORMANCE:")
        key_times = [200, 400, 600, 800, 1000]
        for time_ms in key_times:
            if f'mpjpe_{time_ms}ms' in metrics:
                mean_key = f'mpjpe_{time_ms}ms'
                std_key = f'std_{time_ms}ms'
                print(f"   {time_ms:4d}ms: {metrics[mean_key]:5.1f} ± {metrics[std_key]:4.1f} mm")
        
        if 'degradation_rate' in metrics:
            print(f"   Degradation Rate: {metrics['degradation_rate']:.2f} mm/frame (R²={metrics['degradation_r2']:.3f})")
        
        # === COMPARISON WITH BENCHMARKS ===
        print(f"\n📚 GAZEMOTION PAPER COMPARISON:")
        
        # GazeMotion benchmarks
        gazemotion_baseline = {
            'name': 'GazeMotion siMLPe Baseline',
            'results': [40.6, 72.0, 108.8, 152.6, 201.0],
            'average': 99.5
        }
        
        gazemotion_method = {
            'name': 'GazeMotion + Gaze',
            'results': [25.8, 53.3, 85.8, 122.0, 160.0],
            'average': 75.9
        }
        
        print(f"   GazeMotion Baseline: {gazemotion_baseline['average']:.1f} mm average")
        print(f"   GazeMotion + Gaze:   {gazemotion_method['average']:.1f} mm average")
        
        # Performance comparison
        our_avg = metrics['mean_mpjpe']
        
        if our_avg < gazemotion_baseline['average']:
            improvement = ((gazemotion_baseline['average'] - our_avg) / gazemotion_baseline['average']) * 100
            print(f"   ✅ vs Baseline: {improvement:.1f}% BETTER! ({our_avg:.1f} vs {gazemotion_baseline['average']:.1f})")
        else:
            diff = our_avg - gazemotion_baseline['average']
            print(f"   ⚠️  vs Baseline: {diff:.1f}mm higher ({our_avg:.1f} vs {gazemotion_baseline['average']:.1f})")
        
        if our_avg < gazemotion_method['average']:
            improvement = ((gazemotion_method['average'] - our_avg) / gazemotion_method['average']) * 100
            print(f"   🚀 vs GazeMotion+Gaze: {improvement:.1f}% BETTER! ({our_avg:.1f} vs {gazemotion_method['average']:.1f})")
        else:
            diff = our_avg - gazemotion_method['average']
            print(f"   📈 vs GazeMotion+Gaze: {diff:.1f}mm gap ({our_avg:.1f} vs {gazemotion_method['average']:.1f})")
        
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
        print(f"   Model Type: Pure 3D position training (no gaze)")
        print(f"   Architecture: siMLPe MLP with DCT encoding")
        print(f"   Data: MoGaze 3D positions (21 joints × 3 coords)")
        print(f"   Evaluation: Direct MPJPE (no conversion artifacts)")
        
        print("=" * 100)

def test_comprehensive_3d_evaluation(config, model, dataloader):
    """Test 3D position model with comprehensive metrics collection"""
    model.eval()
    
    # DCT matrices
    dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
    dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)
    
    # Initialize comprehensive metrics
    metrics_collector = ComprehensiveMetrics3D()
    
    print(f"\n🚀 STARTING COMPREHENSIVE 3D EVALUATION")
    print(f"📊 Dataset size: {len(dataloader.dataset)} samples")
    print(f"📐 Data format: 3D positions (21 joints × 3 coords = 63D)")
    print(f"🔄 Direct MPJPE calculation (no conversion needed)")
    
    with torch.no_grad():
        for batch_idx, (motion_input, motion_target) in enumerate(dataloader):
            # Convert numpy arrays to tensors if needed
            if isinstance(motion_input, np.ndarray):
                motion_input = torch.tensor(motion_input).float()
            if isinstance(motion_target, np.ndarray):
                motion_target = torch.tensor(motion_target).float()
                
            motion_input = motion_input.cuda()
            motion_target = motion_target.cuda()
            
            b, n, c = motion_input.shape
            
            # Auto-regressive prediction (same as H36M approach)
            outputs = []
            step = config.motion.h36m_target_length_train  # 10 frames per step
            
            if step == 30:
                num_step = 1
            else:
                num_step = 30 // step + 1
            
            current_input = motion_input.clone()
            
            for idx in range(num_step):
                # Forward pass with exact same logic as training
                if config.deriv_input:
                    motion_input_ = current_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_)
                else:
                    motion_input_ = current_input.clone()
                
                output = model(motion_input_)
                output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
                
                if config.deriv_output:
                    output = output + current_input[:, -1:, :].repeat(1, step, 1)
                
                output = output.reshape(-1, 63)  # 21 joints × 3 positions
                output = output.reshape(b, step, -1)
                outputs.append(output)
                
                # Update input for next iteration
                current_input = torch.cat([current_input[:, step:], output], axis=1)
            
            # Concatenate outputs and take first 30 frames
            motion_pred = torch.cat(outputs, axis=1)[:, :30]  # [batch, 30, 63]
            
            # Convert to numpy for analysis
            motion_pred_np = motion_pred.cpu().numpy()  # [batch, 30, 63]
            motion_target_np = motion_target.cpu().numpy()  # [batch, 30, 63]
            
            # Reshape to [batch, 30, 21, 3] for joint-wise error calculation
            pred_3d = motion_pred_np.reshape(b, 30, 21, 3)  # [batch, 30, 21, 3]
            target_3d = motion_target_np.reshape(b, 30, 21, 3)  # [batch, 30, 21, 3]
            
            # Add to comprehensive metrics
            metrics_collector.add_batch_results(pred_3d, target_3d)
            
            # Progress indicator
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}")
    
    # Compute comprehensive metrics
    metrics = metrics_collector.compute_comprehensive_metrics()
    
    # Print comprehensive report
    metrics_collector.print_comprehensive_report(metrics, "siMLPe 3D Position Baseline")
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-pth', type=str, 
                       default='log_3d/snapshot/model-iter-40000.pth',
                       help='path to trained 3D position model')
    parser.add_argument('--save-results', type=str, default=None,
                       help='save results to CSV file')
    args = parser.parse_args()
    
    print("=" * 100)
    print("🎯 COMPREHENSIVE 3D POSITION EVALUATION")
    print("   Baseline 3D Position Model - No Gaze Information")
    print("=" * 100)
    
    # Initialize model
    model = Model(config)
    
    # Load trained weights
    if os.path.exists(args.model_pth):
        state_dict = torch.load(args.model_pth, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Loaded 3D position model from {args.model_pth}")
    else:
        print(f"❌ Model file not found: {args.model_pth}")
        print("Please train the 3D position model first using train_mogaze_3d.py")
        exit(1)
    
    model.eval()
    model.cuda()
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters()) / 1000000.0
    print(f"📊 Model parameters: {total_params:.2f}M")
    print(f"🏗️  Architecture: Pure siMLPe (no gaze components)")
    
    # Setup test dataset
    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    test_dataset = MoGaze3DEval(config, 'test')
    
    dataloader = DataLoader(test_dataset, batch_size=16,
                          num_workers=0, drop_last=False,
                          shuffle=False, pin_memory=True)
    
    # Run comprehensive evaluation
    metrics = test_comprehensive_3d_evaluation(config, model, dataloader)
    
    # Save results if requested
    if args.save_results:
        results_df = pd.DataFrame([metrics])
        results_df.to_csv(args.save_results, index=False)
        print(f"💾 Results saved to {args.save_results}")
    
    print(f"\n✅ COMPREHENSIVE 3D EVALUATION COMPLETED!")
    
    # Print key takeaway for comparison with gated model
    print(f"\n🔑 KEY TAKEAWAY FOR COMPARISON:")
    print(f"   3D Baseline MPJPE: {metrics['mean_mpjpe']:.1f} ± {metrics['std_mpjpe']:.1f} mm")
    print(f"   This serves as the baseline for measuring gaze information value")
    print(f"   Any gated model improvement over this shows gaze utilization success!")

if __name__ == "__main__":
    main()