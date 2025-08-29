import argparse
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from collections import defaultdict
import pandas as pd
from scipy import stats
import psutil
import threading

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

class EfficiencyTracker:
    """Track CPU usage, memory, and timing metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.cpu_percentages = []
        self.memory_usage = []
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        self.end_time = None
        
    def start_monitoring(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return metrics"""
        self.monitoring = False
        self.end_time = time.time()
        if self.monitor_thread:
            self.monitor_thread.join()
            
        total_time = self.end_time - self.start_time
        return {
            'total_time_s': total_time,
            'avg_cpu_percent': np.mean(self.cpu_percentages) if self.cpu_percentages else 0,
            'peak_cpu_percent': np.max(self.cpu_percentages) if self.cpu_percentages else 0,
            'avg_memory_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
            'peak_memory_mb': np.max(self.memory_usage) if self.memory_usage else 0
        }
        
    def _monitor_resources(self):
        """Internal method to monitor resources"""
        process = psutil.Process()
        while self.monitoring:
            try:
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                self.cpu_percentages.append(cpu_percent)
                self.memory_usage.append(memory_mb)
                
                time.sleep(0.1)  # Monitor every 100ms
            except:
                break

class MPJPEMetrics:
    """Streamlined metrics collection focused on MPJPE and statistics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sample_errors = []  # [num_samples] - per sample average errors
        self.temporal_errors = defaultdict(list)  # per time step errors
        self.gate_weights = []
        
    def add_batch_results(self, pred_joints, target_joints, gate_weight=None):
        """Add results from a batch"""
        # pred_joints: [batch, 30, 21, 3]
        # target_joints: [batch, 30, 21, 3]
        
        batch_size = pred_joints.shape[0]
        
        # Calculate per-joint errors (in mm)
        joint_errors = np.linalg.norm(pred_joints - target_joints, axis=-1) * 1000
        
        # Store sample-wise average errors
        sample_avg_errors = np.mean(joint_errors, axis=(1, 2))
        self.sample_errors.extend(sample_avg_errors)
        
        # Store temporal errors (average over joints and batch)
        for t in range(30):
            frame_error = np.mean(joint_errors[:, t, :])
            self.temporal_errors[t].append(frame_error)
            
        # Store gate weights
        if gate_weight is not None:
            self.gate_weights.extend([gate_weight] * batch_size)
    
    def compute_metrics(self):
        """Compute MPJPE metrics with confidence intervals"""
        sample_errors = np.array(self.sample_errors)
        
        # Basic statistics
        mean_mpjpe = np.mean(sample_errors)
        std_mpjpe = np.std(sample_errors, ddof=1)  # Sample standard deviation
        n_samples = len(sample_errors)
        
        # 95% Confidence Interval for the mean
        confidence_level = 0.95
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df=n_samples-1)
        margin_error = t_critical * (std_mpjpe / np.sqrt(n_samples))
        ci_lower = mean_mpjpe - margin_error
        ci_upper = mean_mpjpe + margin_error
        
        # Temporal breakdown (key time horizons)
        temporal_means = []
        temporal_stds = []
        temporal_cis = []
        
        # MoGaze standard time points (30fps): 200ms=6, 400ms=12, 600ms=18, 800ms=24, 1000ms=30
        key_frames = [5, 11, 17, 23, 29]  # 0-indexed
        key_times = [200, 400, 600, 800, 1000]
        
        temporal_results = {}
        for i, (frame_idx, time_ms) in enumerate(zip(key_frames, key_times)):
            if frame_idx < 30 and frame_idx in self.temporal_errors:
                t_errors = np.array(self.temporal_errors[frame_idx])
                t_mean = np.mean(t_errors)
                t_std = np.std(t_errors, ddof=1)
                
                # 95% CI for this time point
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
                
                temporal_means.append(t_mean)
                temporal_stds.append(t_std)
                temporal_cis.append((t_mean - t_margin, t_mean + t_margin))
        
        # Gate utilization metrics
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

def test_with_efficiency_metrics(config, model, dataloader):
    """Test with efficiency and MPJPE metrics"""
    model.eval()
    
    # Force CPU usage
    device = torch.device('cpu')
    model = model.to(device)
    
    # DCT matrices
    dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
    dct_m = torch.tensor(dct_m).float().to(device).unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().to(device).unsqueeze(0)
    
    # Initialize metrics collectors
    mpjpe_metrics = MPJPEMetrics()
    efficiency_tracker = EfficiencyTracker()
    
    print(f"Testing on CPU with {len(dataloader.dataset)} samples")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Total batches: {len(dataloader)}")
    
    # Start efficiency monitoring
    efficiency_tracker.start_monitoring()
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, (motion_input, motion_target) in enumerate(dataloader):
            # Move to CPU
            motion_input = motion_input.to(device)
            motion_target = motion_target.to(device)
            
            b, n, c = motion_input.shape
            
            # Time single batch inference
            batch_start = time.time()
            
            # Auto-regressive prediction
            outputs = []
            step = config.motion.h36m_target_length_train  # 10 frames per step
            
            if step == 30:
                num_step = 1
            else:
                num_step = 30 // step + 1
            
            current_input = motion_input.clone()
            
            for idx in range(num_step):
                # Forward pass
                if config.deriv_input:
                    motion_input_ = current_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_)
                else:
                    motion_input_ = current_input.clone()
                
                # Single inference step timing
                step_start = time.time()
                output = model(motion_input_)
                step_time = time.time() - step_start
                
                output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
                
                if config.deriv_output:
                    pose_offset = current_input[:, -1:, :63]
                    output = output + pose_offset
                
                # Prepare for next iteration
                batch_size, seq_len, _ = output.shape
                last_gaze = current_input[:, -1:, 63:66]
                extended_gaze = last_gaze.repeat(1, seq_len, 1)
                
                output_with_gaze = torch.cat([output, extended_gaze], dim=2)
                outputs.append(output)
                
                current_input = torch.cat([current_input[:, step:], output_with_gaze], axis=1)
            
            batch_time = time.time() - batch_start
            inference_times.append(batch_time)
            
            # Process results
            motion_pred = torch.cat(outputs, axis=1)[:, :30]  # [batch, 30, 63]
            
            motion_pred_np = motion_pred.cpu().numpy()
            motion_target_np = motion_target.cpu().numpy()
            
            target_pose = motion_target_np[:, :, :63]  # [batch, 30, 63]
            
            # Reshape for joint-wise analysis
            pred_pose_3d = motion_pred_np.reshape(b, 30, 21, 3)
            target_pose_3d = target_pose.reshape(b, 30, 21, 3)
            
            # Add to metrics
            mpjpe_metrics.add_batch_results(
                pred_pose_3d, 
                target_pose_3d, 
                gate_weight=model.get_gate_weight()
            )
            
            # Progress indicator
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Stop monitoring and get results
    efficiency_metrics = efficiency_tracker.stop_monitoring()
    
    # Compute final metrics
    metrics = mpjpe_metrics.compute_metrics()
    
    # Add efficiency metrics
    metrics['efficiency'] = efficiency_metrics
    metrics['efficiency']['avg_inference_time_per_batch'] = np.mean(inference_times)
    metrics['efficiency']['std_inference_time_per_batch'] = np.std(inference_times)
    metrics['efficiency']['inference_fps'] = len(dataloader.dataset) / efficiency_metrics['total_time_s']
    
    return metrics

def print_comparison_report(metrics, model_name="Gated Cross-Attention"):
    """Print focused comparison report"""
    print("=" * 80)
    print(f"MPJPE EVALUATION REPORT - {model_name}")
    print("=" * 80)
    
    overall = metrics['overall']
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  MPJPE: {overall['mean_mpjpe']:.1f} ± {overall['std_mpjpe']:.1f} mm")
    print(f"  95% CI: [{overall['ci_95_lower']:.1f}, {overall['ci_95_upper']:.1f}] mm")
    print(f"  Samples: {overall['n_samples']:,}")
    
    print(f"\nTEMPORAL BREAKDOWN:")
    temporal = metrics['temporal']
    for time_ms in [200, 400, 600, 800, 1000]:
        key = f'mpjpe_{time_ms}ms'
        if key in temporal:
            t = temporal[key]
            print(f"  {time_ms:4d}ms: {t['mean']:5.1f} ± {t['std']:4.1f} mm "
                  f"[{t['ci_lower']:5.1f}, {t['ci_upper']:5.1f}]")
    
    if metrics['gate']:
        gate = metrics['gate']
        print(f"\nGAZE UTILIZATION:")
        print(f"  Gate Weight: {gate['mean_gate_weight']:.4f} ± {gate['std_gate_weight']:.4f}")
        print(f"  Utilization Rate: {gate['gate_utilization_rate']*100:.1f}% (samples with gate > 0.1)")
    
    efficiency = metrics['efficiency']
    print(f"\nEFFICIENCY METRICS (CPU):")
    print(f"  Total Time: {efficiency['total_time_s']:.2f} seconds")
    print(f"  Inference FPS: {efficiency['inference_fps']:.2f} samples/second")
    print(f"  Avg Batch Time: {efficiency['avg_inference_time_per_batch']*1000:.1f} ± "
          f"{efficiency['std_inference_time_per_batch']*1000:.1f} ms")
    print(f"  Peak CPU: {efficiency['peak_cpu_percent']:.1f}%")
    print(f"  Peak Memory: {efficiency['peak_memory_mb']:.1f} MB")
    
    print(f"\nGAZEMOTION COMPARISON BASELINE:")
    print(f"  GazeMotion p7 test: ~84.8mm MPJPE")
    print(f"  Our improvement: {((84.8 - overall['mean_mpjpe']) / 84.8 * 100):.1f}%")
    print(f"  Statistical significance: {overall['ci_95_upper'] < 84.8}")
    
    print("=" * 80)

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
    
    print("STREAMLINED MPJPE AND EFFICIENCY EVALUATION")
    print("Target: Comparison with GazeMotion performance and efficiency")
    
    # Initialize model (force CPU)
    model = create_gated_model(config)
    
    # Load trained weights
    if os.path.exists(args.model_pth):
        try:
            state_dict = torch.load(args.model_pth, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict, strict=True)
        except:
            checkpoint = torch.load(args.model_pth, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                print(f"Loaded checkpoint from: {checkpoint.get('strategy', 'unknown')} training")
            else:
                model.load_state_dict(checkpoint, strict=True)
        print(f"Model loaded from {args.model_pth}")
    else:
        print(f"Model file not found: {args.model_pth}")
        exit(1)
    
    model.eval()
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters()) / 1000000.0
    print(f"Model parameters: {total_params:.2f}M")
    print(f"Current gate weight: {model.get_gate_weight():.4f}")
    print(f"Running on CPU for efficiency comparison")
    
    # Setup test dataset
    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    test_dataset = MoGazeGazeEval(config, 'test')
    
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                          num_workers=0, drop_last=False,
                          shuffle=False, pin_memory=False)  # No pinning for CPU
    
    # Run evaluation
    metrics = test_with_efficiency_metrics(config, model, dataloader)
    
    # Print results
    print_comparison_report(metrics)
    
    # Save results if requested
    if args.save_results:
        # Flatten results for CSV
        results_flat = {}
        results_flat.update(metrics['overall'])
        results_flat.update({f'temporal_{k}': v['mean'] for k, v in metrics['temporal'].items()})
        results_flat.update({f'temporal_{k}_std': v['std'] for k, v in metrics['temporal'].items()})
        results_flat.update(metrics['gate'])
        results_flat.update(metrics['efficiency'])
        
        pd.DataFrame([results_flat]).to_csv(args.save_results, index=False)
        print(f"Results saved to {args.save_results}")

if __name__ == "__main__":
    main()