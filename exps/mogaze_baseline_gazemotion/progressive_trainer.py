import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter

class ProgressiveGazeTrainer:
    """
    Progressive training strategy for gated cross-attention model:
    
    Stage 1: Establish baseline (gate frozen at 0)
    Stage 2: Unlock gate gradually 
    Stage 3: Full multimodal training with alignment loss
    """
    
    def __init__(self, model, config, logger=None):
        self.model = model
        self.config = config
        self.logger = logger
        
        # Training stages configuration
        self.stages = {
            'baseline': {
                'iterations': 15000,  # 15k iterations for stable baseline
                'gate_frozen': True,
                'gate_value': 0.0,
                'alignment_loss_weight': 0.0,
                'description': "Establish 73.3mm baseline (gate=0, pose-only)"
            },
            'progressive': {
                'iterations': 15000,  # 15k iterations for gradual unlocking
                'gate_frozen': False,
                'gate_warmup': True,
                'alignment_loss_weight': 0.1,
                'description': "Progressive gate unlocking with gaze integration"
            },
            'full': {
                'iterations': 10000,  # 10k iterations for full training
                'gate_frozen': False,
                'gate_warmup': False,
                'alignment_loss_weight': 0.5,
                'description': "Full multimodal training with alignment loss"
            }
        }
        
        self.current_stage = 'baseline'
        self.stage_iteration = 0
        self.total_iterations = 0
        
        # Track training progress
        self.best_loss = float('inf')
        self.best_gate_weight = 0.0
        self.baseline_established = False
        
    def get_current_stage_info(self):
        """Get current training stage information"""
        stage_info = self.stages[self.current_stage].copy()
        stage_info['current_stage'] = self.current_stage  # Add this line
        stage_info['current_iteration'] = self.stage_iteration
        stage_info['total_iterations'] = self.total_iterations
        stage_info['progress'] = self.stage_iteration / stage_info['iterations']
        return stage_info
        
    def should_advance_stage(self):
        """Check if we should advance to next stage"""
        current_stage_info = self.stages[self.current_stage]
        return self.stage_iteration >= current_stage_info['iterations']
        
    def advance_stage(self):
        """Advance to next training stage"""
        if self.current_stage == 'baseline':
            self.baseline_established = True
            self.current_stage = 'progressive'
            if self.logger:
                print("🎯 Stage 1 Complete! Baseline established. Advancing to progressive training...")
                    
        elif self.current_stage == 'progressive':
            self.current_stage = 'full'
            if self.logger:
                print("🚀 Stage 2 Complete! Gate unlocked. Advancing to full multimodal training...")
                    
        elif self.current_stage == 'full':
            if self.logger:
                print("✅ Stage 3 Complete! Full training finished.")
            return False  # Training complete
            
        self.stage_iteration = 0
        return True
        
    def prepare_model_for_stage(self):
        """Configure model for current training stage"""
        stage_info = self.stages[self.current_stage]
        
        if stage_info.get('gate_frozen', False):
            # Freeze gate at specific value
            gate_value = stage_info.get('gate_value', 0.0)
            # Convert to logit space (sigmoid^-1)
            if gate_value <= 0:
                gate_logit = -10.0  # Very negative = sigmoid ≈ 0
            elif gate_value >= 1:
                gate_logit = 10.0   # Very positive = sigmoid ≈ 1  
            else:
                gate_logit = np.log(gate_value / (1 - gate_value))
                
            self.model.gaze_gate.data.fill_(gate_logit)
            self.model.gaze_gate.requires_grad = False
            
        else:
            # Unfreeze gate
            self.model.gaze_gate.requires_grad = True
            
    def get_alignment_loss_weight(self):
        """Get current alignment loss weight"""
        stage_info = self.stages[self.current_stage]
        base_weight = stage_info.get('alignment_loss_weight', 0.0)
        
        # Gradually increase alignment loss weight during progressive stage
        if self.current_stage == 'progressive' and stage_info.get('gate_warmup', False):
            progress = self.stage_iteration / stage_info['iterations']
            return base_weight * progress
            
        return base_weight
        
    def get_gate_learning_rate_multiplier(self):
        """Get learning rate multiplier for gate parameter"""
        if self.current_stage == 'baseline':
            return 0.0  # Gate frozen
        elif self.current_stage == 'progressive':
            # Slow gate learning initially
            progress = self.stage_iteration / self.stages['progressive']['iterations']
            return 0.1 + 0.9 * progress  # 0.1 → 1.0
        else:
            return 1.0  # Full learning rate
            
    def compute_progressive_loss(self, motion_pred, motion_target, attention_info=None):
        """Compute loss with progressive weighting"""
        # Base motion loss (always present)
        motion_loss = self._compute_motion_loss(motion_pred, motion_target)
        total_loss = motion_loss
        
        loss_info = {
            'motion_loss': motion_loss.item(),
            'total_loss': motion_loss.item(),
            'alignment_loss': 0.0,
            'gate_weight': self.model.get_gate_weight()
        }
        
        # Add alignment loss if available
        if attention_info and 'alignment_loss' in attention_info:
            alignment_weight = self.get_alignment_loss_weight()
            if alignment_weight > 0:
                alignment_loss = attention_info['alignment_loss']
                weighted_alignment_loss = alignment_weight * alignment_loss
                total_loss = total_loss + weighted_alignment_loss
                
                loss_info['alignment_loss'] = alignment_loss.item()
                loss_info['weighted_alignment_loss'] = weighted_alignment_loss.item()
                loss_info['total_loss'] = total_loss.item()
                
        return total_loss, loss_info
        
    def _compute_motion_loss(self, motion_pred, motion_target):
        """Compute standard motion prediction loss"""
        b, n, c = motion_target.shape
        
        # Extract pose component from target (first 63 dimensions) and move to CUDA
        motion_target_pose = motion_target[:, :, :63].cuda()
        
        # MPJPE loss
        joints = 21  # MoGaze has 21 joints
        motion_pred_joints = motion_pred.reshape(b, n, joints, 3).reshape(-1, 3)
        motion_target_joints = motion_target_pose.reshape(b, n, joints, 3).reshape(-1, 3)
        
        pose_loss = torch.mean(torch.norm(motion_pred_joints - motion_target_joints, 2, 1))
        
        # Velocity loss (if enabled)
        if self.config.use_relative_loss:
            motion_pred_full = motion_pred.reshape(b, n, joints, 3)
            motion_target_full = motion_target_pose.reshape(b, n, joints, 3)
            
            # Compute velocities
            dmotion_pred = motion_pred_full[:, 1:] - motion_pred_full[:, :-1]
            dmotion_target = motion_target_full[:, 1:] - motion_target_full[:, :-1]
            
            velocity_loss = torch.mean(torch.norm((dmotion_pred - dmotion_target).reshape(-1, 3), 2, 1))
            pose_loss = pose_loss + velocity_loss
            
        return pose_loss
        
    def step(self):
        """Advance one training step"""
        self.stage_iteration += 1
        self.total_iterations += 1
        
        # Check if we should advance to next stage
        if self.should_advance_stage():
            self.advance_stage()
            self.prepare_model_for_stage()
            
    def log_progress(self, loss_info, writer=None, iteration=None):
        """Log training progress with stage-specific information"""
        stage_info = self.get_current_stage_info()
        
        if self.logger:
            progress_pct = stage_info['progress'] * 100
            print(f"Stage {self.current_stage.upper()} [{progress_pct:.1f}%] - "
                  f"Gate: {loss_info['gate_weight']:.4f}, "
                  f"Motion Loss: {loss_info['motion_loss']:.4f}, "
                  f"Total Loss: {loss_info['total_loss']:.4f}")
            
        # TensorBoard logging
        if writer and iteration:
            writer.add_scalar(f'Loss/Motion_{self.current_stage}', loss_info['motion_loss'], iteration)
            writer.add_scalar(f'Loss/Total_{self.current_stage}', loss_info['total_loss'], iteration)
            writer.add_scalar(f'Gate/Weight_{self.current_stage}', loss_info['gate_weight'], iteration)
            writer.add_scalar('Stage/Progress', stage_info['progress'], iteration)
            
            if 'alignment_loss' in loss_info and loss_info['alignment_loss'] > 0:
                writer.add_scalar(f'Loss/Alignment_{self.current_stage}', loss_info['alignment_loss'], iteration)
                
    def save_checkpoint(self, path, optimizer=None, additional_info=None):
        """Save training checkpoint with stage information"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'current_stage': self.current_stage,
            'stage_iteration': self.stage_iteration,
            'total_iterations': self.total_iterations,
            'best_loss': self.best_loss,
            'baseline_established': self.baseline_established,
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        if additional_info:
            checkpoint.update(additional_info)
            
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path, optimizer=None):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_stage = checkpoint.get('current_stage', 'baseline')
        self.stage_iteration = checkpoint.get('stage_iteration', 0)
        self.total_iterations = checkpoint.get('total_iterations', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.baseline_established = checkpoint.get('baseline_established', False)
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Prepare model for current stage
        self.prepare_model_for_stage()
        
        return checkpoint
        
    def is_training_complete(self):
        """Check if all training stages are complete"""
        return (self.current_stage == 'full' and 
                self.should_advance_stage())
                
    def get_training_summary(self):
        """Get summary of training progress"""
        total_planned = sum(stage['iterations'] for stage in self.stages.values())
        
        return {
            'current_stage': self.current_stage,
            'stage_description': self.stages[self.current_stage]['description'],
            'stage_progress': f"{self.stage_iteration}/{self.stages[self.current_stage]['iterations']}",
            'total_progress': f"{self.total_iterations}/{total_planned}",
            'baseline_established': self.baseline_established,
            'current_gate_weight': self.model.get_gate_weight(),
            'training_complete': self.is_training_complete()
        }


class GateMonitor:
    """Monitor gate learning dynamics during training"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.gate_history = []
        self.loss_history = []
        
    def update(self, gate_weight, loss):
        self.gate_history.append(gate_weight)
        self.loss_history.append(loss)
        
        # Keep only recent history
        if len(self.gate_history) > self.window_size:
            self.gate_history.pop(0)
            self.loss_history.pop(0)
            
    def get_gate_velocity(self):
        """Get rate of gate change"""
        if len(self.gate_history) < 10:
            return 0.0
        return np.mean(np.diff(self.gate_history[-10:]))
        
    def is_gate_learning(self, threshold=0.001):
        """Check if gate is actively learning"""
        velocity = abs(self.get_gate_velocity())
        return velocity > threshold
        
    def get_statistics(self):
        if not self.gate_history:
            return {}
            
        return {
            'current_gate': self.gate_history[-1],
            'mean_gate': np.mean(self.gate_history),
            'gate_std': np.std(self.gate_history),
            'gate_velocity': self.get_gate_velocity(),
            'is_learning': self.is_gate_learning(),
            'min_gate': np.min(self.gate_history),
            'max_gate': np.max(self.gate_history)
        }