import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter

class AggressiveGateTrainer:
    """
    Aggressive gate-opening schedule for the gated MLP.

    Stages (80k iterations in total):
      baseline   (10k): gate frozen closed, pose pathway only
      aggressive (40k): gate re-initialised to sigmoid(-2) ~ 0.12 and trained with a
                        high learning-rate multiplier (decaying exponentially from
                        max_gate_lr_multiplier to initial_gate_lr_multiplier),
                        gate regularisation and a ramped alignment loss (weight 0.5)
      refinement (30k): alignment loss weight 1.0, gate LR multiplier 2.0
    """

    def __init__(self, model, config, logger=None):
        self.model = model
        self.config = config
        self.logger = logger

        self.stages = {
            'baseline': {
                'iterations': 10000,
                'gate_frozen': True,
                'gate_value': 0.0,
                'alignment_loss_weight': 0.0,
                'description': "Gate frozen closed (pose pathway only)"
            },
            'aggressive': {
                'iterations': 40000,
                'gate_frozen': False,
                'gate_warmup': True,
                'alignment_loss_weight': 0.5,
                'gate_regularization': True,
                'temperature_annealing': True,
                'description': "Gate opening with high gate LR, regularisation and alignment loss"
            },
            'refinement': {
                'iterations': 30000,
                'gate_frozen': False,
                'gate_warmup': False,
                'alignment_loss_weight': 1.0,
                'gate_regularization': False,
                'description': "Joint refinement with alignment loss"
            }
        }

        self.current_stage = 'baseline'
        self.stage_iteration = 0
        self.total_iterations = 0

        # Gate learning-rate multiplier range and regularisation weight
        self.initial_gate_lr_multiplier = 5.0
        self.max_gate_lr_multiplier = 10.0
        self.gate_regularization_weight = 0.1
        self.temperature_schedule = True  # logged only

        self.gate_history = []
        self.gate_gradients = []
        self.best_gate_weight = 0.0
        self.gate_stuck_counter = 0

    def get_current_stage_info(self):
        """Current stage settings plus progress information."""
        stage_info = self.stages[self.current_stage].copy()
        stage_info['current_stage'] = self.current_stage
        stage_info['current_iteration'] = self.stage_iteration
        stage_info['total_iterations'] = self.total_iterations
        stage_info['progress'] = self.stage_iteration / stage_info['iterations']

        stage_info['gate_lr_multiplier'] = self.get_gate_learning_rate_multiplier()
        stage_info['temperature'] = self.get_temperature()

        return stage_info

    def should_advance_stage(self):
        """Check if we should advance to next stage"""
        current_stage_info = self.stages[self.current_stage]
        return self.stage_iteration >= current_stage_info['iterations']

    def advance_stage(self):
        """Move to the next training stage."""
        if self.current_stage == 'baseline':
            self.current_stage = 'aggressive'
            if self.logger:
                print("Entering stage: aggressive")

        elif self.current_stage == 'aggressive':
            self.current_stage = 'refinement'
            if self.logger:
                print("Entering stage: refinement")

        elif self.current_stage == 'refinement':
            if self.logger:
                print("All training stages completed")
            return False

        self.stage_iteration = 0
        return True

    def prepare_model_for_stage(self):
        """Freeze or unfreeze the gate for the current stage."""
        stage_info = self.stages[self.current_stage]

        if stage_info.get('gate_frozen', False):
            # sigmoid(-10) ~ 0: gate closed
            self.model.gaze_gate.data.fill_(-10.0)
            self.model.gaze_gate.requires_grad = False
        else:
            self.model.gaze_gate.requires_grad = True

            # At the start of the aggressive stage, open the gate slightly
            if self.current_stage == 'aggressive' and self.stage_iteration == 0:
                self.model.gaze_gate.data.fill_(-2.0)  # sigmoid(-2) ~ 0.12
                print(f"Gate initialised to {self.model.get_gate_weight():.4f}")

    def get_gate_learning_rate_multiplier(self):
        """Learning-rate multiplier applied to the gate parameter."""
        if self.current_stage == 'baseline':
            return 0.0  # Gate frozen

        elif self.current_stage == 'aggressive':
            progress = self.stage_iteration / self.stages['aggressive']['iterations']

            base_multiplier = self.initial_gate_lr_multiplier
            max_multiplier = self.max_gate_lr_multiplier

            # Exponential decay from max_multiplier to base_multiplier
            current_multiplier = max_multiplier * (base_multiplier / max_multiplier) ** progress

            return max(current_multiplier, 1.0)

        else:  # refinement
            return 2.0

    def get_temperature(self):
        """Temperature schedule (logged only)."""
        if not self.temperature_schedule or self.current_stage == 'baseline':
            return 1.0

        if self.current_stage == 'aggressive':
            progress = self.stage_iteration / self.stages['aggressive']['iterations']
            return 5.0 * (0.2 ** progress)  # 5.0 -> 1.0

        return 1.0

    def get_gate_regularization_loss(self):
        """Gate regularisation: penalise gate values outside [0.2, 0.8]."""
        if self.current_stage != 'aggressive':
            return 0.0

        gate_weight = self.model.get_gate_weight()

        target_range = (0.2, 0.8)

        if gate_weight < target_range[0]:
            reg_loss = (target_range[0] - gate_weight) ** 2
        elif gate_weight > target_range[1]:
            reg_loss = 0.1 * (gate_weight - target_range[1]) ** 2
        else:
            reg_loss = 0.0

        return reg_loss * self.gate_regularization_weight

    def compute_aggressive_loss(self, motion_pred, motion_target, attention_info=None):
        """Motion loss + weighted alignment loss + gate regularisation."""
        motion_loss = self._compute_motion_loss(motion_pred, motion_target)
        total_loss = motion_loss

        loss_info = {
            'motion_loss': motion_loss.item(),
            'total_loss': motion_loss.item(),
            'alignment_loss': 0.0,
            'gate_regularization_loss': 0.0,
            'gate_weight': self.model.get_gate_weight(),
            'temperature': self.get_temperature()
        }

        if attention_info and 'alignment_loss' in attention_info:
            alignment_weight = self.get_alignment_loss_weight()
            if alignment_weight > 0:
                alignment_loss = attention_info['alignment_loss']
                weighted_alignment_loss = alignment_weight * alignment_loss
                total_loss = total_loss + weighted_alignment_loss

                loss_info['alignment_loss'] = alignment_loss.item()
                loss_info['weighted_alignment_loss'] = weighted_alignment_loss.item()

        if self.stages[self.current_stage].get('gate_regularization', False):
            gate_reg_loss = self.get_gate_regularization_loss()
            if gate_reg_loss > 0:
                total_loss = total_loss + gate_reg_loss
                loss_info['gate_regularization_loss'] = gate_reg_loss

        loss_info['total_loss'] = total_loss.item()
        return total_loss, loss_info

    def get_alignment_loss_weight(self):
        """Alignment loss weight; ramped with a sigmoid during the aggressive stage."""
        stage_info = self.stages[self.current_stage]
        base_weight = stage_info.get('alignment_loss_weight', 0.0)

        if self.current_stage == 'aggressive' and stage_info.get('gate_warmup', False):
            progress = self.stage_iteration / stage_info['iterations']
            # Sigmoid ramp centred at 30% of the stage
            ramp_factor = 1.0 / (1.0 + np.exp(-10 * (progress - 0.3)))
            return base_weight * ramp_factor

        return base_weight

    def _compute_motion_loss(self, motion_pred, motion_target):
        """Position loss + velocity loss on the 21 joints."""
        b, n, c = motion_target.shape

        motion_target_pose = motion_target[:, :, :63].cuda()

        joints = 21
        motion_pred_joints = motion_pred.reshape(b, n, joints, 3).reshape(-1, 3)
        motion_target_joints = motion_target_pose.reshape(b, n, joints, 3).reshape(-1, 3)

        pose_loss = torch.mean(torch.norm(motion_pred_joints - motion_target_joints, 2, 1))

        if hasattr(self.config, 'use_relative_loss') and self.config.use_relative_loss:
            motion_pred_full = motion_pred.reshape(b, n, joints, 3)
            motion_target_full = motion_target_pose.reshape(b, n, joints, 3)

            dmotion_pred = motion_pred_full[:, 1:] - motion_pred_full[:, :-1]
            dmotion_target = motion_target_full[:, 1:] - motion_target_full[:, :-1]

            velocity_loss = torch.mean(torch.norm((dmotion_pred - dmotion_target).reshape(-1, 3), 2, 1))
            pose_loss = pose_loss + velocity_loss

        return pose_loss

    def step(self):
        """Advance one iteration and switch stage when the current one is finished."""
        current_gate = self.model.get_gate_weight()

        self.gate_history.append(current_gate)

        if len(self.gate_history) > 100:
            recent_std = np.std(self.gate_history[-100:])
            if recent_std < 0.001:
                self.gate_stuck_counter += 1
            else:
                self.gate_stuck_counter = 0

        if current_gate > self.best_gate_weight:
            self.best_gate_weight = current_gate

        self.stage_iteration += 1
        self.total_iterations += 1

        if self.should_advance_stage():
            self.advance_stage()
            self.prepare_model_for_stage()

    def log_aggressive_progress(self, loss_info, writer=None, iteration=None):
        """Print and TensorBoard-log the gate state."""
        stage_info = self.get_current_stage_info()

        if self.logger:
            progress_pct = stage_info['progress'] * 100
            gate_lr_mult = stage_info['gate_lr_multiplier']
            temp = stage_info['temperature']

            print(f"Stage {self.current_stage} [{progress_pct:.1f}%] - "
                  f"gate: {loss_info['gate_weight']:.4f} (best: {self.best_gate_weight:.4f}) | "
                  f"gate LR x{gate_lr_mult:.1f}")

            if 'gate_regularization_loss' in loss_info and loss_info['gate_regularization_loss'] > 0:
                print(f"   gate regularisation loss: {loss_info['gate_regularization_loss']:.4f}")

            if self.gate_stuck_counter > 50:
                print(f"   gate unchanged for {self.gate_stuck_counter} iterations")

        if writer and iteration:
            writer.add_scalar(f'Aggressive/Gate_Weight_{self.current_stage}', loss_info['gate_weight'], iteration)
            writer.add_scalar(f'Aggressive/Gate_LR_Multiplier_{self.current_stage}', stage_info['gate_lr_multiplier'], iteration)
            writer.add_scalar(f'Aggressive/Temperature_{self.current_stage}', stage_info['temperature'], iteration)
            writer.add_scalar('Aggressive/Best_Gate_Weight', self.best_gate_weight, iteration)

            if 'gate_regularization_loss' in loss_info:
                writer.add_scalar(f'Loss/Gate_Regularization_{self.current_stage}', loss_info['gate_regularization_loss'], iteration)

    def get_aggressive_summary(self):
        """Summary of the training state."""
        total_planned = sum(stage['iterations'] for stage in self.stages.values())

        return {
            'strategy': 'AGGRESSIVE GATE LEARNING',
            'current_stage': self.current_stage,
            'stage_description': self.stages[self.current_stage]['description'],
            'total_progress': f"{self.total_iterations}/{total_planned}",
            'current_gate_weight': self.model.get_gate_weight(),
            'best_gate_weight': self.best_gate_weight,
            'gate_learning_rate_multiplier': self.get_gate_learning_rate_multiplier(),
            'temperature': self.get_temperature(),
            'gate_stuck_counter': self.gate_stuck_counter,
            'training_complete': self.is_training_complete()
        }

    def save_checkpoint(self, path, optimizer=None, additional_info=None):
        """Save model, optimizer and schedule state."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'current_stage': self.current_stage,
            'stage_iteration': self.stage_iteration,
            'total_iterations': self.total_iterations,
            'best_gate_weight': self.best_gate_weight,
            'gate_stuck_counter': self.gate_stuck_counter,
            'gate_history': self.gate_history[-100:],
            'strategy': 'AGGRESSIVE_GATE_LEARNING'
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path, optimizer=None):
        """Restore model, optimizer and schedule state."""
        checkpoint = torch.load(path, map_location='cpu')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_stage = checkpoint.get('current_stage', 'baseline')
        self.stage_iteration = checkpoint.get('stage_iteration', 0)
        self.total_iterations = checkpoint.get('total_iterations', 0)
        self.best_gate_weight = checkpoint.get('best_gate_weight', 0.0)
        self.gate_stuck_counter = checkpoint.get('gate_stuck_counter', 0)
        self.gate_history = checkpoint.get('gate_history', [])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.prepare_model_for_stage()

        return checkpoint

    def is_training_complete(self):
        """True once the last stage has finished."""
        return (self.current_stage == 'refinement' and
                self.should_advance_stage())
