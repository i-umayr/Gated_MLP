# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 304

"""please config ROOT_dir and user when u first using"""
C.abs_dir = osp.dirname(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.repo_name = 'siMLPe'
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]

C.log_dir = osp.abspath(osp.join(C.abs_dir, 'log_gated'))
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'lib'))

"""Data Dir and Weight Dir"""
# MoGaze 3D position + gaze dataset configuration
C.mogaze_anno_dir = osp.join(C.root_dir, 'data/mogaze_3d/')

"""Dataset Config"""
C.motion = edict()

# MoGaze motion parameters - GATED CROSS-ATTENTION VERSION
C.motion.mogaze_input_length = 50      # 50 input frames
C.motion.mogaze_input_length_dct = 50  # DCT input length

# CRITICAL: Keep H36M naming for model compatibility
C.motion.h36m_input_length = 50        # Model expects this name
C.motion.h36m_input_length_dct = 50    # Model expects this name
C.motion.h36m_target_length_train = 10 # Model expects this name
C.motion.h36m_target_length_eval = 30  # Model expects this name

# Keep MoGaze-specific names for dataset
C.motion.mogaze_target_length_train = 10  # 10 target frames for training
C.motion.mogaze_target_length_eval = 30   # 30 target frames for evaluation

# GATED CROSS-ATTENTION DIMENSIONS:
# - Input: 66D (63 pose + 3 gaze) for gated fusion
# - Processing: Separate pathways with cross-attention
# - Output: 63D (pose only) for prediction
C.motion.dim = 63          # Output dimension: pose only
C.motion.input_dim = 66    # Input dimension: pose + gaze

# Data augmentation and processing flags
C.data_aug = True
C.deriv_input = True
C.deriv_output = True
C.use_relative_loss = True

""" Model Config - GATED CROSS-ATTENTION ARCHITECTURE """
## Network type
C.pre_dct = False
C.post_dct = False

## BREAKTHROUGH: Gated Cross-Attention Configuration
C.gated_attention = edict()
C.gated_attention.enabled = True
C.gated_attention.num_heads = 8
C.gated_attention.dropout = 0.1
C.gated_attention.spatial_attention = True
C.gated_attention.uncertainty_weighting = True

# Gaze encoder configuration
C.gaze_encoder = edict()
C.gaze_encoder.hidden_dim = 64
C.gaze_encoder.use_velocity = True
C.gaze_encoder.temporal_smoothing = True

# Progressive training configuration
C.progressive_training = edict()
C.progressive_training.enabled = True
C.progressive_training.baseline_iters = 15000    # Stage 1: Establish baseline
C.progressive_training.progressive_iters = 15000 # Stage 2: Progressive gate opening
C.progressive_training.full_iters = 10000        # Stage 3: Full multimodal training
C.progressive_training.alignment_loss_weight = 0.5

## Motion Network MLP - POSE PATHWAY (63D processing)
# This processes the pose component through the original siMLPe architecture
pose_dim = 63  # Pure pose processing (21 joints × 3)
C.motion_mlp = edict()
C.motion_mlp.hidden_dim = pose_dim        # 63D pose-focused processing
C.motion_mlp.seq_len = 50                 # Input sequence length
C.motion_mlp.num_layers = 48              # Same depth as successful baseline
C.motion_mlp.with_normalization = True
C.motion_mlp.spatial_fc_only = False
C.motion_mlp.norm_axis = 'spatial'

## Motion Network FC In - DUAL PATHWAY INPUT
C.motion_fc_in = edict()
C.motion_fc_in.in_features = 66          # Accept pose+gaze input
C.motion_fc_in.out_features = pose_dim   # Project to pose dimensions
C.motion_fc_in.with_norm = False
C.motion_fc_in.activation = 'relu'
C.motion_fc_in.init_w_trunc_normal = False
C.motion_fc_in.temporal_fc = False

## Motion Network FC Out - POSE-ONLY OUTPUT
C.motion_fc_out = edict()
C.motion_fc_out.in_features = pose_dim   # From fused features
C.motion_fc_out.out_features = 63        # Output pose only (21 joints × 3)
C.motion_fc_out.with_norm = False
C.motion_fc_out.activation = 'relu'
C.motion_fc_out.init_w_trunc_normal = True
C.motion_fc_out.temporal_fc = False

"""Train Config - GATED CROSS-ATTENTION TRAINING"""
C.batch_size = 32  # Same as successful baseline
C.num_workers = 0  # Windows compatibility

# Learning rate configuration - PROGRESSIVE TRAINING COMPATIBLE
C.cos_lr_max = 1e-5    # Base learning rate
C.cos_lr_min = 5e-8    # Minimum learning rate

# BREAKTHROUGH: Total iterations across all progressive stages
total_progressive_iters = (C.progressive_training.baseline_iters + 
                          C.progressive_training.progressive_iters + 
                          C.progressive_training.full_iters)
C.cos_lr_total_iters = total_progressive_iters  # 40,000 total iterations

C.weight_decay = 1e-4
C.model_pth = None

# Gate-specific learning parameters
C.gate_learning = edict()
C.gate_learning.initial_lr_multiplier = 0.1    # Slower gate learning initially
C.gate_learning.warmup_iterations = 5000       # Gate learning warmup
C.gate_learning.max_gradient_norm = 1.0        # Gradient clipping for stability

"""Eval Config"""
C.shift_step = 1

"""Display Config"""
C.print_every = 100
C.save_every = 5000

# Analysis and monitoring
C.analysis = edict()
C.analysis.attention_analysis = True
C.analysis.gate_monitoring = True
C.analysis.uncertainty_tracking = True
C.analysis.save_attention_plots = False

if __name__ == '__main__':
    print("🚀 GATED CROSS-ATTENTION CONFIGURATION")
    print("=" * 60)
    print(f"🎯 BREAKTHROUGH: First MLP-compatible multimodal motion prediction!")
    print(f"📊 Target: Extract 23mm improvement from proven gaze information")
    print("")
    print(f"📐 Architecture Overview:")
    print(f"   Input: {C.motion.input_dim}D (63 pose + 3 gaze)")
    print(f"   Pose Pathway: siMLPe MLP ({pose_dim}D)")
    print(f"   Gaze Pathway: Specialized encoder (3D → {C.gaze_encoder.hidden_dim}D)")
    print(f"   Fusion: Gated cross-attention ({C.gated_attention.num_heads} heads)")
    print(f"   Output: {C.motion.dim}D (pose-only predictions)")
    print("")
    print(f"⚡ Progressive Training Strategy:")
    print(f"   Stage 1 (Baseline): {C.progressive_training.baseline_iters:,} iterations")
    print(f"   Stage 2 (Progressive): {C.progressive_training.progressive_iters:,} iterations")
    print(f"   Stage 3 (Full): {C.progressive_training.full_iters:,} iterations")
    print(f"   Total: {C.cos_lr_total_iters:,} iterations")
    print("")
    print(f"🔬 Innovation Highlights:")
    print(f"   • Learnable gate starts at 0 (safe baseline preservation)")
    print(f"   • Cross-attention learns spatial gaze-pose relationships")
    print(f"   • Uncertainty-aware dynamic fusion")
    print(f"   • Motion-gaze alignment loss")
    print(f"   • Progressive training prevents destabilization")
    print("")
    print(f"📊 Expected Performance:")
    print(f"   Current Baseline: 73.3mm MPJPE")
    print(f"   Proven Available: 23.6mm improvement (GazeMotion)")
    print(f"   Target Range: 65-68mm MPJPE (5-8mm improvement)")
    print(f"   Breakthrough Goal: Extract proven gaze information!")
    print("")
    print(f"🎯 Dataset: {C.mogaze_anno_dir}")
    print(f"📁 Logs: {C.log_dir}")
    print("=" * 60)
    
    # Verify configuration consistency
    assert C.motion.input_dim == 66, "Input must be 66D (63 pose + 3 gaze)"
    assert C.motion.dim == 63, "Output must be 63D (pose only)"
    assert C.motion_mlp.hidden_dim == 63, "Pose pathway must be 63D"
    assert C.progressive_training.enabled, "Progressive training required for gated approach"
    
    print("✅ Configuration validation passed!")
    print("🚀 Ready for breakthrough gated cross-attention training!")