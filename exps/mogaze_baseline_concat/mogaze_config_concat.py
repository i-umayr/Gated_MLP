# encoding: utf-8
"""
Configuration for Concatenation Fusion Ablation Study

Architecture: Input[66] -> FC[66->63] -> 48xMLP[63] -> FC[63->63] -> Output[63]
Same training protocol as gated model: cosine annealing, 80k iters, same loss.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = 304

"""Directory configuration"""
C.abs_dir = osp.dirname(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
# Repository root: two levels above exps/<variant>/
C.root_dir = osp.abspath(osp.join(C.abs_dir, '..', '..'))

C.log_dir = osp.abspath(osp.join(C.abs_dir, 'log_concat'))
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

"""Data Dir"""
C.mogaze_anno_dir = osp.join(C.root_dir, 'data/mogaze_3d/')

"""Dataset Config"""
C.motion = edict()

C.motion.mogaze_input_length = 50
C.motion.mogaze_input_length_dct = 50

# Model compatibility names
C.motion.h36m_input_length = 50
C.motion.h36m_input_length_dct = 50
C.motion.h36m_target_length_train = 10
C.motion.h36m_target_length_eval = 30

C.motion.mogaze_target_length_train = 10
C.motion.mogaze_target_length_eval = 30

# Dimensions
C.motion.dim = 63           # Output: pose only (21 joints x 3)
C.motion.input_dim = 66     # Input: pose + gaze (63 + 3)

# Processing flags — same as gated model
C.data_aug = True
C.deriv_input = True
C.deriv_output = True
C.use_relative_loss = True

"""Model Config"""
C.pre_dct = False
C.post_dct = False

# MLP blocks: 63D hidden, 48 layers — identical to gated model's pose pathway
pose_dim = 63
C.motion_mlp = edict()
C.motion_mlp.hidden_dim = pose_dim
C.motion_mlp.seq_len = 50
C.motion_mlp.num_layers = 48
C.motion_mlp.with_normalization = True
C.motion_mlp.spatial_fc_only = False
C.motion_mlp.norm_axis = 'spatial'

# FC In: 66D input → 63D (spatial projection of concatenated pose+gaze)
C.motion_fc_in = edict()
C.motion_fc_in.in_features = 66
C.motion_fc_in.out_features = pose_dim
C.motion_fc_in.with_norm = False
C.motion_fc_in.activation = 'relu'
C.motion_fc_in.init_w_trunc_normal = False
C.motion_fc_in.temporal_fc = False

# FC Out: 63D → 63D (pose-only output)
C.motion_fc_out = edict()
C.motion_fc_out.in_features = pose_dim
C.motion_fc_out.out_features = 63
C.motion_fc_out.with_norm = False
C.motion_fc_out.activation = 'relu'
C.motion_fc_out.init_w_trunc_normal = True
C.motion_fc_out.temporal_fc = False

"""Train Config — same as gated model"""
C.batch_size = 32
C.num_workers = 0

# Cosine annealing — same schedule as gated/aggressive training
C.cos_lr_max = 3e-4
C.cos_lr_min = 1e-5
C.cos_lr_total_iters = 80000
C.cos_lr_warmup_iters = 2000

C.weight_decay = 1e-4
C.model_pth = None

"""Eval Config"""
C.shift_step = 1

"""Display Config"""
C.print_every = 100
C.save_every = 5000

"""LOPOCV Configuration"""
# The 6 participants used in cross-validation (p3 excluded - calibration issues)
# Each fold: train on 5, test on 1
C.lopocv_participants = ['p1', 'p2', 'p4', 'p5', 'p6', 'p7']

if __name__ == '__main__':
    print("Concatenation Fusion Ablation Configuration")
    print("=" * 60)
    print(f"Architecture: Input[66] -> FC[66->63] -> 48xMLP[63] -> FC[63->63] -> Output[63]")
    print(f"Input: {C.motion.input_dim}D (63 pose + 3 gaze)")
    print(f"Output: {C.motion.dim}D (pose only)")
    print(f"MLP blocks: {C.motion_mlp.num_layers}")
    print(f"Training: {C.cos_lr_total_iters} iterations, cosine annealing")
    print(f"LR: {C.cos_lr_max:.2e} -> {C.cos_lr_min:.2e}")
    print(f"LOPOCV: {len(C.lopocv_participants)} folds")
    print(f"Participants: {C.lopocv_participants}")
