# encoding: utf-8
"""Configuration of the siMLPe baseline (pose only) on MoGaze 3D joint positions."""

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

C.abs_dir = osp.dirname(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.repo_name = 'siMLPe'
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]

C.log_dir = osp.abspath(osp.join(C.abs_dir, 'log_3d'))
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
C.mogaze_anno_dir = osp.join(C.root_dir, 'data/mogaze_3d/')

"""Dataset Config"""
C.motion = edict()

C.motion.mogaze_input_length = 50      # observed frames
C.motion.mogaze_input_length_dct = 50

# The model code reads the H36M-style names
C.motion.h36m_input_length = 50
C.motion.h36m_input_length_dct = 50
C.motion.h36m_target_length_train = 10
C.motion.h36m_target_length_eval = 30

C.motion.mogaze_target_length_train = 10  # predicted frames per training sample
C.motion.mogaze_target_length_eval = 30   # 1 s horizon, predicted auto-regressively

C.motion.dim = 63  # 21 joints x 3 coordinates

C.data_aug = True
C.deriv_input = True
C.deriv_output = True
C.use_relative_loss = True

""" Model Config"""
## Network
C.pre_dct = False
C.post_dct = False

## Motion Network mlp
dim_ = 63
C.motion_mlp = edict()
C.motion_mlp.hidden_dim = dim_
C.motion_mlp.seq_len = 50
C.motion_mlp.num_layers = 48
C.motion_mlp.with_normalization = True
C.motion_mlp.spatial_fc_only = False
C.motion_mlp.norm_axis = 'spatial'

## Motion Network FC In
C.motion_fc_in = edict()
C.motion_fc_in.in_features = C.motion.dim
C.motion_fc_in.out_features = dim_
C.motion_fc_in.with_norm = False
C.motion_fc_in.activation = 'relu'
C.motion_fc_in.init_w_trunc_normal = False
C.motion_fc_in.temporal_fc = False

## Motion Network FC Out
C.motion_fc_out = edict()
C.motion_fc_out.in_features = dim_
C.motion_fc_out.out_features = C.motion.dim
C.motion_fc_out.with_norm = False
C.motion_fc_out.activation = 'relu'
C.motion_fc_out.init_w_trunc_normal = True
C.motion_fc_out.temporal_fc = False

"""Train Config"""
C.batch_size = 32
C.num_workers = 0

# Learning rate: linear warmup followed by cosine decay
C.lr_schedule = 'cosine'  # 'cosine' or 'multistep'

C.cos_lr_max = 3e-4
C.cos_lr_min = 1e-5
C.cos_lr_total_iters = 80000
C.cos_lr_warmup_iters = 2000

# Only used when lr_schedule == 'multistep'
C.lr_milestones = [24000, 48000, 68000]
C.lr_decay_factor = 0.3

C.weight_decay = 1e-4
C.model_pth = None

"""Eval Config"""
C.shift_step = 1

"""Display Config"""
C.print_every = 100
C.save_every = 5000
