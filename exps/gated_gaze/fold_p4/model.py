import copy

import torch
from torch import nn
from mlp import build_mlps
from einops.layers.torch import Rearrange

class siMLPe(nn.Module):
    """
    siMLPe with a 66-D (pose + gaze) input and a 63-D (pose) output.
    Only used to initialise the gated model via --pretrained-baseline / --pretrained-pose.
    """
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(siMLPe, self).__init__()
        seq = self.config.motion_mlp.seq_len
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.motion_mlp = build_mlps(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc

        if self.temporal_fc_in:
            input_length = self.config.motion.h36m_input_length_dct
            self.motion_fc_in = nn.Linear(input_length, input_length)
        else:
            self.motion_fc_in = nn.Linear(66, self.config.motion_mlp.hidden_dim)

        if self.temporal_fc_out:
            input_length = self.config.motion.h36m_input_length_dct
            self.motion_fc_out = nn.Linear(input_length, input_length)
        else:
            self.motion_fc_out = nn.Linear(self.config.motion_mlp.hidden_dim, 63)

        # Separate normalisation of the pose and gaze inputs
        self.pose_norm = nn.LayerNorm(63)
        self.gaze_norm = nn.LayerNorm(3)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input, apply_normalization=True):
        # motion_input: [batch, 50, 66]; output: [batch, 50, 63]
        if apply_normalization:
            pose_input = motion_input[:, :, :63]
            gaze_input = motion_input[:, :, 63:66]

            pose_normalized = self.pose_norm(pose_input)
            gaze_normalized = self.gaze_norm(gaze_input)

            motion_input = torch.cat([pose_normalized, gaze_normalized], dim=2)

        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)
            motion_feats = self.arr0(motion_feats)

        motion_feats = self.motion_mlp(motion_feats)

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)
            motion_feats = self.motion_fc_out(motion_feats)

        return motion_feats
