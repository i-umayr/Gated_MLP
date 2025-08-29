import copy

import torch
from torch import nn
from mlp import build_mlps
from einops.layers.torch import Rearrange

class siMLPe(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(siMLPe, self).__init__()
        seq = self.config.motion_mlp.seq_len
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.motion_mlp = build_mlps(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
        
        # INPUT: 66D (63 pose + 3 gaze) - accepts gaze as additional input
        if self.temporal_fc_in:
            input_length = self.config.motion.h36m_input_length_dct
            self.motion_fc_in = nn.Linear(input_length, input_length)
        else:
            # Accept 66D input (63 pose + 3 gaze)
            self.motion_fc_in = nn.Linear(66, self.config.motion_mlp.hidden_dim)
            
        # OUTPUT: 63D (pose only) - output only poses, not gaze
        if self.temporal_fc_out:
            input_length = self.config.motion.h36m_input_length_dct
            self.motion_fc_out = nn.Linear(input_length, input_length)
        else:
            # Output only 63D poses (21 joints × 3)
            self.motion_fc_out = nn.Linear(self.config.motion_mlp.hidden_dim, 63)

        # FIX 4: Feature normalization layers
        self.pose_norm = nn.LayerNorm(63)
        self.gaze_norm = nn.LayerNorm(3)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input, apply_normalization=True):
        # motion_input is [batch, 50, 66] (63 pose + 3 gaze)
        # output will be [batch, 50, 63] (pose only)

        # FIX 4: Feature normalization - balance pose and gaze scales
        if apply_normalization:
            pose_input = motion_input[:, :, :63]  # [batch, 50, 63]
            gaze_input = motion_input[:, :, 63:66]  # [batch, 50, 3]
            
            # Normalize each modality separately
            pose_normalized = self.pose_norm(pose_input)
            gaze_normalized = self.gaze_norm(gaze_input)
            
            # Recombine normalized features
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

        # Output is [batch, 50, 63] - poses only, no gaze prediction
        return motion_feats