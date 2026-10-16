"""
Concatenation Fusion Model for Ablation Study

The simplest multimodal fusion baseline:
- Concatenate gaze (3D) to pose (63D) at input level → 66D
- First spatial FC projects 66D → 63D
- Standard siMLPe architecture (48 MLP blocks, 63D)
- Output FC 63D → 63D

No gaze encoder, no cross-attention, no gate.
"""

import copy
import torch
from torch import nn
from mlp import build_mlps
from einops.layers.torch import Rearrange


class SiMLPeConcatFusion(nn.Module):
    """
    Concatenation fusion: gaze appended to pose at input, then standard siMLPe.
    
    Input:  [batch, seq_len, 66]  (63 pose + 3 gaze)
    Output: [batch, seq_len, 63]  (pose predictions only)
    """
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(SiMLPeConcatFusion, self).__init__()

        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        # Core siMLPe MLP blocks (63D, 48 layers)
        self.motion_mlp = build_mlps(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc

        # INPUT: 66D → 63D spatial projection (the only difference from pose-only)
        if self.temporal_fc_in:
            input_length = self.config.motion.h36m_input_length_dct
            self.motion_fc_in = nn.Linear(input_length, input_length)
        else:
            # Spatial FC: project concatenated 66D input to 63D for MLP blocks
            self.motion_fc_in = nn.Linear(66, 63)

        # OUTPUT: 63D → 63D
        if self.temporal_fc_out:
            input_length = self.config.motion.h36m_input_length_dct
            self.motion_fc_out = nn.Linear(input_length, input_length)
        else:
            self.motion_fc_out = nn.Linear(63, 63)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input):
        """
        Args:
            motion_input: [batch, seq_len, 66] — 63 pose + 3 gaze concatenated
        Returns:
            motion_pred: [batch, seq_len, 63] — pose predictions
        """
        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)           # [b, 66, seq]
            motion_feats = self.motion_fc_in(motion_feats)   # temporal FC
        else:
            motion_feats = self.motion_fc_in(motion_input)   # [b, seq, 66] → [b, seq, 63]
            motion_feats = self.arr0(motion_feats)           # [b, 63, seq]

        motion_feats = self.motion_mlp(motion_feats)         # 48 MLP blocks

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)           # [b, seq, 63]
            motion_feats = self.motion_fc_out(motion_feats)  # [b, seq, 63]

        return motion_feats


if __name__ == "__main__":
    from mogaze_config_concat import config

    print("Testing Concatenation Fusion Model...")

    model = SiMLPeConcatFusion(config)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.4f}M")

    # Test forward pass
    batch_size, seq_len = 4, 50
    test_input = torch.randn(batch_size, seq_len, 66)  # 63 pose + 3 gaze

    with torch.no_grad():
        output = model(test_input)

    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, 63), "Output shape mismatch!"
    print("Architecture: Input[66] -> FC[66->63] -> 48xMLP[63] -> FC[63->63] -> Output[63]")
    print("Test passed!")
