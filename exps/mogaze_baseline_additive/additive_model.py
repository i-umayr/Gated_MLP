"""
Additive Fusion Model for Ablation Study

Multimodal fusion by additive injection of encoded gaze into the pose features:

    pose (63D) ── spatial FC (63 -> 63) ─────────────┐
                                                      (+)  ->  z0  ->  48x MLP(63D)  ->  FC(63 -> 63)  -> Output(63)
    gaze (3D)  ── GazeEncoder (3 -> 64) -> proj(64->63) ┘

Fusion happens at z0 (after the spatial FC, before the MLP blocks):

    z0 = z0_pose + project(GazeEncoder(gaze))

No attention, no gate. The siMLPe backbone, DCT/IDCT handling, residual learning,
loss, and training protocol are identical to the concatenation ablation and the
gated model; only the fusion mechanism differs. This isolates "simple additive
injection of a gaze encoder" as a fusion strategy.

The GazeEncoder is copied verbatim from gated_attention_model.py so this file is
self-contained and depends only on mlp.py.
"""

import copy
import torch
from torch import nn
from mlp import build_mlps
from einops.layers.torch import Rearrange


class GazeEncoder(nn.Module):
    """Specialized encoder for gaze data with temporal smoothing and velocity features.

    Verbatim from the gated model (gated_attention_model.py) so the additive
    ablation uses an identical gaze pathway. Maps 3D gaze direction -> 64D features.
    """
    def __init__(self, input_dim=3, hidden_dim=64, seq_len=50):
        super(GazeEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Gaze preprocessing: temporal smoothing (depthwise 1D conv)
        self.gaze_smooth = nn.Conv1d(3, 3, kernel_size=3, padding=1, groups=3)

        # Velocity computation (learnable)
        self.velocity_proj = nn.Linear(3, 3)

        # Feature extraction
        self.gaze_proj = nn.Linear(input_dim, hidden_dim)
        self.velocity_proj_out = nn.Linear(input_dim, hidden_dim)

        # Temporal processing
        self.temporal_conv = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, gaze_input):
        # gaze_input: [batch, seq_len, 3]
        batch_size, seq_len, _ = gaze_input.shape

        # Smooth gaze trajectories
        gaze_smooth = self.gaze_smooth(gaze_input.transpose(1, 2)).transpose(1, 2)

        # Compute gaze velocity (direction changes)
        gaze_velocity = torch.zeros_like(gaze_input)
        gaze_velocity[:, 1:] = gaze_input[:, 1:] - gaze_input[:, :-1]
        gaze_velocity = self.velocity_proj(gaze_velocity)

        # Project to hidden dimensions
        gaze_features = self.gaze_proj(gaze_smooth)            # [b, seq, hidden]
        velocity_features = self.velocity_proj_out(gaze_velocity)  # [b, seq, hidden]

        # Combine gaze and velocity features
        combined_features = torch.cat([gaze_features, velocity_features], dim=-1)  # [b, seq, 2*hidden]

        # Temporal convolution
        combined_features = self.temporal_conv(combined_features.transpose(1, 2)).transpose(1, 2)
        combined_features = self.norm(combined_features)

        return combined_features  # [batch, seq_len, hidden_dim]


class SiMLPeAdditiveFusion(nn.Module):
    """
    Additive fusion: encoded gaze is projected to the pose-feature space and added
    element-wise at z0 (after the spatial FC, before the MLP blocks).

    Input:  [batch, seq_len, 66]  (63 pose + 3 gaze)
    Output: [batch, seq_len, 63]  (pose predictions only)
    """
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(SiMLPeAdditiveFusion, self).__init__()

        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        # Core siMLPe MLP blocks (63D, 48 layers) -- identical backbone
        self.motion_mlp = build_mlps(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc

        # --- Pose pathway: spatial FC 63 -> 63 (produces z0_pose) ---
        if self.temporal_fc_in:
            input_length = self.config.motion.h36m_input_length_dct
            self.motion_fc_in = nn.Linear(input_length, input_length)
        else:
            self.motion_fc_in = nn.Linear(63, 63)

        # --- Gaze pathway: encoder (3 -> 64) then projection to pose dim (64 -> 63) ---
        self.gaze_encoder = GazeEncoder(
            input_dim=3,
            hidden_dim=64,
            seq_len=self.config.motion.h36m_input_length_dct,
        )
        self.gaze_fusion_proj = nn.Linear(64, 63)

        # --- Output projection: 63 -> 63 ---
        if self.temporal_fc_out:
            input_length = self.config.motion.h36m_input_length_dct
            self.motion_fc_out = nn.Linear(input_length, input_length)
        else:
            self.motion_fc_out = nn.Linear(63, 63)

        self.reset_parameters()

    def reset_parameters(self):
        # Residual-learning stability: tiny output init (same as siMLPe / concat / gated)
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input):
        """
        Args:
            motion_input: [batch, seq_len, 66] -- 63 pose + 3 gaze, DCT-transformed
                          upstream (identical to the concat/gated protocol).
        Returns:
            motion_pred: [batch, seq_len, 63] -- pose predictions only
        """
        # Split pose and gaze (mirrors the gated model)
        pose_input = motion_input[:, :, :63]    # [b, seq, 63]
        gaze_input = motion_input[:, :, 63:66]  # [b, seq, 3]

        # === Pose pathway: spatial FC -> z0_pose ===
        if self.temporal_fc_in:
            pose_feats = self.arr0(pose_input)            # [b, 63, seq]
            pose_feats = self.motion_fc_in(pose_feats)    # temporal FC on seq
            pose_feats = self.arr1(pose_feats)            # [b, seq, 63]
        else:
            pose_feats = self.motion_fc_in(pose_input)    # [b, seq, 63] (spatial FC)

        # === Gaze pathway: encode -> project to 63D ===
        gaze_feats = self.gaze_encoder(gaze_input)        # [b, seq, 64]
        gaze_feats = self.gaze_fusion_proj(gaze_feats)    # [b, seq, 63]

        # === Additive fusion at z0 ===
        z0 = pose_feats + gaze_feats                      # [b, seq, 63]

        # === MLP blocks (operate on the temporal dimension) ===
        z0 = self.arr0(z0)                                # [b, 63, seq]
        motion_feats = self.motion_mlp(z0)                # 48 MLP blocks
        motion_feats = self.arr1(motion_feats)            # [b, seq, 63]

        # === Output ===
        if self.temporal_fc_out:
            motion_feats = self.arr0(motion_feats)
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.motion_fc_out(motion_feats)  # [b, seq, 63]

        return motion_feats


if __name__ == "__main__":
    from mogaze_config_additive import config

    print("Testing Additive Fusion Model...")

    model = SiMLPeAdditiveFusion(config)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.4f}M")

    # Parameter breakdown
    gaze_enc = sum(p.numel() for p in model.gaze_encoder.parameters())
    gaze_proj = sum(p.numel() for p in model.gaze_fusion_proj.parameters())
    pose_in = sum(p.numel() for p in model.motion_fc_in.parameters())
    mlp = sum(p.numel() for p in model.motion_mlp.parameters())
    out = sum(p.numel() for p in model.motion_fc_out.parameters())
    print(f"  pose_fc_in (63->63):     {pose_in/1e6:.4f}M")
    print(f"  MLP blocks (48 x 63D):   {mlp/1e6:.4f}M")
    print(f"  gaze_encoder (3->64):    {gaze_enc/1e6:.4f}M")
    print(f"  gaze_fusion_proj (64->63): {gaze_proj/1e6:.4f}M")
    print(f"  motion_fc_out (63->63):  {out/1e6:.4f}M")

    # Forward pass
    batch_size, seq_len = 4, 50
    test_input = torch.randn(batch_size, seq_len, 66)  # 63 pose + 3 gaze

    with torch.no_grad():
        output = model(test_input)

    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, 63), "Output shape mismatch!"
    print("Architecture: pose[63]->FC[63->63] (+) proj(GazeEnc(gaze[3]))[63] -> 48xMLP[63] -> FC[63->63] -> Output[63]")
    print("Test passed!")
