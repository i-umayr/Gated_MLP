"""
Cross-Attention (No Gate) Fusion Model -- Ablation Method 3

Identical to the proposed gated cross-attention architecture
(gated_attention_model.py), but with the learnable scalar gate fixed at
alpha = 1.0: the cross-attention output is used directly, with no interpolation
against the pose-only features. This isolates the contribution of the gate.

    pose[63] -> FC(63->63) -> arr0 -> 48x MLP(63D) -> arr1 -> pose_features[b,seq,63] ┐
                                                                                       -> CrossAttn -> FC(63->63) -> Output[63]
    gaze[3] -> GazeEncoder(3->64) -> gaze_features[b,seq,64] ───────────────────────--┘

Differences from the gated model:
  - No learnable gate (gaze_gate removed). Fusion is always full cross-attention
    (equivalent to fixing alpha = 1.0 in (1-alpha)*pose + alpha*crossattn).
  - No alignment loss machinery (motion_predictor / gaze_predictor removed),
    matching the simple training protocol shared by all ablation methods
    (position + velocity loss only, single optimizer group, no gate regularization).

Everything else matches the gated model: GazeEncoder (verbatim), 7-head
cross-attention with spatial-attention and uncertainty weighting (these are part
of the cross-attention block, not the gate, so they are retained), fusion AFTER
the 48 MLP blocks, and 63D pose output. The cross-attention keeps its internal
residual (LayerNorm(attended + pose_features)), so pose information is preserved
even at alpha = 1.0. siMLPe backbone, DCT/IDCT, residual learning, loss, and
training protocol are identical to the concat/additive ablations.

Self-contained: depends only on mlp.py.
"""

import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from mlp import build_mlps
from einops.layers.torch import Rearrange


class GazeEncoder(nn.Module):
    """Specialized encoder for gaze data with temporal smoothing and velocity features.

    Verbatim from the gated model so all multimodal ablations share an identical
    gaze pathway. Maps 3D gaze direction -> 64D features.
    """
    def __init__(self, input_dim=3, hidden_dim=64, seq_len=50):
        super(GazeEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.gaze_smooth = nn.Conv1d(3, 3, kernel_size=3, padding=1, groups=3)
        self.velocity_proj = nn.Linear(3, 3)
        self.gaze_proj = nn.Linear(input_dim, hidden_dim)
        self.velocity_proj_out = nn.Linear(input_dim, hidden_dim)
        self.temporal_conv = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, gaze_input):
        # gaze_input: [batch, seq_len, 3]
        batch_size, seq_len, _ = gaze_input.shape

        gaze_smooth = self.gaze_smooth(gaze_input.transpose(1, 2)).transpose(1, 2)

        gaze_velocity = torch.zeros_like(gaze_input)
        gaze_velocity[:, 1:] = gaze_input[:, 1:] - gaze_input[:, :-1]
        gaze_velocity = self.velocity_proj(gaze_velocity)

        gaze_features = self.gaze_proj(gaze_smooth)
        velocity_features = self.velocity_proj_out(gaze_velocity)

        combined_features = torch.cat([gaze_features, velocity_features], dim=-1)
        combined_features = self.temporal_conv(combined_features.transpose(1, 2)).transpose(1, 2)
        combined_features = self.norm(combined_features)

        return combined_features  # [batch, seq_len, hidden_dim]


class CrossAttention(nn.Module):
    """Multi-head cross-attention where gaze queries pose, with spatial-attention
    and uncertainty weighting. Internals are identical to the gated model's
    GatedCrossAttention; the scalar gate that wrapped it externally is removed.
    """
    def __init__(self, pose_dim=63, gaze_dim=64, num_heads=7, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.pose_dim = pose_dim
        self.gaze_dim = gaze_dim
        self.num_heads = num_heads
        self.head_dim = pose_dim // num_heads

        assert pose_dim % num_heads == 0, "pose_dim must be divisible by num_heads"

        # Cross-attention projections (gaze queries pose)
        self.query_proj = nn.Linear(gaze_dim, pose_dim)
        self.key_proj = nn.Linear(pose_dim, pose_dim)
        self.value_proj = nn.Linear(pose_dim, pose_dim)
        self.out_proj = nn.Linear(pose_dim, pose_dim)

        # Spatial attention (which joints gaze influences)
        self.spatial_attention = nn.Parameter(torch.ones(1, 1, pose_dim) * 0.1)

        # Uncertainty estimation for dynamic per-feature weighting
        self.uncertainty_net = nn.Sequential(
            nn.Linear(pose_dim + gaze_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(pose_dim)

    def forward(self, pose_features, gaze_features):
        # pose_features: [batch, seq_len, pose_dim]; gaze_features: [batch, seq_len, gaze_dim]
        batch_size, seq_len, _ = pose_features.shape

        Q = self.query_proj(gaze_features)
        K = self.key_proj(pose_features)
        V = self.value_proj(pose_features)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attended_values = torch.matmul(attention_weights, V)
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.pose_dim
        )

        attended_output = self.out_proj(attended_values)

        # Spatial attention
        spatial_weights = torch.sigmoid(self.spatial_attention)
        attended_output = attended_output * spatial_weights

        # Uncertainty weighting
        combined_features = torch.cat([pose_features, gaze_features], dim=-1)
        uncertainty = self.uncertainty_net(combined_features)
        attended_output = attended_output * (1 - uncertainty)

        # Internal residual + norm (pose preserved even at alpha = 1.0)
        output = self.norm(attended_output + pose_features)

        return output, attention_weights, uncertainty


class SiMLPeCrossAttnNoGate(nn.Module):
    """
    Ungated cross-attention fusion (alpha fixed at 1.0).

    Input:  [batch, seq_len, 66]  (63 pose + 3 gaze)
    Output: [batch, seq_len, 63]  (pose predictions only)
    """
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(SiMLPeCrossAttnNoGate, self).__init__()

        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        # Pose pathway: 48 MLP blocks (63D) -- identical siMLPe backbone
        self.motion_mlp = build_mlps(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc

        if self.temporal_fc_in:
            input_length = self.config.motion.h36m_input_length_dct
            self.motion_fc_in = nn.Linear(input_length, input_length)
        else:
            self.motion_fc_in = nn.Linear(63, 63)

        # Gaze pathway
        self.gaze_encoder = GazeEncoder(
            input_dim=3,
            hidden_dim=64,
            seq_len=self.config.motion.h36m_input_length_dct,
        )

        # Cross-attention fusion (applied AFTER the MLP blocks, like the gated model)
        self.cross_attention = CrossAttention(
            pose_dim=63,
            gaze_dim=64,
            num_heads=7,   # 63 / 7 = 9
            dropout=0.1,
        )

        # Output projection
        if self.temporal_fc_out:
            input_length = self.config.motion.h36m_input_length_dct
            self.motion_fc_out = nn.Linear(input_length, input_length)
        else:
            self.motion_fc_out = nn.Linear(63, 63)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input, return_attention=False):
        """
        Args:
            motion_input: [batch, seq_len, 66] -- 63 pose + 3 gaze, DCT-transformed
                          upstream (identical to the concat/additive/gated protocol).
        Returns:
            motion_pred: [batch, seq_len, 63] -- pose predictions only
            (optionally also an attention-info dict if return_attention=True)
        """
        pose_input = motion_input[:, :, :63]    # [b, seq, 63]
        gaze_input = motion_input[:, :, 63:66]  # [b, seq, 3]

        # === Pose pathway ===
        if self.temporal_fc_in:
            pose_feats = self.arr0(pose_input)            # [b, 63, seq]
            pose_feats = self.motion_fc_in(pose_feats)    # temporal FC
        else:
            pose_feats = self.motion_fc_in(pose_input)    # [b, seq, 63]
            pose_feats = self.arr0(pose_feats)            # [b, 63, seq]

        pose_features = self.motion_mlp(pose_feats)       # [b, 63, seq] -- 48 MLP blocks
        pose_features = self.arr1(pose_features)          # [b, seq, 63]

        # === Gaze pathway ===
        gaze_features = self.gaze_encoder(gaze_input)     # [b, seq, 64]

        # === Cross-attention fusion (alpha = 1.0: used directly, no gate) ===
        fused, attention_weights, uncertainty = self.cross_attention(
            pose_features, gaze_features
        )

        # === Output ===
        if self.temporal_fc_out:
            fused = self.arr0(fused)
            motion_pred = self.motion_fc_out(fused)
            motion_pred = self.arr1(motion_pred)
        else:
            motion_pred = self.motion_fc_out(fused)       # [b, seq, 63]

        if return_attention:
            return motion_pred, {
                'attention_weights': attention_weights,
                'uncertainty': uncertainty,
            }
        return motion_pred


if __name__ == "__main__":
    from mogaze_config_crossattn import config

    print("Testing Cross-Attention (No Gate) Model...")

    model = SiMLPeCrossAttnNoGate(config)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.4f}M")

    # Parameter breakdown
    mlp = sum(p.numel() for p in model.motion_mlp.parameters())
    pose_in = sum(p.numel() for p in model.motion_fc_in.parameters())
    gaze_enc = sum(p.numel() for p in model.gaze_encoder.parameters())
    xattn = sum(p.numel() for p in model.cross_attention.parameters())
    out = sum(p.numel() for p in model.motion_fc_out.parameters())
    print(f"  pose_fc_in (63->63):     {pose_in/1e6:.4f}M")
    print(f"  MLP blocks (48 x 63D):   {mlp/1e6:.4f}M")
    print(f"  gaze_encoder (3->64):    {gaze_enc/1e6:.4f}M")
    print(f"  cross_attention:         {xattn/1e6:.4f}M")
    print(f"  motion_fc_out (63->63):  {out/1e6:.4f}M")

    # Forward pass
    batch_size, seq_len = 4, 50
    test_input = torch.randn(batch_size, seq_len, 66)

    model.eval()
    with torch.no_grad():
        output = model(test_input)
        output2, attn = model(test_input, return_attention=True)

    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {tuple(attn['attention_weights'].shape)}")
    print(f"Uncertainty shape:       {tuple(attn['uncertainty'].shape)}")
    assert output.shape == (batch_size, seq_len, 63), "Output shape mismatch!"
    print("Architecture: pose[63]->FC->48xMLP[63] + GazeEnc(gaze)[64] -> CrossAttn(alpha=1) -> FC -> Output[63]")
    print("Test passed!")
