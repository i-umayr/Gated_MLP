import copy
import torch
from torch import nn
import torch.nn.functional as F
from mlp import build_mlps
from einops.layers.torch import Rearrange
import math

class GazeEncoder(nn.Module):
    """Encodes the gaze direction sequence: temporal smoothing, velocity features and a temporal convolution."""
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

        # Frame-to-frame change of the gaze direction
        gaze_velocity = torch.zeros_like(gaze_input)
        gaze_velocity[:, 1:] = gaze_input[:, 1:] - gaze_input[:, :-1]
        gaze_velocity = self.velocity_proj(gaze_velocity)

        gaze_features = self.gaze_proj(gaze_smooth)  # [batch, seq_len, hidden_dim]
        velocity_features = self.velocity_proj_out(gaze_velocity)  # [batch, seq_len, hidden_dim]

        combined_features = torch.cat([gaze_features, velocity_features], dim=-1)  # [batch, seq_len, hidden_dim*2]

        combined_features = self.temporal_conv(combined_features.transpose(1, 2)).transpose(1, 2)
        combined_features = self.norm(combined_features)

        return combined_features  # [batch, seq_len, hidden_dim]


class GatedCrossAttention(nn.Module):
    """Multi-head cross-attention in which gaze features query pose features."""
    def __init__(self, pose_dim=66, gaze_dim=64, num_heads=8, dropout=0.1):
        super(GatedCrossAttention, self).__init__()
        self.pose_dim = pose_dim
        self.gaze_dim = gaze_dim
        self.num_heads = num_heads
        self.head_dim = pose_dim // num_heads

        assert pose_dim % num_heads == 0, "pose_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(gaze_dim, pose_dim)  # queries from gaze
        self.key_proj = nn.Linear(pose_dim, pose_dim)    # keys from pose
        self.value_proj = nn.Linear(pose_dim, pose_dim)  # values from pose

        self.out_proj = nn.Linear(pose_dim, pose_dim)

        # Learnable per-dimension weights of the attended output
        self.spatial_attention = nn.Parameter(torch.ones(1, 1, pose_dim) * 0.1)

        # Per-frame uncertainty in [0, 1] that down-weights the attended output
        self.uncertainty_net = nn.Sequential(
            nn.Linear(pose_dim + gaze_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(pose_dim)

    def forward(self, pose_features, gaze_features):
        # pose_features: [batch, seq_len, pose_dim]
        # gaze_features: [batch, seq_len, gaze_dim]
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

        spatial_weights = torch.sigmoid(self.spatial_attention)
        attended_output = attended_output * spatial_weights

        combined_features = torch.cat([pose_features, gaze_features], dim=-1)
        uncertainty = self.uncertainty_net(combined_features)  # [batch, seq_len, 1]

        attended_output = attended_output * (1 - uncertainty)

        # Residual connection and layer normalisation
        output = self.norm(attended_output + pose_features)

        return output, attention_weights, uncertainty


class SiMLPeWithGatedGaze(nn.Module):
    """
    Gated MLP: siMLPe pose pathway + gaze encoder fused by gated cross-attention.

    Input [batch, seq_len, 66] (63 pose + 3 gaze), output [batch, seq_len, 63] (pose).
    The fused features are (1 - a) * pose + a * cross_attention(pose, gaze) with
    a = sigmoid(gaze_gate).
    """
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(SiMLPeWithGatedGaze, self).__init__()

        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        # Pose pathway: siMLPe MLP blocks on the 63 pose dimensions
        pose_config = copy.deepcopy(config.motion_mlp)
        pose_config.hidden_dim = 63
        self.pose_mlp = build_mlps(pose_config)

        self.gaze_encoder = GazeEncoder(
            input_dim=3,
            hidden_dim=64,
            seq_len=config.motion.h36m_input_length_dct
        )

        self.cross_attention = GatedCrossAttention(
            pose_dim=63,
            gaze_dim=64,
            num_heads=7,  # 63 / 7 = 9 dims per head
            dropout=0.1
        )

        # Learnable gate (logit); the training schedules set its value
        self.gaze_gate = nn.Parameter(torch.tensor(0.0))

        # Motion-direction predictors used by the alignment loss
        self.motion_predictor = nn.Linear(63, 3)
        self.gaze_predictor = nn.Linear(64, 3)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc

        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(config.motion.h36m_input_length_dct, config.motion.h36m_input_length_dct)
        else:
            self.pose_fc_in = nn.Linear(63, 63)
            self.gaze_fc_in = nn.Linear(3, 3)

        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(config.motion.h36m_input_length_dct, config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_out = nn.Linear(63, 63)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def get_gate_weight(self):
        """Current gate value sigmoid(gaze_gate) as a Python float."""
        return torch.sigmoid(self.gaze_gate).item()

    def compute_alignment_loss(self, pose_features, gaze_features):
        """1 - cosine similarity between motion directions predicted from pose and from gaze."""
        pose_motion = self.motion_predictor(pose_features)  # [batch, seq, 3]
        gaze_motion = self.gaze_predictor(gaze_features)    # [batch, seq, 3]

        pose_motion = F.normalize(pose_motion, dim=-1)
        gaze_motion = F.normalize(gaze_motion, dim=-1)

        alignment_loss = 1 - F.cosine_similarity(pose_motion, gaze_motion, dim=-1)
        return alignment_loss.mean()

    def forward(self, motion_input, return_attention=False, return_gate_weight=False):
        # motion_input: [batch, seq_len, 66] (63 pose + 3 gaze)
        pose_input = motion_input[:, :, :63]
        gaze_input = motion_input[:, :, 63:66]

        # Pose pathway
        if self.temporal_fc_in:
            pose_feats = self.arr0(pose_input)
            pose_feats = self.motion_fc_in(pose_feats)
        else:
            pose_feats = self.pose_fc_in(pose_input)
            pose_feats = self.arr0(pose_feats)

        pose_features = self.pose_mlp(pose_feats)  # [batch, 63, seq_len]
        pose_features = self.arr1(pose_features)   # [batch, seq_len, 63]

        # Gaze pathway
        gaze_features = self.gaze_encoder(gaze_input)  # [batch, seq_len, 64]

        # Gated cross-attention fusion
        gaze_enhanced_pose, attention_weights, uncertainty = self.cross_attention(
            pose_features, gaze_features
        )

        gate_weight = torch.sigmoid(self.gaze_gate)
        fused_features = (1 - gate_weight) * pose_features + gate_weight * gaze_enhanced_pose

        # Output projection
        if self.temporal_fc_out:
            fused_features = self.arr0(fused_features)
            motion_pred = self.motion_fc_out(fused_features)
            motion_pred = self.arr1(motion_pred)
        else:
            motion_pred = self.motion_fc_out(fused_features)

        returns = [motion_pred]

        if return_attention:
            returns.append({
                'attention_weights': attention_weights,
                'uncertainty': uncertainty,
                'pose_features': pose_features,
                'gaze_features': gaze_features,
                'alignment_loss': self.compute_alignment_loss(pose_features, gaze_features)
            })

        if return_gate_weight:
            returns.append(gate_weight.item())

        return returns[0] if len(returns) == 1 else tuple(returns)


def create_gated_model(config, pretrained_pose_model=None):
    """Create the gated model, optionally copying shape-compatible weights from a pose-only model."""
    model = SiMLPeWithGatedGaze(config)

    if pretrained_pose_model is not None:
        pretrained_dict = pretrained_pose_model.state_dict()
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                          if k in model_dict and model_dict[k].shape == v.shape}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(f"Loaded {len(pretrained_dict)} compatible parameters from the pose-only model")

    return model
