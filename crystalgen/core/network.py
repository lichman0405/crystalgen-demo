import torch
import torch.nn as nn
import torch.nn.functional as F

class AtomEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)  # [B, N, H]

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        x: [B, N, H]
        """
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.shape[-1] ** 0.5)  # [B, N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, V)  # [B, N, H]
        return self.output_proj(attended) + x  # residual

class CrystalNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.encoder = AtomEncoder(input_dim, hidden_dim)
        self.attn = SelfAttention(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # importance score
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: Tensor of shape [B, 256, 4]
        returns: [B, 256, 1] importance scores ∈ (0, 1)
        """
        h = self.encoder(x)
        h = self.attn(h)
        score = self.head(h)
        return score
