# model/encoder.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


def sinusoidal_2d_pos_encoding(h, w, d_model, device):
    """生成 2D 正弦位置编码 (h, w, d_model)"""
    assert d_model % 4 == 0
    pe = torch.zeros(h, w, d_model, device=device)
    d_half = d_model // 2
    div_term = torch.exp(
        torch.arange(0, d_half, 2, device=device).float() * (-math.log(10000.0) / d_half)
    )  # (d_half/2,)

    y_pos = torch.arange(h, device=device).float().unsqueeze(1)  # (h, 1)
    x_pos = torch.arange(w, device=device).float().unsqueeze(0).unsqueeze(-1)  # (1, w, 1)

    # Y -> first half channels
    pe_y_sin = torch.sin(y_pos * div_term)  # (h, d_half/2)
    pe_y_cos = torch.cos(y_pos * div_term)
    pe[:, :, 0:d_half:2] = pe_y_sin.unsqueeze(1).expand(-1, w, -1)
    pe[:, :, 1:d_half:2] = pe_y_cos.unsqueeze(1).expand(-1, w, -1)

    # X -> second half channels
    pe_x_sin = torch.sin(x_pos * div_term)  # (1, w, d_half/2)
    pe_x_cos = torch.cos(x_pos * div_term)
    pe[:, :, d_half::2] = pe_x_sin.expand(h, -1, -1)
    pe[:, :, d_half + 1::2] = pe_x_cos.expand(h, -1, -1)
    return pe


class RelativeGeometryBias(nn.Module):
    """论文 Eq.(2)(3) 的相对几何偏置"""
    def __init__(self, n_head, d_g=cfg.GEO_HEAD_DIM):
        super().__init__()
        self.n_head = n_head
        self.W_g = nn.ModuleList([nn.Linear(2, d_g) for _ in range(n_head)])
        self.w_g = nn.ParameterList([nn.Parameter(torch.randn(d_g)) for _ in range(n_head)])

    def forward(self, h, w, device):
        # 归一化坐标
        ys = torch.linspace(0, 1, h, device=device)
        xs = torch.linspace(0, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (N, 2)
        N = coords.size(0)

        # pairwise log distances
        dx = (coords[:, 0].unsqueeze(1) - coords[:, 0].unsqueeze(0)).abs() + 1e-6
        dy = (coords[:, 1].unsqueeze(1) - coords[:, 1].unsqueeze(0)).abs() + 1e-6
        r = torch.stack([dx.log(), dy.log()], dim=-1)  # (N, N, 2)

        biases = []
        for h_idx in range(self.n_head):
            # λ_ij = w^T ReLU(W r)
            hidden = F.relu(self.W_g[h_idx](r))          # (N, N, d_g)
            bias = torch.einsum("ijd,d->ij", hidden, self.w_g[h_idx])
            biases.append(bias)
        # (n_head, N, N)
        return torch.stack(biases, dim=0)


class GeometryAwareEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.D_MODEL, cfg.N_HEAD, dropout=cfg.DROPOUT, batch_first=True)
        self.geo_bias = RelativeGeometryBias(cfg.N_HEAD)
        self.ff = nn.Sequential(
            nn.Linear(cfg.D_MODEL, cfg.D_FF),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.D_FF, cfg.D_MODEL),
            nn.Dropout(cfg.DROPOUT),
        )
        self.norm1 = nn.LayerNorm(cfg.D_MODEL)
        self.norm2 = nn.LayerNorm(cfg.D_MODEL)
        self.dropout = nn.Dropout(cfg.DROPOUT)

        # 我们手动加 bias，所以需要拆开 attention
        self.q_proj = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)
        self.k_proj = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)
        self.v_proj = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)
        self.out_proj = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)

    def forward(self, x, h, w):
        """
        x: (B, N, D)  flattened feature
        h, w: feature map 空间尺寸
        """
        B, N, D = x.shape
        residual = x
        x = self.norm1(x)

        # QKV
        q = self.q_proj(x).view(B, N, cfg.N_HEAD, D // cfg.N_HEAD).transpose(1, 2)  # (B,H,N,d)
        k = self.k_proj(x).view(B, N, cfg.N_HEAD, D // cfg.N_HEAD).transpose(1, 2)
        v = self.v_proj(x).view(B, N, cfg.N_HEAD, D // cfg.N_HEAD).transpose(1, 2)

        # content energy
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D // cfg.N_HEAD)  # (B,H,N,N)

        # + relative geometry bias (论文 Eq.4)
        geo = self.geo_bias(h, w, x.device)  # (H, N, N)
        attn_logits = attn_logits + geo.unsqueeze(0)

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B,H,N,d)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        x = residual + self.dropout(out)

        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        return x


class GeometryAwareEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([GeometryAwareEncoderLayer() for _ in range(cfg.N_ENC_LAYERS)])
        self.norm = nn.LayerNorm(cfg.D_MODEL)

    def forward(self, feat_map):
        """
        feat_map: (B, D, h, w)
        return: memory (B, N, D), (h, w)
        """
        B, D, h, w = feat_map.shape
        # 加 2D 位置编码
        pe = sinusoidal_2d_pos_encoding(h, w, D, feat_map.device)
        x = feat_map.permute(0, 2, 3, 1) + pe.unsqueeze(0)  # (B,h,w,D)
        x = x.view(B, h * w, D)

        for layer in self.layers:
            x = layer(x, h, w)
        x = self.norm(x)
        return x, (h, w)
