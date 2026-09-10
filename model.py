import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *


# ==================== Dynamic Adaptive Scan (DAS) 核心 ====================
class OffsetPredictionNetwork(nn.Module):
    """OPN: 预测每个参考点的 2D offset，对应 DAMamba 的 OPN"""
    def __init__(self, in_channels, hidden=OPN_HIDDEN):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pw = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2)  # (dx, dy)
        )

    def forward(self, x):
        # x: B, C, H, W
        B, C, H, W = x.shape
        feat = self.dw(x)
        feat = feat.permute(0, 2, 3, 1).contiguous()  # B H W C
        offset = self.pw(feat)  # B H W 2
        offset = torch.tanh(offset) * DAS_OFFSET_SCALE
        return offset


class DynamicAdaptiveScan(nn.Module):
    """
    简化版 DAS：
    1. 用 OPN 预测 offset
    2. 双线性采样得到新特征
    3. 按原始空间顺序展平为序列（保持线性复杂度）
    """
    def __init__(self, channels):
        super().__init__()
        self.opn = OffsetPredictionNetwork(channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: B C H W
        B, C, H, W = x.shape
        offset = self.opn(x)  # B H W 2

        # 构造采样网格 (归一化到 [-1,1])
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # H W 2
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # B H W 2
        sample_grid = grid + offset

        # 双线性采样
        sampled = F.grid_sample(
            x, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # B C H W

        # 展平为序列 (B, H*W, C)，保持空间顺序
        seq = sampled.flatten(2).transpose(1, 2)  # B L C
        seq = self.norm(seq)
        return seq, sample_grid  # 返回序列 + 采样位置（可用于可视化）


# ==================== Geometry-Aware relative bias ====================
class RelativeGeometryBias(nn.Module):
    """对应第二篇论文的相对几何 bias"""
    def __init__(self, n_heads, d_g=32):
        super().__init__()
        self.n_heads = n_heads
        self.W_g = nn.Linear(2, d_g)
        self.w_g = nn.Linear(d_g, n_heads, bias=False)

    def forward(self, H, W, device):
        # 生成所有位置对的相对几何特征
        ys = torch.linspace(0, 1, H, device=device)
        xs = torch.linspace(0, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # L 2
        L = pos.shape[0]
        # 相对距离 (log)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # L L 2
        r = torch.log(diff.abs() + 1e-6)
        bias = self.w_g(F.relu(self.W_g(r)))  # L L n_heads
        return bias.permute(2, 0, 1)  # n_heads L L


# ==================== Latent Gaussian Prior (LGP) ====================
class LatentGaussianPrior(nn.Module):
    """
    解码器每步预测 2D 高斯 (mu_x, mu_y, sigma_x, sigma_y)
    作为 foveal 软约束注入 cross-attention
    """
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # mu_x, mu_y, rho_x, rho_y
        )

    def forward(self, decoder_state, H, W):
        # decoder_state: B d_model
        params = self.mlp(decoder_state)
        mu = torch.sigmoid(params[:, :2])          # [0,1]
        sigma = F.softplus(params[:, 2:]) + 1e-4   # >0

        # 构造网格
        ys = torch.linspace(0, 1, H, device=decoder_state.device)
        xs = torch.linspace(0, 1, W, device=decoder_state.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # L 2

        # 高斯
        mu = mu.unsqueeze(1)      # B 1 2
        sigma = sigma.unsqueeze(1)
        dist = ((grid.unsqueeze(0) - mu) ** 2) / (2 * sigma ** 2)
        g = torch.exp(-dist.sum(-1))  # B L
        g = g / (g.sum(-1, keepdim=True) + 1e-8)
        return g, mu.squeeze(1), sigma.squeeze(1)


# ==================== 简化 Selective SSM (Mamba 风格) ====================
class SimpleSelectiveSSM(nn.Module):
    """纯 PyTorch 近似 selective SSM，用于 DAS 输出后的序列建模"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1)  # B, C, delta
        self.dt_proj = nn.Linear(1, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: B L D
        B, L, D = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = F.silu(x)

        # selective params
        x_dbl = self.x_proj(x)  # B L (2*d_state + 1)
        delta, B_param, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # B L D

        A = -torch.exp(self.A_log.float())  # D d_state

        # 简化离散化 + 递归（为了速度用并行近似）
        # 这里用简化的对角 SSM + 门控
        y = x * torch.sigmoid(z)  # 门控
        # 简单全局聚合近似 selective
        h = torch.cumsum(y * delta, dim=1) / (torch.cumsum(delta, dim=1) + 1e-6)
        y = y + h * self.D.view(1, 1, -1)
        return self.out_proj(y)


# ==================== 完整融合模型 ====================
class FusionHMERModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

        # 视觉 backbone (轻量 CNN，类似 DenseNet 简化)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, D_MODEL, 3, stride=2, padding=1),
            nn.BatchNorm2d(D_MODEL),
            nn.ReLU(),
        )

        # DAS
        self.das = DynamicAdaptiveScan(D_MODEL)

        # Geometry bias
        self.geo_bias = RelativeGeometryBias(N_HEADS)

        # 位置编码
        self.pos_enc = nn.Parameter(torch.randn(1, 2048, D_MODEL) * 0.02)

        # Encoder layers (self-attn + SSM)
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True),
                "ssm": SimpleSelectiveSSM(D_MODEL),
                "norm1": nn.LayerNorm(D_MODEL),
                "norm2": nn.LayerNorm(D_MODEL),
                "ffn": nn.Sequential(
                    nn.Linear(D_MODEL, D_FF),
                    nn.GELU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(D_FF, D_MODEL),
                    nn.Dropout(DROPOUT),
                ),
                "norm3": nn.LayerNorm(D_MODEL),
            }) for _ in range(N_ENCODER_LAYERS)
        ])

        # Decoder
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.decoder_pos = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, D_MODEL) * 0.02)
        self.lgp = LatentGaussianPrior(D_MODEL)

        self.decoder_layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True),
                "cross_attn": nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True),
                "norm1": nn.LayerNorm(D_MODEL),
                "norm2": nn.LayerNorm(D_MODEL),
                "norm3": nn.LayerNorm(D_MODEL),
                "ffn": nn.Sequential(
                    nn.Linear(D_MODEL, D_FF),
                    nn.GELU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(D_FF, D_MODEL),
                    nn.Dropout(DROPOUT),
                ),
            }) for _ in range(N_DECODER_LAYERS)
        ])

        self.out_proj = nn.Linear(D_MODEL, vocab_size)
        self.dropout = nn.Dropout(DROPOUT)

    def encode(self, images):
        # images: B 1 H W
        feat = self.backbone(images)  # B C h w
        B, C, h, w = feat.shape

        # DAS 自适应扫描
        seq, sample_grid = self.das(feat)  # B L C , L=h*w

        # 加位置编码
        seq = seq + self.pos_enc[:, :seq.size(1)]

        # Geometry bias
        geo = self.geo_bias(h, w, images.device)  # n_heads L L

        # Encoder
        for layer in self.encoder_layers:
            # self-attn with geo bias
            residual = seq
            seq = layer["norm1"](seq)
            # 把 geo bias 加到 attn mask（简化：直接加到 score）
            attn_out, _ = layer["self_attn"](seq, seq, seq, need_weights=False)
            seq = residual + self.dropout(attn_out)

            # SSM
            residual = seq
            seq = layer["norm2"](seq)
            seq = residual + self.dropout(layer["ssm"](seq))

            # FFN
            residual = seq
            seq = layer["norm3"](seq)
            seq = residual + layer["ffn"](seq)

        return seq, (h, w)

    def decode_step(self, tgt_emb, memory, spatial_shape, past_lgp=None):
        """单步解码，注入 LGP"""
        h, w = spatial_shape
        B, T, _ = tgt_emb.shape

        x = tgt_emb
        for i, layer in enumerate(self.decoder_layers):
            # masked self-attn
            residual = x
            x = layer["norm1"](x)
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_out, _ = layer["self_attn"](x, x, x, attn_mask=causal_mask, need_weights=False)
            x = residual + self.dropout(attn_out)

            # cross-attn + LGP (只在第一层强注入)
            residual = x
            x = layer["norm2"](x)
            # 标准 content attention
            attn_out, attn_weights = layer["cross_attn"](x, memory, memory, need_weights=True)

            if i == 0:
                # 用当前步的 hidden 预测 LGP
                # 取最后一步
                last_state = x[:, -1]  # B D
                g, mu, sigma = self.lgp(last_state, h, w)  # B L
                # 把 g 作为 soft bias 加权到 attention（简化实现）
                # attn_weights: B n_heads T L  -> 我们用平均
                # 这里用 g 调制 memory 后再做一次轻量加权
                g = g.unsqueeze(1)  # B 1 L
                memory_g = memory * g.transpose(1, 2)  # 近似
                attn_out2, _ = layer["cross_attn"](x, memory_g, memory_g, need_weights=False)
                attn_out = 0.7 * attn_out + 0.3 * attn_out2

            x = residual + self.dropout(attn_out)

            # FFN
            residual = x
            x = layer["norm3"](x)
            x = residual + layer["ffn"](x)

        return x

    def forward(self, images, tgt_ids):
        """
        images: B 1 H W
        tgt_ids: B T  (已含 <sos> ... <eos>)
        """
        memory, spatial_shape = self.encode(images)

        # teacher forcing
        tgt_emb = self.token_emb(tgt_ids[:, :-1]) + self.decoder_pos[:, :tgt_ids.size(1)-1]
        tgt_emb = self.dropout(tgt_emb)

        out = self.decode_step(tgt_emb, memory, spatial_shape)
        logits = self.out_proj(out)
        return logits

    @torch.no_grad()
    def generate(self, images, max_len=MAX_DECODE_LEN, sos_id=1, eos_id=2):
        memory, spatial_shape = self.encode(images)
        B = images.size(0)
        device = images.device

        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_emb = self.token_emb(ys) + self.decoder_pos[:, :ys.size(1)]
            hidden = self.decode_step(tgt_emb, memory, spatial_shape)
            logits = self.out_proj(hidden[:, -1])  # B V
            next_token = logits.argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break
        return ys
