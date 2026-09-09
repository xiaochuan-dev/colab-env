# model/decoder.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


class LatentGaussianPrior(nn.Module):
    """论文 3.3 Spatial Prior for Grounded Decoding (LGP)"""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.D_MODEL, cfg.LGP_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.LGP_HIDDEN, 4),  # μx, μy, ρx, ρy
        )

    def forward(self, decoder_state, h, w):
        """
        decoder_state: (B, T, D) 或 (B, D)
        return: normalized Gaussian map (B, T, N) 或 (B, N)
        """
        squeeze = False
        if decoder_state.dim() == 2:
            decoder_state = decoder_state.unsqueeze(1)
            squeeze = True

        B, T, _ = decoder_state.shape
        params = self.mlp(decoder_state)  # (B, T, 4)
        mu_x = torch.sigmoid(params[..., 0])
        mu_y = torch.sigmoid(params[..., 1])
        # Softplus 保证正方差
        sigma_x = F.softplus(params[..., 2]) + 1e-4
        sigma_y = F.softplus(params[..., 3]) + 1e-4

        # 网格坐标
        ys = torch.linspace(0, 1, h, device=decoder_state.device)
        xs = torch.linspace(0, 1, w, device=decoder_state.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.reshape(-1)  # (N,)
        grid_y = grid_y.reshape(-1)

        # G_t,i = exp( - (x-μx)^2 / 2σx² - (y-μy)^2 / 2σy² )
        dx = grid_x.view(1, 1, -1) - mu_x.unsqueeze(-1)   # (B,T,N)
        dy = grid_y.view(1, 1, -1) - mu_y.unsqueeze(-1)
        log_g = - (dx ** 2) / (2 * sigma_x.unsqueeze(-1) ** 2) \
                - (dy ** 2) / (2 * sigma_y.unsqueeze(-1) ** 2)
        G = torch.exp(log_g)
        G = G / (G.sum(dim=-1, keepdim=True) + 1e-8)

        if squeeze:
            G = G.squeeze(1)
            mu_x = mu_x.squeeze(1)
            mu_y = mu_y.squeeze(1)
        return G, (mu_x, mu_y)


class AttentionRefinementModule(nn.Module):
    """CoMER 的 ARM（简化版 coverage）"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, cfg.ARM_DIM, cfg.ARM_KERNEL, padding=cfg.ARM_KERNEL // 2)
        self.proj = nn.Linear(cfg.ARM_DIM, 1)

    def forward(self, prev_attn, h, w):
        """
        prev_attn: (B, N) 上一时刻 attention
        return: coverage feature (B, N)
        """
        B = prev_attn.size(0)
        feat = prev_attn.view(B, 1, h, w)
        feat = F.relu(self.conv(feat))
        feat = feat.view(B, cfg.ARM_DIM, -1).transpose(1, 2)  # (B, N, C)
        cov = self.proj(feat).squeeze(-1)  # (B, N)
        return cov


class SpatiallyGroundedDecoderLayer(nn.Module):
    def __init__(self, use_lgp=True):
        super().__init__()
        self.use_lgp = use_lgp
        self.self_attn = nn.MultiheadAttention(cfg.D_MODEL, cfg.N_HEAD, dropout=cfg.DROPOUT, batch_first=True)
        self.cross_q = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)
        self.cross_k = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)
        self.cross_v = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)
        self.cross_out = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)

        self.ff = nn.Sequential(
            nn.Linear(cfg.D_MODEL, cfg.D_FF),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.D_FF, cfg.D_MODEL),
            nn.Dropout(cfg.DROPOUT),
        )
        self.norm1 = nn.LayerNorm(cfg.D_MODEL)
        self.norm2 = nn.LayerNorm(cfg.D_MODEL)
        self.norm3 = nn.LayerNorm(cfg.D_MODEL)
        self.dropout = nn.Dropout(cfg.DROPOUT)

        if use_lgp:
            self.lgp = LatentGaussianPrior()
            self.arm = AttentionRefinementModule()

    def forward(self, tgt, memory, memory_hw, tgt_mask=None, is_entity=None, prev_attn=None):
        """
        tgt: (B, T, D)
        memory: (B, N, D)
        memory_hw: (h, w)
        is_entity: (B, T)  1=entity, 0=structural
        prev_attn: (B, N) 用于 ARM
        """
        h, w = memory_hw
        B, T, D = tgt.shape
        N = memory.size(1)

        # 1. Masked self-attention
        residual = tgt
        tgt = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, need_weights=False)
        tgt = residual + self.dropout(tgt2)

        # 2. Cross-attention with optional LGP
        residual = tgt
        tgt = self.norm2(tgt)

        q = self.cross_q(tgt).view(B, T, cfg.N_HEAD, D // cfg.N_HEAD).transpose(1, 2)
        k = self.cross_k(memory).view(B, N, cfg.N_HEAD, D // cfg.N_HEAD).transpose(1, 2)
        v = self.cross_v(memory).view(B, N, cfg.N_HEAD, D // cfg.N_HEAD).transpose(1, 2)

        # content energy
        energy = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D // cfg.N_HEAD)  # (B,H,T,N)

        if self.use_lgp:
            # 用当前层输入预测高斯（简化：用 tgt 的 mean 或最后一步）
            # 论文在每个 step 用 decoder state 预测，这里对整个序列一起算
            G, (mu_x, mu_y) = self.lgp(tgt, h, w)  # (B, T, N)
            log_G = torch.log(G + 1e-8)

            # ARM coverage（用上一时刻或累积）
            if prev_attn is not None:
                cov = self.arm(prev_attn, h, w)  # (B, N)
            else:
                cov = torch.zeros(B, N, device=tgt.device)

            # 论文 Eq.(19) 门控：只对 entity token 加空间约束
            if is_entity is not None:
                # is_entity: (B, T) -> (B, 1, T, 1)
                gate = is_entity.unsqueeze(1).unsqueeze(-1).float()
            else:
                gate = 1.0

            # energy = content + gate * logG - gate * cov
            energy = energy + gate * log_G.unsqueeze(1) - gate * cov.unsqueeze(1).unsqueeze(2)

            attn_weights = F.softmax(energy, dim=-1)  # (B,H,T,N)
            # 取平均 head 作为 coverage 输入
            avg_attn = attn_weights.mean(dim=1).mean(dim=1)  # (B, N) 简化
        else:
            attn_weights = F.softmax(energy, dim=-1)
            avg_attn = attn_weights.mean(dim=1).mean(dim=1)
            mu_x = mu_y = None

        out = torch.matmul(attn_weights, v)  # (B,H,T,d)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.cross_out(out)
        tgt = residual + self.dropout(out)

        # 3. FFN
        residual = tgt
        tgt = self.norm3(tgt)
        tgt = residual + self.ff(tgt)

        return tgt, avg_attn, (mu_x, mu_y)


class SpatiallyGroundedDecoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.D_MODEL, padding_idx=0)
        self.pos_enc = PositionalEncoding(cfg.D_MODEL, cfg.DROPOUT)
        # 第一层使用 LGP，后面层普通（论文建议）
        self.layers = nn.ModuleList([
            SpatiallyGroundedDecoderLayer(use_lgp=(i == 0))
            for i in range(cfg.N_DEC_LAYERS)
        ])
        self.norm = nn.LayerNorm(cfg.D_MODEL)
        self.proj = nn.Linear(cfg.D_MODEL, vocab_size)

    def forward(self, tgt_ids, memory, memory_hw, vocab=None):
        """
        tgt_ids: (B, T)
        memory: (B, N, D)
        """
        B, T = tgt_ids.shape
        x = self.embed(tgt_ids) * math.sqrt(cfg.D_MODEL)
        x = self.pos_enc(x)

        # causal mask
        causal = torch.triu(torch.ones(T, T, device=tgt_ids.device), diagonal=1).bool()

        # entity mask
        is_entity = None
        if vocab is not None:
            is_entity = torch.ones(B, T, device=tgt_ids.device)
            for b in range(B):
                for t in range(T):
                    if vocab.is_struct(tgt_ids[b, t].item()):
                        is_entity[b, t] = 0

        prev_attn = None
        mus = []
        for layer in self.layers:
            x, prev_attn, mu = layer(x, memory, memory_hw, tgt_mask=causal,
                                     is_entity=is_entity, prev_attn=prev_attn)
            if mu[0] is not None:
                mus.append(mu)

        x = self.norm(x)
        logits = self.proj(x)
        return logits, mus


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
