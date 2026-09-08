import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import ContinuousTransformerWrapper, Decoder


class ConvBackbone(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 192, 3, 2, 1),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.Conv2d(192, dim, 3, 2, 1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class GeometrySelfAttention(nn.Module):
    def __init__(self, dim=256, heads=8, geo_dim=64, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.dk = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.geo = nn.Sequential(
            nn.Linear(2, geo_dim),
            nn.ReLU(),
            nn.Linear(geo_dim, heads),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w, mask=None):
        b, n, d = x.shape
        y = self.norm(x)

        q, k, v = self.qkv(y).chunk(3, dim=-1)
        q = q.view(b, n, self.heads, self.dk).transpose(1, 2)
        k = k.view(b, n, self.heads, self.dk).transpose(1, 2)
        v = v.view(b, n, self.heads, self.dk).transpose(1, 2)

        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=x.device),
            torch.linspace(0, 1, w, device=x.device),
            indexing="ij",
        )
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        rel = coords[:, None, :] - coords[None, :, :]
        rel = torch.sign(rel) * torch.log1p(rel.abs() * max(h, w))
        geometry_bias = self.geo(rel).permute(2, 0, 1).unsqueeze(0)

        energy = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        energy = energy + geometry_bias

        if mask is not None:
            energy = energy.masked_fill(
                ~mask[:, None, None, :], -torch.finfo(energy.dtype).max
            )

        attention = energy.softmax(dim=-1)
        z = (attention @ v).transpose(1, 2).reshape(b, n, d)
        return x + self.drop(self.proj(z))


class FFN(nn.Module):
    def __init__(self, dim=256, mult=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class GeometryEncoder(nn.Module):
    def __init__(self, dim=256, depth=3, heads=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [GeometrySelfAttention(dim, heads), FFN(dim)]
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat):
        _, _, h, w = feat.shape
        x = feat.flatten(2).transpose(1, 2)
        for attention, ffn in self.layers:
            x = attention(x, h, w)
            x = ffn(x)
        return self.norm(x), h, w


class GaussianCrossAttention(nn.Module):
    """First decoder cross-attention with an explicit Gaussian spatial prior."""

    def __init__(self, dim=256, heads=8, gaussian_hidden=128, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.dk = dim // heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.gaussian = nn.Sequential(
            nn.Linear(dim, gaussian_hidden),
            nn.GELU(),
            nn.Linear(gaussian_hidden, 4),
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x,
        context,
        grid_h,
        grid_w,
        context_mask=None,
        entity_mask=None,
    ):
        b, t, d = x.shape
        n = context.shape[1]
        if n != grid_h * grid_w:
            raise ValueError(
                f"Context length {n} != grid_h*grid_w ({grid_h}*{grid_w})"
            )

        z = self.norm(x)
        q = self.q(z).view(b, t, self.heads, self.dk).transpose(1, 2)
        k = self.k(context).view(b, n, self.heads, self.dk).transpose(1, 2)
        v = self.v(context).view(b, n, self.heads, self.dk).transpose(1, 2)

        energy = q @ k.transpose(-2, -1) / math.sqrt(self.dk)

        params = self.gaussian(z)
        mu = torch.sigmoid(params[..., :2])
        sigma = F.softplus(params[..., 2:]) + 1e-3

        ys = torch.linspace(0, 1, grid_h, device=x.device)
        xs = torch.linspace(0, 1, grid_w, device=x.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gx.flatten(), gy.flatten()], dim=-1)

        dx = coords[None, None, :, 0] - mu[:, :, None, 0]
        dy = coords[None, None, :, 1] - mu[:, :, None, 1]
        gaussian = torch.exp(
            -0.5
            * (
                (dx / sigma[:, :, None, 0]) ** 2
                + (dy / sigma[:, :, None, 1]) ** 2
            )
        )
        gaussian = gaussian / (gaussian.sum(-1, keepdim=True) + 1e-6)

        if entity_mask is None:
            entity_mask = torch.ones(
                b, t, device=x.device, dtype=torch.bool
            )

        spatial_bias = torch.log(gaussian.clamp_min(1e-6)).unsqueeze(1)
        energy = energy + spatial_bias * entity_mask[:, None, :, None].float()

        if context_mask is not None:
            energy = energy.masked_fill(
                ~context_mask[:, None, None, :], -torch.finfo(energy.dtype).max
            )

        attention = energy.softmax(dim=-1)
        out = (attention @ v).transpose(1, 2).reshape(b, t, d)
        return x + self.drop(self.o(out)), attention, mu


class SelfBlock(nn.Module):
    def __init__(self, dim=256, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.n1 = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        t = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1
        )
        y, _ = self.attn(
            self.n1(x),
            self.n1(x),
            self.n1(x),
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + self.drop(y)
        return x + self.drop(self.ff(self.ff_norm(x)))


class SpatialHMER(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=256,
        heads=8,
        decoder_depth=3,
        max_len=150,
        pad_id=0,
        entity_ids=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.pad_id = pad_id

        self.backbone = ConvBackbone(dim)
        self.encoder = GeometryEncoder(dim, depth=3, heads=heads)

        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

        self.self1 = SelfBlock(dim, heads)
        self.lgp = GaussianCrossAttention(dim, heads)

        self.xt_decoder = ContinuousTransformerWrapper(
            dim_in=dim,
            dim_out=dim,
            max_seq_len=max_len,
            attn_layers=Decoder(
                dim=dim,
                depth=max(1, decoder_depth - 1),
                heads=heads,
                cross_attend=True,
                ff_glu=True,
                rotary_pos_emb=True,
            ),
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.entity_ids = set(entity_ids or [])

    def entity_mask(self, tokens):
        if not self.entity_ids:
            return torch.ones_like(tokens, dtype=torch.bool)

        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for token_id in self.entity_ids:
            mask |= tokens == token_id
        return mask

    def encode(self, images):
        features = self.backbone(images)
        return self.encoder(features)

    def decode_hidden(self, tokens, memory, grid_h, grid_w):
        if tokens.shape[1] > self.max_len:
            raise ValueError(
                f"Sequence length {tokens.shape[1]} exceeds max_len={self.max_len}"
            )

        x = self.token_emb(tokens) + self.pos[:, : tokens.shape[1]]
        x = self.self1(x)
        x, attention, mu = self.lgp(
            x,
            memory,
            grid_h=grid_h,
            grid_w=grid_w,
            entity_mask=self.entity_mask(tokens),
        )
        x = self.xt_decoder(x, context=memory)
        return self.norm(x), attention, mu

    def forward(self, images, tokens):
        memory, grid_h, grid_w = self.encode(images)
        hidden, attention, mu = self.decode_hidden(
            tokens, memory, grid_h, grid_w
        )
        logits = self.head(hidden)
        return logits, {"attention": attention, "mu": mu}

    @torch.no_grad()
    def generate(self, image, bos_id, eos_id, max_len=150):
        self.eval()
        memory, grid_h, grid_w = self.encode(image)

        ids = torch.full(
            (image.size(0), 1),
            bos_id,
            device=image.device,
            dtype=torch.long,
        )

        for _ in range(max_len - 1):
            hidden, _, _ = self.decode_hidden(
                ids, memory, grid_h, grid_w
            )
            logits = self.head(hidden[:, -1])
            next_token = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        return ids
