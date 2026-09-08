import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from x_transformers import (
    ContinuousTransformerWrapper,
    Decoder,
)


class ConvBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                1,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(
                128,
                192,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(192),
            nn.GELU(),

            nn.Conv2d(
                192,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        dropout=0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GeometrySelfAttention(nn.Module):
    """
    Geometry-aware self attention。

    使用二维 feature-grid 坐标作为 relative geometry bias。
    """

    def __init__(
        self,
        dim=256,
        heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        assert dim % heads == 0

        self.qkv = nn.Linear(
            dim,
            dim * 3,
        )

        self.out = nn.Linear(
            dim,
            dim,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.geometry = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, heads),
        )

    def forward(
        self,
        x,
        coords,
        mask=None,
    ):
        """
        x:
            [B, N, D]

        coords:
            [N, 2]

        mask:
            [B, N]
            True = valid
        """

        b, n, d = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            b,
            n,
            3,
            self.heads,
            self.head_dim,
        )

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4,
        )

        q, k, v = qkv

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        delta = (
            coords[:, None, :]
            - coords[None, :, :]
        )

        abs_delta = delta.abs()

        geo_input = torch.cat(
            [
                delta,
                abs_delta,
            ],
            dim=-1,
        )

        geometry_bias = self.geometry(
            geo_input
        )

        geometry_bias = geometry_bias.permute(
            2,
            0,
            1,
        )

        scores = (
            scores
            + geometry_bias.unsqueeze(0)
        )

        if mask is not None:
            key_mask = ~mask

            scores = scores.masked_fill(
                key_mask[:, None, None, :],
                -torch.finfo(scores.dtype).max,
            )

        attn = F.softmax(
            scores,
            dim=-1,
        )

        attn = self.dropout(attn)

        out = torch.matmul(
            attn,
            v,
        )

        out = out.transpose(
            1,
            2,
        ).reshape(
            b,
            n,
            d,
        )

        out = self.out(out)

        return out


class GeometryEncoderLayer(nn.Module):
    def __init__(
        self,
        dim=256,
        heads=8,
        ff_dim=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = GeometrySelfAttention(
            dim=dim,
            heads=heads,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(dim)

        self.ff = FeedForward(
            dim,
            ff_dim,
            dropout,
        )

    def forward(
        self,
        x,
        coords,
        mask=None,
    ):
        x = x + self.attn(
            self.norm1(x),
            coords,
            mask,
        )

        x = x + self.ff(
            self.norm2(x)
        )

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return x


class GeometryEncoder(nn.Module):
    def __init__(
        self,
        dim=256,
        depth=3,
        heads=8,
        ff_dim=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                GeometryEncoderLayer(
                    dim=dim,
                    heads=heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x,
        coords,
        mask=None,
    ):
        for layer in self.layers:
            x = layer(
                x,
                coords,
                mask,
            )

        return x


class GaussianPredictor(nn.Module):
    """
    LGP:
        hidden state -> mu_x, mu_y, sigma_x, sigma_y
    """

    def __init__(
        self,
        dim=256,
        hidden_dim=128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x):
        params = self.net(x)

        mu = torch.sigmoid(
            params[..., :2]
        )

        sigma = F.softplus(
            params[..., 2:4]
        ) + 1e-3

        return mu, sigma


class GaussianCrossAttention(nn.Module):
    """
    Cross attention + Gaussian spatial prior。

    ARM coverage 也在这里加入。
    """

    def __init__(
        self,
        dim=256,
        heads=8,
        dropout=0.1,
        use_arm=True,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.use_arm = use_arm

        assert dim % heads == 0

        self.q = nn.Linear(
            dim,
            dim,
        )

        self.k = nn.Linear(
            dim,
            dim,
        )

        self.v = nn.Linear(
            dim,
            dim,
        )

        self.out = nn.Linear(
            dim,
            dim,
        )

        self.lgp = GaussianPredictor(
            dim=dim,
            hidden_dim=128,
        )

        if use_arm:
            self.arm = nn.Conv2d(
                1,
                32,
                kernel_size=5,
                padding=2,
            )

            self.arm_out = nn.Conv2d(
                32,
                1,
                kernel_size=1,
            )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        memory,
        grid_h,
        grid_w,
        memory_mask=None,
        coverage=None,
    ):
        """
        x:
            [B, T, D]

        memory:
            [B, N, D]
        """

        b, t, d = x.shape

        n = memory.shape[1]

        q = self.q(x)
        k = self.k(memory)
        v = self.v(memory)

        q = q.reshape(
            b,
            t,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.reshape(
            b,
            n,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.reshape(
            b,
            n,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        # ------------------------------------------------
        # LGP
        # ------------------------------------------------

        mu, sigma = self.lgp(x)

        device = x.device

        yy, xx = torch.meshgrid(
            torch.linspace(
                0,
                1,
                grid_h,
                device=device,
            ),
            torch.linspace(
                0,
                1,
                grid_w,
                device=device,
            ),
            indexing="ij",
        )

        coords = torch.stack(
            [
                xx.reshape(-1),
                yy.reshape(-1),
            ],
            dim=-1,
        )

        coords = coords.unsqueeze(0).unsqueeze(0)

        mu = mu.unsqueeze(2)

        sigma = sigma.unsqueeze(2)

        diff = coords - mu

        gaussian = -0.5 * (
            diff[..., 0] ** 2
            / sigma[..., 0] ** 2
            +
            diff[..., 1] ** 2
            / sigma[..., 1] ** 2
        )

        gaussian = gaussian.unsqueeze(1)

        scores = scores + gaussian

        # ------------------------------------------------
        # ARM
        # ------------------------------------------------

        if (
            self.use_arm
            and coverage is not None
        ):
            arm_map = coverage.reshape(
                b,
                1,
                grid_h,
                grid_w,
            )

            arm_map = self.arm(
                arm_map
            )

            arm_map = self.arm_out(
                arm_map
            )

            arm_map = arm_map.reshape(
                b,
                1,
                1,
                n,
            )

            scores = scores + arm_map

        # ------------------------------------------------
        # visual mask
        # ------------------------------------------------

        if memory_mask is not None:
            scores = scores.masked_fill(
                ~memory_mask[:, None, None, :],
                -torch.finfo(scores.dtype).max,
            )

        attention = F.softmax(
            scores,
            dim=-1,
        )

        attention = self.dropout(
            attention
        )

        out = torch.matmul(
            attention,
            v,
        )

        out = out.transpose(
            1,
            2,
        ).reshape(
            b,
            t,
            d,
        )

        out = self.out(out)

        # 用 head 平均 attention 更新 coverage
        mean_attention = attention.mean(
            dim=1
        )

        return (
            out,
            mean_attention,
            mu,
        )


class DecoderSelfAttention(nn.Module):
    def __init__(
        self,
        dim=256,
        heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        padding_mask=None,
    ):
        t = x.shape[1]

        causal_mask = torch.triu(
            torch.ones(
                t,
                t,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        y = self.norm(x)

        y, _ = self.attn(
            y,
            y,
            y,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        return x + self.dropout(y)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim=256,
        heads=8,
        ff_dim=1024,
        dropout=0.1,
        use_arm=False,
    ):
        super().__init__()

        self.self_attn = DecoderSelfAttention(
            dim=dim,
            heads=heads,
            dropout=dropout,
        )

        self.cross_norm = nn.LayerNorm(
            dim
        )

        self.cross_attn = GaussianCrossAttention(
            dim=dim,
            heads=heads,
            dropout=dropout,
            use_arm=use_arm,
        )

        self.ff_norm = nn.LayerNorm(
            dim
        )

        self.ff = FeedForward(
            dim=dim,
            hidden_dim=ff_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x,
        memory,
        grid_h,
        grid_w,
        token_padding_mask=None,
        memory_mask=None,
        coverage=None,
    ):
        x = self.self_attn(
            x,
            padding_mask=token_padding_mask,
        )

        cross_out, attention, mu = (
            self.cross_attn(
                self.cross_norm(x),
                memory,
                grid_h,
                grid_w,
                memory_mask=memory_mask,
                coverage=coverage,
            )
        )

        x = x + cross_out

        x = x + self.ff(
            self.ff_norm(x)
        )

        return (
            x,
            attention,
            mu,
        )


class SpatialHMER(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        model_dim=256,
        num_heads=8,
        decoder_depth=3,
        ff_dim=1024,
        max_len=150,
        dropout=0.1,
    ):
        super().__init__()

        self.model_dim = model_dim

        self.pad_id = pad_id

        self.max_len = max_len

        # ------------------------------------------------
        # Encoder
        # ------------------------------------------------

        self.backbone = ConvBackbone()

        self.encoder_proj = nn.Conv2d(
            256,
            model_dim,
            kernel_size=1,
        )

        self.geometry_encoder = GeometryEncoder(
            dim=model_dim,
            depth=3,
            heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # ------------------------------------------------
        # Decoder
        # ------------------------------------------------

        self.token_emb = nn.Embedding(
            vocab_size,
            model_dim,
            padding_idx=pad_id,
        )

        self.pos = nn.Parameter(
            torch.randn(
                1,
                max_len,
                model_dim,
            ) * 0.02
        )

        self.decoder_layers = nn.ModuleList()

        for i in range(decoder_depth):
            # ARM 从第二层开始
            use_arm = i >= 1

            self.decoder_layers.append(
                DecoderBlock(
                    dim=model_dim,
                    heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    use_arm=use_arm,
                )
            )

        self.norm = nn.LayerNorm(
            model_dim
        )

        self.head = nn.Linear(
            model_dim,
            vocab_size,
        )

        # ------------------------------------------------
        # x_transformers
        #
        # 保留一个轻量的 FF/Transformer refinement。
        # ------------------------------------------------

        self.refine = ContinuousTransformerWrapper(
            dim=model_dim,
            max_seq_len=max_len,
            attn_layers=Decoder(
                dim=model_dim,
                depth=1,
                heads=num_heads,
                ff_mult=4,
                cross_attend=False,
            ),
        )

    # ====================================================
    # Encoder
    # ====================================================

    def _downsample_mask(
        self,
        image_mask,
        grid_h,
        grid_w,
    ):
        """
        image_mask:
            [B, W]

        CNN 一共 4 次 stride=2，
        所以 feature width 大约为 W/16。

        使用 adaptive max pooling 保证：
        只要一个 feature cell 对应的原图区域存在，
        就认为该 cell 有效。
        """

        mask = image_mask.float()

        mask = F.adaptive_max_pool1d(
            mask.unsqueeze(1),
            grid_w,
        )

        mask = mask.squeeze(1) > 0.5

        return mask

    def encode(
        self,
        images,
        image_mask=None,
    ):
        x = self.backbone(
            images
        )

        x = self.encoder_proj(
            x
        )

        b, d, h, w = x.shape

        grid_h = h
        grid_w = w

        # --------------------------------------------
        # visual mask
        # --------------------------------------------

        if image_mask is not None:
            memory_mask = (
                self._downsample_mask(
                    image_mask,
                    grid_h,
                    grid_w,
                )
            )
        else:
            memory_mask = torch.ones(
                b,
                h * w,
                dtype=torch.bool,
                device=x.device,
            )

        # --------------------------------------------
        # flatten
        # --------------------------------------------

        x = x.flatten(
            2
        ).transpose(
            1,
            2,
        )

        # --------------------------------------------
        # geometry coordinates
        # --------------------------------------------

        yy, xx = torch.meshgrid(
            torch.linspace(
                0,
                1,
                grid_h,
                device=x.device,
            ),
            torch.linspace(
                0,
                1,
                grid_w,
                device=x.device,
            ),
            indexing="ij",
        )

        coords = torch.stack(
            [
                xx.flatten(),
                yy.flatten(),
            ],
            dim=-1,
        )

        x = self.geometry_encoder(
            x,
            coords,
            memory_mask,
        )

        x = x * memory_mask.unsqueeze(
            -1
        )

        return (
            x,
            grid_h,
            grid_w,
            memory_mask,
        )

    # ====================================================
    # Decoder
    # ====================================================

    def decode_hidden(
        self,
        tokens,
        memory,
        grid_h,
        grid_w,
        memory_mask=None,
    ):
        b, t = tokens.shape

        if t > self.max_len:
            raise ValueError(
                f"Sequence length {t} > max_len {self.max_len}"
            )

        x = self.token_emb(
            tokens
        )

        x = x + self.pos[
            :,
            :t,
        ]

        token_padding_mask = (
            tokens == self.pad_id
        )

        coverage = None

        last_attention = None
        last_mu = None

        for layer in self.decoder_layers:
            (
                x,
                attention,
                mu,
            ) = layer(
                x,
                memory,
                grid_h,
                grid_w,
                token_padding_mask=token_padding_mask,
                memory_mask=memory_mask,
                coverage=coverage,
            )

            last_attention = attention
            last_mu = mu

            # attention:
            # [B, T, N]
            #
            # 每个 decoder token 对视觉区域的 attention
            if attention is not None:
                current_coverage = attention

                if coverage is None:
                    coverage = current_coverage
                else:
                    coverage = (
                        coverage
                        + current_coverage
                    )

        # x_transformers refinement
        x = self.refine(
            x
        )

        x = self.norm(
            x
        )

        return (
            x,
            last_attention,
            last_mu,
        )

    def forward(
        self,
        images,
        tokens,
        image_mask=None,
    ):
        (
            memory,
            grid_h,
            grid_w,
            memory_mask,
        ) = self.encode(
            images,
            image_mask,
        )

        hidden, attention, mu = (
            self.decode_hidden(
                tokens,
                memory,
                grid_h,
                grid_w,
                memory_mask,
            )
        )

        logits = self.head(
            hidden
        )

        return (
            logits,
            attention,
        )

    # ====================================================
    # Autoregressive generation
    # ====================================================

    @torch.no_grad()
    def generate(
        self,
        image,
        bos_id,
        eos_id,
        max_len=150,
        image_mask=None,
    ):
        self.eval()

        (
            memory,
            grid_h,
            grid_w,
            memory_mask,
        ) = self.encode(
            image,
            image_mask,
        )

        batch_size = image.shape[0]

        ids = torch.full(
            (
                batch_size,
                1,
            ),
            bos_id,
            dtype=torch.long,
            device=image.device,
        )

        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=image.device,
        )

        for _ in range(
            max_len - 1
        ):
            hidden, _, _ = (
                self.decode_hidden(
                    ids,
                    memory,
                    grid_h,
                    grid_w,
                    memory_mask,
                )
            )

            logits = self.head(
                hidden[:, -1]
            )

            next_token = logits.argmax(
                dim=-1
            )

            next_token = torch.where(
                finished,
                torch.full_like(
                    next_token,
                    eos_id,
                ),
                next_token,
            )

            ids = torch.cat(
                [
                    ids,
                    next_token.unsqueeze(1),
                ],
                dim=1,
            )

            finished |= (
                next_token == eos_id
            )

            if finished.all():
                break

        return ids