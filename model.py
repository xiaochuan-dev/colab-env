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
        # H, W 必须是 int（空间尺寸），不能是 Parameter
        H = int(H)
        W = int(W)
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

        return seq, (int(h), int(w))

    def _extract_kv(self, attn_module, x):
        """从 MultiheadAttention 中手动提取 K/V，用于缓存。
        返回 (key, value)，shape 均为 (B, L, D_model)。
        """
        D = x.size(-1)
        weight = attn_module.in_proj_weight
        bias = attn_module.in_proj_bias
        # Q 占 [0:D), K 占 [D:2D), V 占 [2D:3D)
        w_k = weight[D:2 * D]
        w_v = weight[2 * D:3 * D]
        b_k = bias[D:2 * D] if bias is not None else None
        b_v = bias[2 * D:3 * D] if bias is not None else None
        k = F.linear(x, w_k, b_k)
        v = F.linear(x, w_v, b_v)
        return k, v

    def decode_step(self, tgt_emb, memory, spatial_shape, past_kv=None, use_cache=False):
        """
        解码一步（或完整序列）。

        past_kv: list of length N_DECODER_LAYERS，每个元素为
                 (self_k, self_v, cross_k, cross_v) 或 None
                 self_k/v : (B, past_len, D)
                 cross_k/v: (B, mem_len, D)  — memory 投影可跨步复用
        use_cache: True 时只对当前新 token 计算，并返回 updated past_kv

        返回:
            x: (B, T_cur, D)
            new_past_kv: 仅当 use_cache=True 时返回
        """
        spat_h, spat_w = spatial_shape  # 避免与后面 weight 变量冲突
        spat_h, spat_w = int(spat_h), int(spat_w)
        B, T, _ = tgt_emb.shape
        device = tgt_emb.device

        if past_kv is None:
            past_kv = [None] * len(self.decoder_layers)

        new_past_kv = [] if use_cache else None
        x = tgt_emb

        for i, layer in enumerate(self.decoder_layers):
            # ---------- Self-Attention (with optional KV cache) ----------
            residual = x
            x = layer["norm1"](x)

            if use_cache:
                past = past_kv[i]
                if past is not None:
                    past_self_k, past_self_v, past_cross_k, past_cross_v = past
                else:
                    past_self_k = past_self_v = past_cross_k = past_cross_v = None

                # 当前步 K/V
                cur_k, cur_v = self._extract_kv(layer["self_attn"], x)
                if past_self_k is not None:
                    self_k = torch.cat([past_self_k, cur_k], dim=1)
                    self_v = torch.cat([past_self_v, cur_v], dim=1)
                else:
                    self_k, self_v = cur_k, cur_v

                # Q 只来自当前 token
                D = x.size(-1)
                weight = layer["self_attn"].in_proj_weight
                bias = layer["self_attn"].in_proj_bias
                w_q = weight[:D]
                b_q = bias[:D] if bias is not None else None
                q = F.linear(x, w_q, b_q)  # (B, T, D)

                n_heads = layer["self_attn"].num_heads
                head_dim = D // n_heads

                def reshape_heads(t):
                    # (B, L, D) -> (B, n_heads, L, head_dim)
                    return t.view(B, -1, n_heads, head_dim).transpose(1, 2)

                q_h = reshape_heads(q)
                k_h = reshape_heads(self_k)
                v_h = reshape_heads(self_v)

                scale = head_dim ** -0.5
                attn_score = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale  # (B, H, T, past+T)
                if T > 1:
                    # 全量时仍需 causal mask
                    causal_mask = torch.triu(
                        torch.ones(T, self_k.size(1), device=device, dtype=torch.bool),
                        diagonal=1
                    )
                    attn_score = attn_score.masked_fill(
                        causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                    )
                attn_prob = F.softmax(attn_score, dim=-1)
                attn_prob = F.dropout(attn_prob, p=DROPOUT, training=self.training)
                attn_out_h = torch.matmul(attn_prob, v_h)  # (B, H, T, head_dim)
                attn_out = attn_out_h.transpose(1, 2).contiguous().view(B, T, D)
                attn_out = layer["self_attn"].out_proj(attn_out)
            else:
                # 全量计算（训练 / teacher-forcing）
                causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
                attn_out, _ = layer["self_attn"](x, x, x, attn_mask=causal_mask, need_weights=False)
                self_k = self_v = None

            x = residual + self.dropout(attn_out)

            # ---------- Cross-Attention + LGP ----------
            residual = x
            x = layer["norm2"](x)

            if use_cache:
                # cross K/V 只依赖 memory，首次计算后复用
                if past is not None and past_cross_k is not None:
                    cross_k, cross_v = past_cross_k, past_cross_v
                else:
                    cross_k, cross_v = self._extract_kv(layer["cross_attn"], memory)

                D = x.size(-1)
                weight = layer["cross_attn"].in_proj_weight
                bias = layer["cross_attn"].in_proj_bias
                w_q = weight[:D]
                b_q = bias[:D] if bias is not None else None
                q = F.linear(x, w_q, b_q)

                n_heads = layer["cross_attn"].num_heads
                head_dim = D // n_heads

                def reshape_heads(t):
                    return t.view(B, -1, n_heads, head_dim).transpose(1, 2)

                q_h = reshape_heads(q)
                k_h = reshape_heads(cross_k)
                v_h = reshape_heads(cross_v)

                scale = head_dim ** -0.5
                attn_score = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
                attn_prob = F.softmax(attn_score, dim=-1)
                attn_prob = F.dropout(attn_prob, p=DROPOUT, training=self.training)
                attn_out_h = torch.matmul(attn_prob, v_h)
                attn_out = attn_out_h.transpose(1, 2).contiguous().view(B, T, D)
                attn_out = layer["cross_attn"].out_proj(attn_out)

                # LGP 注入（仅第一层）
                if i == 0:
                    last_state = x[:, -1]
                    g, mu, sigma = self.lgp(last_state, spat_h, spat_w)  # B L
                    g = g.unsqueeze(1)  # B 1 L
                    memory_g = memory * g.transpose(1, 2)
                    cross_k_g, cross_v_g = self._extract_kv(layer["cross_attn"], memory_g)
                    k_h_g = reshape_heads(cross_k_g)
                    v_h_g = reshape_heads(cross_v_g)
                    attn_score_g = torch.matmul(q_h, k_h_g.transpose(-2, -1)) * scale
                    attn_prob_g = F.softmax(attn_score_g, dim=-1)
                    attn_out_h_g = torch.matmul(attn_prob_g, v_h_g)
                    attn_out2 = attn_out_h_g.transpose(1, 2).contiguous().view(B, T, D)
                    attn_out2 = layer["cross_attn"].out_proj(attn_out2)
                    attn_out = 0.7 * attn_out + 0.3 * attn_out2
            else:
                # 全量路径
                attn_out, _ = layer["cross_attn"](x, memory, memory, need_weights=False)
                if i == 0:
                    last_state = x[:, -1]
                    g, mu, sigma = self.lgp(last_state, spat_h, spat_w)
                    g = g.unsqueeze(1)
                    memory_g = memory * g.transpose(1, 2)
                    attn_out2, _ = layer["cross_attn"](x, memory_g, memory_g, need_weights=False)
                    attn_out = 0.7 * attn_out + 0.3 * attn_out2
                cross_k = cross_v = None

            x = residual + self.dropout(attn_out)

            # ---------- FFN ----------
            residual = x
            x = layer["norm3"](x)
            x = residual + layer["ffn"](x)

            if use_cache:
                new_past_kv.append((self_k, self_v, cross_k, cross_v))

        if use_cache:
            return x, new_past_kv
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

        out = self.decode_step(tgt_emb, memory, spatial_shape, use_cache=False)
        logits = self.out_proj(out)
        return logits

    @torch.no_grad()
    def generate(self, images, max_len=MAX_DECODE_LEN, sos_id=1, eos_id=2,
                 beam_size=None, length_penalty=None):
        """
        解码入口。
        beam_size<=1: 带 KV Cache 的贪心
        beam_size>1 : Beam Search（带长度惩罚）
        输入会自动搬到与模型参数相同的 device，避免 DataParallel 多卡 mismatch。
        """
        param_device = next(self.parameters()).device
        if images.device != param_device:
            images = images.to(param_device)

        beam_size = BEAM_SIZE if beam_size is None else beam_size
        length_penalty = LENGTH_PENALTY if length_penalty is None else length_penalty

        if beam_size is None or beam_size <= 1:
            return self._generate_greedy(images, max_len, sos_id, eos_id)
        return self._generate_beam(images, max_len, sos_id, eos_id, beam_size, length_penalty)

    @torch.no_grad()
    def _generate_greedy(self, images, max_len=MAX_DECODE_LEN, sos_id=1, eos_id=2):
        """带 KV Cache 的贪心解码。"""
        memory, spatial_shape = self.encode(images)
        B = images.size(0)
        device = images.device

        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        past_kv = None

        for step in range(max_len):
            cur_ids = ys[:, -1:]
            pos = ys.size(1) - 1
            tgt_emb = self.token_emb(cur_ids) + self.decoder_pos[:, pos:pos+1]

            hidden, past_kv = self.decode_step(
                tgt_emb, memory, spatial_shape,
                past_kv=past_kv, use_cache=True
            )
            logits = self.out_proj(hidden[:, -1])
            next_token = logits.argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break
        return ys

    @torch.no_grad()
    def _generate_beam(self, images, max_len, sos_id, eos_id, beam_size, length_penalty):
        """
        Beam Search + 每条 beam 独立 KV Cache。
        每步只对最新 token 做 decode，避免整段重算。
        验证建议单卡调用（DataParallel 时用 model.module）。
        """
        memory, spatial_shape = self.encode(images)
        B = images.size(0)
        device = images.device
        final_seqs = []

        for b in range(B):
            mem_b = memory[b:b+1]  # (1, Lm, D)

            # 每条 beam: seq(list[int]), score(float), past_kv
            beams = [{"seq": [sos_id], "score": 0.0, "past_kv": None}]
            finished_beams = []

            for step in range(max_len):
                active = [bm for bm in beams if bm["seq"][-1] != eos_id]
                if not active:
                    break

                candidates = []
                for bm in active:
                    seq = bm["seq"]
                    pos = len(seq) - 1
                    cur = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
                    tgt_emb = self.token_emb(cur) + self.decoder_pos[:, pos:pos + 1]

                    hidden, new_past = self.decode_step(
                        tgt_emb, mem_b, spatial_shape,
                        past_kv=bm["past_kv"], use_cache=True
                    )
                    log_probs = F.log_softmax(self.out_proj(hidden[:, -1]), dim=-1)[0]
                    topk_logp, topk_ids = log_probs.topk(beam_size)

                    for logp, tid in zip(topk_logp.tolist(), topk_ids.tolist()):
                        tid = int(tid)
                        new_seq = seq + [tid]
                        new_score = bm["score"] + float(logp)
                        if tid == eos_id:
                            L = max(len(new_seq) - 1, 1)
                            finished_beams.append(
                                (new_seq, new_score / (L ** length_penalty))
                            )
                        else:
                            candidates.append({
                                "seq": new_seq,
                                "score": new_score,
                                "past_kv": new_past,
                            })

                candidates.sort(key=lambda x: x["score"], reverse=True)
                beams = candidates[:beam_size]

                if len(finished_beams) >= beam_size and not beams:
                    break

            for bm in beams:
                L = max(len(bm["seq"]) - 1, 1)
                finished_beams.append(
                    (bm["seq"], bm["score"] / (L ** length_penalty))
                )

            if not finished_beams:
                finished_beams = [([sos_id, eos_id], -1e9)]

            finished_beams.sort(key=lambda x: x[1], reverse=True)
            final_seqs.append(finished_beams[0][0])

        max_l = max(len(s) for s in final_seqs)
        out = torch.full((B, max_l), 0, dtype=torch.long, device=device)
        for i, seq in enumerate(final_seqs):
            out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        return out
