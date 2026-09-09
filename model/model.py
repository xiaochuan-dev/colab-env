# model/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
from model.backbone import DenseNetFeature
from model.encoder import GeometryAwareEncoder
from model.decoder import SpatiallyGroundedDecoder


class LGP_HMER(nn.Module):
    """Spatially-Grounded Gaussian-Prior Attention 完整模型"""
    def __init__(self, vocab_size):
        super().__init__()
        self.backbone = DenseNetFeature()
        self.encoder = GeometryAwareEncoder()
        self.decoder = SpatiallyGroundedDecoder(vocab_size)
        self.vocab_size = vocab_size

    def forward(self, images, tgt_ids, vocab=None):
        """
        images: (B, 3, H, W)
        tgt_ids: (B, T)  teacher forcing
        return: logits (B, T, V), aux (mus)
        """
        feat = self.backbone(images)               # (B, D, h, w)
        memory, hw = self.encoder(feat)            # (B, N, D)
        # decoder 输入是 tgt 前移一位
        logits, mus = self.decoder(tgt_ids[:, :-1], memory, hw, vocab=vocab)
        return logits, mus

    @torch.no_grad()
    def greedy_decode(self, images, vocab, max_len=None):
        max_len = max_len or cfg.MAX_DECODE_LEN
        self.eval()
        device = images.device
        B = images.size(0)

        feat = self.backbone(images)
        memory, hw = self.encoder(feat)

        ys = torch.full((B, 1), vocab.sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits, _ = self.decoder(ys, memory, hw, vocab=vocab)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == vocab.eos_id)
            if finished.all():
                break
        return ys
