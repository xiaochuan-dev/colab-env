# train.py  —— 直接运行: python train.py
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import config as cfg
from utils.vocab import Vocab
from dataset import build_dataloaders, load_samples
from model.model import LGP_HMER


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_exp_rate(preds, gts, vocab):
    """简单 Exact Match"""
    correct = 0
    total = len(preds)
    for p, g in zip(preds, gts):
        pred_str = vocab.decode(p.tolist())
        if pred_str.replace(" ", "") == g.replace(" ", ""):
            correct += 1
    return correct / max(total, 1)


def train_one_epoch(model, loader, optimizer, scaler, criterion, vocab, device, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")
    for imgs, tgt, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=cfg.USE_AMP):
            logits, mus = model(imgs, tgt, vocab=vocab)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))

            if cfg.LAMBDA_SG > 0 and mus:
                pass

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, vocab, device, epoch, do_decode=True):
    """
    do_decode=False 时只算 val loss（快），
    do_decode=True  时额外做 greedy decode 算 ExpRate（慢）。
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_gts = []
    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]")
    for imgs, tgt, raws in pbar:
        imgs = imgs.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        with autocast("cuda", enabled=cfg.USE_AMP):
            logits, _ = model(imgs, tgt, vocab=vocab)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        total_loss += loss.item()

        if do_decode:
            pred_ids = model.greedy_decode(imgs, vocab)
            all_preds.extend(pred_ids.cpu())
            all_gts.extend(raws)

    exp_rate = compute_exp_rate(all_preds, all_gts, vocab) if do_decode else 0.0
    return total_loss / max(len(loader), 1), exp_rate


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = Vocab()
    if os.path.exists(cfg.VOCAB_FILE):
        vocab.load(cfg.VOCAB_FILE)
        print(f"Loaded vocab size={len(vocab)}")
    else:
        samples = load_samples(cfg.FORMULA_FILE, cfg.IMAGE_DIR)
        formulas = [s[1] for s in samples] if samples else ["a + b = c", "\\frac{1}{2}"]
        vocab.build(formulas)
        vocab.save(cfg.VOCAB_FILE)

    train_loader, val_loader, test_loader = build_dataloaders(vocab)

    model = LGP_HMER(len(vocab)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    def lr_lambda(epoch):
        if epoch < cfg.WARMUP_EPOCHS:
            return (epoch + 1) / cfg.WARMUP_EPOCHS
        progress = (epoch - cfg.WARMUP_EPOCHS) / max(1, cfg.NUM_EPOCHS - cfg.WARMUP_EPOCHS)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler("cuda", enabled=cfg.USE_AMP)

    best_exp = 0.0
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, vocab, device, epoch
        )
        scheduler.step()

        do_decode = (epoch % cfg.EVAL_EVERY == 0) or (epoch == cfg.NUM_EPOCHS)
        val_loss, exp_rate = evaluate(
            model, val_loader, criterion, vocab, device, epoch, do_decode=do_decode
        )

        lr = optimizer.param_groups[0]["lr"]
        if do_decode:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"ExpRate={exp_rate*100:.2f}%  lr={lr:.6f}  time={time.time()-t0:.1f}s"
            )
            if exp_rate > best_exp:
                best_exp = exp_rate
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "vocab_size": len(vocab),
                    "exp_rate": exp_rate,
                }
                path = os.path.join(cfg.CHECKPOINT_DIR, "best.pt")
                torch.save(ckpt, path)
                print(f"  >> saved best to {path}")
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"(skip decode)  lr={lr:.6f}  time={time.time()-t0:.1f}s"
            )

        if epoch % cfg.SAVE_EVERY == 0:
            path = os.path.join(cfg.CHECKPOINT_DIR, f"epoch_{epoch}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict()}, path)

    print(f"Training finished. Best ExpRate = {best_exp*100:.2f}%")


if __name__ == "__main__":
    main()