import os
import glob
import random
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import *
from dataset import build_tokenizer_from_train, build_dataloaders, LaTeXTokenizer
from model import FusionHMERModel
from metrics import evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_resume_ckpt(filename):
    """在 /kaggle/input/ 下递归找指定文件名的 checkpoint"""
    return glob.glob(f"/kaggle/input/**/{filename}", recursive=True)


def main():
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Train] device = {device}")
    print(f"[Train] pwd = {os.getcwd()}")
    print(f"[Train] SAVE_DIR = {SAVE_DIR}")

    # 直接用 config 里的 SAVE_DIR，不写死
    os.makedirs(SAVE_DIR, exist_ok=True)
    LAST_CKPT_PATH = os.path.join(SAVE_DIR, "last.pth")
    BEST_CKPT_PATH = BEST_MODEL_PATH  # 就是 ./checkpoints/best_model.pt

    # 自动在 /kaggle/input/ 下找挂载的旧 checkpoint
    found_last = find_resume_ckpt("last.pth")
    found_best = find_resume_ckpt("best_model.pt")
    print(f"[Resume] found last.pth: {found_last}")
    print(f"[Resume] found best_model.pt: {found_best}")

    # 如果 working 里没有，就从挂载的 Dataset 复制过来
    if not os.path.exists(LAST_CKPT_PATH):
        if found_last:
            shutil.copy(found_last[0], LAST_CKPT_PATH)
            print(f"[Resume] copied {found_last[0]} -> {LAST_CKPT_PATH}")
        elif found_best:
            shutil.copy(found_best[0], LAST_CKPT_PATH)
            print(f"[Resume] copied {found_best[0]} -> {LAST_CKPT_PATH}")

    if not os.path.exists(BEST_CKPT_PATH) and found_best:
        shutil.copy(found_best[0], BEST_CKPT_PATH)
        print(f"[Resume] copied {found_best[0]} -> {BEST_CKPT_PATH}")

    # 1. Tokenizer
    if os.path.exists(TOKENIZER_PATH):
        tokenizer = LaTeXTokenizer()
        tokenizer.load(TOKENIZER_PATH)
        print(f"[Train] loaded tokenizer, vocab={len(tokenizer.token2id)}")
    else:
        tokenizer = build_tokenizer_from_train()

    vocab_size = len(tokenizer.token2id)

    # 2. Data
    train_loader, val_loader, test_loader = build_dataloaders(tokenizer)
    print(f"[Train] train batches={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    # 3. Model
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
    model = FusionHMERModel(vocab_size)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] model params = {total_params / 1e6:.2f} M")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    # ========== 恢复训练 ==========
    start_epoch = 1
    global_step = 0
    best_exprate = -1.0

    if os.path.exists(LAST_CKPT_PATH):
        print(f"[Resume] loading checkpoint from {LAST_CKPT_PATH}")
        ckpt = torch.load(LAST_CKPT_PATH, map_location=device)

        state = ckpt["model"]
        if isinstance(model, nn.DataParallel):
            try:
                model.module.load_state_dict(state)
            except RuntimeError:
                model.load_state_dict(state)
        else:
            model.load_state_dict(state)

        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_exprate = ckpt.get("exprate", -1.0)
        global_step = ckpt.get("global_step", 0)

        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            for _ in range(ckpt["epoch"]):
                scheduler.step()

        print(f"[Resume] resume from epoch {start_epoch}, best_exprate={best_exprate:.2f}")
    else:
        print("[Resume] no checkpoint found, training from scratch")

    # ========== 训练循环 ==========
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for imgs, ids, _ in pbar:
            imgs = imgs.to(device)
            ids = ids.to(device)

            optimizer.zero_grad()
            logits = model(imgs, ids)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                ids[:, 1:].reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1
            if global_step % LOG_INTERVAL == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] train loss = {avg_loss:.4f}")

        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

        # 每轮都存 last.pth
        torch.save({
            "epoch": epoch,
            "model": state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "exprate": best_exprate,
            "vocab_size": vocab_size,
        }, LAST_CKPT_PATH)
        print(f"[Save] last.pth saved (epoch={epoch}) -> {LAST_CKPT_PATH}")

        # 评估 & 存 best
        if epoch % EVAL_INTERVAL == 0:
            metrics = evaluate(model, val_loader, tokenizer, device, desc=f"Val Epoch {epoch}", beam_size=1)
            print(f"[Val] BLEU={metrics['BLEU']:.2f}  "
                  f"EditDist={metrics['EditDistance']:.4f}  "
                  f"ExpRate(norm)={metrics['ExpRate']:.2f}%  "
                  f"ExpRate(raw)={metrics['ExpRate_raw']:.2f}%  "
                  f"(n={metrics['num_samples']})")

            if metrics["ExpRate"] > best_exprate:
                best_exprate = metrics["ExpRate"]
                torch.save({
                    "epoch": epoch,
                    "model": state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": global_step,
                    "exprate": best_exprate,
                    "vocab_size": vocab_size,
                }, BEST_CKPT_PATH)
                print(f"[Save] best model saved (ExpRate={best_exprate:.2f}) -> {BEST_CKPT_PATH}")

    # ========== 最终测试 ==========
    print("\n[Test] loading best model and evaluating on test set ...")
    ckpt = torch.load(BEST_CKPT_PATH, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, tokenizer, device, desc="Test", beam_size=BEAM_SIZE)
    print(f"[Test] BLEU={test_metrics['BLEU']:.2f}  "
          f"EditDist={test_metrics['EditDistance']:.4f}  "
          f"ExpRate(norm)={test_metrics['ExpRate']:.2f}%  "
          f"ExpRate(raw)={test_metrics['ExpRate_raw']:.2f}%")


if __name__ == "__main__":
    main()