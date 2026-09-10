import os
import random
import math
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


def main():
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Train] device = {device}")

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
        model = nn.DataParallel(model, device_ids=[0, 1])

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] model params = {total_params / 1e6:.2f} M")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    best_exprate = -1.0
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for imgs, ids, _ in pbar:
            imgs = imgs.to(device)
            ids = ids.to(device)

            optimizer.zero_grad()
            logits = model(imgs, ids)  # B T-1 V
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

        # 评估
        if epoch % EVAL_INTERVAL == 0:
            metrics = evaluate(model, val_loader, tokenizer, device, desc=f"Val Epoch {epoch}")
            print(f"[Val] BLEU={metrics['BLEU']:.2f}  "
                  f"EditDist={metrics['EditDistance']:.4f}  "
                  f"ExpRate={metrics['ExpRate']:.2f}%  "
                  f"(n={metrics['num_samples']})")

            if metrics["ExpRate"] > best_exprate:
                best_exprate = metrics["ExpRate"]
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "exprate": best_exprate,
                    "vocab_size": vocab_size,
                }, BEST_MODEL_PATH)
                print(f"[Save] best model saved (ExpRate={best_exprate:.2f})")

    # 最终测试
    print("\n[Test] loading best model and evaluating on test set ...")
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, tokenizer, device, desc="Test")
    print(f"[Test] BLEU={test_metrics['BLEU']:.2f}  "
          f"EditDist={test_metrics['EditDistance']:.4f}  "
          f"ExpRate={test_metrics['ExpRate']:.2f}%")


if __name__ == "__main__":
    main()
