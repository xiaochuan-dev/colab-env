import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Im2LatexHF, download_im2latex100k, collate_fn
from model import SpatialHMER
from vocab import Vocab, build_vocab_from_formulas, tokenize_latex


# ============================================================
# Configuration: edit these values directly. No CLI arguments.
# ============================================================
DATASET_NAME = "yuntian-deng/im2latex-100k"
DATA_CACHE_DIR = Path("./data/huggingface")
OUTPUT_DIR = Path("./checkpoints/lgp")

EPOCHS = 25
BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_LEN = 150
IMAGE_HEIGHT = 128
MAX_IMAGE_WIDTH = 800
MIN_TOKEN_FREQ = 1
SEED = 42

MODEL_DIM = 256
NUM_HEADS = 8
DECODER_DEPTH = 3
GRAD_CLIP = 5.0

# Set to a checkpoint path to resume, otherwise leave None.
RESUME_CHECKPOINT = None

# ============================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_validation_split(dataset):
    if "val" in dataset:
        return dataset["val"]
    if "validation" in dataset:
        return dataset["validation"]
    raise KeyError(f"Cannot find validation split. Available: {list(dataset.keys())}")


def build_or_load_vocab(train_split):
    vocab_path = OUTPUT_DIR / "vocab.json"
    if vocab_path.exists():
        print(f"Loading vocabulary: {vocab_path}")
        return Vocab.load(vocab_path)

    print("Building vocabulary from training formulas...")
    vocab = build_vocab_from_formulas(
        (str(row["formula"]) for row in train_split),
        min_freq=MIN_TOKEN_FREQ,
    )
    vocab.save(vocab_path)
    return vocab


def make_entity_ids(vocab):
    structural_tokens = {
        "{",
        "}",
        "^",
        "_",
        "\\frac",
        "\\sqrt",
        "\\left",
        "\\right",
        "\\begin",
        "\\end",
        "\\over",
    }
    special_tokens = {"<pad>", "<bos>", "<eos>", "<unk>"}
    return [
        token_id
        for token, token_id in vocab.stoi.items()
        if token not in structural_tokens and token not in special_tokens
    ]


def make_loader(split, vocab, shuffle):
    dataset = Im2LatexHF(
        split,
        vocab,
        max_len=MAX_LEN,
        image_height=IMAGE_HEIGHT,
        max_width=MAX_IMAGE_WIDTH,
    )

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: collate_fn(batch, vocab.pad_id),
    )

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **loader_kwargs)


def compute_loss(logits, targets, pad_id):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
    )


@torch.no_grad()
def evaluate(model, loader, pad_id, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for images, tokens, _, _ in loader:
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)

        logits, _ = model(images, tokens[:, :-1])
        loss = compute_loss(logits, tokens[:, 1:], pad_id)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / max(total_samples, 1)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "config": {
                "max_len": MAX_LEN,
                "model_dim": MODEL_DIM,
                "heads": NUM_HEADS,
                "decoder_depth": DECODER_DEPTH,
            },
        },
        path,
    )


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Cache: {DATA_CACHE_DIR.resolve()}")

    # Automatically downloads the dataset on first run.
    dataset = download_im2latex100k()
    train_split = dataset["train"]
    val_split = choose_validation_split(dataset)

    print(f"Train samples: {len(train_split):,}")
    print(f"Validation samples: {len(val_split):,}")

    vocab = build_or_load_vocab(train_split)
    print(f"Vocabulary size: {len(vocab):,}")

    train_loader = make_loader(train_split, vocab, shuffle=True)
    val_loader = make_loader(val_split, vocab, shuffle=False)

    entity_ids = make_entity_ids(vocab)
    model = SpatialHMER(
        vocab_size=len(vocab),
        dim=MODEL_DIM,
        heads=NUM_HEADS,
        decoder_depth=DECODER_DEPTH,
        max_len=MAX_LEN,
        pad_id=vocab.pad_id,
        entity_ids=entity_ids,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    start_epoch = 0
    best_val = float("inf")

    if RESUME_CHECKPOINT:
        checkpoint_path = Path(RESUME_CHECKPOINT)
        print(f"Resuming from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val = checkpoint.get("best_val", best_val)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        sample_count = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
        )

        for images, tokens, _, _ in progress:
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(images, tokens[:, :-1])
            loss = compute_loss(logits, tokens[:, 1:], vocab.pad_id)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        train_loss = running_loss / max(sample_count, 1)
        val_loss = evaluate(model, val_loader, vocab.pad_id, device)
        lr = scheduler.get_last_lr()[0]

        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} "
            f"lr={lr:.3e}"
        )

        save_checkpoint(
            OUTPUT_DIR / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch + 1,
            best_val,
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                OUTPUT_DIR / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_val,
            )
            print(f"Saved best checkpoint: {OUTPUT_DIR / 'best.pt'}")

    print("Training finished.")
    print(f"Best checkpoint: {OUTPUT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
