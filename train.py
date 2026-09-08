import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Im2LatexHF, download_im2latex100k, collate_fn
from model import SpatialHMER
from vocab import Vocab, build_vocab_from_formulas


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "yuntian-deng/im2latex-100k"

DATA_CACHE_DIR = Path("./data/huggingface")
OUTPUT_DIR = Path("./checkpoints/lgp")

EPOCHS = 25

BATCH_SIZE = 64
NUM_WORKERS = 2

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

# ============================================================
# Autoregressive validation
#
# 每个 epoch 只对固定数量的样本做真正的 autoregressive
# generation。
#
# 500:
#   速度较快
#
# 1000:
#   更稳定
#
# 5000:
#   更准确，但明显更慢
# ============================================================

GEN_EVAL_SAMPLES = 500

GEN_MAX_LEN = MAX_LEN

# 设置 checkpoint 路径可以继续训练
# None 表示从头开始
RESUME_CHECKPOINT = None


# ============================================================
# Random seed
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 对固定尺寸/类似尺寸的 CNN 输入通常更快
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


# ============================================================
# Dataset
# ============================================================

def choose_validation_split(dataset):
    if "val" in dataset:
        return dataset["val"]

    if "validation" in dataset:
        return dataset["validation"]

    raise KeyError(
        "Cannot find validation split. "
        f"Available: {list(dataset.keys())}"
    )


def build_or_load_vocab(train_split):
    vocab_path = OUTPUT_DIR / "vocab.json"

    if vocab_path.exists():
        print(
            f"Loading vocabulary: {vocab_path}"
        )
        return Vocab.load(vocab_path)

    print(
        "Building vocabulary from training formulas..."
    )

    vocab = build_vocab_from_formulas(
        (
            str(row["formula"])
            for row in train_split
        ),
        min_freq=MIN_TOKEN_FREQ,
    )

    vocab.save(vocab_path)

    return vocab


# ============================================================
# Entity tokens
# ============================================================

def make_entity_ids(vocab):
    structural_tokens = {
        "{",
        "}",
        "^",
        "_",
        r"\frac",
        r"\sqrt",
        r"\left",
        r"\right",
        r"\begin",
        r"\end",
        r"\over",
    }

    special_tokens = {
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
    }

    entity_ids = []

    for token, token_id in vocab.stoi.items():
        if token in structural_tokens:
            continue

        if token in special_tokens:
            continue

        entity_ids.append(token_id)

    return entity_ids


# ============================================================
# DataLoader
# ============================================================

def make_loader(
    split,
    vocab,
    shuffle,
):
    dataset = Im2LatexHF(
        split,
        vocab,
        max_len=MAX_LEN,
        image_height=IMAGE_HEIGHT,
        max_width=MAX_IMAGE_WIDTH,
    )

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": lambda batch: collate_fn(
            batch,
            vocab.pad_id,
        ),
    }

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(
        dataset,
        **loader_kwargs,
    )


# ============================================================
# Loss
# ============================================================

def compute_loss(
    logits,
    targets,
    pad_id,
):
    return F.cross_entropy(
        logits.reshape(
            -1,
            logits.size(-1),
        ),
        targets.reshape(-1),
        ignore_index=pad_id,
    )


# ============================================================
# Validation
#
# 一次 forward 同时计算：
#
#   1. val_loss
#   2. token_acc
#   3. tf_seq_acc
#
# 原来的代码这里实际上跑了两遍 validation。
# 现在合并成一遍。
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    pad_id,
    device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    total_tokens = 0
    correct_tokens = 0

    correct_sequences = 0
    total_sequences = 0

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    for images, tokens, _, _ in progress:

        images = images.to(
            device,
            non_blocking=True,
        )

        tokens = tokens.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Teacher forcing
        # ----------------------------------------------------

        decoder_input = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits, _ = model(
            images,
            decoder_input,
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = compute_loss(
            logits,
            targets,
            pad_id,
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predictions = logits.argmax(
            dim=-1
        )

        mask = targets != pad_id

        # ----------------------------------------------------
        # Token accuracy
        # ----------------------------------------------------

        correct = (
            predictions == targets
        ) & mask

        correct_tokens += (
            correct.sum().item()
        )

        total_tokens += (
            mask.sum().item()
        )

        # ----------------------------------------------------
        # Teacher-forced sequence accuracy
        #
        # 一个序列所有有效 token 都正确，
        # 才算整个 expression 正确。
        # ----------------------------------------------------

        sequence_correct = (
            (predictions == targets)
            | ~mask
        ).all(dim=1)

        correct_sequences += (
            sequence_correct.sum().item()
        )

        total_sequences += (
            tokens.size(0)
        )

        # ----------------------------------------------------
        # Loss statistics
        # ----------------------------------------------------

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    # ========================================================
    # Final metrics
    # ========================================================

    avg_loss = (
        total_loss /
        max(total_samples, 1)
    )

    token_accuracy = (
        correct_tokens /
        max(total_tokens, 1)
        * 100.0
    )

    tf_sequence_accuracy = (
        correct_sequences /
        max(total_sequences, 1)
        * 100.0
    )

    return (
        avg_loss,
        token_accuracy,
        tf_sequence_accuracy,
    )


# ============================================================
# Fast autoregressive generation
#
# 注意：
#
# 这里仍然是 autoregressive。
#
# 但是 Encoder 只运行一次：
#
# image
#   ↓
# encoder
#   ↓
# memory
#
# 然后 decoder 才逐 token 生成。
# ============================================================

@torch.no_grad()
def generate_batch(
    model,
    images,
    bos_id,
    eos_id,
    max_len,
):
    model.eval()

    batch_size = images.size(0)

    # --------------------------------------------------------
    # Encoder 只运行一次
    # --------------------------------------------------------

    memory, grid_h, grid_w = model.encode(
        images
    )

    # --------------------------------------------------------
    # 初始 token
    # --------------------------------------------------------

    ids = torch.full(
        (
            batch_size,
            1,
        ),
        bos_id,
        dtype=torch.long,
        device=images.device,
    )

    # --------------------------------------------------------
    # 每个样本是否已经结束
    # --------------------------------------------------------

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=images.device,
    )

    # ========================================================
    # Autoregressive decoding
    # ========================================================

    for _ in range(max_len - 1):

        hidden, _, _ = model.decode_hidden(
            ids,
            memory,
            grid_h,
            grid_w,
        )

        logits = model.head(
            hidden[:, -1]
        )

        next_token = logits.argmax(
            dim=-1
        )

        # ----------------------------------------------------
        # 已经结束的样本强制 EOS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 更新 finished
        # ----------------------------------------------------

        finished |= (
            next_token == eos_id
        )

        # ----------------------------------------------------
        # 整个 batch 都结束
        # ----------------------------------------------------

        if finished.all():
            break

    return ids


# ============================================================
# Real autoregressive sequence accuracy
#
# 只计算 GEN_EVAL_SAMPLES 个样本。
#
# 这个指标才是真正的：
#
# image
#   ↓
# BOS
#   ↓
# token1
#   ↓
# token2
#   ↓
# ...
#
# 而不是 teacher forcing。
# ============================================================

@torch.no_grad()
def evaluate_generation_sequence_accuracy(
    model,
    loader,
    vocab,
    device,
    max_samples,
):
    model.eval()

    correct_sequences = 0
    total_sequences = 0

    processed_samples = 0

    progress = tqdm(
        loader,
        desc=(
            "Autoregressive validation"
        ),
        leave=False,
    )

    for images, tokens, _, _ in progress:

        # ----------------------------------------------------
        # 达到样本数量后停止
        # ----------------------------------------------------

        if processed_samples >= max_samples:
            break

        remaining = (
            max_samples -
            processed_samples
        )

        # ----------------------------------------------------
        # 最后一批只取需要的数量
        # ----------------------------------------------------

        if images.size(0) > remaining:
            images = images[:remaining]
            tokens = tokens[:remaining]

        images = images.to(
            device,
            non_blocking=True,
        )

        tokens = tokens.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Real autoregressive generation
        # ----------------------------------------------------

        generated = generate_batch(
            model=model,
            images=images,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            max_len=GEN_MAX_LEN,
        )

        # ----------------------------------------------------
        # Remove BOS
        # ----------------------------------------------------

        generated = generated[:, 1:]

        targets = tokens[:, 1:]

        batch_size = tokens.size(0)

        # ====================================================
        # Compare every expression
        # ====================================================

        for i in range(batch_size):

            pred = generated[i]
            target = targets[i]

            # ------------------------------------------------
            # Target EOS
            # ------------------------------------------------

            eos_positions = (
                target == vocab.eos_id
            ).nonzero(
                as_tuple=False
            )

            if eos_positions.numel() > 0:

                target_end = (
                    eos_positions[0].item()
                    + 1
                )

            else:

                valid_positions = (
                    target != vocab.pad_id
                ).nonzero(
                    as_tuple=False
                )

                if valid_positions.numel() == 0:
                    target_end = 0
                else:
                    target_end = (
                        valid_positions[-1].item()
                        + 1
                    )

            target = target[:target_end]

            # ------------------------------------------------
            # Prediction EOS
            # ------------------------------------------------

            pred_eos_positions = (
                pred == vocab.eos_id
            ).nonzero(
                as_tuple=False
            )

            if pred_eos_positions.numel() > 0:

                pred_end = (
                    pred_eos_positions[0].item()
                    + 1
                )

                pred = pred[:pred_end]

            # ------------------------------------------------
            # Exact sequence match
            # ------------------------------------------------

            if torch.equal(
                pred,
                target,
            ):
                correct_sequences += 1

            total_sequences += 1

        processed_samples += batch_size

        progress.set_postfix(
            samples=processed_samples
        )

    return (
        correct_sequences /
        max(total_sequences, 1)
        * 100.0
    )


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val,
    best_acc,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "best_acc": best_acc,
            "config": {
                "max_len": MAX_LEN,
                "model_dim": MODEL_DIM,
                "heads": NUM_HEADS,
                "decoder_depth": DECODER_DEPTH,
            },
        },
        path,
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Seed
    # ========================================================

    set_seed(SEED)

    # ========================================================
    # Directories
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Dataset: {DATASET_NAME}"
    )

    print(
        f"Cache: {DATA_CACHE_DIR.resolve()}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    dataset = download_im2latex100k()

    train_split = dataset["train"]

    val_split = choose_validation_split(
        dataset
    )

    print(
        f"Train samples: "
        f"{len(train_split):,}"
    )

    print(
        f"Validation samples: "
        f"{len(val_split):,}"
    )

    # ========================================================
    # Vocabulary
    # ========================================================

    vocab = build_or_load_vocab(
        train_split
    )

    print(
        f"Vocabulary size: "
        f"{len(vocab):,}"
    )

    # ========================================================
    # DataLoader
    # ========================================================

    train_loader = make_loader(
        train_split,
        vocab,
        shuffle=True,
    )

    val_loader = make_loader(
        val_split,
        vocab,
        shuffle=False,
    )

    # ========================================================
    # Model
    # ========================================================

    entity_ids = make_entity_ids(
        vocab
    )

    model = SpatialHMER(
        vocab_size=len(vocab),
        dim=MODEL_DIM,
        heads=NUM_HEADS,
        decoder_depth=DECODER_DEPTH,
        max_len=MAX_LEN,
        pad_id=vocab.pad_id,
        entity_ids=entity_ids,
    ).to(device)

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
        )
    )

    # ========================================================
    # Resume
    # ========================================================

    start_epoch = 0

    best_val = float("inf")
    best_acc = 0.0

    if RESUME_CHECKPOINT:

        checkpoint_path = Path(
            RESUME_CHECKPOINT
        )

        print(
            f"Resuming from: "
            f"{checkpoint_path}"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        if "scheduler" in checkpoint:

            scheduler.load_state_dict(
                checkpoint["scheduler"]
            )

        start_epoch = checkpoint.get(
            "epoch",
            0,
        )

        best_val = checkpoint.get(
            "best_val",
            best_val,
        )

        best_acc = checkpoint.get(
            "best_acc",
            best_acc,
        )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        start_epoch,
        EPOCHS,
    ):

        model.train()

        running_loss = 0.0
        sample_count = 0

        # ====================================================
        # Train progress
        # ====================================================

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS}"
            ),
        )

        # ====================================================
        # Training loop
        # ====================================================

        for images, tokens, _, _ in progress:

            images = images.to(
                device,
                non_blocking=True,
            )

            tokens = tokens.to(
                device,
                non_blocking=True,
            )

            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Teacher forcing
            # ------------------------------------------------

            decoder_input = tokens[:, :-1]

            targets = tokens[:, 1:]

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            logits, _ = model(
                images,
                decoder_input,
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = compute_loss(
                logits,
                targets,
                vocab.pad_id,
            )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            # ------------------------------------------------
            # Optimizer
            # ------------------------------------------------

            optimizer.step()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            batch_size = images.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            sample_count += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        # ====================================================
        # Scheduler
        # ====================================================

        scheduler.step()

        train_loss = (
            running_loss /
            max(sample_count, 1)
        )

        # ====================================================
        # Validation 1
        #
        # 一次 forward 获得：
        #
        #   val_loss
        #   token_acc
        #   tf_seq_acc
        #
        # 不再重复验证。
        # ====================================================

        (
            val_loss,
            token_acc,
            tf_seq_acc,
        ) = evaluate(
            model,
            val_loader,
            vocab.pad_id,
            device,
        )

        # ====================================================
        # Validation 2
        #
        # Real autoregressive generation
        #
        # 只测试固定数量样本。
        # ====================================================

        gen_seq_acc = (
            evaluate_generation_sequence_accuracy(
                model=model,
                loader=val_loader,
                vocab=vocab,
                device=device,
                max_samples=GEN_EVAL_SAMPLES,
            )
        )

        # ====================================================
        # Learning rate
        # ====================================================

        lr = scheduler.get_last_lr()[0]

        # ====================================================
        # Print metrics
        # ====================================================

        print()

        print(
            "=" * 80
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"train_loss : {train_loss:.5f}"
        )

        print(
            f"val_loss   : {val_loss:.5f}"
        )

        print(
            f"token_acc  : {token_acc:.2f}%"
        )

        print(
            f"tf_seq_acc : {tf_seq_acc:.2f}%"
        )

        print(
            f"gen_seq_acc: {gen_seq_acc:.2f}% "
            f"(first {GEN_EVAL_SAMPLES} samples)"
        )

        print(
            f"lr         : {lr:.3e}"
        )

        print(
            "=" * 80
        )

        print()

        # ====================================================
        # Save last checkpoint
        # ====================================================

        save_checkpoint(
            OUTPUT_DIR / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch + 1,
            best_val,
            best_acc,
        )

        # ====================================================
        # Best validation loss
        # ====================================================

        if val_loss < best_val:

            best_val = val_loss

            save_checkpoint(
                OUTPUT_DIR / "best_loss.pt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_val,
                best_acc,
            )

            print(
                "Saved best loss checkpoint: "
                f"{OUTPUT_DIR / 'best_loss.pt'}"
            )

        # ====================================================
        # Best autoregressive accuracy
        # ====================================================

        if gen_seq_acc > best_acc:

            best_acc = gen_seq_acc

            save_checkpoint(
                OUTPUT_DIR / "best_acc.pt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_val,
                best_acc,
            )

            print(
                "Saved best generation accuracy checkpoint: "
                f"{OUTPUT_DIR / 'best_acc.pt'}"
            )

    # ========================================================
    # Finished
    # ========================================================

    print()

    print(
        "Training finished."
    )

    print(
        f"Best loss checkpoint: "
        f"{OUTPUT_DIR / 'best_loss.pt'}"
    )

    print(
        f"Best generation accuracy checkpoint: "
        f"{OUTPUT_DIR / 'best_acc.pt'}"
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()