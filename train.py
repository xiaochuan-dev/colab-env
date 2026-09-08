import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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

# ------------------------------------------------------------
# Autoregressive validation
#
# 不需要每个 epoch 对整个验证集做 generate。
# 固定使用前 N 个验证样本计算真实自回归准确率。
#
# 如果你想更准确，可以改成 2000 / 5000。
# 如果想更快，可以改成 500。
# ------------------------------------------------------------
GEN_EVAL_SAMPLES = 1000

# ------------------------------------------------------------
# 自回归生成最多生成多少 token
# ------------------------------------------------------------
GEN_MAX_LEN = MAX_LEN

# Set a checkpoint path to resume, otherwise None.
RESUME_CHECKPOINT = None


# ============================================================
# Random seed
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
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
        f"Cannot find validation split. Available: {list(dataset.keys())}"
    )


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

    return [
        token_id
        for token, token_id in vocab.stoi.items()
        if token not in structural_tokens
        and token not in special_tokens
    ]


# ============================================================
# DataLoader
# ============================================================

def make_loader(split, vocab, shuffle):
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

def compute_loss(logits, targets, pad_id):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
    )


# ============================================================
# Validation
#
# val_loss + token accuracy
#
# 这个是 Teacher Forcing。
# 整个序列一次性并行计算，因此很快。
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

    for images, tokens, _, _ in loader:
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

        loss = compute_loss(
            logits,
            targets,
            pad_id,
        )

        # ----------------------------------------------------
        # Token accuracy
        # ----------------------------------------------------

        predictions = logits.argmax(dim=-1)

        mask = targets != pad_id

        correct = (
            predictions == targets
        ) & mask

        correct_tokens += correct.sum().item()
        total_tokens += mask.sum().item()

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        batch_size = images.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    avg_loss = (
        total_loss /
        max(total_samples, 1)
    )

    token_accuracy = (
        correct_tokens /
        max(total_tokens, 1)
        * 100.0
    )

    return avg_loss, token_accuracy


# ============================================================
# Teacher-forced sequence accuracy
#
# 注意：
# 这个不是论文意义上的真实 autoregressive accuracy。
#
# 它使用 ground-truth prefix。
# ============================================================

@torch.no_grad()
def evaluate_teacher_forced_sequence_accuracy(
    model,
    loader,
    pad_id,
    device,
):
    model.eval()

    correct_sequences = 0
    total_sequences = 0

    for images, tokens, _, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        tokens = tokens.to(
            device,
            non_blocking=True,
        )

        decoder_input = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits, _ = model(
            images,
            decoder_input,
        )

        predictions = logits.argmax(dim=-1)

        mask = targets != pad_id

        sequence_correct = (
            (predictions == targets) | ~mask
        ).all(dim=1)

        correct_sequences += (
            sequence_correct.sum().item()
        )

        total_sequences += tokens.size(0)

    return (
        correct_sequences /
        max(total_sequences, 1)
        * 100.0
    )


# ============================================================
# Autoregressive generation
#
# 这个函数专门用于验证。
#
# 注意：
# 这里仍然是标准 autoregressive decoding。
#
# [BOS]
#   ↓
# [BOS, y1]
#   ↓
# [BOS, y1, y2]
#   ↓
# ...
#
# 但会使用 finished mask：
#
# 某个样本已经 EOS 后，就不再继续生成有效 token。
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

    memory, grid_h, grid_w = model.encode(images)

    ids = torch.full(
        (
            batch_size,
            1,
        ),
        bos_id,
        device=images.device,
        dtype=torch.long,
    )

    finished = torch.zeros(
        batch_size,
        device=images.device,
        dtype=torch.bool,
    )

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
        # 已经结束的样本强制保持 EOS
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
        # 整个 batch 都 EOS
        # 立即停止
        # ----------------------------------------------------

        if finished.all():
            break

    return ids


# ============================================================
# Autoregressive sequence accuracy
#
# 只在固定的 GEN_EVAL_SAMPLES 上测试。
#
# 这样每个 epoch 不会因为整个 validation set 的
# autoregressive decoding 耗费大量时间。
# ============================================================

@torch.no_grad()
def evaluate_generation_sequence_accuracy(
    model,
    loader,
    vocab,
    device,
    max_samples=GEN_EVAL_SAMPLES,
):
    model.eval()

    correct_sequences = 0
    total_sequences = 0

    processed_samples = 0

    progress = tqdm(
        loader,
        desc="Generating validation",
        leave=False,
    )

    for images, tokens, _, _ in progress:

        # ----------------------------------------------------
        # 如果已经达到验证数量，停止
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
            max_len=max_len,
        )

        # generated:
        #
        # [BOS, token1, token2, ..., EOS]
        #
        # Ground truth:
        #
        # [BOS, token1, token2, ..., EOS, PAD, PAD]

        generated = generated[:, 1:]
        targets = tokens[:, 1:]

        batch_size = tokens.size(0)

        # ----------------------------------------------------
        # Compare each sample
        # ----------------------------------------------------

        for i in range(batch_size):

            pred = generated[i]
            target = targets[i]

            # ------------------------------------------------
            # Find EOS in target
            # ------------------------------------------------

            eos_positions = (
                target == vocab.eos_id
            ).nonzero(
                as_tuple=False
            )

            if eos_positions.numel() > 0:
                target_end = (
                    eos_positions[0].item() + 1
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
            # Find EOS in prediction
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
            samples=processed_samples,
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

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        f"Train samples: {len(train_split):,}"
    )

    print(
        f"Validation samples: {len(val_split):,}"
    )

    # ========================================================
    # Vocabulary
    # ========================================================

    vocab = build_or_load_vocab(
        train_split
    )

    print(
        f"Vocabulary size: {len(vocab):,}"
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

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS}"
            ),
        )

        # ====================================================
        # Train
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

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Teacher forcing
            # ------------------------------------------------

            decoder_input = tokens[:, :-1]
            targets = tokens[:, 1:]

            logits, _ = model(
                images,
                decoder_input,
            )

            loss = compute_loss(
                logits,
                targets,
                vocab.pad_id,
            )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

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
        # Full validation:
        #
        # val_loss
        # token_acc
        # ====================================================

        val_loss, token_acc = evaluate(
            model,
            val_loader,
            vocab.pad_id,
            device,
        )

        # ====================================================
        # Validation 2
        #
        # Full validation:
        #
        # Teacher-forced sequence accuracy
        # ====================================================

        tf_seq_acc = (
            evaluate_teacher_forced_sequence_accuracy(
                model,
                val_loader,
                vocab.pad_id,
                device,
            )
        )

        # ====================================================
        # Validation 3
        #
        # Real autoregressive generation
        #
        # 只使用 GEN_EVAL_SAMPLES
        # ====================================================

        gen_seq_acc = (
            evaluate_generation_sequence_accuracy(
                model,
                val_loader,
                vocab,
                device,
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
        # Best generation accuracy
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

