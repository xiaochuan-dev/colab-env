from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset import (
    download_im2latex100k,
    Im2LatexHF,
    collate_fn,
)

from vocab import Vocab, tokenize_latex

from model import SpatialHMER


# =========================================================
# Config
# =========================================================

DATASET_NAME = "yuntian-deng/im2latex-100k"

DATA_CACHE_DIR = Path(
    "./data/huggingface"
)

OUTPUT_DIR = Path(
    "./checkpoints/lgp"
)

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

FF_DIM = 1024

DROPOUT = 0.1

GRAD_CLIP = 5.0

RESUME_CHECKPOINT = None

GEN_EVAL_SAMPLES = 500

GEN_MAX_LEN = MAX_LEN


# =========================================================
# Seed
# =========================================================

def seed_everything(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# =========================================================
# Device
# =========================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =========================================================
# Vocab
# =========================================================

def build_vocab(dataset):
    formulas = []

    train_ds = dataset["train"]

    for row in train_ds:
        formulas.append(
            str(
                row["formula"]
            ).strip()
        )

    vocab = Vocab.build(
        formulas,
        min_freq=MIN_TOKEN_FREQ,
    )

    return vocab


# =========================================================
# Loss
# =========================================================

def compute_loss(
    logits,
    targets,
    pad_id,
):
    """
    logits:
        [B, T, V]

    targets:
        [B, T]
    """

    return F.cross_entropy(
        logits.reshape(
            -1,
            logits.shape[-1],
        ),
        targets.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=0.0,
    )


# =========================================================
# Teacher forcing metrics
# =========================================================

def teacher_forcing_metrics(
    logits,
    targets,
    pad_id,
):
    predictions = logits.argmax(
        dim=-1
    )

    mask = (
        targets != pad_id
    )

    correct = (
        predictions == targets
    ) & mask

    total_tokens = mask.sum().item()

    correct_tokens = correct.sum().item()

    token_acc = (
        correct_tokens
        / max(total_tokens, 1)
    )

    sequence_correct = (
        (predictions == targets)
        | (~mask)
    ).all(
        dim=1
    )

    seq_acc = (
        sequence_correct.float()
        .mean()
        .item()
    )

    return (
        token_acc,
        seq_acc,
    )


# =========================================================
# Sequence normalize
# =========================================================

def strip_sequence(
    ids,
    bos_id,
    eos_id,
    pad_id,
):
    result = []

    for token in ids:
        token = int(token)

        if token == bos_id:
            continue

        if token == eos_id:
            result.append(
                token
            )
            break

        if token == pad_id:
            break

        result.append(
            token
        )

    return result


# =========================================================
# Edit distance
# =========================================================

def edit_distance(
    a,
    b,
):
    """
    Levenshtein distance
    """

    n = len(a)
    m = len(b)

    if n == 0:
        return m

    if m == 0:
        return n

    prev = list(
        range(m + 1)
    )

    for i in range(
        1,
        n + 1,
    ):
        cur = [
            i
        ]

        for j in range(
            1,
            m + 1,
        ):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1

            cur.append(
                min(
                    cur[j - 1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + cost,
                )
            )

        prev = cur

    return prev[-1]


# =========================================================
# Generation evaluation
# =========================================================

@torch.no_grad()
def evaluate_generation(
    model,
    loader,
    device,
    vocab,
    max_samples=500,
):
    model.eval()

    exact = 0

    total = 0

    total_distance = 0

    total_ref_length = 0

    le1 = 0

    le2 = 0

    for (
        images,
        tokens,
        image_mask,
        token_mask,
        formulas,
    ) in loader:

        if total >= max_samples:
            break

        remaining = (
            max_samples - total
        )

        if images.shape[0] > remaining:
            images = images[
                :remaining
            ]

            tokens = tokens[
                :remaining
            ]

            image_mask = image_mask[
                :remaining
            ]

        images = images.to(
            device,
            non_blocking=True,
        )

        tokens = tokens.to(
            device,
            non_blocking=True,
        )

        image_mask = image_mask.to(
            device,
            non_blocking=True,
        )

        predictions = model.generate(
            images,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            max_len=GEN_MAX_LEN,
            image_mask=image_mask,
        )

        for i in range(
            images.shape[0]
        ):
            pred = strip_sequence(
                predictions[i].tolist(),
                vocab.bos_id,
                vocab.eos_id,
                vocab.pad_id,
            )

            target = strip_sequence(
                tokens[i].tolist(),
                vocab.bos_id,
                vocab.eos_id,
                vocab.pad_id,
            )

            if pred == target:
                exact += 1

            distance = edit_distance(
                pred,
                target,
            )

            total_distance += distance

            total_ref_length += max(
                len(target),
                1,
            )

            if distance <= 1:
                le1 += 1

            if distance <= 2:
                le2 += 1

            total += 1

    if total == 0:
        return {
            "gen_seq_acc": 0.0,
            "ned": 1.0,
            "le1": 0.0,
            "le2": 0.0,
        }

    return {
        "gen_seq_acc": exact / total,
        "ned": (
            total_distance
            / max(total_ref_length, 1)
        ),
        "le1": le1 / total,
        "le2": le2 / total,
    }


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    vocab,
    gen_samples=500,
):
    model.eval()

    total_loss = 0.0

    total_batches = 0

    total_correct = 0

    total_tokens = 0

    total_seq_correct = 0

    total_sequences = 0

    generation_done = 0

    gen_exact = 0

    gen_distance = 0

    gen_ref_length = 0

    gen_le1 = 0

    gen_le2 = 0

    for (
        images,
        tokens,
        image_mask,
        token_mask,
        formulas,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        tokens = tokens.to(
            device,
            non_blocking=True,
        )

        image_mask = image_mask.to(
            device,
            non_blocking=True,
        )

        # =================================================
        # Teacher forcing
        # =================================================

        decoder_input = tokens[
            :,
            :-1,
        ]

        targets = tokens[
            :,
            1:,
        ]

        logits, _ = model(
            images,
            decoder_input,
            image_mask=image_mask,
        )

        loss = compute_loss(
            logits,
            targets,
            vocab.pad_id,
        )

        total_loss += loss.item()

        total_batches += 1

        predictions = logits.argmax(
            dim=-1
        )

        mask = (
            targets != vocab.pad_id
        )

        correct = (
            predictions == targets
        ) & mask

        total_correct += (
            correct.sum().item()
        )

        total_tokens += (
            mask.sum().item()
        )

        seq_correct = (
            (predictions == targets)
            | (~mask)
        ).all(
            dim=1
        )

        total_seq_correct += (
            seq_correct.sum().item()
        )

        total_sequences += (
            targets.shape[0]
        )

        # =================================================
        # Autoregressive generation
        #
        # 与 TF 共用相同的 validation batch，
        # 避免之前 first 500 / full val 的不公平比较。
        # =================================================

        if generation_done < gen_samples:

            remaining = (
                gen_samples
                - generation_done
            )

            take = min(
                remaining,
                images.shape[0],
            )

            gen_images = images[
                :take
            ]

            gen_tokens = tokens[
                :take
            ]

            gen_masks = image_mask[
                :take
            ]

            generated = model.generate(
                gen_images,
                bos_id=vocab.bos_id,
                eos_id=vocab.eos_id,
                max_len=GEN_MAX_LEN,
                image_mask=gen_masks,
            )

            for i in range(take):

                pred = strip_sequence(
                    generated[i].tolist(),
                    vocab.bos_id,
                    vocab.eos_id,
                    vocab.pad_id,
                )

                target = strip_sequence(
                    gen_tokens[i].tolist(),
                    vocab.bos_id,
                    vocab.eos_id,
                    vocab.pad_id,
                )

                if pred == target:
                    gen_exact += 1

                distance = edit_distance(
                    pred,
                    target,
                )

                gen_distance += (
                    distance
                )

                gen_ref_length += max(
                    len(target),
                    1,
                )

                if distance <= 1:
                    gen_le1 += 1

                if distance <= 2:
                    gen_le2 += 1

            generation_done += take

    token_acc = (
        total_correct
        / max(total_tokens, 1)
    )

    tf_seq_acc = (
        total_seq_correct
        / max(total_sequences, 1)
    )

    gen_seq_acc = (
        gen_exact
        / max(generation_done, 1)
    )

    gen_ned = (
        gen_distance
        / max(gen_ref_length, 1)
    )

    return {
        "loss": (
            total_loss
            / max(total_batches, 1)
        ),
        "token_acc": token_acc,
        "tf_seq_acc": tf_seq_acc,
        "gen_seq_acc": gen_seq_acc,
        "gen_ned": gen_ned,
        "gen_le1": (
            gen_le1
            / max(generation_done, 1)
        ),
        "gen_le2": (
            gen_le2
            / max(generation_done, 1)
        ),
    }


# =========================================================
# Training
# =========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    pad_id,
):
    model.train()

    total_loss = 0.0

    total_batches = 0

    for (
        images,
        tokens,
        image_mask,
        token_mask,
        formulas,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        tokens = tokens.to(
            device,
            non_blocking=True,
        )

        image_mask = image_mask.to(
            device,
            non_blocking=True,
        )

        decoder_input = tokens[
            :,
            :-1,
        ]

        targets = tokens[
            :,
            1:,
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        if scaler is not None:

            with torch.cuda.amp.autocast(
                enabled=True
            ):
                logits, _ = model(
                    images,
                    decoder_input,
                    image_mask=image_mask,
                )

                loss = compute_loss(
                    logits,
                    targets,
                    pad_id,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            logits, _ = model(
                images,
                decoder_input,
                image_mask=image_mask,
            )

            loss = compute_loss(
                logits,
                targets,
                pad_id,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

        total_loss += loss.item()

        total_batches += 1

    return (
        total_loss
        / max(total_batches, 1)
    )


# =========================================================
# Save
# =========================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    vocab,
    best_loss,
    best_acc,
):
    checkpoint = {
        "epoch": epoch,

        "model": model.state_dict(),

        "optimizer": optimizer.state_dict(),

        "scheduler": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),

        "vocab_stoi": vocab.stoi,

        "best_loss": best_loss,

        "best_acc": best_acc,

        "config": {
            "model_dim": MODEL_DIM,
            "num_heads": NUM_HEADS,
            "decoder_depth": DECODER_DEPTH,
            "ff_dim": FF_DIM,
            "max_len": MAX_LEN,
            "image_height": IMAGE_HEIGHT,
            "max_image_width": MAX_IMAGE_WIDTH,
        },
    }

    torch.save(
        checkpoint,
        path,
    )


# =========================================================
# Main
# =========================================================

def main():
    seed_everything(
        SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    # =====================================================
    # Dataset download
    # =====================================================

    print(
        "Loading Im2LaTeX-100K..."
    )

    dataset = (
        download_im2latex100k()
    )

    print(
        dataset
    )

    # =====================================================
    # Vocabulary
    # =====================================================

    print(
        "Building vocabulary..."
    )

    vocab = build_vocab(
        dataset
    )

    print(
        f"Vocabulary size: {len(vocab)}"
    )

    print(
        f"PAD: {vocab.pad_id}"
    )

    print(
        f"BOS: {vocab.bos_id}"
    )

    print(
        f"EOS: {vocab.eos_id}"
    )

    print(
        f"UNK: {vocab.unk_id}"
    )

    # =====================================================
    # Dataset
    # =====================================================

    train_dataset = Im2LatexHF(
        dataset["train"],
        vocab,
        max_len=MAX_LEN,
        image_height=IMAGE_HEIGHT,
        max_width=MAX_IMAGE_WIDTH,
    )

    if "validation" in dataset:
        val_split = dataset[
            "validation"
        ]
    else:
        val_split = dataset[
            "val"
        ]

    val_dataset = Im2LatexHF(
        val_split,
        vocab,
        max_len=MAX_LEN,
        image_height=IMAGE_HEIGHT,
        max_width=MAX_IMAGE_WIDTH,
    )

    # =====================================================
    # DataLoader
    # =====================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        collate_fn=lambda batch:
            collate_fn(
                batch,
                vocab.pad_id,
            ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        collate_fn=lambda batch:
            collate_fn(
                batch,
                vocab.pad_id,
            ),
    )

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Val samples: {len(val_dataset)}"
    )

    # =====================================================
    # Model
    # =====================================================

    model = SpatialHMER(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        decoder_depth=DECODER_DEPTH,
        ff_dim=FF_DIM,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    )

    model = model.to(
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {parameter_count:,}"
    )

    # =====================================================
    # Optimizer
    # =====================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(
            0.9,
            0.98,
        ),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    # =====================================================
    # AMP
    # =====================================================

    scaler = (
        torch.cuda.amp.GradScaler()
        if device.type == "cuda"
        else None
    )

    # =====================================================
    # Resume
    # =====================================================

    start_epoch = 1

    best_loss = float(
        "inf"
    )

    best_acc = 0.0

    if RESUME_CHECKPOINT is not None:

        checkpoint = torch.load(
            RESUME_CHECKPOINT,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        if (
            scheduler is not None
            and checkpoint.get(
                "scheduler"
            ) is not None
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler"]
            )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        best_loss = checkpoint.get(
            "best_loss",
            float("inf"),
        )

        best_acc = checkpoint.get(
            "best_acc",
            0.0,
        )

        print(
            f"Resumed from epoch "
            f"{start_epoch}"
        )

    # =====================================================
    # Training loop
    # =====================================================

    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            vocab.pad_id,
        )

        metrics = evaluate(
            model,
            val_loader,
            device,
            vocab,
            gen_samples=GEN_EVAL_SAMPLES,
        )

        scheduler.step()

        lr = optimizer.param_groups[
            0
        ]["lr"]

        print()

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"train_loss : {train_loss:.5f}"
        )

        print(
            f"val_loss   : {metrics['loss']:.5f}"
        )

        print(
            f"token_acc  : "
            f"{metrics['token_acc'] * 100:.2f}%"
        )

        print(
            f"tf_seq_acc : "
            f"{metrics['tf_seq_acc'] * 100:.2f}%"
        )

        print(
            f"gen_seq_acc: "
            f"{metrics['gen_seq_acc'] * 100:.2f}%"
        )

        print(
            f"gen_NED    : "
            f"{metrics['gen_ned']:.4f}"
        )

        print(
            f"gen_<=1    : "
            f"{metrics['gen_le1'] * 100:.2f}%"
        )

        print(
            f"gen_<=2    : "
            f"{metrics['gen_le2'] * 100:.2f}%"
        )

        print(
            f"lr         : {lr:.3e}"
        )

        # =================================================
        # last
        # =================================================

        save_checkpoint(
            OUTPUT_DIR / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            vocab,
            best_loss,
            best_acc,
        )

        # =================================================
        # best loss
        # =================================================

        if metrics["loss"] < best_loss:

            best_loss = metrics["loss"]

            save_checkpoint(
                OUTPUT_DIR
                / "best_loss.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                vocab,
                best_loss,
                best_acc,
            )

            print(
                "Saved best_loss.pt"
            )

        # =================================================
        # best generation accuracy
        # =================================================

        if (
            metrics["gen_seq_acc"]
            > best_acc
        ):

            best_acc = (
                metrics["gen_seq_acc"]
            )

            save_checkpoint(
                OUTPUT_DIR
                / "best_acc.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                vocab,
                best_loss,
                best_acc,
            )

            print(
                "Saved best_acc.pt"
            )


if __name__ == "__main__":
    main()