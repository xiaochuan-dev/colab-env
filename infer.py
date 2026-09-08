from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import SpatialHMER
from vocab import Vocab


# ============================================================
# Inference configuration: edit these paths directly.
# ============================================================
CHECKPOINT_PATH = Path("./checkpoints/lgp/best.pt")
VOCAB_PATH = Path("./checkpoints/lgp/vocab.json")
IMAGE_PATH = Path("./test.png")

MAX_LEN = 150
IMAGE_HEIGHT = 128
MAX_IMAGE_WIDTH = 800
# ============================================================


def load_image(path):
    image = Image.open(path).convert("L")
    w, h = image.size
    scale = IMAGE_HEIGHT / max(h, 1)
    new_width = max(8, int(round(w * scale)))
    new_width = min(MAX_IMAGE_WIDTH, new_width)

    image = image.resize(
        (new_width, IMAGE_HEIGHT),
        Image.Resampling.BILINEAR,
    )
    x = 1.0 - torch.from_numpy(np.asarray(image)).float() / 255.0
    return x.unsqueeze(0).unsqueeze(0)


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(VOCAB_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    entity_ids = make_entity_ids(vocab)
    model = SpatialHMER(
        vocab_size=len(vocab),
        dim=256,
        heads=8,
        decoder_depth=3,
        max_len=MAX_LEN,
        pad_id=vocab.pad_id,
        entity_ids=entity_ids,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image = load_image(IMAGE_PATH).to(device)
    token_ids = model.generate(
        image,
        vocab.bos_id,
        vocab.eos_id,
        MAX_LEN,
    )[0].tolist()

    print(vocab.decode(token_ids))


if __name__ == "__main__":
    main()
