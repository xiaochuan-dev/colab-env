from pathlib import Path

import numpy as np
import torch
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from vocab import tokenize_latex


DATASET_NAME = "yuntian-deng/im2latex-100k"
DATA_CACHE_DIR = Path("./data/huggingface")


def download_im2latex100k():
    """Download/cache Im2LaTeX-100K automatically through Hugging Face Datasets."""
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_NAME, cache_dir=str(DATA_CACHE_DIR))

    # The currently published dataset uses train/val/test. Keep validation naming
    # compatible with mirrors that expose validation instead of val.
    if "val" in dataset and "validation" not in dataset:
        dataset["validation"] = dataset["val"]
    elif "validation" in dataset and "val" not in dataset:
        dataset["val"] = dataset["validation"]

    return dataset


class Im2LatexHF(Dataset):
    def __init__(self, split_dataset, vocab, max_len=150, image_height=128, max_width=800):
        self.ds = split_dataset
        self.vocab = vocab
        self.max_len = max_len
        self.image_height = image_height
        self.max_width = max_width

    def __len__(self):
        return len(self.ds)

    def _load_image(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        image = image.convert("L")
        w, h = image.size
        scale = self.image_height / max(h, 1)
        nw = max(8, int(round(w * scale)))
        nw = min(self.max_width, nw)

        image = image.resize((nw, self.image_height), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(image)).float() / 255.0
        x = 1.0 - x
        return x.unsqueeze(0)

    def __getitem__(self, idx):
        row = self.ds[idx]
        image = self._load_image(row["image"])
        formula = str(row["formula"]).strip()
        ids = self.vocab.encode(tokenize_latex(formula), self.max_len)
        return image, torch.tensor(ids, dtype=torch.long), formula


def collate_fn(batch, pad_id):
    images, sequences, formulas = zip(*batch)

    height = images[0].shape[-2]
    max_width = max(image.shape[-1] for image in images)
    image_batch = torch.zeros(len(images), 1, height, max_width, dtype=torch.float32)
    image_mask = torch.zeros(len(images), max_width, dtype=torch.bool)

    for i, image in enumerate(images):
        width = image.shape[-1]
        image_batch[i, :, :, :width] = image
        image_mask[i, :width] = True

    max_tokens = max(sequence.numel() for sequence in sequences)
    token_batch = torch.full(
        (len(sequences), max_tokens), pad_id, dtype=torch.long
    )
    for i, sequence in enumerate(sequences):
        token_batch[i, : sequence.numel()] = sequence

    return image_batch, token_batch, image_mask, formulas
