import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import numpy as np
from collections import Counter
import json
import os
import re
from config import *


def normalize_latex(s: str) -> str:
    """与 metrics.py 保持一致的规范化，训练时对 GT 也规范化。"""
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\^([^{\\])", r"^{\1}", s)
    s = re.sub(r"_([^{\\])", r"_{\1}", s)
    s = re.sub(r"(\\[a-zA-Z]+)\^([^{\\])", r"\1^{\2}", s)
    s = re.sub(r"(\\[a-zA-Z]+)_([^{\\])", r"\1_{\2}", s)
    replacements = {
        r"\geq": r"\ge",
        r"\leq": r"\le",
        r"\neq": r"\ne",
        r"\rightarrow": r"\to",
        r"\longrightarrow": r"\to",
        r"\ldots": r"\dots",
        r"\cdots": r"\dots",
        r"\varphi": r"\phi",
        r"\varepsilon": r"\epsilon",
        r"\varnothing": r"\emptyset",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


class LaTeXTokenizer:
    def __init__(self):
        self.token2id = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.special = set(self.token2id.keys())

    def build_vocab(self, formulas, min_freq=VOCAB_MIN_FREQ):
        counter = Counter()
        for f in formulas:
            tokens = self._tokenize(normalize_latex(f))
            counter.update(tokens)
        for tok, cnt in counter.most_common():
            if cnt >= min_freq and tok not in self.token2id:
                idx = len(self.token2id)
                self.token2id[tok] = idx
                self.id2token[idx] = tok
        print(f"[Tokenizer] vocab size = {len(self.token2id)}")

    def _tokenize(self, formula: str):
        formula = formula.strip()
        tokens = []
        i = 0
        s = formula
        while i < len(s):
            if s[i].isspace():
                i += 1
                continue
            if s[i] == "\\" and i + 1 < len(s):
                j = i + 1
                while j < len(s) and (s[j].isalpha() or s[j] in "^*_{}"):
                    j += 1
                tokens.append(s[i:j])
                i = j
            elif s[i] in "{}()[]^_":
                tokens.append(s[i])
                i += 1
            else:
                j = i + 1
                while j < len(s) and not s[j].isspace() and s[j] not in "{}()[]^_\\":
                    j += 1
                tokens.append(s[i:j])
                i = j
        return tokens

    def encode(self, formula, max_len=MAX_FORMULA_LEN):
        formula = normalize_latex(formula)
        tokens = self._tokenize(formula)[:max_len - 2]
        ids = [self.token2id.get(t, self.token2id["<unk>"]) for t in tokens]
        ids = [self.token2id["<sos>"]] + ids + [self.token2id["<eos>"]]
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i == self.token2id["<eos>"]:
                break
            if i in (self.token2id["<sos>"], self.token2id["<pad>"]):
                continue
            tokens.append(self.id2token.get(i, "<unk>"))
        return " ".join(tokens)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"token2id": self.token2id, "id2token": {str(k): v for k, v in self.id2token.items()}}, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.token2id = data["token2id"]
        self.id2token = {int(k): v for k, v in data["id2token"].items()}


def get_image_transform():
    """纯 PIL + numpy 实现，不依赖 torchvision"""
    target_h, target_w = IMG_SIZE

    def transform(img: Image.Image):
        img = img.convert("L")
        w, h = img.size
        scale = min(target_w / max(w, 1), target_h / max(h, 1))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("L", (target_w, target_h), 255)
        left = (target_w - new_w) // 2
        top = (target_h - new_h) // 2
        canvas.paste(img, (left, top))

        arr = np.array(canvas, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor

    return transform


class Im2LatexDataset(Dataset):
    def __init__(self, split="train", tokenizer=None, transform=None, max_samples=None):
        print(f"[Data] loading {DATASET_NAME} split={split} ...")
        self.ds = load_dataset(DATASET_NAME, split=split)

        # ========== 快速验证用：限制样本数 ==========
        if max_samples is not None and max_samples < len(self.ds):
            self.ds = self.ds.select(range(max_samples))
            print(f"[Data] {split} truncated to {max_samples} samples")
        # ==========================================

        self.tokenizer = tokenizer
        self.transform = transform or get_image_transform()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"].convert("L")
        formula = item["formula"]
        img_t = self.transform(img)
        if self.tokenizer is None:
            return img_t, formula
        ids = self.tokenizer.encode(formula)
        # 返回规范化后的 formula 便于评测对齐
        return img_t, torch.tensor(ids, dtype=torch.long), normalize_latex(formula)


def collate_fn(batch):
    imgs, ids_list, formulas = zip(*batch)
    imgs = torch.stack(imgs, 0)

    max_len = min(max(len(x) for x in ids_list), MAX_SEQ_LEN)
    B = len(ids_list)
    ids = torch.full((B, max_len), 0, dtype=torch.long)
    for i, seq in enumerate(ids_list):
        L = min(len(seq), max_len)
        ids[i, :L] = seq[:L]
    return imgs, ids, formulas


def build_dataloaders(tokenizer):
    # 快速验证：train 只用 1000 张，val/test 也缩小
    train_ds = Im2LatexDataset("train", tokenizer, max_samples=1000)
    val_ds   = Im2LatexDataset("val",   tokenizer, max_samples=200)
    test_ds  = Im2LatexDataset("test",  tokenizer, max_samples=200)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )
    return train_loader, val_loader, test_loader


def build_tokenizer_from_train():
    print("[Data] building tokenizer from train split ...")
    ds = load_dataset(DATASET_NAME, split="train")
    formulas = [x["formula"] for x in ds]
    tok = LaTeXTokenizer()
    tok.build_vocab(formulas)
    tok.save(TOKENIZER_PATH)
    return tok