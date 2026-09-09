# dataset.py
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config as cfg
from utils.vocab import Vocab


class Im2LatexDataset(Dataset):
    def __init__(self, samples, vocab: Vocab, is_train=True):
        """
        samples: list of (img_path, latex_str)
        """
        self.samples = samples
        self.vocab = vocab
        self.is_train = is_train

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=cfg.NORMALIZE_MEAN, std=cfg.NORMALIZE_STD)

    def __len__(self):
        return len(self.samples)

    def _resize_keep_ratio(self, img: Image.Image):
        w, h = img.size
        scale = cfg.IMG_HEIGHT / h
        new_w = min(int(w * scale), cfg.IMG_MAX_WIDTH)
        new_h = cfg.IMG_HEIGHT
        img = img.resize((new_w, new_h), Image.BILINEAR)
        # pad to fixed max width for batching convenience (or dynamic pad later)
        return img

    def __getitem__(self, idx):
        img_path, latex = self.samples[idx]
        try:
            if img_path == "dummy" or not os.path.exists(img_path):
                raise FileNotFoundError
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # 坏图 / 假数据 fallback
            img = Image.new("RGB", (cfg.IMG_HEIGHT * 4, cfg.IMG_HEIGHT), (255, 255, 255))

        if self.is_train and cfg.USE_AUG:
            # 简单 scale 增强
            scale = random.uniform(*cfg.SCALE_RANGE)
            w, h = img.size
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)

        img = self._resize_keep_ratio(img)
        img = self.to_tensor(img)
        img = self.normalize(img)

        # pad 到固定宽度（简化 batch）
        c, h, w = img.shape
        if w < cfg.IMG_MAX_WIDTH:
            pad = torch.zeros(c, h, cfg.IMG_MAX_WIDTH - w)
            img = torch.cat([img, pad], dim=2)
        else:
            img = img[:, :, :cfg.IMG_MAX_WIDTH]

        ids = self.vocab.encode(latex)
        if len(ids) > cfg.MAX_SEQ_LEN:
            ids = ids[:cfg.MAX_SEQ_LEN - 1] + [self.vocab.eos_id]
        return img, torch.tensor(ids, dtype=torch.long), latex


def collate_fn(batch):
    imgs, seqs, raws = zip(*batch)
    imgs = torch.stack(imgs, dim=0)

    max_len = max(s.size(0) for s in seqs)
    padded = torch.full((len(seqs), max_len), fill_value=0, dtype=torch.long)  # pad_id=0
    for i, s in enumerate(seqs):
        padded[i, :s.size(0)] = s
    return imgs, padded, raws


def load_samples(formula_file, image_dir):
    samples = []
    if not os.path.exists(formula_file):
        print(f"[WARN] {formula_file} 不存在，请准备 im2latexv2 数据")
        return samples
    with open(formula_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            name, latex = parts[0].strip(), parts[1].strip()
            path = os.path.join(image_dir, name)
            if not os.path.exists(path):
                # 尝试常见后缀
                for ext in [".png", ".jpg", ".jpeg", ".PNG"]:
                    if os.path.exists(path + ext):
                        path = path + ext
                        break
                    if os.path.exists(os.path.join(image_dir, name + ext)):
                        path = os.path.join(image_dir, name + ext)
                        break
            if os.path.exists(path):
                samples.append((path, latex))
    print(f"[Data] loaded {len(samples)} samples")
    return samples


def build_dataloaders(vocab: Vocab):
    samples = load_samples(cfg.FORMULA_FILE, cfg.IMAGE_DIR)
    if len(samples) == 0:
        # 生成假数据方便调试代码结构
        print("[Data] 使用假数据做结构测试")
        samples = [("dummy", "a + b = c")] * 64

    random.seed(42)
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * cfg.TRAIN_RATIO)
    n_val = int(n * cfg.VAL_RATIO)
    train_s = samples[:n_train]
    val_s = samples[n_train:n_train + n_val]
    test_s = samples[n_train + n_val:]

    train_ds = Im2LatexDataset(train_s, vocab, is_train=True)
    val_ds = Im2LatexDataset(val_s, vocab, is_train=False)
    test_ds = Im2LatexDataset(test_s, vocab, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )
    return train_loader, val_loader, test_loader
