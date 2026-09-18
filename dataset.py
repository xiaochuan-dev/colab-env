import re
import json
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config import *


# 多行/复杂环境：cleaned 里很多，当前模型不适合，默认过滤
def strip_env(s: str) -> str:
    """去掉外层 \\begin{xxx}...\\end{xxx}（cleaned 几乎全是 align*）。"""
    if not s:
        return s
    # 反复剥最外层 begin/end
    for _ in range(3):
        m = re.match(
            r"^\s*\\begin\s*\{[a-zA-Z*]+\}\s*([\s\S]*?)\s*\\end\s*\{[a-zA-Z*]+\}\s*$",
            s,
        )
        if not m:
            break
        s = m.group(1).strip()
    return s


def normalize_latex(s: str) -> str:
    """轻度规范化 + 剥 align* 外壳。"""
    if s is None:
        return ""
    s = s.strip()
    s = strip_env(s)
    s = s.replace("$$", " ").replace("$", " ")
    s = re.sub(r"\s+", " ", s)
    # 单字符上下标补花括号
    s = re.sub(r"\^([^{\\\s])", r"^{\1}", s)
    s = re.sub(r"_([^{\\\s])", r"_{\1}", s)
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
        r"\left": r"",
        r"\right": r"",
        r"\displaystyle": r"",
        r"\limits": r"",
        r"\mbox": r"\text",
        r"{\rm ": r"\mathrm{",
        r"\rm ": r"\mathrm ",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def latex_tokenize(formula: str):
    """
    统一 LaTeX 分词（连续公式 / 空格公式都适用）:
      \\alpha  \\,  \\{  以及单字符 { } _ ^ & = 等
    """
    s = formula.strip()
    if not s:
        return []
    tokens = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "\\":
            if i + 1 >= n:
                tokens.append("\\")
                i += 1
                continue
            nxt = s[i + 1]
            if nxt.isalpha():
                j = i + 1
                while j < n and s[j].isalpha():
                    j += 1
                tokens.append(s[i:j])
                i = j
            else:
                # \, \; \! \{ \} \| 等单符号命令
                tokens.append(s[i : i + 2])
                i += 2
            continue
        # 结构/运算符单字符
        if ch in "{}()[]^_&=+-*/,.:;!|<>'~":
            tokens.append(ch)
            i += 1
            continue
        # 连续数字合成一个 token
        if ch.isdigit():
            j = i + 1
            while j < n and s[j].isdigit():
                j += 1
            tokens.append(s[i:j])
            i = j
            continue
        # 普通字母：单字符（避免把整段英文粘成稀有大 token）
        tokens.append(ch)
        i += 1
    return tokens


# 剥壳后仍含这些 → 多行/嵌套，仍过滤
_COMPLEX_MARKERS = (
    r"\begin",
    r"\end",
    r"\substack",
    r"\\\\",  # align 多行
)


def is_good_formula(formula: str) -> bool:
    if not formula or not formula.strip():
        return False
    # 先剥壳再判断
    core = normalize_latex(formula)
    if not core:
        return False
    if FILTER_COMPLEX_ENV:
        low = core.lower()
        if any(m in low for m in (r"\begin", r"\end", r"\substack")):
            return False
        # 多行 align（剥壳后仍有 \\）
        if r"\\" in core:
            return False
    if len(core) > MAX_FORMULA_CHARS:
        return False
    if len(latex_tokenize(core)) < 1:
        return False
    return True


class LaTeXTokenizer:
    def __init__(self):
        self.token2id = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.special = set(self.token2id.keys())

    def build_vocab(self, formulas, min_freq=VOCAB_MIN_FREQ):
        counter = Counter()
        for f in formulas:
            counter.update(latex_tokenize(normalize_latex(f)))
        for tok, cnt in counter.most_common():
            if cnt >= min_freq and tok not in self.token2id:
                idx = len(self.token2id)
                self.token2id[tok] = idx
                self.id2token[idx] = tok
        print(f"[Tokenizer] vocab size = {len(self.token2id)}")
        # 统计 unk 风险：低频被丢掉的比例
        total = sum(counter.values())
        kept = sum(c for t, c in counter.items() if t in self.token2id)
        print(f"[Tokenizer] token coverage = {kept / max(total, 1) * 100:.2f}%")

    def _tokenize(self, formula: str):
        return latex_tokenize(formula)

    def encode(self, formula, max_len=MAX_FORMULA_LEN):
        formula = normalize_latex(formula)
        tokens = self._tokenize(formula)[: max_len - 2]
        ids = [self.token2id.get(t, self.token2id["<unk>"]) for t in tokens]
        return [self.token2id["<sos>"]] + ids + [self.token2id["<eos>"]]

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
            json.dump(
                {
                    "token2id": self.token2id,
                    "id2token": {str(k): v for k, v in self.id2token.items()},
                },
                f,
            )

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.token2id = data["token2id"]
        self.id2token = {int(k): v for k, v in data["id2token"].items()}


def get_image_transform():
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
        return torch.from_numpy(arr).unsqueeze(0)

    return transform


def _extract_image_and_text(item):
    img = item.get("image")
    if img is None:
        raise KeyError("sample missing image")
    text = (
        item.get("text")
        or item.get("formula")
        or item.get("latex_formula")
        or item.get("latex")
        or ""
    )
    return img, text


def _text_column_name(column_names):
    for c in ("latex_formula", "text", "formula", "latex"):
        if c in column_names:
            return c
    raise KeyError(f"no formula column in {column_names}")


_CLEANED_SPLITS = None


def _get_cleaned_splits():
    """下载 cleaned_formulas → 过滤复杂环境 → 截断 MAX_SAMPLES → 90/5/5。"""
    global _CLEANED_SPLITS
    if _CLEANED_SPLITS is not None:
        return _CLEANED_SPLITS
    from datasets import load_dataset

    print(f"[Data] loading {CLEANED_DATASET}/{CLEANED_CONFIG} (once) ...")
    full = load_dataset(CLEANED_DATASET, CLEANED_CONFIG, split="train")
    col = _text_column_name(full.column_names)
    n0 = len(full)

    if FILTER_COMPLEX_ENV:
        def _ok(ex):
            return is_good_formula(ex[col])

        full = full.filter(_ok, num_proc=1)
        print(f"[Data] after filter complex/long: {n0} -> {len(full)}")

    n = len(full)
    max_n = int(MAX_SAMPLES) if MAX_SAMPLES is not None else n
    if max_n < n:
        full = full.shuffle(seed=SEED).select(range(max_n))
        print(f"[Data] limited samples: {n} -> {len(full)} (MAX_SAMPLES={max_n})")
    else:
        print(f"[Data] using samples: {len(full)}")

    full = full.train_test_split(test_size=0.1, seed=SEED)
    rest = full["test"].train_test_split(test_size=0.5, seed=SEED)
    _CLEANED_SPLITS = {
        "train": full["train"],
        "val": rest["train"],
        "test": rest["test"],
    }
    print(
        f"[Data] cleaned splits: train={len(_CLEANED_SPLITS['train'])}, "
        f"val={len(_CLEANED_SPLITS['val'])}, test={len(_CLEANED_SPLITS['test'])}"
    )
    return _CLEANED_SPLITS


class HFFormulaDataset(Dataset):
    def __init__(self, split="train", tokenizer=None, transform=None, backend=None):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.transform = transform or get_image_transform()
        backend = backend or DATASET_BACKEND
        self.backend = backend

        if backend == "cleaned":
            key = "val" if split in ("val", "validation") else split
            if key not in ("train", "val", "test"):
                key = "test"
            self.ds = _get_cleaned_splits()[key]

        elif backend == "latex_ocr":
            hf_split = "validation" if split in ("val", "validation", "test") else "train"
            if split == "test":
                print("[Data] lukbl 无独立 test，使用 validation")
            print(f"[Data] loading {LATEX_OCR_DATASET} split={hf_split} ...")
            self.ds = load_dataset(LATEX_OCR_DATASET, split=hf_split)

        elif backend == "hf_100k":
            name = {"train": "train", "val": "val", "validation": "val", "test": "test"}.get(
                split, split
            )
            print(f"[Data] loading {DATASET_NAME} split={name} ...")
            self.ds = load_dataset(DATASET_NAME, split=name)

        else:
            raise ValueError(
                f"不支持 DATASET_BACKEND={backend}，可选: cleaned / latex_ocr / hf_100k"
            )

        print(f"[Data] {backend} split={split} samples={len(self.ds)}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        n = len(self.ds)
        last_err = None
        for k in range(8):
            j = (idx + k) % n
            try:
                item = self.ds[j]
                img, formula = _extract_image_and_text(item)
                if not isinstance(img, Image.Image):
                    img = Image.open(img).convert("L")
                else:
                    img = img.convert("L")
                img_t = self.transform(img)
                formula = normalize_latex(formula)
                if self.tokenizer is None:
                    return img_t, formula
                ids = self.tokenizer.encode(formula)
                return img_t, torch.tensor(ids, dtype=torch.long), formula
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"failed to load sample near idx={idx}: {last_err}")


def make_dataset(split, tokenizer):
    if DATASET_BACKEND not in ("cleaned", "latex_ocr", "hf_100k"):
        raise ValueError(f"Unknown DATASET_BACKEND={DATASET_BACKEND}")
    return HFFormulaDataset(split, tokenizer, backend=DATASET_BACKEND)


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
    train_ds = make_dataset("train", tokenizer)
    val_ds = make_dataset("val", tokenizer)
    test_ds = make_dataset("test", tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def build_tokenizer_from_train():
    """只读文本列建词表，不解码图片。"""
    print("[Data] building tokenizer from train split (text only) ...")
    ds = make_dataset("train", tokenizer=None)
    col = _text_column_name(ds.ds.column_names)
    formulas = ds.ds[col]
    print(f"[Data] collected {len(formulas)} formulas (column={col})")
    # 打印一个分词样例，方便确认不是整句 <unk>
    demo = normalize_latex(formulas[0])
    print(f"[Data] tokenize demo: {demo[:80]}")
    print(f"[Data] tokens: {latex_tokenize(demo)[:40]}")
    tok = LaTeXTokenizer()
    tok.build_vocab(formulas)
    tok.save(TOKENIZER_PATH)
    # unk 率抽查
    unk = tok.token2id["<unk>"]
    rates = []
    for f in list(formulas)[:500]:
        ids = tok.encode(f)
        body = ids[1:-1]
        if body:
            rates.append(sum(1 for x in body if x == unk) / len(body))
    if rates:
        print(f"[Tokenizer] avg unk rate on 500 samples = {sum(rates)/len(rates)*100:.2f}%")
    return tok
