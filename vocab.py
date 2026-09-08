import json
import re
from collections import Counter
from pathlib import Path

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class Vocab:
    def __init__(self, stoi):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}
        self.pad_id = stoi["<pad>"]
        self.bos_id = stoi["<bos>"]
        self.eos_id = stoi["<eos>"]
        self.unk_id = stoi["<unk>"]

    def __len__(self):
        return len(self.stoi)

    def encode(self, tokens, max_len):
        if max_len < 2:
            raise ValueError("max_len must be >= 2")
        ids = [self.bos_id]
        ids += [self.stoi.get(t, self.unk_id) for t in tokens[: max_len - 2]]
        ids += [self.eos_id]
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            token = self.itos.get(int(i), "<unk>")
            if token == "<eos>":
                break
            if token not in SPECIALS:
                out.append(token)
        return " ".join(out)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.stoi, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def tokenize_latex(s):
    """Tokenize both the space-separated HF formulas and compact LaTeX."""
    s = s.strip()
    if not s:
        return []

    if " " in s:
        return s.split()

    return re.findall(r"\\[A-Za-z]+|[{}_^&]|\\.|[^\s]", s)


def build_vocab_from_formulas(formulas, min_freq=1, max_items=None):
    counter = Counter()
    for formula in formulas:
        counter.update(tokenize_latex(formula))

    items = [token for token, count in counter.most_common() if count >= min_freq]
    if max_items is not None:
        items = items[:max_items]

    stoi = {token: i for i, token in enumerate(SPECIALS)}
    for token in items:
        if token not in stoi:
            stoi[token] = len(stoi)
    return Vocab(stoi)
