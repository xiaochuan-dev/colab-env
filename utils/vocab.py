# utils/vocab.py
import json
import os
from collections import Counter
import config as cfg


class Vocab:
    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3

    def build(self, formulas, min_freq=1):
        counter = Counter()
        for latex in formulas:
            tokens = self.tokenize(latex)
            counter.update(tokens)

        self.token2id = {t: i for i, t in enumerate(cfg.SPECIAL_TOKENS)}
        for tok, freq in counter.most_common():
            if freq >= min_freq and tok not in self.token2id:
                self.token2id[tok] = len(self.token2id)

        self.id2token = {i: t for t, i in self.token2id.items()}
        self.pad_id = self.token2id[cfg.PAD_TOKEN]
        self.sos_id = self.token2id[cfg.SOS_TOKEN]
        self.eos_id = self.token2id[cfg.EOS_TOKEN]
        self.unk_id = self.token2id[cfg.UNK_TOKEN]
        print(f"[Vocab] size = {len(self.token2id)}")

    def tokenize(self, latex: str):
        """简单空格分词 + 常见宏保护。生产环境可换成更精细的 LaTeX tokenizer。"""
        # 保证常见命令不被拆开
        latex = latex.strip()
        tokens = []
        i = 0
        while i < len(latex):
            if latex[i] == "\\":
                j = i + 1
                while j < len(latex) and (latex[j].isalpha() or latex[j] in "'"):
                    j += 1
                tokens.append(latex[i:j])
                i = j
            elif latex[i].isspace():
                i += 1
            else:
                tokens.append(latex[i])
                i += 1
        return tokens

    def encode(self, latex: str, add_sos_eos=True):
        tokens = self.tokenize(latex)
        ids = [self.token2id.get(t, self.unk_id) for t in tokens]
        if add_sos_eos:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special=True):
        tokens = []
        for i in ids:
            if i == self.eos_id:
                break
            tok = self.id2token.get(i, cfg.UNK_TOKEN)
            if skip_special and tok in cfg.SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.token2id = json.load(f)
        self.id2token = {int(i): t for t, i in self.token2id.items()}
        # reverse may have str keys after json
        self.id2token = {int(k): v for k, v in self.id2token.items()}
        self.pad_id = self.token2id[cfg.PAD_TOKEN]
        self.sos_id = self.token2id[cfg.SOS_TOKEN]
        self.eos_id = self.token2id[cfg.EOS_TOKEN]
        self.unk_id = self.token2id[cfg.UNK_TOKEN]

    def __len__(self):
        return len(self.token2id)

    def is_struct(self, token_id):
        tok = self.id2token.get(token_id, "")
        return tok in cfg.STRUCT_TOKENS
