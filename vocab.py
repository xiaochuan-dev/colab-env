import re
from collections import Counter


SPECIALS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
]


def tokenize_latex(s: str):
    """
    同时兼容：

        \\frac { x } { y }

    和：

        \\frac{x}{y}

    这种 LaTeX。
    """
    s = s.strip()

    if not s:
        return []

    # Im2LaTeX-100K 原始数据很多公式已经是空格分词形式。
    # 但是这里不直接简单 split，避免遇到混合格式。
    pattern = r"\\[A-Za-z]+|\\.|[{}\_^&]|[^\s{}_^&\\]|[^\s]"

    tokens = re.findall(pattern, s)

    return tokens


class Vocab:
    def __init__(self, stoi):
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}

        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    @classmethod
    def build(
        cls,
        formulas,
        min_freq=1,
    ):
        counter = Counter()

        for formula in formulas:
            counter.update(tokenize_latex(formula))

        tokens = [
            token
            for token, freq in counter.items()
            if freq >= min_freq
        ]

        # 保证确定性的 vocab
        tokens.sort()

        itos = SPECIALS + tokens

        stoi = {
            token: idx
            for idx, token in enumerate(itos)
        }

        return cls(stoi)

    def __len__(self):
        return len(self.stoi)

    def encode(
        self,
        tokens,
        max_len,
    ):
        """
        [BOS] token... [EOS]

        max_len 是最终 sequence 长度。
        """

        if max_len < 2:
            raise ValueError("max_len must be >= 2")

        tokens = tokens[: max_len - 2]

        ids = [
            self.bos_id,
            *[
                self.stoi.get(token, self.unk_id)
                for token in tokens
            ],
            self.eos_id,
        ]

        return ids

    def decode(
        self,
        ids,
        remove_special=True,
    ):
        tokens = []

        for idx in ids:
            token = self.itos.get(int(idx), "<unk>")

            if remove_special and token in SPECIALS:
                continue

            tokens.append(token)

        return tokens

    def decode_latex(self, ids):
        """
        用于最终查看预测结果。
        """
        tokens = self.decode(ids)

        output = ""

        for token in tokens:
            if token.startswith("\\"):
                output += token
            elif token in {
                "{",
                "}",
                "_",
                "^",
            }:
                output += token
            else:
                output += token

        return output