import re
import torch
import torch.nn as nn
import editdistance
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from config import *


def normalize_latex(s: str) -> str:
    """
    轻量级 LaTeX canonical normalization：
    - 统一空白
    - 单字符上下标补花括号: x^2 -> x^{2}, a_1 -> a_{1}
    - 常见等价命令映射
    - 去掉多余的空花括号等
    用于评测时让等价写法算作匹配。
    """
    if s is None:
        return ""
    s = s.strip()
    # 多空格 -> 单空格
    s = re.sub(r"\s+", " ", s)

    # 单 token 上下标补 {}
    # 注意：已有 ^{...} 的不要再改
    s = re.sub(r"\^([^{\\])", r"^{\1}", s)
    s = re.sub(r"_([^{\\])", r"_{\1}", s)
    # 命令后的单字符上下标: \alpha_1 -> \alpha_{1}（简单处理）
    s = re.sub(r"(\\[a-zA-Z]+)\^([^{\\])", r"\1^{\2}", s)
    s = re.sub(r"(\\[a-zA-Z]+)_([^{\\])", r"\1_{\2}", s)

    # 等价命令（选较短/更常见的作为标准）
    replacements = {
        r"\geq": r"\ge",
        r"\leq": r"\le",
        r"\neq": r"\ne",
        r"\rightarrow": r"\to",
        r"\longrightarrow": r"\to",
        r"\cdot": r"\cdot",  # 保持
        r"\ldots": r"\dots",
        r"\cdots": r"\dots",
        r"\varphi": r"\phi",
        r"\varepsilon": r"\epsilon",
        r"\varnothing": r"\emptyset",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)

    # 去掉 \left \right（视觉上常等价于普通括号）
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")

    # 再次压缩空格
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_bleu(refs, hyps):
    """
    refs, hyps: list of str (空格分词后的 LaTeX)
    返回 corpus BLEU
    """
    # 规范化后再算，减少等价写法带来的惩罚
    refs_n = [normalize_latex(r) for r in refs]
    hyps_n = [normalize_latex(h) for h in hyps]
    bleu = BLEU(tokenize="none")
    score = bleu.corpus_score(hyps_n, [refs_n])
    return score.score  # 0-100


def compute_edit_distance(refs, hyps):
    """
    平均归一化 Levenshtein 距离 (越小越好)
    """
    total = 0.0
    for r, h in zip(refs, hyps):
        r_toks = normalize_latex(r).split()
        h_toks = normalize_latex(h).split()
        if len(r_toks) == 0:
            dist = 0 if len(h_toks) == 0 else 1
        else:
            dist = editdistance.eval(r_toks, h_toks) / max(len(r_toks), 1)
        total += dist
    return total / max(len(refs), 1)


def compute_exprate(refs, hyps, use_normalize=True):
    """
    Expression Rate: 完全匹配的比例
    use_normalize=True 时先做 canonical normalization 再比较
    """
    if use_normalize:
        correct = sum(
            1 for r, h in zip(refs, hyps)
            if normalize_latex(r) == normalize_latex(h)
        )
    else:
        correct = sum(1 for r, h in zip(refs, hyps) if r.strip() == h.strip())
    return correct / max(len(refs), 1) * 100.0


def evaluate(model, dataloader, tokenizer, device, desc="Eval", beam_size=None):
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    all_refs = []
    all_hyps = []
    beam_size = beam_size if beam_size is not None else BEAM_SIZE

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=True)
        for imgs, ids, formulas in pbar:
            imgs = imgs.to(device)
            # beam search 解码
            pred_ids = real_model.generate(imgs, beam_size=beam_size)
            for i in range(imgs.size(0)):
                hyp = tokenizer.decode(pred_ids[i].cpu().tolist())
                ref = formulas[i]
                all_hyps.append(hyp)
                all_refs.append(ref)
            pbar.set_postfix(samples=len(all_refs))

    bleu = compute_bleu(all_refs, all_hyps)
    ed = compute_edit_distance(all_refs, all_hyps)
    exprate = compute_exprate(all_refs, all_hyps, use_normalize=True)
    exprate_raw = compute_exprate(all_refs, all_hyps, use_normalize=False)

    return {
        "BLEU": bleu,
        "EditDistance": ed,
        "ExpRate": exprate,          # 规范化后
        "ExpRate_raw": exprate_raw,  # 原始字符串
        "num_samples": len(all_refs)
    }
