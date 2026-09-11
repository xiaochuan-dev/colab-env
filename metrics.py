import re
import torch
import torch.nn as nn
import editdistance
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from config import *


def normalize_latex(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\^([^{\\])", r"^{\1}", s)
    s = re.sub(r"_([^{\\])", r"_{\1}", s)
    s = re.sub(r"(\\[a-zA-Z]+)\^([^{\\])", r"\1^{\2}", s)
    s = re.sub(r"(\\[a-zA-Z]+)_([^{\\])", r"\1_{\2}", s)
    replacements = {
        r"\geq": r"\ge", r"\leq": r"\le", r"\neq": r"\ne",
        r"\rightarrow": r"\to", r"\longrightarrow": r"\to",
        r"\ldots": r"\dots", r"\cdots": r"\dots",
        r"\varphi": r"\phi", r"\varepsilon": r"\epsilon",
        r"\varnothing": r"\emptyset",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    s = s.replace(r"\left", "").replace(r"\right", "")
    return re.sub(r"\s+", " ", s).strip()


def compute_bleu(refs, hyps):
    refs_n = [normalize_latex(r) for r in refs]
    hyps_n = [normalize_latex(h) for h in hyps]
    return BLEU(tokenize="none").corpus_score(hyps_n, [refs_n]).score


def compute_edit_distance(refs, hyps):
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
    if use_normalize:
        correct = sum(1 for r, h in zip(refs, hyps) if normalize_latex(r) == normalize_latex(h))
    else:
        correct = sum(1 for r, h in zip(refs, hyps) if r.strip() == h.strip())
    return correct / max(len(refs), 1) * 100.0


def evaluate(model, dataloader, tokenizer, device, desc="Eval", beam_size=1):
    """
    默认 beam_size=1（贪心+KV cache，快）。
    最终测试可传 beam_size=5。
    单卡：输入对齐 model.module 参数设备。
    """
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    param_device = next(real_model.parameters()).device

    all_refs, all_hyps = [], []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"{desc}(beam={beam_size})", leave=True)
        for imgs, ids, formulas in pbar:
            imgs = imgs.to(param_device, non_blocking=True)
            pred_ids = real_model.generate(imgs, beam_size=beam_size)
            for i in range(imgs.size(0)):
                all_hyps.append(tokenizer.decode(pred_ids[i].cpu().tolist()))
                all_refs.append(formulas[i])
            pbar.set_postfix(samples=len(all_refs))

    return {
        "BLEU": compute_bleu(all_refs, all_hyps),
        "EditDistance": compute_edit_distance(all_refs, all_hyps),
        "ExpRate": compute_exprate(all_refs, all_hyps, True),
        "ExpRate_raw": compute_exprate(all_refs, all_hyps, False),
        "num_samples": len(all_refs),
    }
