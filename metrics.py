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
        r"\geq": r"\ge",
        r"\leq": r"\le",
        r"\neq": r"\ne",
        r"\rightarrow": r"\to",
        r"\longrightarrow": r"\to",
        r"\cdot": r"\cdot",
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


def compute_bleu(refs, hyps):
    refs_n = [normalize_latex(r) for r in refs]
    hyps_n = [normalize_latex(h) for h in hyps]
    bleu = BLEU(tokenize="none")
    score = bleu.corpus_score(hyps_n, [refs_n])
    return score.score


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
        correct = sum(
            1 for r, h in zip(refs, hyps)
            if normalize_latex(r) == normalize_latex(h)
        )
    else:
        correct = sum(1 for r, h in zip(refs, hyps) if r.strip() == h.strip())
    return correct / max(len(refs), 1) * 100.0


def evaluate(model, dataloader, tokenizer, device, desc="Eval", beam_size=None):
    """
    关键 DataParallel 包装的 model，绝不解包。
    通过 model(..., mode="generate") 让 DataParallel 自动把 batch 切到两张卡。
    """
    model.eval()
    beam_size = beam_size if beam_size is not None else BEAM_SIZE

    all_refs = []
    all_hyps = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=True)
        for imgs, ids, formulas in pbar:
            imgs = imgs.to(device, non_blocking=True)

            # 关键 DataParallel 的 __call__ → 自动 scatter 到多卡
            pred_ids = model(
                imgs,
                mode="generate",
                beam_size=beam_size
            )

            for i in range(pred_ids.size(0)):
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
        "ExpRate": exprate,
        "ExpRate_raw": exprate_raw,
        "num_samples": len(all_refs)
    }