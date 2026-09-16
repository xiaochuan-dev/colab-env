import re
import torch
import torch.nn as nn
import editdistance
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from config import *


def normalize_latex(s: str) -> str:
    """只用于「连续 LaTeX 字符串」（GT），不要用于 decode 后的空格分词结果。"""
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\^([^{\\\s])", r"^{\1}", s)
    s = re.sub(r"_([^{\\\s])", r"_{\1}", s)
    s = re.sub(r"(\\[a-zA-Z]+)\^([^{\\\s])", r"\1^{\2}", s)
    s = re.sub(r"(\\[a-zA-Z]+)_([^{\\\s])", r"\1_{\2}", s)
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


def latex_tokenize(formula: str):
    """与 LaTeXTokenizer._tokenize 一致；先 normalize 再切。"""
    s = normalize_latex(formula)
    tokens = []
    i = 0
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


def ref_tokens(ref: str):
    return latex_tokenize(ref)


def hyp_tokens(hyp: str):
    """decode 输出已是空格分词，禁止再 normalize。"""
    return hyp.split()


def compute_bleu(refs, hyps):
    refs_t = [" ".join(ref_tokens(r)) for r in refs]
    hyps_t = [" ".join(hyp_tokens(h)) for h in hyps]
    return BLEU(tokenize="none").corpus_score(hyps_t, [refs_t]).score


def compute_edit_distance(refs, hyps):
    total = 0.0
    for r, h in zip(refs, hyps):
        r_toks = ref_tokens(r)
        h_toks = hyp_tokens(h)
        if len(r_toks) == 0:
            dist = 0 if len(h_toks) == 0 else 1
        else:
            dist = editdistance.eval(r_toks, h_toks) / max(len(r_toks), 1)
        total += dist
    return total / max(len(refs), 1)


def compute_exprate(refs, hyps, use_normalize=True):
    correct = sum(
        1 for r, h in zip(refs, hyps)
        if ref_tokens(r) == hyp_tokens(h)
    )
    return correct / max(len(refs), 1) * 100.0


def compute_exprate_raw(refs, hyps):
    correct = sum(1 for r, h in zip(refs, hyps) if r.strip() == h.strip())
    return correct / max(len(refs), 1) * 100.0


def evaluate(model, dataloader, tokenizer, device, desc="Eval", beam_size=1, debug_n=3):
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

    if debug_n > 0:
        print("\n[Debug] sample predictions:")
        for i in range(min(debug_n, len(all_refs))):
            rt = ref_tokens(all_refs[i])
            ht = hyp_tokens(all_hyps[i])
            print(f"  REF: {' '.join(rt)}")
            print(f"  HYP: {' '.join(ht)}")
            print(f"  match={rt == ht}")
            print("  ---")

    return {
        "BLEU": compute_bleu(all_refs, all_hyps),
        "EditDistance": compute_edit_distance(all_refs, all_hyps),
        "ExpRate": compute_exprate(all_refs, all_hyps, True),
        "ExpRate_raw": compute_exprate_raw(all_refs, all_hyps),
        "num_samples": len(all_refs),
    }
