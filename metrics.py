import torch
import torch.nn as nn
import editdistance
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from config import *
from dataset import normalize_latex, latex_tokenize


def ref_tokens(ref: str):
    return latex_tokenize(normalize_latex(ref))


def hyp_tokens(hyp: str):
    """decode 输出已是空格分词，禁止再 normalize。"""
    return [t for t in hyp.split() if t]


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
    correct = sum(1 for r, h in zip(refs, hyps) if ref_tokens(r) == hyp_tokens(h))
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
