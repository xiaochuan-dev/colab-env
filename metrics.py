import torch
import torch.nn as nn
import editdistance
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from config import *


def compute_bleu(refs, hyps):
    """
    refs, hyps: list of str (空格分词后的 LaTeX)
    返回 corpus BLEU
    """
    bleu = BLEU(tokenize="none")  # 已经是 token 级别
    # sacrebleu 期望 refs 是 list of list
    score = bleu.corpus_score(hyps, [refs])
    return score.score  # 0-100


def compute_edit_distance(refs, hyps):
    """
    平均归一化 Levenshtein 距离 (越小越好)
    返回 1 - normalized_ed 作为相似度，或直接返回平均 ed
    """
    total = 0.0
    for r, h in zip(refs, hyps):
        r_toks = r.split()
        h_toks = h.split()
        if len(r_toks) == 0:
            dist = 0 if len(h_toks) == 0 else 1
        else:
            dist = editdistance.eval(r_toks, h_toks) / max(len(r_toks), 1)
        total += dist
    return total / max(len(refs), 1)


def compute_exprate(refs, hyps):
    """
    Expression Rate: 完全匹配的比例
    """
    correct = sum(1 for r, h in zip(refs, hyps) if r.strip() == h.strip())
    return correct / max(len(refs), 1) * 100.0


def evaluate(model, dataloader, tokenizer, device, desc="Eval"):
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()
    all_refs = []
    all_hyps = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=True)
        for imgs, ids, formulas in pbar:
            imgs = imgs.to(device)
            # generate
            pred_ids = real_model.generate(imgs)
            for i in range(imgs.size(0)):
                hyp = tokenizer.decode(pred_ids[i].cpu().tolist())
                ref = formulas[i]
                all_hyps.append(hyp)
                all_refs.append(ref)
            pbar.set_postfix(samples=len(all_refs))

    bleu = compute_bleu(all_refs, all_hyps)
    ed = compute_edit_distance(all_refs, all_hyps)
    exprate = compute_exprate(all_refs, all_hyps)

    return {
        "BLEU": bleu,
        "EditDistance": ed,
        "ExpRate": exprate,
        "num_samples": len(all_refs)
    }
