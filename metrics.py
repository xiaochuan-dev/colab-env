import re
import torch
import torch.nn as nn
import editdistance
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from config import *
from concurrent.futures import ThreadPoolExecutor

def normalize_latex(s: str) -> str:
    # ……（保持你原来的实现不变）……
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

def _generate_on_device(model, imgs, beam_size, device):
    """在指定 device 上跑 generate"""
    imgs = imgs.to(device)
    with torch.cuda.device(device):
        pred_ids = model.generate(imgs, beam_size=beam_size)
    return pred_ids.cpu()

def evaluate(model, dataloader, tokenizer, device, desc="Eval", beam_size=None):
    # 始终拿到真正的模型（解包）
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    real_model.eval()

    beam_size = beam_size if beam_size is not None else BEAM_SIZE
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    use_multi_gpu = num_gpus >= 2

    all_refs = []
    all_hyps = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=True)
        for imgs, ids, formulas in pbar:
            B = imgs.size(0)

            if use_multi_gpu and B >= 2:
                # 把 batch 均分到两张卡
                mid = B // 2
                imgs1, imgs2 = imgs[:mid], imgs[mid:]
                formulas1, formulas2 = formulas[:mid], formulas[mid:]

                # 并行在两张卡上跑
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut1 = executor.submit(
                        _generate_on_device, real_model, imgs1, beam_size, torch.device("cuda:0")
                    )
                    fut2 = executor.submit(
                        _generate_on_device, real_model, imgs2, beam_size, torch.device("cuda:1")
                    )
                    pred1 = fut1.result()
                    pred2 = fut2.result()

                pred_ids = torch.cat([pred1, pred2], dim=0)
                cur_formulas = formulas1 + formulas2
            else:
                # 单卡 fallback
                imgs = imgs.to(device)
                pred_ids = real_model.generate(imgs, beam_size=beam_size)
                pred_ids = pred_ids.cpu()
                cur_formulas = formulas

            for i in range(pred_ids.size(0)):
                hyp = tokenizer.decode(pred_ids[i].tolist())
                ref = cur_formulas[i]
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