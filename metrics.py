import re
import torch
import torch.nn as nn
import editdistance
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from config import *
from concurrent.futures import ThreadPoolExecutor
import copy

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

def _generate_on_device(model, imgs, beam_size):
    """模型已经在对应 device 上，直接跑"""
    with torch.no_grad():
        pred_ids = model.generate(imgs, beam_size=beam_size)
    return pred_ids.cpu()

def evaluate(model, dataloader, tokenizer, device, desc="Eval", beam_size=None):
    # 拿到原始模型（解包 DataParallel）
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    base_model.eval()

    beam_size = beam_size if beam_size is not None else BEAM_SIZE
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    use_multi_gpu = num_gpus >= 2

    # 为两张卡各准备一份模型（只在第一次或需要时做）
    if use_multi_gpu:
        # 把当前权重复制到两张卡
        model0 = copy.deepcopy(base_model).to("cuda:0").eval()
        model1 = copy.deepcopy(base_model).to("cuda:1").eval()
    else:
        model0 = base_model.to(device).eval()
        model1 = None

    all_refs = []
    all_hyps = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=True)
        for imgs, ids, formulas in pbar:
            B = imgs.size(0)

            if use_multi_gpu and B >= 2:
                mid = (B + 1) // 2          # 尽量均衡
                imgs0 = imgs[:mid].to("cuda:0", non_blocking=True)
                imgs1 = imgs[mid:].to("cuda:1", non_blocking=True)
                formulas0 = formulas[:mid]
                formulas1 = formulas[mid:]

                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut0 = executor.submit(_generate_on_device, model0, imgs0, beam_size)
                    fut1 = executor.submit(_generate_on_device, model1, imgs1, beam_size)
                    pred0 = fut0.result()
                    pred1 = fut1.result()

                pred_ids = torch.cat([pred0, pred1], dim=0)
                cur_formulas = formulas0 + formulas1
            else:
                # 单卡 fallback
                imgs = imgs.to(device, non_blocking=True)
                pred_ids = _generate_on_device(model0, imgs, beam_size)
                cur_formulas = formulas

            for i in range(pred_ids.size(0)):
                hyp = tokenizer.decode(pred_ids[i].tolist())
                ref = cur_formulas[i]
                all_hyps.append(hyp)
                all_refs.append(ref)

            pbar.set_postfix(samples=len(all_refs))

    # 清理临时模型，释放显存
    if use_multi_gpu:
        del model0, model1
        torch.cuda.empty_cache()

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