
#!/usr/bin/env python3
"""
从 Hugging Face 自动下载公式识别数据集，并整理成项目需要的格式：
  data/im2latexv2/
    images/
    formulas.txt   # 每行: image_name.png\\t latex

默认使用 yuntian-deng/im2latex-100k（经典 im2latex-100k，HF 上快）。
也可切换为更大的 OleehyO/latex-formulas (cleaned_formulas, ~550K)。

用法:
  pip install datasets pillow
  python download_data.py
"""

import os
import sys
from pathlib import Path

# -------------------- 配置 --------------------
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "im2latexv2"
IMAGE_DIR = DATA_ROOT / "images"
FORMULA_FILE = DATA_ROOT / "formulas.txt"

# 可选数据集（在 HF 上）
# 1) 经典 im2latex-100k（推荐，体积小、经典基准）
HF_DATASET = "yuntian-deng/im2latex-100k"
HF_CONFIG = None          # 无 config 名
HF_SPLIT = None           # 加载全部 split 再合并

# 2) 更大清洗版（约 550K），取消下面注释即可切换：
# HF_DATASET = "OleehyO/latex-formulas"
# HF_CONFIG = "cleaned_formulas"
# HF_SPLIT = "train"

# 最多保存多少条（None=全部；调试可设 5000）
MAX_SAMPLES = None

# 国内可设镜像（可选）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def main():
    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError:
        print("请先安装: pip install datasets pillow")
        sys.exit(1)

    print("=" * 60)
    print("从 Hugging Face 下载公式数据集")
    print(f"  dataset : {HF_DATASET}")
    print(f"  config  : {HF_CONFIG}")
    print(f"  输出目录: {DATA_ROOT}")
    print("=" * 60)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # 加载
    print("[1/3] load_dataset ...")
    if HF_CONFIG:
        ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT or "train")
    else:
        # yuntian-deng/im2latex-100k 通常有 train/validation/test
        raw = load_dataset(HF_DATASET)
        if hasattr(raw, "keys"):
            from datasets import concatenate_datasets
            parts = []
            for split_name in raw.keys():
                print(f"  + split: {split_name} ({len(raw[split_name])})")
                parts.append(raw[split_name])
            ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        else:
            ds = raw

    n_total = len(ds)
    n = n_total if MAX_SAMPLES is None else min(n_total, MAX_SAMPLES)
    print(f"[2/3] 共 {n_total} 条，将写入 {n} 条")

    sample0 = ds[0]
    keys = list(sample0.keys())
    print(f"  字段: {keys}")

    def get_latex(ex):
        for k in ("formula", "text", "latex", "latex_formula", "label"):
            if k in ex and ex[k] is not None:
                return str(ex[k]).strip()
        raise KeyError(f"找不到公式字段，可用字段: {list(ex.keys())}")

    def get_image(ex):
        if "image" in ex:
            return ex["image"]
        raise KeyError("找不到 image 字段")

    def get_name(ex, idx):
        for k in ("filename", "id", "file_name", "image_path"):
            if k in ex and ex[k]:
                name = str(ex[k])
                if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                    name = name + ".png"
                return Path(name).name
        return f"{idx:07d}.png"

    lines = []
    print("[3/3] 写出图片 + formulas.txt ...")
    for i in range(n):
        ex = ds[i]
        latex = get_latex(ex)
        img = get_image(ex)
        name = get_name(ex, i)

        if hasattr(img, "convert"):
            img = img.convert("RGB")
        else:
            img = Image.open(img).convert("RGB")

        out_path = IMAGE_DIR / name
        if out_path.exists():
            name = f"{i:07d}_{name}"
            out_path = IMAGE_DIR / name
        img.save(out_path, format="PNG")

        latex = " ".join(latex.split())
        lines.append(f"{name}\t{latex}")

        if (i + 1) % 2000 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n}")

    FORMULA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FORMULA_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print()
    print("=" * 60)
    print("完成！")
    print(f"  images/      : {IMAGE_DIR}  ({n} 张)")
    print(f"  formulas.txt : {FORMULA_FILE}")
    print()
    print("接下来运行:  python train.py")
    print("=" * 60)
    print()
    print("切换更大数据集: 编辑本脚本，改用")
    print('  HF_DATASET = "OleehyO/latex-formulas"')
    print('  HF_CONFIG  = "cleaned_formulas"')
    print("国内加速可设置: export HF_ENDPOINT=https://hf-mirror.com")


if __name__ == "__main__":
    main()