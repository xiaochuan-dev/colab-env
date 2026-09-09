#!/usr/bin/env python3
"""
自动下载并整理公式识别数据集，适配本项目的 data/im2latexv2/ 格式。

说明：
  - 官方 MathNet im2latexv2 完整版约 40GB+（Zenodo Part1+Part2），体积过大，
    本脚本默认下载更常用、体积可控的 **im2latex-100k 处理后版本**（Harvard / Deng et al.）。
  - 如需完整 im2latexv2，请手动从下面链接下载后按 README 整理。

完整 im2latexv2（MathNet）:
  Part1: https://zenodo.org/records/11230382  (~40.6 GB)
  Part2: https://zenodo.org/records/11296280
  解压脚本: unpack_im2latexv2.py（随 Part1 提供）

本脚本下载的是经典 im2latex-100k 处理后数据，可直接训练。
"""

import os
import sys
import tarfile
import zipfile
import urllib.request
import shutil
from pathlib import Path

# -------------------- 配置（与 config.py 对齐） --------------------
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "im2latexv2"
IMAGE_DIR = DATA_ROOT / "images"
FORMULA_FILE = DATA_ROOT / "formulas.txt"
TMP_DIR = ROOT / "data" / "_tmp_download"

# 经典 im2latex-100k 处理后资源（公开镜像，体积相对可控）
# 来源: https://im2markup.yuntiandeng.com/data/  与 Zenodo 56198
URLS = {
    # 处理后图片（cropped / padded，便于训练）
    "images": "https://zenodo.org/records/56198/files/formula_images.tar.gz?download=1",
    # 归一化公式列表
    "formulas": "https://zenodo.org/records/56198/files/im2latex_formulas.norm.lst?download=1",
    # 划分文件（可选）
    "train": "https://zenodo.org/records/56198/files/im2latex_train.lst?download=1",
    "val": "https://zenodo.org/records/56198/files/im2latex_validate.lst?download=1",
    "test": "https://zenodo.org/records/56198/files/im2latex_test.lst?download=1",
}

# 备用：如果上面失败，可改用其他镜像（用户自行替换）
# 例如 Kaggle / HuggingFace 等需要额外鉴权的源


def download(url: str, dest: Path, desc: str = ""):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] 已存在: {dest}")
        return
    print(f"[download] {desc or url}")
    print(f"  -> {dest}")
    try:
        def _progress(block_num, block_size, total_size):
            if total_size <= 0:
                return
            downloaded = block_num * block_size
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  {pct:5.1f}%  ({mb:.1f} MB)")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
    except Exception as e:
        print(f"\n[ERROR] 下载失败: {e}")
        print("请手动下载后放到 data/_tmp_download/ 目录，或更换镜像 URL。")
        raise


def extract_tar(tar_path: Path, out_dir: Path):
    print(f"[extract] {tar_path.name} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(out_dir)


def build_formulas_txt(formulas_lst: Path, image_root: Path, out_file: Path):
    """
    把 im2latex 原始列表转成项目需要的:
      image_name.png\\t latex
    原始 norm.lst 每行一条公式；图片名通常是 数字.png 或与 lst 行号对应。
    经典结构：formula_images/ 下有 xxx.png，lst 文件给出 formula 与 image 对应关系。
    """
    print("[convert] 生成 formulas.txt ...")
    # 尝试多种常见布局
    # 布局1: 图片已解压到 image_root，文件名与某种 index 对应
    # 布局2: 有 im2latex_*.lst 记录 "formula_idx image_name render_type"

    # 先读所有公式
    formulas = []
    with open(formulas_lst, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                formulas.append(line)

    # 收集图片
    img_files = {}
    for p in image_root.rglob("*"):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            img_files[p.stem] = p
            img_files[p.name] = p

    lines = []
    # 若图片名是 0.png, 1.png ... 与公式行号对齐
    matched = 0
    for i, latex in enumerate(formulas):
        candidates = [
            str(i),
            f"{i}.png",
            f"{i:07d}",
            f"{i:07d}.png",
        ]
        found = None
        for c in candidates:
            if c in img_files:
                found = img_files[c]
                break
            # 也试纯数字 stem
            if Path(c).stem in img_files:
                found = img_files[Path(c).stem]
                break
        if found is None:
            continue
        # 复制/软链到统一 images/ 目录（用相对名）
        rel_name = found.name
        target = IMAGE_DIR / rel_name
        if not target.exists():
            try:
                os.link(found, target)  # hardlink 省空间
            except OSError:
                shutil.copy2(found, target)
        # latex 里空格分隔的 token 拼回（norm.lst 通常已空格分词）
        latex_clean = latex.replace(" ", "")
        # 更稳妥：保留空格分词形式也可，本项目 vocab 支持两种
        lines.append(f"{rel_name}\t{latex}")
        matched += 1

    if matched == 0:
        # 回退：只要有图片就按文件名顺序硬配对（仅用于冒烟）
        print("[WARN] 无法按 index 对齐，尝试按文件名排序硬配对（仅调试用）")
        imgs_sorted = sorted([p for p in IMAGE_DIR.glob("*") if p.suffix.lower() == ".png"])
        for i, img in enumerate(imgs_sorted):
            if i >= len(formulas):
                break
            lines.append(f"{img.name}\t{formulas[i]}")
            matched += 1

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[done] 写入 {out_file} ，共 {matched} 条样本")


def main():
    print("=" * 60)
    print("im2latex 数据自动下载脚本")
    print("=" * 60)
    print(f"目标目录: {DATA_ROOT}")
    print()
    print("注意: 完整 MathNet im2latexv2 约 40GB+，本脚本下载经典 im2latex-100k。")
    print("完整 v2 请手动下载:")
    print("  https://zenodo.org/records/11230382  (Part1)")
    print("  https://zenodo.org/records/11296280  (Part2)")
    print()

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 下载
    img_tar = TMP_DIR / "formula_images.tar.gz"
    formulas_lst = TMP_DIR / "im2latex_formulas.norm.lst"

    try:
        download(URLS["images"], img_tar, "formula images (tar.gz)")
        download(URLS["formulas"], formulas_lst, "normalized formulas")
    except Exception:
        print("\n若 Zenodo 下载失败，可手动把文件放到:")
        print(f"  {img_tar}")
        print(f"  {formulas_lst}")
        print("然后重新运行本脚本。")
        return

    # 2. 解压图片
    extract_root = TMP_DIR / "extracted_images"
    if not any(extract_root.rglob("*.png")):
        extract_tar(img_tar, extract_root)
    else:
        print("[skip] 图片已解压")

    # 找到真正的图片目录
    pngs = list(extract_root.rglob("*.png"))
    if not pngs:
        print("[ERROR] 解压后未找到 png，请检查 tar 内容")
        return
    # 把图片集中到 IMAGE_DIR
    print(f"[copy] 共发现 {len(pngs)} 张图片，写入 {IMAGE_DIR}")
    for p in pngs:
        target = IMAGE_DIR / p.name
        if not target.exists():
            try:
                os.link(p, target)
            except OSError:
                shutil.copy2(p, target)

    # 3. 生成 formulas.txt
    build_formulas_txt(formulas_lst, IMAGE_DIR, FORMULA_FILE)

    print()
    print("=" * 60)
    print("完成！数据已准备好:")
    print(f"  images/     -> {IMAGE_DIR}")
    print(f"  formulas.txt -> {FORMULA_FILE}")
    print()
    print("接下来直接运行:  python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
