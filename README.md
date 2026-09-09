# Spatially-Grounded Gaussian-Prior Attention for Formula Recognition

基于论文 **Spatially-Grounded Gaussian-Prior Attention for Handwritten Mathematical Expression Recognition**（Haj Ali & Mouchère）的实现，适配 **im2latexv2** 印刷体数据集。

## 核心技术点（论文复现）

1. **Geometry-Aware Encoder**  
   - 在 DenseNet 特征图上加入 2D 正弦位置编码  
   - Grid-Augmented Multi-head Self-Attention + 相对几何偏置（Relative Geometry Bias，论文 Eq.2-4）

2. **Latent Gaussian Prior (LGP)**  
   - 解码时用轻量 MLP 预测每个时间步的 2D 高斯中心与方差  
   - 作为 foveal anchor 注入 cross-attention energy（log-space）

3. **Gated Refinement**  
   - 区分 Entity token 与 Structural token  
   - 仅对实体符号施加空间约束与 coverage，结构 token 回退到纯 content attention

4. **Attention Refinement Module (ARM)**  
   - 来自 CoMER 的 coverage 机制，与 LGP 结合

## 目录结构

```
lgp_hmer/
├── config.py          # 所有超参数与路径（无命令行参数）
├── train.py           # 直接运行入口
├── dataset.py
├── model/
│   ├── backbone.py    # DenseNet
│   ├── encoder.py     # Geometry-Aware Encoder
│   ├── decoder.py     # LGP + Gated Decoder
│   └── model.py
├── utils/
│   └── vocab.py
├── data/im2latexv2/   # 请把数据放这里
│   ├── images/
│   └── formulas.txt   # 格式: image_name\tlatex
├── checkpoints/
└── requirements.txt
```

## 数据准备（im2latexv2）

1. 下载 im2latexv2（MathNet 发布版推荐，已归一化）：
   - 参考 https://github.com/felix-schmitt/MathNet 或 Zenodo 链接
2. 整理成：
   ```
   data/im2latexv2/
     images/          # 所有 png/jpg
     formulas.txt     # 每行:  xxx.png\t normalized_latex
   ```
3. 如果还没有 `vocab.json`，首次运行 `train.py` 会自动从 formulas.txt 构建。

## 训练

```bash
cd lgp_hmer
pip install -r requirements.txt
python train.py
```

所有超参数在 `config.py`，**无需任何命令行参数**。

主要可调常量：
- `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`
- `LAMBDA_SG = 0.0`（im2latexv2 无 symbol bbox 时保持 0；有 bbox 可设 0.1）
- `IMG_HEIGHT`, `IMG_MAX_WIDTH`
- `STRUCT_TOKENS`（结构符号集合，用于 gating）

## 推理示例

```python
import torch
from model.model import LGP_HMER
from utils.vocab import Vocab
import config as cfg

vocab = Vocab()
vocab.load(cfg.VOCAB_FILE)
model = LGP_HMER(len(vocab))
ckpt = torch.load("checkpoints/best.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

# images: (B,3,H,W) 已 normalize
pred_ids = model.greedy_decode(images, vocab)
print(vocab.decode(pred_ids[0].tolist()))
```

## 注意

- 论文原始实验在 CROHME（手写）上，本实现把同一套空间 grounding 机制迁移到印刷体 im2latexv2。
- 印刷体通常更干净，LGP 的正则效果依然有助于减少结构漂移（上下标、分数线等）。
- 若有 symbol-level 坐标，可在 `train_one_epoch` 中启用 `LAMBDA_SG > 0` 并计算 SmoothL1。
- 首次运行若无真实数据会使用假样本做结构冒烟测试。
