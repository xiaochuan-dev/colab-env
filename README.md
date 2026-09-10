# Fusion of DAMamba (Dynamic Adaptive Scan) + Spatially-Grounded Gaussian-Prior for Im2LaTeX

完整可运行训练代码，融合两篇论文核心技术：

1. **DAMamba - Dynamic Adaptive Scan (DAS)**
   - Offset Prediction Network (OPN)
   - 双线性插值采样
   - 自适应重排成序列（保持线性复杂度）

2. **Spatially-Grounded Gaussian-Prior Attention**
   - Geometry-Aware Encoder（相对几何 bias）
   - Latent Gaussian Prior (LGP) 注入 cross-attention
   - 简化 entity/structural 门控思想

## 数据集
自动从 Hugging Face 下载 `yuntian-deng/im2latex-100k`（约 55k train / 6k val / 6.8k test）。

## 评估指标
- **BLEU**（corpus，token-level）
- **Edit Distance**（归一化 Levenshtein，越小越好）
- **ExpRate**（完全匹配率 %）

## 快速开始

```bash
cd im2latex_fusion
pip install -r requirements.txt

# 所有超参在 config.py，无需命令行参数
python train.py
```

## 配置
所有常量集中在 `config.py`：
- 图像尺寸、模型维度、DAS 采样点数、LGP 权重
- batch size、学习率、epoch 数等

## 模型结构概览
```
Image → CNN Backbone → DAS (自适应扫描) → Geometry-Aware Encoder (Self-Attn + SSM)
                                              ↓
                                    Transformer Decoder + LGP (Gaussian Prior)
                                              ↓
                                         LaTeX tokens
```

## 注意事项
- 本实现用纯 PyTorch 近似 Selective SSM，便于无额外 CUDA kernel 运行。
- 真实生产环境可替换为官方 `mamba-ssm` 以获得更高速度与精度。
- im2latex-100k 无符号级 bbox，因此 `USE_SPATIAL_SUPERVISION=False`（弱监督模式）。
- 训练需要 GPU，推荐至少 12GB 显存（batch=16）。可在 config.py 调小 BATCH_SIZE / D_MODEL。
- 首次运行会自动下载数据集并构建词表。

## 文件说明
- `config.py`      : 所有超参数
- `dataset.py`     : HF 数据集加载 + LaTeX 分词器
- `model.py`       : DAS + LGP + Geometry Bias + 简化 SSM 融合模型
- `metrics.py`     : BLEU / EditDistance / ExpRate
- `train.py`       : 完整训练 + 验证 + 测试循环
