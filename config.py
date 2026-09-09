# ============================================================
# config.py  ——  所有超参数与路径常量（无需命令行参数）
# ============================================================

import os

# -------------------- 路径 --------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT_DIR, "data", "im2latexv2")   # 请把数据集放到这里
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
FORMULA_FILE = os.path.join(DATA_ROOT, "formulas.txt")     # 每行: image_name\tlatex
VOCAB_FILE = os.path.join(DATA_ROOT, "vocab.json")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -------------------- 数据 --------------------
# im2latexv2 建议先做 LaTeX 归一化（参考 MathNet）
MAX_SEQ_LEN = 200          # 超过截断
IMG_HEIGHT = 64            # 输入图像统一高度（保持宽高比后 pad）
IMG_MAX_WIDTH = 800
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
# 剩余作为 test

# -------------------- 模型结构 --------------------
# DenseNet backbone
DENSENET_GROWTH = 16
DENSENET_BLOCKS = [16, 16, 16]   # 每个 dense block 的层数
DENSENET_THETA = 0.5            # transition reduction

# Encoder
D_MODEL = 256
N_HEAD = 8
D_FF = 1024
N_ENC_LAYERS = 3
GEO_HEAD_DIM = 64               # relative geometry bias 的隐藏维

# Decoder
N_DEC_LAYERS = 3
LGP_HIDDEN = 128                # Latent Gaussian Prior MLP 隐藏维
ARM_KERNEL = 5
ARM_DIM = 32
DROPOUT = 0.1

# 实体 token vs 结构 token（用于 gating）
# 结构 token 不施加空间高斯约束
STRUCT_TOKENS = {
    "{", "}", "^", "_", "\\frac", "\\sqrt", "\\left", "\\right",
    "\\bigl", "\\bigr", "\\Bigl", "\\Bigr", "\\biggl", "\\biggr",
    "\\Biggl", "\\Biggr", "\\overline", "\\underline", "\\hat",
    "\\tilde", "\\bar", "\\dot", "\\ddot", "\\vec", "\\mathbf",
    "\\mathrm", "\\mathit", "\\mathcal", "\\mathbb", "\\boldsymbol",
    "\\begin", "\\end", "\\matrix", "\\pmatrix", "\\bmatrix",
    "\\array", "\\cases", "&", "\\\\", "\\limits", "\\nolimits",
    "\\displaystyle", "\\textstyle", "\\scriptstyle", "\\scriptscriptstyle",
    "\\,", "\\;", "\\!", "\\:", "\\quad", "\\qquad",
}

# -------------------- 训练 --------------------
BATCH_SIZE = 16
EVAL_EVERY = 5
MAX_DECODE_LEN = 120

NUM_EPOCHS = 100
LEARNING_RATE = 0.08
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
LAMBDA_SG = 0.0                 # im2latexv2 通常无 symbol bbox，设为 0（纯弱监督）
# 如果有 bbox 标注可改为 0.1

# 数据增强
SCALE_RANGE = (0.7, 1.4)
USE_AUG = True

# 优化器 / 调度
USE_AMP = True                  # 混合精度
GRAD_CLIP = 5.0
SAVE_EVERY = 5                  # 每多少 epoch 存一次

# 推理
BEAM_SIZE = 5

# 设备
DEVICE = "cuda"                 # 自动检测会在 train.py 里覆盖

# 特殊 token
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
