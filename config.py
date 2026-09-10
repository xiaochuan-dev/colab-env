# ==================== 所有训练常量配置（无命令行参数） ====================
import os

# 数据集
DATASET_NAME = "yuntian-deng/im2latex-100k"
MAX_FORMULA_LEN = 150          # 截断过长公式
IMG_SIZE = (128, 512)          # (H, W) 统一 resize，保持宽高比后 pad
VOCAB_MIN_FREQ = 2             # 词频过滤

# 模型结构（已扩大容量）
D_MODEL = 384
N_HEADS = 8
N_ENCODER_LAYERS = 4
N_DECODER_LAYERS = 4
D_FF = 1536
DROPOUT = 0.1
MAX_SEQ_LEN = 160              # 含 <sos> <eos>

# DAS (Dynamic Adaptive Scan) 相关
DAS_OFFSET_SCALE = 0.1         # 初始 offset 幅度
DAS_NUM_POINTS = 64            # 自适应采样点数（近似 H*W 的子集）
OPN_HIDDEN = 128

# LGP (Latent Gaussian Prior) 相关
LGP_LAMBDA = 0.1               # 空间 grounding loss 权重（弱监督时为 0）
USE_SPATIAL_SUPERVISION = False  # im2latex-100k 无 bbox，设为 False
ENTITY_TOKENS = set()          # 运行时根据 vocab 填充（数字/字母/常见符号）

# 训练
BATCH_SIZE = 24                # 模型变大后略减 batch，避免 OOM；显存够可调回 32
NUM_WORKERS = 4
LR = 2e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
WARMUP_EPOCHS = 2
GRAD_CLIP = 1.0
SEED = 42
DEVICE = "cuda"                # 自动 fallback 到 cpu
SAVE_DIR = "./checkpoints"
LOG_INTERVAL = 50
EVAL_INTERVAL = 1              # 每多少 epoch 评估一次

# 评估
BEAM_SIZE = 5                  # beam search 宽度
MAX_DECODE_LEN = 150
LENGTH_PENALTY = 0.6           # beam 长度惩罚 alpha

# 路径
os.makedirs(SAVE_DIR, exist_ok=True)
TOKENIZER_PATH = os.path.join(SAVE_DIR, "tokenizer.json")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pt")
