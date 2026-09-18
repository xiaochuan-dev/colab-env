# ==================== 所有训练常量配置（无命令行参数） ====================
import os

# 数据集后端:
#   "cleaned"    -> OleehyO/latex-formulas cleaned_formulas（推荐，~55万，HF 自动下载）
#   "latex_ocr"  -> lukbl/LaTeX-OCR-dataset
#   "hf_100k"    -> yuntian-deng/im2latex-100k
DATASET_BACKEND = "cleaned"

LATEX_OCR_DATASET = "lukbl/LaTeX-OCR-dataset"
DATASET_NAME = "yuntian-deng/im2latex-100k"
CLEANED_DATASET = "OleehyO/latex-formulas"
CLEANED_CONFIG = "cleaned_formulas"

MAX_FORMULA_LEN = 150
IMG_SIZE = (128, 512)
VOCAB_MIN_FREQ = 2

D_MODEL = 384
N_HEADS = 8
N_ENCODER_LAYERS = 4
N_DECODER_LAYERS = 4
D_FF = 1536
DROPOUT = 0.1
MAX_SEQ_LEN = 160

DAS_OFFSET_SCALE = 0.1
DAS_NUM_POINTS = 64
OPN_HIDDEN = 128

LGP_LAMBDA = 0.1
USE_SPATIAL_SUPERVISION = False
ENTITY_TOKENS = set()

BATCH_SIZE = 24
NUM_WORKERS = 4
LR = 2e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
WARMUP_EPOCHS = 2
GRAD_CLIP = 1.0
SEED = 42
DEVICE = "cuda"
SAVE_DIR = "./checkpoints"
LOG_INTERVAL = 50
EVAL_INTERVAL = 1

BEAM_SIZE = 5
MAX_DECODE_LEN = 120
LENGTH_PENALTY = 0.6

os.makedirs(SAVE_DIR, exist_ok=True)
TOKENIZER_PATH = os.path.join(SAVE_DIR, "tokenizer.json")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pt")
