# Im2LaTeX-100K + Spatially-Grounded Gaussian-Prior Attention

This project is a directly runnable PyTorch implementation of the architecture discussed in the supplied paper, adapted to the public `yuntian-deng/im2latex-100k` dataset.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Train

There are **no command-line training arguments**.

Just run:

```bash
python train.py
```

`train.py` automatically downloads/caches `yuntian-deng/im2latex-100k` with Hugging Face Datasets. The dataset is image + formula text in Parquet format and currently exposes about 67.9k examples. The repository data files are about 337 MB. 

The local Hugging Face cache is:

```text
./data/huggingface/
```

Training checkpoints are written to:

```text
./checkpoints/lgp/
├── vocab.json
├── last.pt
└── best.pt
```

Change hyperparameters directly at the top of `train.py`.

## 3. Inference

Put a formula image at `./test.png`, then edit paths at the top of `infer.py` if necessary:

```text
CHECKPOINT_PATH = Path("./checkpoints/lgp/best.pt")
VOCAB_PATH = Path("./checkpoints/lgp/vocab.json")
IMAGE_PATH = Path("./test.png")
```

Then:

```bash
python infer.py
```

## 4. Architecture

The implementation contains:

- CNN visual backbone
- geometry-aware self-attention encoder
- explicit Gaussian latent spatial prior in the first decoder cross-attention
- entity/structural token gating
- x-transformers decoder for the remaining decoder layers
- autoregressive LaTeX generation

The Gaussian cross-attention receives the actual CNN feature-grid height and width rather than guessing a square/rectangular grid from the number of visual tokens.

## 5. Dataset note

The Hugging Face processed dataset is used directly; there is no need to manually download and unpack the old `.tar.gz`/`.lst` release.

If you want the original classic file layout instead, the official Im2LaTeX project also publishes processed files and split lists, but this code intentionally uses the Hugging Face version so `python train.py` can perform the download automatically.
