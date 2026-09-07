from pathlib import Path
from huggingface_hub import hf_hub_download
import os
import zipfile

from paddleocr import PaddleOCRVL

def download_zip():
    if not os.path.exists('./imgs.zip'):
        zip_path = hf_hub_download(
            repo_id="xiaochuan-dev/latex",
            filename="imgs.zip",
            repo_type="dataset"
        )

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
            
        print("解压完成！图片在 . 目录下")

download_zip()
output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)

pipeline = PaddleOCRVL()
output = pipeline.predict("./imgs/a/formulas/00000.png")
for res in output:
    res.print() 