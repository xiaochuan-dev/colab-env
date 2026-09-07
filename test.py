from pathlib import Path

from paddleocr import PaddleOCRVL

output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)

pipeline = PaddleOCRVL()
output = pipeline.predict("./imgs/a/formulas/00000.png")
for res in output:
    res.print() 