from pathlib import Path
from huggingface_hub import hf_hub_download
import os
import zipfile
import json

from paddleocr import PaddleOCRVL


def download_zip():
    if not os.path.exists("./imgs.zip"):
        zip_path = hf_hub_download(
            repo_id="xiaochuan-dev/latex",
            filename="imgs.zip",
            repo_type="dataset",
        )

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")

        print("解压完成！图片在 ./imgs 目录下")


def recognize_formulas():
    formula_dir = Path("./imgs/a/formulas")

    img_paths = sorted(
        [
            str(path)
            for path in formula_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        ]
    )

    if not img_paths:
        print(f"目录中没有找到图片: {formula_dir}")
        return

    print(f"找到 {len(img_paths)} 张公式图片")

    # 初始化模型
    pipeline = PaddleOCRVL()

    # 一次性传入多张图片
    output = pipeline.predict(img_paths)

    results = {}

    # PaddleOCRVL 的输出顺序与输入图片路径对应
    for img_path, res in zip(img_paths, output):
        try:
            data = res.json

            # 提取识别结果
            parts = []

            for block in data.get("parsing_res_list", []):
                content = block.get("block_content", "").strip()

                if content:
                    parts.append(content)

            value = "\n".join(parts).strip()

        except Exception as e:
            print(f"处理失败: {img_path}")
            print(e)
            value = ""

        results[img_path] = value

    # 保存 JSON
    output_path = Path("./output/formulas.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"识别完成！")
    print(f"结果保存到: {output_path}")


if __name__ == "__main__":
    download_zip()
    recognize_formulas()
