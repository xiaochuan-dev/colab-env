
from pathlib import Path
from huggingface_hub import hf_hub_download
import os
import zipfile
import json

from paddleocr import PaddleOCRVL


BATCH_SIZE = 1000


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
    output_path = Path("./output/formulas.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取所有图片
    img_paths = sorted(
        [
            str(path)
            for path in formula_dir.iterdir()
            if path.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".webp",
            }
        ]
    )

    total = len(img_paths)

    if total == 0:
        print(f"目录中没有找到图片: {formula_dir}")
        return

    print(f"共找到 {total} 张公式图片")
    print(f"每批处理 {BATCH_SIZE} 张")

    # 如果之前已经有结果，则继续使用已有结果
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        print(f"已加载 {len(results)} 条已有结果")
    else:
        results = {}

    # 初始化模型，只初始化一次
    pipeline = PaddleOCRVL()

    # 按 1000 张一批
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        batch_paths = img_paths[start:end]

        # 跳过已经识别过的图片
        batch_paths = [
            path for path in batch_paths
            if path not in results
        ]

        if not batch_paths:
            print(
                f"[{start}:{end}] 已经处理过，跳过"
            )
            continue

        print(
            f"\n处理第 {start + 1} ~ {end} 张，"
            f"本批 {len(batch_paths)} 张"
        )

        try:
            # 一次传入这一批图片
            output = pipeline.predict(batch_paths)

            # 保存本批结果
            for img_path, res in zip(batch_paths, output):
                try:
                    data = res.json

                    parts = []

                    for block in data.get(
                        "parsing_res_list", []
                    ):
                        content = block.get(
                            "block_content", ""
                        ).strip()

                        if content:
                            parts.append(content)

                    value = "\n".join(parts).strip()

                    results[img_path] = value

                except Exception as e:
                    print(f"处理结果失败: {img_path}")
                    print(f"错误: {e}")

                    results[img_path] = ""

            # ★ 每批处理完成后立即保存
            with open(
                output_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    results,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(
                f"第 {start + 1} ~ {end} 张处理完成"
            )
            print(
                f"当前已保存 {len(results)}/{total} 条"
            )

        except Exception as e:
            print(
                f"\n第 {start + 1} ~ {end} 批处理失败:"
            )
            print(e)

            # 即使这一批失败，也保存之前的结果
            with open(
                output_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    results,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print("已有结果已保存")
            raise

    print("\n全部识别完成！")
    print(f"结果文件: {output_path}")


if __name__ == "__main__":
    download_zip()
    recognize_formulas()
