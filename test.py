
from pathlib import Path
from huggingface_hub import hf_hub_download
import os
import zipfile
import json

from paddleocr import PaddleOCRVL


BATCH_SIZE = 100


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

    d = 'a'

    formula_dir = Path(f"./imgs/{d}/formulas")
    output_path = Path(f"./output/formulas_{d}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取所有图片
    img_paths = sorted(
        str(path)
        for path in formula_dir.iterdir()
        if path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".webp",
        }
    )

    total = len(img_paths)

    if total == 0:
        print(f"目录中没有找到图片: {formula_dir}")
        return

    print(f"共找到 {total} 张公式图片")
    print(f"每批处理 {BATCH_SIZE} 张")

    # 读取已有结果，实现断点续跑
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        print(f"已有 {len(results)} 条结果")
    else:
        results = {}

    # 模型只初始化一次
    pipeline = PaddleOCRVL()

    # 每 1000 张一批
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        batch_paths = img_paths[start:end]

        # 跳过已经处理过的图片
        batch_paths = [
            path for path in batch_paths
            if path not in results
        ]

        if not batch_paths:
            print(f"[{start + 1}-{end}] 已处理，跳过")
            continue

        print(
            f"\n========== "
            f"处理 {start + 1}-{end} / {total} "
            f"=========="
        )

        try:
            # 一次性传入这一批图片
            output = pipeline.predict(batch_paths)

            # 处理这一批的结果
            for img_path, res in zip(batch_paths, output):

                try:
                    # 你的 PaddleOCRVL 返回结构：
                    #
                    # {
                    #     "res": {
                    #         "parsing_res_list": [...]
                    #     }
                    # }
                    data = res.json["res"]

                    parts = []

                    for block in data.get("parsing_res_list", []):
                        content = block.get(
                            "block_content",
                            ""
                        ).strip()

                        if content:
                            parts.append(content)

                    value = "\n".join(parts)

                    results[img_path] = value

                except Exception as e:
                    print(f"解析结果失败: {img_path}")
                    print(f"错误: {e}")

                    # 失败也记录，避免下次无限重复
                    results[img_path] = ""

            # ==================================================
            # ★ 每一批处理完成后立即保存 JSON
            # ==================================================
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
                f"本批完成：{start + 1}-{end}"
            )
            print(
                f"当前进度：{len(results)} / {total}"
            )
            print(
                f"JSON 已保存：{output_path}"
            )

        except Exception as e:
            print(
                f"\n第 {start + 1}-{end} 批识别失败"
            )
            print(f"错误: {e}")

            # 即使当前批失败，也保存之前已经完成的结果
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

            print("之前已经完成的结果已保存")
            raise

    print("\n================================")
    print("全部识别完成！")
    print(f"总图片数：{total}")
    print(f"结果数量：{len(results)}")
    print(f"结果文件：{output_path}")
    print("================================")


if __name__ == "__main__":
    download_zip()
    recognize_formulas()
