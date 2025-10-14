#!/usr/bin/env python3
"""
可视化 SAM 自动掩膜 (AMG) 生成结果。
输入：原始图像 + COCO RLE JSON
输出：叠加mask的彩色图像
"""

import argparse
import os
import json
import cv2
import numpy as np
from pycocotools import mask as mask_utils

def visualize_masks(image_path: str, json_path: str, output_path: str):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ 无法读取图像: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 读取JSON
    with open(json_path, "r") as f:
        masks = json.load(f)

    overlay = image.copy()
    for m in masks:
        # 解析COCO RLE格式的mask
        mask = mask_utils.decode(m["segmentation"]).astype(bool)
        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        overlay[mask] = color

    # 叠加
    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"✅ 叠加图已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SAM Automatic Mask Generator (AMG) output JSON + original image."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--json", type=str, required=True, help="Path to COCO-format JSON mask file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save output overlay image.")

    args = parser.parse_args()

    visualize_masks(args.image, args.json, args.output)


if __name__ == "__main__":
    main()
