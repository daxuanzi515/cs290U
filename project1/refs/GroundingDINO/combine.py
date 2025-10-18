#!/usr/bin/env python3
# coding: utf-8
import argparse
import os
import torch
import cv2
import numpy as np
import json
from groundingdino.util.inference import load_model, load_image, predict
from pycocotools import mask as mask_utils
import sys

# ✅ 引入本地 SAM 源码
sys.path.append("/home/cxx/HWs/CS290U/project1/refs/segment-anything")
from segment_anything import sam_model_registry, SamPredictor


def run_dino_sam(args):
    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1️⃣ Load GroundingDINO
    print(f"[INFO] Loading GroundingDINO model...")
    model = load_model(args.dino_config, args.dino_weights)
    image_source, image = load_image(args.image)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=args.text,
        box_threshold=args.box_thresh,
        text_threshold=args.text_thresh,
    )

    if len(boxes) == 0:
        print(f"[WARN] ❌ No boxes detected for {args.image}\n")
        return

    print(f"[INFO] ✅ DINO detected {len(boxes)} objects.")
    print(f"[INFO] Boxes & Phrases:")
    for i, (box, phrase) in enumerate(zip(boxes, phrases)):
        x1, y1, x2, y2 = box.tolist()
        print(f"  {i+1}. '{phrase}' box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

    # 👉 绘制 DINO 检测框用于调试
    debug_img = image_source.copy()
    for box, phrase in zip(boxes, phrases):
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug_img, phrase, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    base = os.path.splitext(os.path.basename(args.image))[0]
    debug_path = os.path.join(args.output, f"{base}_dino_box.jpg")
    cv2.imwrite(debug_path, debug_img)
    print(f"[DEBUG] DINO bounding boxes saved to {debug_path}")

    # 2️⃣ Load SAM
    print(f"[INFO] Loading SAM model...")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_weights)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))

    overlay = image_source.copy()
    results = []

    # 3️⃣ SAM segmentation per detected box
    for box, phrase in zip(boxes, phrases):
        # box = box.cpu().numpy()
        # multimask_flag = bool(args.multimask)

        # masks, scores, _ = predictor.predict(
        #     box=box,
        #     multimask_output=multimask_flag
        # )
        box = box.cpu().numpy()
        h, w, _ = image_source.shape  # 原图尺寸

        # GroundingDINO 输出中心坐标+宽高(0~1范围)
        x_center, y_center, bw, bh = box
        x_min = (x_center - bw / 2) * w
        y_min = (y_center - bh / 2) * h
        x_max = (x_center + bw / 2) * w
        y_max = (y_center + bh / 2) * h

        # SAM 需要 [x_min, y_min, x_max, y_max]
        input_box = np.array([x_min, y_min, x_max, y_max])

        multimask_flag = bool(args.multimask)
        masks, scores, _ = predictor.predict(
            box=input_box,
            multimask_output=multimask_flag
        )


        print(f"[INFO] '{phrase}': 生成 {len(masks)} 个 mask (multimask={multimask_flag})")

        img_area = image_source.shape[0] * image_source.shape[1]

        if multimask_flag:
            for idx, (mask, score) in enumerate(zip(masks, scores)):
                mask_area = np.sum(mask)
                area_ratio = 100 * mask_area / img_area
                print(f"    mask#{idx+1}: score={score:.3f}, area={area_ratio:.2f}% of image")

                color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                overlay[mask] = color

                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")

                results.append({
                    "phrase": phrase,
                    "mask_index": idx + 1,
                    "score": float(score),
                    "area_ratio": float(area_ratio),
                    "segmentation": rle
                })
        else:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            mask_area = np.sum(mask)
            area_ratio = 100 * mask_area / img_area
            print(f"    best mask: score={scores[best_idx]:.3f}, area={area_ratio:.2f}% of image")

            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            overlay[mask] = color

            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")

            results.append({
                "phrase": phrase,
                "score": float(scores[best_idx]),
                "area_ratio": float(area_ratio),
                "segmentation": rle
            })

    # 4️⃣ Save results
    json_path = os.path.join(args.output, f"{base}.json")
    overlay_path = os.path.join(args.output, f"{base}_overlay.jpg")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    cv2.imwrite(overlay_path, overlay)

    print(f"✅ {base}: saved {json_path}, {overlay_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GroundingDINO + SAM 联动")
    parser.add_argument("--dino-config", type=str, required=True)
    parser.add_argument("--dino-weights", type=str, required=True)
    parser.add_argument("--sam-weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--box-thresh", type=float, default=0.35)
    parser.add_argument("--text-thresh", type=float, default=0.25)
    parser.add_argument("--multimask", type=int, default=0, help="是否输出多个mask（1=True, 0=False）")
    args = parser.parse_args()
    run_dino_sam(args)
