#!/usr/bin/env python3
"""
Analyze SAM Everything-mode JSON mask outputs.
Generates a summary CSV showing average metrics and success/failure classification.
"""

import os
import json
import csv
import numpy as np
import argparse

def analyze_json(json_path, iou_thresh=0.85, stability_thresh=0.9):
    """计算单张图片的mask统计信息"""
    try:
        with open(json_path, "r") as f:
            masks = json.load(f)
        if len(masks) == 0:
            return 0, 0.0, 0.0, "FAIL (empty)"
    except Exception as e:
        return 0, 0.0, 0.0, f"ERROR: {e}"

    ious = [m.get("predicted_iou", 0) for m in masks]
    stabs = [m.get("stability_score", 0) for m in masks]

    avg_iou = np.mean(ious)
    avg_stab = np.mean(stabs)
    n_masks = len(masks)

    # 判断是否为成功案例
    if avg_iou >= iou_thresh and avg_stab >= stability_thresh and n_masks > 0:
        status = "SUCCESS ✅"
    else:
        status = "FAIL ❌"

    return n_masks, avg_iou, avg_stab, status


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SAM JSON mask results and generate summary CSV."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing SAM JSON files.")
    parser.add_argument("--output-csv", type=str, default="summary.csv", help="Path to save summary CSV.")
    parser.add_argument("--iou-thresh", type=float, default=0.85, help="Average IoU threshold for success.")
    parser.add_argument("--stab-thresh", type=float, default=0.9, help="Average stability threshold for success.")
    args = parser.parse_args()

    rows = [["Image", "Num_Masks", "Avg_IoU", "Avg_Stability", "Result"]]

    json_files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
    if not json_files:
        print(f"❌ No JSON files found in {args.input_dir}")
        return

    for f in sorted(json_files):
        json_path = os.path.join(args.input_dir, f)
        name = os.path.splitext(f)[0]
        n_masks, avg_iou, avg_stab, status = analyze_json(
            json_path, args.iou_thresh, args.stab_thresh
        )
        rows.append([name, n_masks, round(avg_iou, 4), round(avg_stab, 4), status])

    # 保存到CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"✅ Summary saved to: {args.output_csv}")
    print(f"Analyzed {len(json_files)} files from: {args.input_dir}")


if __name__ == "__main__":
    main()
