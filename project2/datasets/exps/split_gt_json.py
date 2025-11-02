#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a unified HorizonGS/MatrixCity gt.json into multiple per-view JSONs
based on 'file_path' prefix (e.g., aerial_h, aerial_q, street_cam1, etc.)
"""

import os
import json
import argparse
from collections import defaultdict

def split_gt_json(input_path, output_dir):
    with open(input_path, "r") as f:
        data = json.load(f)

    frames_by_prefix = defaultdict(list)

    for frame in data["frames"]:
        file_path = frame["file_path"]
        prefix = file_path.split("/")[0]  # e.g. aerial_h, street_cam1
        frames_by_prefix[prefix].append(frame)

    os.makedirs(output_dir, exist_ok=True)

    for prefix, frames in frames_by_prefix.items():
        subset = {
            "camera_model": data.get("camera_model", "SIMPLE_PINHOLE"),
            "orientation_override": data.get("orientation_override", "none"),
            "frames": frames
        }
        out_path = os.path.join(output_dir, f"gt_{prefix}.json")
        with open(out_path, "w") as f:
            json.dump(subset, f, indent=2)
        print(f"✅ Saved {len(frames):4d} frames → {out_path}")

    print("\nAll subsets have been written to:", os.path.abspath(output_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split multi-view GT JSON by folder prefix.")
    parser.add_argument("--input", required=True, help="Path to unified gt.json")
    parser.add_argument("--output_dir", default="gt_split", help="Output directory for subsets")
    args = parser.parse_args()
    split_gt_json(args.input, args.output_dir)
