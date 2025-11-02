#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert COLMAP images.txt into demo.txt (SuperGlue evaluation format)
Output format:
image_path_1 image_path_2 0 0
"""

import os
import re
import argparse


def parse_colmap_images(images_txt):
    """Extract image filenames from COLMAP images.txt."""
    image_names = []
    with open(images_txt, 'r') as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            if re.match(r".*\.(jpg|jpeg|png|bmp|tif)$", parts[-1], re.IGNORECASE):
                image_names.append(parts[-1])
    return image_names


def generate_pairs(image_list, mode="sequential"):
    """Generate image pairs from the image list."""
    pairs = []
    n = len(image_list)
    if mode == "sequential":
        for i in range(n - 1):
            pairs.append((image_list[i], image_list[i + 1]))
    elif mode == "full":
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((image_list[i], image_list[j]))
    else:
        raise ValueError("mode must be 'sequential' or 'full'")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Convert COLMAP images.txt into demo.txt format")
    parser.add_argument("--colmap_dir", type=str, required=True,
                        help="Path to COLMAP reconstruction folder (contains images.txt)")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Root path prefix for your images (e.g. data/mobile_data/av_frames)")
    parser.add_argument("--output", type=str, default="demo.txt",
                        help="Output filename (default: demo.txt)")
    parser.add_argument("--mode", type=str, default="sequential",
                        choices=["sequential", "full"],
                        help="Pairing mode: sequential or full")
    args = parser.parse_args()

    images_txt = os.path.join(args.colmap_dir, "images.txt")
    if not os.path.exists(images_txt):
        raise FileNotFoundError(f"Cannot find images.txt in {args.colmap_dir}")

    image_names = parse_colmap_images(images_txt)
    print(f"Found {len(image_names)} images in {images_txt}")

    pairs = generate_pairs(image_names, args.mode)
    print(f"Generated {len(pairs)} pairs using mode = {args.mode}")

    with open(args.output, "w") as f:
        for a, b in pairs:
            f.write(f"{args.image_root}/{a} {args.image_root}/{b} 0 0\n")

    print(f"✅ Saved to {args.output}")


if __name__ == "__main__":
    main()


# import numpy as np

# path = "project2/colmap_workspace/results/0/demo_eval.txt"
# line = open(path).readlines()[0]
# parts = line.strip().split()
# T_flat = list(map(float, parts[-16:]))
# T = np.array(T_flat).reshape(4,4)
# print("T_0to1 =\n", T)
# print("Rotation part:\n", T[:3,:3])
# print("Translation norm:", np.linalg.norm(T[:3,3]))
