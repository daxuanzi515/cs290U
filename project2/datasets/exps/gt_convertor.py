#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert dataset JSON (MatrixCity / HorizonGS) into SuperGlue --eval format
Each line: imageA imageB 0 0 [K0_0...K0_8] [K1_0...K1_8] [T_0to1_0...T_0to1_15]
"""

import os
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_frames(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    frames = data["frames"]
    cameras = []
    for frame in frames:
        fx, fy, cx, cy = frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])
        T_c2w = np.array(frame["transform_matrix"])
        T_w2c = np.linalg.inv(T_c2w)
        cameras.append({
            "path": frame["file_path"],
            "K": K,
            "T": T_w2c
        })
    return cameras


def compute_angle_and_dist(T0, T1):
    """Compute rotation angle (deg) and translation distance"""
    R0, R1 = T0[:3, :3], T1[:3, :3]
    R_rel = R1 @ R0.T
    angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
    t0, t1 = T0[:3, 3], T1[:3, 3]
    dist = np.linalg.norm(t1 - t0)
    return angle, dist

def safe_join(root, relpath):
    if os.path.isabs(relpath):
        return relpath
    rel_clean = relpath
    if relpath.startswith(os.path.basename(root) + "/"):
        rel_clean = relpath.split("/", 1)[1]
    return os.path.join(root, rel_clean)


def main():
    parser = argparse.ArgumentParser(description="Convert MatrixCity/HorizonGS JSON to SuperGlue --eval input")
    parser.add_argument("--json_path", required=True, help="Path to transforms.json")
    parser.add_argument("--image_root", required=True, help="Root dir for image files")
    parser.add_argument("--output", default="gt_eval.txt", help="Output txt path")
    parser.add_argument("--skip", type=int, default=2, help="Frame interval")
    parser.add_argument("--min_angle", type=float, default=1.0, help="Min rotation diff to include pair")
    parser.add_argument("--min_dist", type=float, default=0.02, help="Min translation diff to include pair")
    args = parser.parse_args()

    cams = load_frames(args.json_path)
    n = len(cams)
    pairs = []

    for i in range(0, n - args.skip):
        c0, c1 = cams[i], cams[i + args.skip]
        angle, dist = compute_angle_and_dist(c0["T"], c1["T"])
        if angle < args.min_angle and dist < args.min_dist:
            continue
        pairs.append((c0, c1, angle, dist))

    # with open(args.output, "w") as f:
    #     for c0, c1, angle, dist in pairs:
    #         K0, K1 = c0["K"].flatten(), c1["K"].flatten()
    #         T_0to1 = np.linalg.inv(c1["T"]) @ c0["T"]
    #         T_flat = T_0to1.flatten()
    #         # f.write(f"{os.path.join(args.image_root, c0['path'])} {os.path.join(args.image_root, c1['path'])} 0 0 ")
    #         # fix
    #         f.write(f"{safe_join(args.image_root, c0['path'])} {safe_join(args.image_root, c1['path'])} 0 0 ")
    #         f.write(" ".join(f"{x:.6f}" for x in K0) + " ")
    #         f.write(" ".join(f"{x:.6f}" for x in K1) + " ")
    #         f.write(" ".join(f"{x:.6f}" for x in T_flat) + "\n")

    # print(f"✅ Generated {len(pairs)} pairs from {n} frames.")
    # print(f"   Avg rotation: {np.mean([p[2] for p in pairs]):.2f}°, avg dist: {np.mean([p[3] for p in pairs]):.3f}m")
    # print(f"→ Output saved to {args.output}")


    with open(args.output, "w") as f:
        valid_count, missing_count = 0, 0

        for c0, c1, angle, dist in pairs:
            # 拼接路径（使用 safe_join 修复重复前缀）
            path0 = safe_join(args.image_root, c0["path"])
            path1 = safe_join(args.image_root, c1["path"])

            # ✅ 检查图像是否存在
            if not (os.path.exists(path0) and os.path.exists(path1)):
                print(f"[skip] Missing image(s): {path0} or {path1}")
                missing_count += 1
                continue

            # 写入有效匹配对
            K0, K1 = c0["K"].flatten(), c1["K"].flatten()
            # T_0to1 = np.linalg.inv(c1["T"]) @ c0["T"]
            T_0to1 = c1["T"] @ np.linalg.inv(c0["T"])
            T_flat = T_0to1.flatten()

            f.write(f"{path0} {path1} 0 0 ")
            f.write(" ".join(f"{x:.6f}" for x in K0) + " ")
            f.write(" ".join(f"{x:.6f}" for x in K1) + " ")
            f.write(" ".join(f"{x:.6f}" for x in T_flat) + "\n")

            valid_count += 1

        print(f"✅ Generated {valid_count} valid pairs from {len(pairs)} total.")
        print(f"🚫 Skipped {missing_count} pairs (missing image files).")
        print(f"   Avg rotation: {np.mean([p[2] for p in pairs]):.2f}°, avg dist: {np.mean([p[3] for p in pairs]):.3f}m")
        print(f"→ Output saved to {args.output}")


if __name__ == "__main__":
    main()
