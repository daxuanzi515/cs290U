#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced COLMAP → SuperGlue (--eval) converter

Features:
- Supports --skip to control frame interval.
- Computes relative pose T_0to1 = inv(T1) @ T0 (COLMAP world→cam).
- Filters pairs with too small rotation (<1°) or translation (<0.01 m).
- Prints statistics of accepted pairs.
Each output line:
  imageA imageB 0 0 [K0_0...K0_8] [K1_0...K1_8] [T_0to1_0...T_0to1_15]
"""

import os
import re
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_cameras(cameras_txt):
    """Parse camera intrinsics from cameras.txt"""
    cameras = {}
    with open(cameras_txt, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = map(float, parts[2:4])
            params = list(map(float, parts[4:]))
            if model in ['PINHOLE', 'SIMPLE_PINHOLE']:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model == 'SIMPLE_RADIAL':
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                raise ValueError(f"Unsupported camera model: {model}")
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])
            cameras[cam_id] = K
    return cameras


def parse_images(images_txt):
    """Parse poses from images.txt"""
    images = {}
    with open(images_txt, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if re.match(r".*\.(jpg|jpeg|png|bmp|tif)$", parts[-1], re.IGNORECASE):
                img_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                cam_id = int(parts[8])
                name = parts[9]
                rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
                trans = np.array([[tx], [ty], [tz]])
                T = np.eye(4)
                T[:3, :3] = rot
                T[:3, 3] = trans.flatten()
                images[name] = {'id': img_id, 'cam': cam_id, 'T': T}
    return images


def compute_angle_and_distance(T0, T1):
    """Compute rotation (deg) and translation distance between two camera poses"""
    R0, R1 = T0[:3, :3], T1[:3, :3]
    R_rel = R1 @ R0.T
    angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
    t0, t1 = T0[:3, 3], T1[:3, 3]
    dist = np.linalg.norm(t1 - t0)
    return angle, dist


def main():
    parser = argparse.ArgumentParser(description="Generate SuperGlue --eval input from COLMAP outputs (enhanced)")
    parser.add_argument("--colmap_dir", required=True, help="Path to COLMAP reconstruction folder")
    parser.add_argument("--image_root", required=True, help="Root directory for images")
    parser.add_argument("--output", default="demo_eval.txt", help="Output txt file path")
    parser.add_argument("--mode", default="sequential", choices=["sequential", "full"], help="Pairing mode")
    parser.add_argument("--skip", type=int, default=1, help="Frame interval (default=1, i.e., consecutive frames)")
    parser.add_argument("--min_angle", type=float, default=1.0, help="Minimum rotation difference (deg) to keep pair")
    parser.add_argument("--min_dist", type=float, default=0.01, help="Minimum translation (m) to keep pair")
    args = parser.parse_args()

    cameras = parse_cameras(os.path.join(args.colmap_dir, "cameras.txt"))
    images = parse_images(os.path.join(args.colmap_dir, "images.txt"))
    image_names = sorted(images.keys())

    pairs = []
    if args.mode == "sequential":
        for i in range(len(image_names) - args.skip):
            pairs.append((image_names[i], image_names[i + args.skip]))
    elif args.mode == "full":
        for i in range(len(image_names)):
            for j in range(i + args.skip, len(image_names)):
                pairs.append((image_names[i], image_names[j]))

    kept_pairs = []
    angles, dists = [], []

    with open(args.output, "w") as f:
        for name0, name1 in pairs:
            im0 = images[name0]
            im1 = images[name1]
            T0, T1 = im0['T'], im1['T']
            angle, dist = compute_angle_and_distance(T0, T1)
            if angle < args.min_angle and dist < args.min_dist:
                continue  # skip almost identical frames

            K0 = cameras[im0['cam']].flatten()
            K1 = cameras[im1['cam']].flatten()
            # Correct relative pose: from cam0 to cam1
            T_0to1 = np.linalg.inv(T1) @ T0
            T_flat = T_0to1.flatten()

            f.write(f"{args.image_root}/{name0} {args.image_root}/{name1} 0 0 ")
            f.write(" ".join(map(lambda x: f"{x:.6f}", K0)) + " ")
            f.write(" ".join(map(lambda x: f"{x:.6f}", K1)) + " ")
            f.write(" ".join(map(lambda x: f"{x:.6f}", T_flat)) + "\n")

            kept_pairs.append((name0, name1))
            angles.append(angle)
            dists.append(dist)

    if len(kept_pairs) == 0:
        print("⚠️ No valid pairs found (check skip/min_angle/min_dist thresholds)")
        return

    print(f"✅ Generated {len(kept_pairs)} pairs (filtered from {len(pairs)})")
    print(f"   Average rotation diff: {np.mean(angles):.2f}°  "
          f"(min {np.min(angles):.2f}, max {np.max(angles):.2f})")
    print(f"   Average translation:   {np.mean(dists):.3f} m  "
          f"(min {np.min(dists):.3f}, max {np.max(dists):.3f})")
    print(f"→ Output saved to {args.output}")


if __name__ == "__main__":
    main()
