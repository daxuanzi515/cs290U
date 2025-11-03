#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperGlue Matching Evaluation (GT-based Reprojection Error)
适用于 SuperGlue 在原图分辨率 (--resize -1) 下运行的匹配结果。
计算匹配点的重投影误差、AUC@1/3/5/10/20、平均误差。
"""

import os
import argparse
import numpy as np
import csv
from tqdm import tqdm


def parse_gt_line(line):
    items = line.strip().split()
    img0, img1 = items[0], items[1]

    # 跳过前两个 0
    nums = list(map(float, items[4:]))

    K0 = np.array(nums[0:9]).reshape(3, 3)
    K1 = np.array(nums[9:18]).reshape(3, 3)

    # 注意：这里 R + t 共 12 个
    Rt = np.array(nums[18:30]).reshape(3, 4)
    R = Rt[:, :3]
    t = Rt[:, 3:4]

    return img0, img1, K0, K1, R, t



def parse_gt_file(gt_file):
    pairs = []
    with open(gt_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            pairs.append(parse_gt_line(line))
    return pairs


def project(K, R, t, pts3d):
    """将3D点投影到图像平面"""
    pts_cam = (R @ pts3d.T + t).T
    pts_img = (K @ pts_cam.T).T
    return pts_img[:, :2] / pts_img[:, 2:3]


def backproject(uv, depth, K):
    """2D坐标反投影为3D点（假设固定深度）"""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fx = fx if fx != 0 else 1.0
    fy = fy if fy != 0 else 1.0
    X = (uv[:, 0] - cx) / fx * depth
    Y = (uv[:, 1] - cy) / fy * depth
    Z = depth
    return np.stack([X, Y, Z], axis=1)


def reprojection_error(K0, K1, R, t, pts1, pts2, depth=10.0):
    """计算重投影误差"""
    pts3d = backproject(pts1, np.ones(len(pts1)) * depth, K0)
    proj = project(K1, R, t, pts3d)
    err = np.linalg.norm(proj - pts2, axis=1)
    return err


def load_superglue_match(npz_path):
    """加载SuperGlue匹配结果"""
    data = np.load(npz_path)
    kpts0 = data["keypoints0"]
    kpts1 = data["keypoints1"]
    matches = data["matches"]
    conf = data["match_confidence"]
    valid = matches > -1
    pts1 = kpts0[valid]
    pts2 = kpts1[matches[valid]]
    return pts1, pts2, conf[valid]


# ---------- 主函数 ----------
def main(args):
    pairs = parse_gt_file(args.gt_file)
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    auc_levels = [1, 3, 5, 10, 20]

    csv_path = os.path.join(args.output_dir, "superglue_eval.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["img0", "img1", "num_matches", "mean_err", "AUC@1", "AUC@3", "AUC@5", "AUC@10", "AUC@20"]
    )

    for (img0, img1, K0, K1, R, t) in tqdm(pairs, desc="Evaluating SuperGlue"):
        stem = os.path.splitext(os.path.basename(img0))[0] + "_" + os.path.splitext(
            os.path.basename(img1)
        )[0]
        npz_path = os.path.join(args.match_dir, f"{stem}_matches.npz")
        if not os.path.exists(npz_path):
            continue

        pts1, pts2, conf = load_superglue_match(npz_path)
        if len(pts1) < 5:
            continue

        # ---- Auto-fix K (防止全零矩阵) ----
        if np.isclose(K0[0, 0], 0) or np.isclose(K1[0, 0], 0):
            print(f"[WARN] Invalid K detected for {stem}, using identity fallback.")
            K0 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=float)
            K1 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=float)

        # ---- 不缩放：SuperGlue 运行在原分辨率 (--resize -1) ----
        K0 = K0.copy()
        K1 = K1.copy()

        # ---- 计算重投影误差 ----
        err = reprojection_error(K0, K1, R, t, pts1, pts2, depth=args.depth)
        err = err[np.isfinite(err)]
        if len(err) == 0:
            continue

        mean_err = np.mean(err)
        aucs = [np.mean(err < thr) * 100.0 for thr in auc_levels]

        results.append({"mean_err": mean_err, "aucs": aucs})
        writer.writerow([os.path.basename(img0), os.path.basename(img1), len(pts1), mean_err, *aucs])
        csv_file.flush()

    csv_file.close()

    if not results:
        print("[WARN] No valid pairs found.")
        return

    mean_err = np.mean([r["mean_err"] for r in results])
    aucs = np.mean([r["aucs"] for r in results], axis=0)
    summary = (
        f"SuperGlue Evaluation ({len(results)} pairs)\n"
        f"AUC@1\tAUC@3\tAUC@5\tAUC@10\tAUC@20\tMError(px)\n"
        f"{aucs[0]:.2f}\t{aucs[1]:.2f}\t{aucs[2]:.2f}\t{aucs[3]:.2f}\t{aucs[4]:.2f}\t{mean_err:.2f}\n"
    )

    print("\n" + summary)
    with open(os.path.join(args.output_dir, "superglue_summary.txt"), "w") as f:
        f.write(summary)
    print(f"[INFO] Saved summary → {args.output_dir}/superglue_summary.txt")


# ---------- 入口 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SuperGlue matches using reprojection error")
    parser.add_argument("--gt_file", type=str, required=True)
    parser.add_argument("--match_dir", type=str, required=True, help="Path to *_matches.npz files")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--depth", type=float, default=10.0, help="Assumed scene depth for backprojection (m)")
    args = parser.parse_args()
    main(args)

# python refs/SuperGluePretrainedNetwork/eval.py \
#   --gt_file datasets/exps/inputs/gt_aerial_h.txt \
#   --match_dir datasets/exps/matches/superglue/aerial_h \
#   --output_dir datasets/exps/matches/superglue/new
