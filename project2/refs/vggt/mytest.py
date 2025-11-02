#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGGT 3D Reprojection Evaluation (SuperGlue-aligned) + Visualization (fixed scaling)
"""

import argparse
import numpy as np
import torch, os, csv, cv2
from tqdm import tqdm
from PIL import Image


def parse_gt_line(line):
    items = line.strip().split()
    img0, img1 = items[0], items[1]
    nums = list(map(float, items[2:]))
    K0 = np.array(nums[0:9]).reshape(3, 3)
    K1 = np.array(nums[9:18]).reshape(3, 3)
    R = np.array(nums[18:27]).reshape(3, 3)
    t = np.array(nums[27:30]).reshape(3, 1)
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
    pts_cam = (R @ pts3d.T + t).T
    pts_img = (K @ pts_cam.T).T
    return pts_img[:, :2] / pts_img[:, 2:3]


def resize_vggt_image(img, max_size=1600, patch=14):
    """resize 并返回缩放比例"""
    W, H = img.size
    scale = min(max_size / max(W, H), 1.0)
    new_W = (int(W * scale) // patch) * patch
    new_H = (int(H * scale) // patch) * patch
    img_resized = img.resize((new_W, new_H), Image.Resampling.BICUBIC)
    return img_resized, (new_W / W, new_H / H)


def draw_vggt_matches(img0_path, img1_path, proj1, proj2, max_draw=400, save_path=None):
    img0 = np.array(Image.open(img0_path).convert("RGB"))
    img1 = np.array(Image.open(img1_path).convert("RGB"))
    H0, W0, _ = img0.shape
    H1, W1, _ = img1.shape
    canvas = np.zeros((max(H0, H1), W0 + W1, 3), dtype=np.uint8)
    canvas[:H0, :W0] = img0
    canvas[:H1, W0:W0 + W1] = img1
    N = min(max_draw, len(proj1))
    if N == 0:
        return
    idxs = np.random.choice(len(proj1), N, replace=False)
    for i in idxs:
        x0, y0 = proj1[i]
        x1, y1 = proj2[i]
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(canvas, (int(x0), int(y0)), (int(x1) + W0, int(y1)), color, 1)
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved match visualization → {save_path}")

# ---------------- Visualization Add-ons ----------------

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_pointcloud_change(pts3d_pred, R, t, save_path):
    """3D点云位姿变化"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    pts_cam1 = pts3d_pred
    pts_cam2 = (R @ pts3d_pred.T + t).T

    ax.scatter(pts_cam1[:, 0], pts_cam1[:, 1], pts_cam1[:, 2],
               s=1, c='r', label='View1')
    ax.scatter(pts_cam2[:, 0], pts_cam2[:, 1], pts_cam2[:, 2],
               s=1, c='b', label='View2')

    ax.legend()
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title("3D Point Cloud before and after transformation")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved 3D transform visualization → {save_path}")


def draw_motion_field(img_path, proj1, proj2, save_path):
    """投影变化热力图"""
    img = np.array(Image.open(img_path).convert("RGB"))
    h, w, _ = img.shape
    flow = np.linalg.norm(proj1 - proj2, axis=1)
    heat = np.zeros((h, w))
    idx = np.round(proj1).astype(int)
    valid = (idx[:, 0] >= 0) & (idx[:, 0] < w) & (idx[:, 1] >= 0) & (idx[:, 1] < h)
    idx = idx[valid]
    heat[idx[:, 1], idx[:, 0]] = flow[valid]
    heat = cv2.GaussianBlur(heat, (15, 15), 0)
    if np.max(heat) > 0:
        heat = (heat / np.max(heat) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved reprojection heatmap → {save_path}")


def draw_proj_points(img0, img1, proj1, proj2, save_path):
    """输入图与投影点对比"""
    img0 = np.array(Image.open(img0).convert("RGB"))
    img1 = np.array(Image.open(img1).convert("RGB"))
    for p in proj1[::500]:
        cv2.circle(img0, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)
    for p in proj2[::500]:
        cv2.circle(img1, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
    combined = np.hstack([img0, img1])
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved projected points comparison → {save_path}")


@torch.no_grad()
def vggt_reproj_eval(model, pairs, device, output_csv, img_size=1600, num_samples=5):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pairs = pairs[:num_samples]
    print(f"[INFO] Evaluating {len(pairs)} pairs...")

    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    # 原有 SuperGlue-style 统计
    auc_levels = [1, 3, 5, 10, 25]
    auc_accum = {k: [] for k in auc_levels}
    mean_errs = []

    # === MAST3R-style 统计 ===
    auc_levels_mast = [5, 10, 20]   # AUC@5/10/20
    prec_thresh = 3                 # Prec@3px
    auc_mast_accum = {k: [] for k in auc_levels_mast}
    prec3_list = []

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img0", "img1", "mean_err",
                         "AUC@1", "AUC@3", "AUC@5", "AUC@10", "AUC@25"])

        for (img0, img1, K0, K1, R, t) in tqdm(pairs, desc="Reprojection Eval"):
            img0_pil, scale0 = resize_vggt_image(Image.open(img0).convert("RGB"), img_size)
            img1_pil, scale1 = resize_vggt_image(Image.open(img1).convert("RGB"), img_size)

            K0_scaled = K0.copy()
            K0_scaled[0, :] *= scale0[0]
            K0_scaled[1, :] *= scale0[1]
            K1_scaled = K1.copy()
            K1_scaled[0, :] *= scale1[0]
            K1_scaled[1, :] *= scale1[1]

            im0 = torch.from_numpy(np.array(img0_pil).transpose(2, 0, 1)).float() / 255.0
            im1 = torch.from_numpy(np.array(img1_pil).transpose(2, 0, 1)).float() / 255.0
            images = torch.stack([im0, im1], dim=0).unsqueeze(0).to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                agg, ps_idx = model.aggregator(images)
                point_map, _ = model.point_head(agg, images, ps_idx)

            pts3d_pred = point_map[0, 0].permute(1, 2, 0).cpu().numpy().reshape(-1, 3)
            proj1 = project(K0_scaled, np.eye(3), np.zeros((3, 1)), pts3d_pred)
            proj2 = project(K1_scaled, R, t, pts3d_pred)

            valid = np.isfinite(proj1).all(axis=1) & np.isfinite(proj2).all(axis=1)
            proj1, proj2 = proj1[valid], proj2[valid]
            err = np.linalg.norm(proj1 - proj2, axis=1)

            # 原有 SuperGlue-style 统计
            mean_err = float(np.mean(err))
            mean_errs.append(mean_err)
            for k in auc_levels:
                auc_accum[k].append(np.mean(err < k))

            # === MAST3R-style 累计 ===
            for k in auc_levels_mast:
                auc_mast_accum[k].append(float(np.mean(err < k)))
            prec3_list.append(float(np.mean(err < prec_thresh)))

            # 逐对写 CSV
            writer.writerow([os.path.basename(img0), os.path.basename(img1),
                             mean_err, *[auc_accum[k][-1] for k in auc_levels]])

            # 可视化（避免重复保存同一张 matches）
            vis_base = os.path.splitext(output_csv)[0] + f"_{os.path.basename(img0)}_to_{os.path.basename(img1)}"
            # draw_vggt_matches(img0, img1, proj1, proj2, save_path=vis_base + "_matches.png")
            # visualize_pointcloud_change(pts3d_pred, R, t, vis_base + "_3d.png")
            # draw_motion_field(img0, proj1, proj2, vis_base + "_flow.png")
            # draw_proj_points(img0, img1, proj1, proj2, vis_base + "_proj.png")

        # === 写入 MAST3R 风格的 summary.txt（百分比 + MError） ===
    auc5_avg  = np.mean(auc_mast_accum[5])  * 100.0
    auc10_avg = np.mean(auc_mast_accum[10]) * 100.0
    auc20_avg = np.mean(auc_mast_accum[20]) * 100.0
    prec3_avg = np.mean(prec3_list)         * 100.0
    merror = np.mean(mean_errs)             # 保留像素单位，不乘百分比

    summary_txt = os.path.splitext(output_csv)[0] + "_summary.txt"
    with open(summary_txt, "w") as sf:
        sf.write("AUC@5\t AUC@10\t AUC@20\t Prec\t MError\t\n")
        sf.write(f"{auc5_avg:.2f}\t {auc10_avg:.2f}\t {auc20_avg:.2f}\t {prec3_avg:.2f}\t {merror:.2f}\t\n")

    # 同步打印
    print("VGGT Match Pairs Evaluation (SuperGlue-aligned):")
    print("\nAUC@5\t AUC@10\t AUC@20\t Prec\t MError\t")
    print(f"{auc5_avg:.2f}\t {auc10_avg:.2f}\t {auc20_avg:.2f}\t {prec3_avg:.2f}\t {merror:.2f}\t")
    print(f"[INFO] Saved summary → {summary_txt}")
    print("[INFO] Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=1600)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(args.device)
    pairs = parse_gt_file(args.gt_file)
    vggt_reproj_eval(model, pairs, args.device, args.output_csv,
                     img_size=args.img_size, num_samples=args.num_samples)
