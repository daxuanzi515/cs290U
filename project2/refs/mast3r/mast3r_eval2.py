#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASt3R Dense Matching Evaluator (v4)
- Confidence normalization (sigmoid)
- Camera intrinsic scaling (for 1600x1066 → resized input)
- Realistic AUC thresholds
- Outputs: per-pair CSV + summary TXT + conf visualizations
"""

import argparse
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os, csv
import matplotlib.pyplot as plt
from mast3r.model import AsymmetricMASt3R


# =============================
# --- Utility functions ---
# =============================

def parse_gt_line(line):
    """Parse one line from GT file (SuperGlue format)."""
    items = line.strip().split()
    img0, img1 = items[0], items[1]
    nums = list(map(float, items[2:]))
    K0 = np.array(nums[0:9]).reshape(3, 3)
    K1 = np.array(nums[9:18]).reshape(3, 3)
    R = np.array(nums[18:27]).reshape(3, 3)
    t = np.array(nums[27:30]).reshape(3, 1)
    return img0, img1, K0, K1, R, t


def project(K, R, t, pts3d):
    """Project 3D points into image plane."""
    pts_cam = (R @ pts3d.T + t).T
    pts_img = (K @ pts_cam.T).T
    return pts_img[:, :2] / pts_img[:, 2:3]


def compute_reprojection_metrics(pred_pts3d, K0, K1, R, t, H, W):
    """Compute reprojection error and AUC metrics in pixels."""
    pts = pred_pts3d.reshape(-1, 3)
    proj1 = project(K0, np.eye(3), np.zeros((3, 1)), pts)
    proj2 = project(K1, R, t, pts)

    valid = np.logical_and.reduce([
        proj1[:, 0] > 0, proj1[:, 0] < W,
        proj1[:, 1] > 0, proj1[:, 1] < H
    ])
    proj1, proj2 = proj1[valid], proj2[valid]
    err = np.linalg.norm(proj1 - proj2, axis=1)

    auc_thresholds = [1, 2, 5, 10]  # pixels
    auc = [np.mean(err < tau) for tau in auc_thresholds]

    return {
        "mean_err": float(err.mean()),
        "AUC@1": auc[0],
        "AUC@2": auc[1],
        "AUC@5": auc[2],
        "AUC@10": auc[3],
    }


def save_conf_map(conf_map, out_path_base):
    """Save confidence map as .png and .npy."""
    conf = conf_map.detach().cpu().numpy()
    conf = np.squeeze(conf)
    if conf.ndim == 1:
        L = int(np.sqrt(conf.shape[0]))
        conf = conf.reshape(L, L) if L * L == conf.shape[0] else conf[np.newaxis, :]
    elif conf.ndim == 3:
        conf = conf.mean(axis=0)

    plt.figure(figsize=(5, 5))
    plt.imshow(conf, cmap='plasma', aspect='auto')
    plt.colorbar(label='Confidence')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path_base + ".png", bbox_inches='tight')
    plt.close()
    np.save(out_path_base + ".npy", conf)


# =============================
# --- Main evaluation ---
# =============================
def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Loading model: {args.model}")
    model = AsymmetricMASt3R.from_pretrained(args.model).to(device).eval()

    to_tensor = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    # Load pairs
    with open(args.gt_file) as f:
        pairs = [parse_gt_line(l) for l in f if not l.strip().startswith("#")]
    total_pairs = len(pairs)
    print(f"[INFO] Found {total_pairs} pairs to evaluate.\n")

    csv_path = os.path.join(args.output_dir, "mast3r_results.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["img0", "img1", "mean_conf", "mean_err", "AUC@1", "AUC@2", "AUC@5", "AUC@10"])

    all_metrics = []

    # Process all image pairs
    for (img0, img1, K0, K1, R, t) in tqdm(pairs, desc="Evaluating"):
        try:
            im0 = to_tensor(Image.open(img0).convert("RGB")).unsqueeze(0).to(device)
            im1 = to_tensor(Image.open(img1).convert("RGB")).unsqueeze(0).to(device)
        except Exception as e:
            print(f"[WARN] Failed to load pair {img0}, {img1}: {e}")
            continue

        # === Step 1: Scale intrinsics for resized input ===
        orig_w, orig_h = 1600.0, 1066.0  # original image resolution
        scale_x = args.img_size / orig_w
        scale_y = args.img_size / orig_h

        K0_scaled = K0.copy()
        K1_scaled = K1.copy()
        K0_scaled[0, :] *= scale_x
        K0_scaled[1, :] *= scale_y
        K1_scaled[0, :] *= scale_x
        K1_scaled[1, :] *= scale_y

        # === Step 2: Prepare model input ===
        view1 = {"img": im0, "true_shape": torch.tensor([[args.img_size, args.img_size]]).to(device),
                 "instance": torch.tensor([0]).to(device)}
        view2 = {"img": im1, "true_shape": torch.tensor([[args.img_size, args.img_size]]).to(device),
                 "instance": torch.tensor([0]).to(device)}

        # === Step 3: Forward model ===
        with torch.no_grad():
            res1, res2 = model(view1, view2)

        # === Step 4: Normalize confidence ===
        conf_map = torch.sigmoid(res1["conf"])
        mean_conf = float(conf_map.mean().item())

        # === Step 5: Compute reprojection metrics ===
        pts3d_pred = res2["pts3d_in_other_view"][0].permute(1, 2, 0).cpu().numpy()
        metrics = compute_reprojection_metrics(pts3d_pred, K0_scaled, K1_scaled, R, t,
                                               args.img_size, args.img_size)
        metrics["mean_conf"] = mean_conf
        all_metrics.append(metrics)

        # === Step 6: Save conf maps ===
        conf_name = f"conf_{os.path.basename(img0)}_{os.path.basename(img1)}"
        save_conf_map(conf_map, os.path.join(args.output_dir, conf_name))

        # === Step 7: Write per-pair CSV ===
        writer.writerow([
            os.path.basename(img0), os.path.basename(img1),
            mean_conf, metrics["mean_err"],
            metrics["AUC@1"], metrics["AUC@2"], metrics["AUC@5"], metrics["AUC@10"]
        ])
        csv_file.flush()

    csv_file.close()
    print(f"[INFO] Per-pair results saved to {csv_path}")

    # --- Step 8: Summarize results ---
    if all_metrics:
        mean_conf = np.mean([m["mean_conf"] for m in all_metrics])
        mean_err = np.mean([m["mean_err"] for m in all_metrics])
        auc1 = np.mean([m["AUC@1"] for m in all_metrics])
        auc2 = np.mean([m["AUC@2"] for m in all_metrics])
        auc5 = np.mean([m["AUC@5"] for m in all_metrics])
        auc10 = np.mean([m["AUC@10"] for m in all_metrics])

        summary_text = (
            f"Evaluation Results (mean over {len(all_metrics)} pairs):\n"
            f"AUC@1\tAUC@2\tAUC@5\tAUC@10\tPrec(=AUC@5)\tMScore(=MeanConf)\n"
            f"{auc1*100:6.2f}\t{auc2*100:6.2f}\t{auc5*100:6.2f}\t{auc10*100:6.2f}\t"
            f"{auc5*100:6.2f}\t{mean_conf*100:6.2f}\n"
            f"\nMean Reprojection Error: {mean_err:.4f} px\n"
        )

        print("\n" + summary_text)

        summary_path = os.path.join(args.output_dir, "mast3r_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary_text)

        print(f"[INFO] Summary written to {summary_path}")
    else:
        print("[WARN] No valid pairs were processed.")


# =============================
# --- Entry Point ---
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MASt3R dense matching with GT camera data")
    parser.add_argument("--gt_file", type=str, required=True, help="Path to gt file (SuperGlue style)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--model", type=str,
                        default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                        help="MASt3R pretrained model name")
    parser.add_argument("--img_size", type=int, default=512, help="Input image size (must be multiple of 16)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()
    main(args)
