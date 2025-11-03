#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASt3R Dense Matching Evaluator (GT-aligned Reprojection Version)
- Fixed 512x512 input (for MASt3R model)
- Scale-compensated reprojection to original 1600x1066 space
- Supports --num_samples to limit evaluated pairs
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
    items = line.strip().split()
    img0, img1 = items[0], items[1]
    nums = list(map(float, items[2:]))
    K0 = np.array(nums[0:9]).reshape(3, 3)
    K1 = np.array(nums[9:18]).reshape(3, 3)
    R = np.array(nums[18:27]).reshape(3, 3)
    t = np.array(nums[27:30]).reshape(3, 1)
    return img0, img1, K0, K1, R, t


def project(K, R, t, pts3d):
    pts_cam = (R @ pts3d.T + t).T
    pts_img = (K @ pts_cam.T).T
    return pts_img[:, :2] / pts_img[:, 2:3]


def compute_reprojection_metrics(pred_pts3d, K0, K1, R, t,
                                 H_in, W_in, H_orig, W_orig):
    """Compute reprojection error (px) and AUC metrics in original image scale."""
    pts = pred_pts3d.reshape(-1, 3)

    # Project in model (512x512) space
    proj1 = project(K0, np.eye(3), np.zeros((3, 1)), pts)
    proj2 = project(K1, R, t, pts)

    # === Scale predictions back to original image space ===
    scale_x = W_orig / W_in
    scale_y = H_orig / H_in
    proj1[:, 0] *= scale_x
    proj1[:, 1] *= scale_y
    proj2[:, 0] *= scale_x
    proj2[:, 1] *= scale_y

    # === Valid mask and error ===
    valid = np.logical_and.reduce([
        proj1[:, 0] > 0, proj1[:, 0] < W_orig,
        proj1[:, 1] > 0, proj1[:, 1] < H_orig
    ])
    proj1, proj2 = proj1[valid], proj2[valid]
    err = np.linalg.norm(proj1 - proj2, axis=1)

    auc_thresholds = [1, 2, 5, 10]  # in original px
    auc = [np.mean(err < tau) for tau in auc_thresholds]

    return {
        "mean_err": float(np.mean(err)),
        "AUC@1": auc[0],
        "AUC@2": auc[1],
        "AUC@5": auc[2],
        "AUC@10": auc[3],
    }


def save_conf_map(conf_map, out_path_base):
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

    # --- Load pairs ---
    with open(args.gt_file) as f:
        pairs = [parse_gt_line(l) for l in f if not l.strip().startswith("#")]
    total_pairs = len(pairs)

    # Limit number of samples if specified
    if args.num_samples > 0:
        pairs = pairs[:min(args.num_samples, total_pairs)]
    print(f"[INFO] Found {total_pairs} total pairs, evaluating {len(pairs)} of them.\n")

    csv_path = os.path.join(args.output_dir, "mast3r_results.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["img0", "img1", "mean_conf", "mean_err(px)",
                         "AUC@1", "AUC@2", "AUC@5", "AUC@10"])

        all_metrics = []

        for (img0, img1, K0, K1, R, t) in tqdm(pairs, desc="Evaluating"):
            try:
                im0 = to_tensor(Image.open(img0).convert("RGB")).unsqueeze(0).to(device)
                im1 = to_tensor(Image.open(img1).convert("RGB")).unsqueeze(0).to(device)
            except Exception as e:
                print(f"[WARN] Failed to load pair {img0}, {img1}: {e}")
                continue

            # === Step 1: Model inference ===
            view1 = {"img": im0, "true_shape": torch.tensor([[args.img_size, args.img_size]]).to(device),
                     "instance": torch.tensor([0]).to(device)}
            view2 = {"img": im1, "true_shape": torch.tensor([[args.img_size, args.img_size]]).to(device),
                     "instance": torch.tensor([0]).to(device)}
            with torch.no_grad():
                res1, res2 = model(view1, view2)

            # === Step 2: Confidence normalization ===
            conf_map = torch.sigmoid(res1["conf"])
            mean_conf = float(conf_map.mean().item())

            # === Step 3: Reprojection metrics (scale compensated) ===
            pts3d_pred = res2["pts3d_in_other_view"][0].permute(1, 2, 0).cpu().numpy()
            metrics = compute_reprojection_metrics(
                pts3d_pred, K0, K1, R, t,
                args.img_size, args.img_size, 1066, 1600
            )
            metrics["mean_conf"] = mean_conf
            all_metrics.append(metrics)

            # === Step 4: Save confidence maps ===
            conf_name = f"conf_{os.path.basename(img0)}_{os.path.basename(img1)}"
            save_conf_map(conf_map, os.path.join(args.output_dir, conf_name))

            # === Step 5: Write to CSV ===
            writer.writerow([
                os.path.basename(img0), os.path.basename(img1),
                f"{mean_conf:.4f}", f"{metrics['mean_err']:.3f}",
                f"{metrics['AUC@1']:.3f}", f"{metrics['AUC@2']:.3f}",
                f"{metrics['AUC@5']:.3f}", f"{metrics['AUC@10']:.3f}"
            ])
            csv_file.flush()

    # === Step 6: Summary ===
    if all_metrics:
        mean_conf = np.mean([m["mean_conf"] for m in all_metrics])
        mean_err = np.mean([m["mean_err"] for m in all_metrics])
        auc1 = np.mean([m["AUC@1"] for m in all_metrics])
        auc2 = np.mean([m["AUC@2"] for m in all_metrics])
        auc5 = np.mean([m["AUC@5"] for m in all_metrics])
        auc10 = np.mean([m["AUC@10"] for m in all_metrics])

        summary_text = (
            f"\nEvaluation Results (mean over {len(all_metrics)} pairs, original pixel scale):\n"
            f"AUC@1\tAUC@2\tAUC@5\tAUC@10\tPrec(=AUC@5)\tMScore(=MeanConf)\n"
            f"{auc1*100:6.2f}\t{auc2*100:6.2f}\t{auc5*100:6.2f}\t{auc10*100:6.2f}\t"
            f"{auc5*100:6.2f}\t{mean_conf*100:6.2f}\n"
            f"\nMean Reprojection Error: {mean_err:.4f} px\n"
        )

        print(summary_text)
        with open(os.path.join(args.output_dir, "mast3r_summary.txt"), "w") as f:
            f.write(summary_text)
    else:
        print("[WARN] No valid pairs were processed.")


# =============================
# --- Entry Point ---
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MASt3R with scale-compensated reprojection")
    parser.add_argument("--gt_file", type=str, required=True, help="Path to gt file (SuperGlue format)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--model", type=str,
                        default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                        help="MASt3R pretrained model name")
    parser.add_argument("--img_size", type=int, default=512, help="Input image size (must be multiple of 16)")
    parser.add_argument("--num_samples", type=int, default=-1, help="Limit number of pairs to evaluate (-1 = all)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()
    main(args)


# python evaluate_mast3r_reproj.py \
#   --gt_file datasets/GS/road/gt/gt_aerial_h.txt \
#   --output_dir results/mast3r_aerial_h_test \
#   --num_samples 5 \
#   --device cuda