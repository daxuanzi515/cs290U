#!/usr/bin/env python3
import argparse
import numpy as np
import torch, os, csv, random
from tqdm import tqdm
from mast3r.model import AsymmetricMASt3R
from PIL import Image
from torchvision import transforms

# ---------- utility functions ----------

def parse_gt_line(line):
    items = line.strip().split()
    img0, img1 = items[0], items[1]
    nums = list(map(float, items[2:]))
    K0 = np.array(nums[0:9]).reshape(3,3)
    K1 = np.array(nums[9:18]).reshape(3,3)
    R  = np.array(nums[18:27]).reshape(3,3)
    t  = np.array(nums[27:30]).reshape(3,1)
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

# ---------- main evaluation ----------

@torch.no_grad()
def mast3r_reproj_eval(model, pairs, device, output_csv, img_size=512, num_samples=10):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    tform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor()
    ])

    # sample first N pairs for quick evaluation
    if len(pairs) > num_samples:
        pairs = pairs[:num_samples]
    print(f"[INFO] Evaluating {len(pairs)} pairs...")

    auc_levels = [1,3,5,10,25]
    auc_accum = {k:[] for k in auc_levels}
    mean_errs = []
    prec_list = []   # precision@3px 列表

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img0","img1","mean_conf","mean_err",
                         "AUC@1","AUC@3","AUC@5","AUC@10","AUC@25","Prec@3px"])

        for (img0, img1, K0, K1, R, t) in tqdm(pairs, desc="Reprojection Eval"):
            im0 = tform(Image.open(img0).convert("RGB")).unsqueeze(0).to(device)
            im1 = tform(Image.open(img1).convert("RGB")).unsqueeze(0).to(device)

            view1 = {"img": im0, "true_shape": torch.tensor([[img_size,img_size]]).to(device), "instance": torch.tensor([0]).to(device)}
            view2 = {"img": im1, "true_shape": torch.tensor([[img_size,img_size]]).to(device), "instance": torch.tensor([0]).to(device)}

            res1, res2 = model(view1, view2)
            pts3d_pred = res2["pts3d_in_other_view"][0].permute(1,2,0).cpu().numpy().reshape(-1,3)
            conf = res1["conf"][0].mean().item()

            proj1 = project(K0, np.eye(3), np.zeros((3,1)), pts3d_pred)
            proj2 = project(K1, R, t, pts3d_pred)

            valid = np.isfinite(proj1).all(axis=1) & np.isfinite(proj2).all(axis=1)
            proj1, proj2 = proj1[valid], proj2[valid]
            err = np.linalg.norm(proj1 - proj2, axis=1)
            mean_err = np.mean(err)
            mean_errs.append(mean_err)

            for k in auc_levels:
                auc_accum[k].append(np.mean(err < k))

            prec = np.mean(err < 3.0)
            prec_list.append(prec)

            writer.writerow([
                os.path.basename(img0), os.path.basename(img1),
                conf, mean_err,
                *[np.mean(err < k) for k in auc_levels], prec
            ])

    # summary
    mean_err_total = np.mean(mean_errs)
    auc_means = {k: np.mean(v) for k,v in auc_accum.items()}
    mean_prec = np.mean(prec_list) * 100   # Prec@3px (%)
    auc5 = auc_means[5] * 100
    auc10 = auc_means[10] * 100
    auc20 = auc_means[25] * 100  # AUC@25 近似 AUC@20

    print("\nMASt3R Match Pairs Evaluation (SuperGlue-aligned):")
    print("AUC@5\tAUC@10\tAUC@20\tPrec@3px\tMeanErr(px)")
    print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
        auc5, auc10, auc20, mean_prec, mean_err_total))

    summary_path = os.path.splitext(output_csv)[0] + "_summary.txt"
    with open(summary_path, "w") as sf:
        sf.write("Evaluation Summary ({} pairs)\n".format(len(pairs)))
        sf.write("AUC@5\tAUC@10\tAUC@20\tPrec@3px\tMeanErr(px)\n")
        sf.write("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(
            auc5, auc10, auc20, mean_prec, mean_err_total))

    print(f"[INFO] Summary saved to {summary_path}")
    print(f"[INFO] Per-pair CSV saved to {output_csv}")


# ---------- entry point ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MASt3R 3D reprojection evaluation")
    parser.add_argument("--gt_file", type=str, required=True,
                        help="Path to GT file (e.g., datasets/exps/inputs/gt_aerial_h.txt)")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output CSV path for results")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of pairs to evaluate (default: 10)")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Input image resize size (must be multiple of 16)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (cuda or cpu)")
    args = parser.parse_args()

    device = args.device
    print(f"[INFO] Using device: {device}")
    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).to(device).eval()

    pairs = parse_gt_file(args.gt_file)
    mast3r_reproj_eval(model, pairs, device, args.output_csv,
                       img_size=args.img_size, num_samples=args.num_samples)




# python refs/mast3r/mytest.py \
#   --gt_file datasets/exps/inputs/gt_aerial_h.txt \
#   --output_csv datasets/exps/matches/mast3r/mast3r_reproj10.csv \
#   --num_samples 10

# python refs/mast3r/mytest.py \
#   --gt_file datasets/exps/inputs/gt_aerial_h.txt \
#   --output_csv datasets/exps/matches/mast3r/mast3r_reproj20.csv \
#   --num_samples 20

# python refs/mast3r/mytest.py \
#   --gt_file datasets/exps/inputs/gt_aerial_h.txt \
#   --output_csv datasets/exps/matches/mast3r/mast3r_reproj100.csv \
#   --num_samples 100