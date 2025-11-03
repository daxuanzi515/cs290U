#!/usr/bin/env python3
import os
import numpy as np
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r_visloc.localization import run_pnp
from dust3r_visloc.evaluation import get_pose_error
from dust3r_visloc.datasets.utils import get_HW_resolution

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.inference import inference, loss_of_one_batch
from dust3r.utils.geometry import geotrf, colmap_to_opencv_intrinsics, opencv_to_colmap_intrinsics
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r_visloc.datasets import *
from dust3r_visloc.localization import run_pnp
from dust3r_visloc.evaluation import get_pose_error, aggregate_stats, export_results
from dust3r_visloc.datasets.utils import get_HW_resolution, rescale_points3d


import numpy as np
import cv2

def _world2cam(cam_to_world):
    """ cam_to_world (4x4) -> world_to_cam (3x4) """
    W2C = np.linalg.inv(cam_to_world)
    return W2C[:3, :]  # 3x4

def _build_P(K, cam_to_world):
    """ 投影矩阵 P = K · [R|t] ，其中 [R|t] = world_to_cam(3x4) """
    W2C = _world2cam(cam_to_world)   # 3x4
    return K @ W2C                   # 3x4

def triangulate_points_opencv(Kq, Km, cam2world_q, cam2world_m, pts_q, pts_m):
    """
    用 OpenCV 线性三角化：
      - Kq, Km: 3x3
      - cam2world_*: 4x4
      - pts_q, pts_m: (N,2) 像素坐标（对应同一组匹配）
    返回：
      - Xw: (N,3) 世界坐标系下的三维点
      - mask_valid: (N,) 是否满足双目可见的“正深度 + 投影合理性”的简单检查
    """
    assert pts_q.shape[0] >= 2 and pts_q.shape == pts_m.shape

    # 投影矩阵
    Pq = _build_P(Kq, cam2world_q)  # 3x4
    Pm = _build_P(Km, cam2world_m)  # 3x4

    # OpenCV 需要 2xN
    xq = pts_q.T.astype(np.float32)  # 2xN
    xm = pts_m.T.astype(np.float32)  # 2xN

    X_h = cv2.triangulatePoints(Pq, Pm, xq, xm)  # 4xN
    X   = (X_h[:3] / X_h[3:]).T                  # N x 3  (世界坐标)

    # 简单有效性检查：两相机坐标系下 z 都要 > 0
    def z_in_cam(P, Xw):
        # X_cam ~ [R|t] * Xw_homog
        Xw_h = np.hstack([Xw, np.ones((Xw.shape[0], 1), dtype=Xw.dtype)])  # N x 4
        Xc   = (P @ Xw_h.T).T  # N x 3  up to scale; 但 z 符号仍可用
        return Xc[:, 2]

    z_q = z_in_cam(_build_P(np.eye(3), cam2world_q), X)  # 用未乘K的 [R|t] 更严谨，但符号判断足矣
    z_m = z_in_cam(_build_P(np.eye(3), cam2world_m), X)
    mask_valid = (z_q > 0) & (z_m > 0)

    return X, mask_valid


def debug_visualize_depth(depth, out_path):
    d = depth.copy()
    d[np.isnan(d)] = 0
    d = np.clip(d, 0, np.percentile(d, 99))
    d = (d / d.max() * 255).astype(np.uint8)
    d_color = cv2.applyColorMap(d, cv2.COLORMAP_JET)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, d_color)

def debug_visloc(dataset, model, device="cuda", max_samples=5, pnp_mode="cv2", pnp_max_points=100000):
    print(f"[DEBUG] Running dataset with max {max_samples} samples...\n")

    pred_poses = []
    gt_poses = []
    img_names = []

    for idx in tqdm(range(min(len(dataset), max_samples))):
        try:
            views = dataset[idx]
        except Exception as e:
            print(f"⚠️ Failed to load sample {idx}: {e}")
            continue

        query_view = views[0]
        map_views = views[1:]
        print(f"\n===== [DEBUG] {query_view['image_name']} =====")

        for map_view in map_views:
            name = map_view['image_name']
            print(f"\n[Matching] {query_view['image_name']} ↔ {name}")

            try:
                # === Step 1. MASt3R inference to get correspondences ===
                pair = inference(model, [query_view, map_view], device=device)
                corr_q = pair["matches0"]  # (N,2) points in query
                corr_m = pair["matches1"]  # (N,2) points in map
                conf = pair.get("confidence", None)

                if corr_q is None or len(corr_q) < 8:
                    print(f"⚠️ No valid correspondences for {name}")
                    continue

                print(f"[Match] {len(corr_q)} matches found")

                # === Step 2. Triangulate 3D points ===
                Kq = query_view["intrinsics"]
                Km = map_view["intrinsics"]
                R = map_view["R"]
                t = map_view["t"]

                pts4d_h = triangulate_points(Kq, Km, R, t, corr_q.T, corr_m.T)
                pts3d = pts4d_h[:3] / pts4d_h[3]  # [3, N]
                pts3d = pts3d.T

                if pts3d.shape[0] < 8:
                    print(f"⚠️ Too few triangulated points for {name}")
                    continue

                # === Step 3. Run PnP with 2D–3D pairs ===
                success, cam_to_world = run_pnp(
                    corr_q.astype(np.float32),
                    pts3d.astype(np.float32),
                    Kq,
                    None,
                    pnp_mode,
                    reprojectionError=5.0,
                    img_size=query_view["image"].shape[:2][::-1],
                )

                if success:
                    print(f"✅ PnP succeeded for {name} using {len(pts3d)} triangulated points")
                    pred_poses.append(cam_to_world)
                    gt_poses.append(map_view["cam_to_world"])
                    img_names.append(name)
                else:
                    print(f"⚠️ PnP failed for {name}")

            except Exception as e:
                print(f"⚠️ Error in matching/PnP for {name}: {e}")
                continue


    # ---------- 计算误差指标 ----------
    # pose_errors = []
    # angular_errors = []
    # for pred_pose, gt_pose in zip(pred_poses, gt_poses):
    #     te, ae = get_pose_error(pred_pose, gt_pose)
    #     pose_errors.append(te)
    #     angular_errors.append(ae)

    pose_errors, angular_errors = [], []

    for idx, (pred_pose, gt_pose) in enumerate(zip(pred_poses, gt_poses)):
        te, ae = get_pose_error(pred_pose, gt_pose)
        pose_errors.append(te)
        angular_errors.append(ae)

    # stats = aggregate_stats("VislocHorizonGS_debug", pose_errors, angular_errors)
    # print("\n=== Evaluation Summary ===")
    stats = aggregate_stats("VislocHorizonGS_debug", pose_errors, angular_errors)
    print("\n=== Evaluation Summary ===")
    print(stats)
# def debug_visloc(dataset, model, device="cuda", max_samples=10, pnp_mode="cv2", pnp_max_points=100000):
#     print(f"[DEBUG] Running dataset with max {max_samples} samples...\n")
#     fast_nn_params = dict(device=device, dist='dot', block_size=2**13)

#     for idx in tqdm(range(min(len(dataset), max_samples))):
#         try:
#             views = dataset[idx]
#         except Exception as e:
#             print(f"⚠️ Failed to load sample {idx}: {e}")
#             continue

#         query_view = views[0]
#         map_views = views[1:]
#         print(f"\n===== [DEBUG] {query_view['image_name']} =====")

#         query_pts2d = []
#         query_pts3d = []

#         for map_view in map_views:
#             name = map_view['image_name']
#             depth = map_view.get('depth', None)

#             if depth is not None:
#                 print(f"[Depth] {name} mean={np.nanmean(depth):.2f} range=({np.nanmin(depth):.2f},{np.nanmax(depth):.2f}) valid_ratio={(np.isfinite(depth)).mean():.3f}")
#                 debug_visualize_depth(depth, f"debug_depths/{os.path.basename(name)}.png")

#             valid = map_view.get('valid', None)
#             if valid is not None:
#                 print(f"[Valid mask] {name} valid_count={valid.sum()} / {valid.size}")

#             try:
#                 # Dummy matching (simulate a few 2D-3D points for test)
#                 if 'pts3d' in map_view and map_view['pts3d'] is not None:
#                     pts3d = map_view['pts3d'].reshape(-1, 3).cpu().numpy()
#                     h, w = depth.shape[:2]
#                     y, x = np.mgrid[0:h:10, 0:w:10]
#                     pts2d = np.stack([x.flatten(), y.flatten()], axis=-1).astype(np.float32)
#                     pts3d_sampled = pts3d[::max(1, len(pts3d)//len(pts2d))][:len(pts2d)]

#                     success, cam_to_world = run_pnp(
#                         pts2d, pts3d_sampled,
#                         map_view['intrinsics'], map_view.get('distortion', None),
#                         pnp_mode, reprojectionError=5.0, img_size=[w, h]
#                     )



#                     if success:
#                         print(f"✅ PnP succeeded for {name}")
#                     else:
#                         print(f"⚠️ PnP failed for {name}")
#                 else:
#                     print(f"⚠️ No pts3d found for {name}")

#             except Exception as e:
#                 print(f"⚠️ error during pnp for {name}: {type(e).__name__}: {e}")

#     print("\n[DEBUG] Finished 10-sample run.")


#     pose_errors = []
#     angular_errors = []
#     for qv in dataset.pairs[:10]:  # 只取前 10 样本
#         query_pose = np.eye(4)  # 假设真值
#         # 如果你保存了预测位姿，可替换这里
#         pred_pose = np.eye(4)
#         te, ae = get_pose_error(pred_pose, query_pose)
#         pose_errors.append(te)
#         angular_errors.append(ae)

#     print(aggregate_stats('VislocHorizonGS_debug', pose_errors, angular_errors))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pnp_mode", type=str, default="cv2")
    args = parser.parse_args()

    model = AsymmetricMASt3R.from_pretrained("naver/" + args.model_name).to(args.device)
    dataset = eval(args.dataset)
    dataset.set_resolution(model)

    debug_visloc(dataset, model, device=args.device, pnp_mode=args.pnp_mode)
