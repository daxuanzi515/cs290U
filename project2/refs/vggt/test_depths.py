#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image
import cv2

# === 修改路径 ===
img_name = "H_009_00006.JPG"
img_path = f"datasets/GS/road/images/aerial_h/{img_name}"
depth_path = f"datasets/GS/road/depths/aerial_h/{os.path.splitext(img_name)[0]}.png"

# === 相机内参 ===
K = np.array([
    [2493.977086, 0.000000, 802.115139],
    [0.000000, 2493.977086, 535.496804],
    [0.000000, 0.000000, 1.000000]
], dtype=np.float64)

# === 加载 RGB 和深度 ===
rgb = np.array(Image.open(img_path).convert("RGB"))
depth_img = np.array(Image.open(depth_path), dtype=np.float32)
print(f"[RGB] {rgb.shape}, [Depth] {depth_img.shape}, dtype={depth_img.dtype}")

# === 解码 RGB-encoded 深度 (0–255, 3通道) ===
if depth_img.ndim == 3 and depth_img.shape[2] == 3:
    depth = depth_img[:, :, 0]*256*256 + depth_img[:, :, 1]*256 + depth_img[:, :, 2]
    depth = depth.astype(np.float32)
    if depth.max() > 1e4:
        depth /= 1000.0  # mm→m
else:
    depth = depth_img
print(f"[Depth Decoded] range=({depth.min():.3f}, {depth.max():.3f}), mean={depth.mean():.3f}")

# === 构建3D点 ===
H, W = depth.shape
ys, xs = np.indices((H, W), dtype=np.float32)
Z = depth
fx, fy = float(K[0, 0]), float(K[1, 1])
cx, cy = float(K[0, 2]), float(K[1, 2])
valid = np.isfinite(Z) & (Z > 0)
print(f"Valid ratio = {np.count_nonzero(valid)/valid.size:.3f}")

X = (xs - cx) / fx * Z
Y = (ys - cy) / fy * Z
pts3d = np.stack([X, Y, Z], axis=-1)[valid]
pts2d = np.stack([xs, ys], axis=-1)[valid]

# 随机采样一部分点（太多会卡）
if len(pts3d) > 5000:
    idx = np.random.choice(len(pts3d), 5000, replace=False)
    pts3d = pts3d[idx]
    pts2d = pts2d[idx]

print(f"PnP input: {len(pts3d)} points")

# === 尝试运行 PnP ===
try:
    retval, rvec, tvec = cv2.solvePnP(
        pts3d.astype(np.float64),
        pts2d.astype(np.float64),
        K.astype(np.float64),
        distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    print("✅ PnP succeeded")
    print("rvec =", rvec.flatten())
    print("tvec =", tvec.flatten())
except Exception as e:
    print("❌ PnP failed:", e)

from PIL import Image
import numpy as np

depth_path = "datasets/GS/road/depths/aerial_h/H_009_00002.png"
img = Image.open(depth_path)
arr = np.array(img)
print("dtype:", arr.dtype)
print("shape:", arr.shape)
print("min:", arr.min(), "max:", arr.max(), "mean:", arr.mean())
