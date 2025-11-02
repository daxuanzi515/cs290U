import numpy as np
import argparse, json, re, random
from pathlib import Path


# ---------- 四元数转旋转矩阵 ----------
def qvec2rotmat(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2*q2 + q3*q3),   2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),       1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),       2*(q2*q3 + q0*q1),     1 - 2*(q1*q1 + q2*q2)]
    ], dtype=float)


# ---------- 读取 camera 参数 ----------
def read_cameras_txt(path):
    cams = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            cam_id = int(p[0])
            params = list(map(float, p[4:]))
            if len(params) >= 4:
                fx, fy, cx, cy = params[:4]
            elif len(params) == 3:
                fx = fy = params[0]; cx, cy = params[1:3]
            else:
                raise ValueError("无法识别的 camera 参数格式")
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=float)
            cams[cam_id] = K
    return cams


# ---------- 读取 frames.txt ----------
def read_frames_txt(path):
    frames = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            qw, qx, qy, qz = map(float, parts[2:6])
            tx, ty, tz = map(float, parts[6:9])
            frame_idx = parts[-1]
            frames[frame_idx] = {
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz])
            }
    return frames


def compute_relative_pose(a, b):
    R_a, t_a = qvec2rotmat(a["qvec"]), a["tvec"]
    R_b, t_b = qvec2rotmat(b["qvec"]), b["tvec"]

    R = R_b @ R_a.T
    t = t_b - R @ t_a

    # 归一化方向
    if np.linalg.norm(t) > 1e-8:
        t /= np.linalg.norm(t)

    # ✅ 自动方向判定：保证 z 分量朝负方向（相机前方）
    if t[2] > 0:
        t = -t

    return R, t


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.findall(r'\d+|\D+', str(s))]


# ---------- 主流程 ----------
def main(args):
    cams = read_cameras_txt(Path(args.input_path) / "cameras.txt")
    frames = read_frames_txt(Path(args.input_path) / "frames.txt")

    cam_id = list(cams.keys())[0]
    K = cams[cam_id]
    names = sorted(frames.keys(), key=natural_key)

    # ✳️ 限制前 N 帧
    if args.max_frames:
        names = names[:args.max_frames]

    # 构建配对
    pairs = [(names[i], names[i+1]) for i in range(len(names)-1)]

    # ✳️ 随机采样
    if args.sample_pairs:
        pairs = random.sample(pairs, min(args.sample_pairs, len(pairs)))

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = {"pairs": []}
    for n0, n1 in pairs:
        R, t = compute_relative_pose(frames[n0], frames[n1])
        dataset["pairs"].append({
            "image0": f"{args.image_dir}/frame_{int(n0):02d}.png",
            "image1": f"{args.image_dir}/frame_{int(n1):02d}.png",
            "K0": K.tolist(),
            "K1": K.tolist(),
            "R": R.tolist(),
            "t": t.tolist()
        })

    # ---------- 输出 ----------
    json.dump(dataset, open(args.output, "w"), indent=2)

    txt_out = out_dir / "test.txt"
    gt_out = out_dir / "test_with_gt.txt"

    with open(txt_out, "w") as f:
        for p in dataset["pairs"]:
            f.write(f"{p['image0']} {p['image1']}\n")

    with open(gt_out, "w") as f:
        for p in dataset["pairs"]:
            R = np.array(p["R"])
            t = np.array(p["t"]).reshape(3, 1)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()
            line = [p["image0"], p["image1"], "0", "0"]
            line += [f"{v:.8f}" for v in np.array(p["K0"]).flatten()]
            line += [f"{v:.8f}" for v in np.array(p["K1"]).flatten()]
            line += [f"{v:.8f}" for v in T.flatten()]
            f.write(" ".join(line) + "\n")

    print(f"✅ 成功生成 {len(pairs)} 对")
    print(f"📄 无评估: {txt_out}")
    print(f"📄 评估:   {gt_out}")
    print(f"🧩 调试JSON: {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True, help="包含 cameras.txt 和 frames.txt 的路径")
    p.add_argument("--image_dir", required=True, help="原始图像所在目录")
    p.add_argument("--output", required=True, help="输出 JSON 路径")
    p.add_argument("--max_frames", type=int, default=None, help="限制导出前 N 帧（快速 demo）")
    p.add_argument("--sample_pairs", type=int, default=None, help="随机选取 N 对相邻帧")
    args = p.parse_args()
    main(args)

# python colmap_workspace/test_quick.py \
#   --input_path colmap_workspace/results/2 \
#   --image_dir data/mobile_data/front_frames \
#   --output colmap_workspace/results/2/test_demo.json \
#   --max_frames 8
