# # import numpy as np
# # import random, re
# # from pathlib import Path

# # # ---------- quaternion → rotation matrix ----------
# # def qvec2rotmat(q):
# #     q0, q1, q2, q3 = q
# #     return np.array([
# #         [1 - 2*(q2*q2 + q3*q3),   2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
# #         [2*(q1*q2 + q0*q3),       1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1)],
# #         [2*(q1*q3 - q0*q2),       2*(q2*q3 + q0*q1),     1 - 2*(q1*q1 + q2*q2)]
# #     ], dtype=float)

# # def read_frames_txt(path):
# #     frames = {}
# #     with open(path, "r") as f:
# #         for line in f:
# #             if line.startswith("#") or not line.strip():
# #                 continue
# #             parts = line.split()
# #             if len(parts) < 10:
# #                 continue
# #             qw, qx, qy, qz = map(float, parts[2:6])
# #             tx, ty, tz = map(float, parts[6:9])
# #             frame_idx = parts[-1]
# #             frames[frame_idx] = {
# #                 "qvec": np.array([qw, qx, qy, qz]),
# #                 "tvec": np.array([tx, ty, tz])
# #             }
# #     return frames

# # # ---------- compute relative pose ----------
# # def compute_relative_pose(a, b):
# #     R_a, t_a = qvec2rotmat(a["qvec"]), a["tvec"]
# #     R_b, t_b = qvec2rotmat(b["qvec"]), b["tvec"]
# #     R = R_b @ R_a.T
# #     t = R_b @ (t_a - t_b)
# #     return R, t

# # def rad2deg(r): return r * 180 / np.pi
# # def rotation_angle(R):
# #     return rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))

# # # ---------- main ----------
# # def main(frames_path, sample=10):
# #     frames = read_frames_txt(frames_path)
# #     names = sorted(frames.keys(), key=lambda s: [int(t) if t.isdigit() else t for t in re.findall(r'\d+|\D+', str(s))])
# #     print(f"📸 共 {len(names)} 帧，随机检查 {sample} 对相邻帧:\n")

# #     for i in random.sample(range(len(names)-1), min(sample, len(names)-1)):
# #         f0, f1 = names[i], names[i+1]
# #         R, t = compute_relative_pose(frames[f0], frames[f1])
# #         dR = rotation_angle(R)
# #         print(f"{f0:>6} → {f1:<6}  ΔR={dR:6.3f}°   |t|={np.linalg.norm(t):.5f}   t={t.round(5)}")

# # if __name__ == "__main__":
# #     frames_path = "/home/cxx/HWs/CS290U/project2/colmap_workspace/results/2/frames.txt"
# #     main(frames_path, sample=10)

import numpy as np, re, json
from pathlib import Path

def qvec2rotmat(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2*q2 + q3*q3),   2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),       1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),       2*(q2*q3 + q0*q1),     1 - 2*(q1*q1 + q2*q2)]
    ], dtype=float)

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

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.findall(r'\d+|\D+', str(s))]

def compute_relative_pose(a, b):
    R_a, t_a = qvec2rotmat(a["qvec"]), a["tvec"]
    R_b, t_b = qvec2rotmat(b["qvec"]), b["tvec"]
    R = R_b @ R_a.T
    t = R_b @ (t_a - t_b)
    return R, t

def rad2deg(r): return r * 180 / np.pi
def rotation_angle(R): return rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))

def main(frames_path, image_dir, out_dir, fx=6763.18, fy=640.0, cx=360.0, cy=10.45, t_thresh=0.2, r_thresh=2.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = read_frames_txt(frames_path)
    names = sorted(frames.keys(), key=natural_key)
    print(f"📸 共 {len(names)} 帧，开始过滤异常相邻对...")

    valid_pairs = []
    for i in range(len(names)-1):
        n0, n1 = names[i], names[i+1]
        R, t = compute_relative_pose(frames[n0], frames[n1])
        dR = rotation_angle(R)
        dist = np.linalg.norm(t)

        if dist > t_thresh or dR > r_thresh:
            print(f"⚠️ 过滤异常对 {n0}->{n1}  ΔR={dR:.2f}°  |t|={dist:.3f}")
            continue
        valid_pairs.append((n0, n1, R, t))

    print(f"✅ 保留 {len(valid_pairs)} 对，过滤 {len(names)-1 - len(valid_pairs)} 对")

    # 相机内参矩阵
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    # 输出 JSON（调试）
    dataset = {"pairs": []}
    for n0, n1, R, t in valid_pairs:
        dataset["pairs"].append({
            "image0": f"{image_dir}/frame_{int(n0):02d}.png",
            "image1": f"{image_dir}/frame_{int(n1):02d}.png",
            "K0": K.tolist(),
            "K1": K.tolist(),
            "R": R.tolist(),
            "t": t.tolist()
        })
    json_path = out_dir / "test_filtered.json"
    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # 输出 SuperGlue GT 文件（38列）
    txt_path = out_dir / "test_with_gt_filtered.txt"
    with open(txt_path, "w") as f:
        for d in dataset["pairs"]:
            R = np.array(d["R"])
            t = np.array(d["t"]).reshape(3, 1)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()
            line = [d["image0"], d["image1"], "0", "0"]
            line += [f"{v:.8f}" for v in np.array(d["K0"]).flatten()]
            line += [f"{v:.8f}" for v in np.array(d["K1"]).flatten()]
            line += [f"{v:.8f}" for v in T.flatten()]
            f.write(" ".join(line) + "\n")

    print(f"📄 已输出: {txt_path}")
    print(f"🧩 调试JSON: {json_path}")

if __name__ == "__main__":
    root = "/home/cxx/HWs/CS290U/project2/"
    frames_path = root+"colmap_workspace/results/2/frames.txt"
    image_dir = root+"data/mobile_data/front_frames"
    out_dir = root+"colmap_workspace/results/2"
    main(frames_path, image_dir, out_dir)


