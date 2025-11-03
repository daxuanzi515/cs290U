#!/usr/bin/env python3
import argparse, json, numpy as np, os, struct
from pygltflib import GLTF2
import open3d as o3d
import matplotlib.pyplot as plt

def extract_binary_blob(gltf_path):
    """Manually extract the binary blob from a .glb file (compatible with all pygltflib versions)."""
    with open(gltf_path, "rb") as f:
        data = f.read()
    # GLB header = 12 bytes, chunk header = 8 bytes
    if data[0:4] != b'glTF':
        raise ValueError("Not a valid GLB file")
    length = struct.unpack_from("<I", data, 8)[0]
    # Skip 20 bytes header (magic + version + length + chunk header)
    json_length = struct.unpack_from("<I", data, 12)[0]
    json_start = 20
    bin_start = 20 + json_length + 8
    bin_blob = data[bin_start:length]
    return bin_blob

# def extract_vertices(gltf_path, gltf):
#     """Try to extract vertex positions from mesh primitives"""
#     try:
#         bin_blob = gltf.binary_blob()
#     except Exception:
#         print("[!] gltf.binary_blob() failed, using manual binary extraction")
#         bin_blob = extract_binary_blob(gltf_path)

#     all_points = []
#     if not gltf.meshes:
#         print("[!] No meshes in GLB file")
#         return np.zeros((0, 3))

#     for mesh in gltf.meshes:
#         for prim in mesh.primitives:
#             if not prim.attributes or "POSITION" not in prim.attributes:
#                 continue
#             acc_id = prim.attributes["POSITION"]
#             accessor = gltf.accessors[acc_id]
#             bv = gltf.bufferViews[accessor.bufferView]
#             start = (bv.byteOffset or 0) + (accessor.byteOffset or 0)
#             length = accessor.count * 12  # 3 floats * 4 bytes
#             raw = bin_blob[start:start+length]
#             try:
#                 pts = np.array(struct.unpack("<" + "f"*(length//4), raw)).reshape(-1,3)
#                 all_points.append(pts)
#             except Exception as e:
#                 print("[!] Failed to unpack vertices:", e)

#     if all_points:
#         points = np.concatenate(all_points, axis=0)
#         print(f"[✓] Extracted {len(points)} vertex points from GLB mesh data")
#         return points
#     else:
#         print("[!] No vertex POSITION attributes found in GLB")
#         return np.zeros((0, 3))

def extract_vertices(gltf_path, gltf):
    """Try to extract vertex positions from mesh primitives, compatible with all pygltflib versions."""
    try:
        bin_blob = gltf.binary_blob()
    except Exception:
        print("[!] gltf.binary_blob() failed, using manual binary extraction")
        bin_blob = extract_binary_blob(gltf_path)

    all_points = []
    if not gltf.meshes:
        print("[!] No meshes in GLB file")
        return np.zeros((0, 3))

    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            # 兼容 Attributes 对象 和 dict 两种格式
            pos_attr = None
            if hasattr(prim.attributes, "POSITION"):  # 新版
                pos_attr = prim.attributes.POSITION
            elif isinstance(prim.attributes, dict) and "POSITION" in prim.attributes:  # 旧版
                pos_attr = prim.attributes["POSITION"]

            if pos_attr is None:
                continue

            acc_id = pos_attr
            accessor = gltf.accessors[acc_id]
            bv = gltf.bufferViews[accessor.bufferView]
            start = (bv.byteOffset or 0) + (accessor.byteOffset or 0)
            length = accessor.count * 12  # 3 floats * 4 bytes
            raw = bin_blob[start:start+length]
            try:
                pts = np.array(struct.unpack("<" + "f" * (length // 4), raw)).reshape(-1, 3)
                all_points.append(pts)
            except Exception as e:
                print("[!] Failed to unpack vertices:", e)

    if all_points:
        points = np.concatenate(all_points, axis=0)
        print(f"[✓] Extracted {len(points)} vertex points from GLB mesh data")
        return points
    else:
        print("[!] No vertex POSITION attributes found in GLB")
        return np.zeros((0, 3))

def analyze(glb_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gltf = GLTF2().load(glb_path)

    # --- extract camera poses ---
    poses = []
    for node in gltf.nodes:
        if node.matrix:
            poses.append(np.array(node.matrix).reshape(4, 4))
    pose_list = [p.tolist() for p in poses]
    with open(f"{out_dir}/poses.json", "w") as f:
        json.dump(pose_list, f, indent=2)
    print(f"[✓] Saved {len(poses)} camera poses → poses.json")

    # --- try to load point cloud ---
    pcd = o3d.io.read_point_cloud(glb_path)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        print("[!] Open3D could not find a point cloud, extracting manually...")
        points = extract_vertices(glb_path, gltf)

    if len(points) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(f"{out_dir}/sparse_points.ply", pcd)

        centroid = points.mean(axis=0)
        extent = points.max(axis=0) - points.min(axis=0)
        stats = {
            "num_poses": len(poses),
            "num_points": len(points),
            "centroid": centroid.tolist(),
            "extent": extent.tolist()
        }
        with open(f"{out_dir}/stats.txt", "w") as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
        print(f"[✓] Stats saved → stats.txt")

        # 可视化
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2], s=1, c='blue')
        for p in poses:
            ax.scatter(p[0,3], p[1,3], p[2,3], c='red', marker='^')
        plt.title("MASt3R Sparse Reconstruction")
        plt.savefig(f"{out_dir}/reconstruction_view.png", dpi=300)
        print(f"[✓] 3D plot saved → reconstruction_view.png")
    else:
        print("[×] Still no vertex data found in GLB — likely camera-only reconstruction.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glb", required=True)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    analyze(args.input_glb, args.out_dir)


# python /home/cxx/HWs/CS290U/project2/refs/mast3r/analyze_glb.py --input_glb /home/cxx/HWs/CS290U/project2/refs/mast3r/outputs/1_small_city_20p/tmp3k9crmhe_scene.glb --out_dir /home/cxx/HWs/CS290U/project2/refs/mast3r/outputs/1_small_city_20p/results/