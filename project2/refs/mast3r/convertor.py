#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, numpy as np

def load_transforms(p):
    with open(p, "r") as f:
        return json.load(f)

def as_float_matrix(M, expect=(4,4)):
    M = np.array(M, dtype=np.float64)
    if M.shape == expect:
        return M
    raise ValueError(f"Bad matrix shape {M.shape}, expect {expect}")

def ensure_cam2world(M, matrix_type):
    if matrix_type == 'cam2world':
        return M
    elif matrix_type == 'world2cam':
        return np.linalg.inv(M)
    else:
        raise ValueError("matrix_type must be 'cam2world' or 'world2cam'")

def build_K(fl_x, fl_y, cx, cy):
    return np.array([[fl_x,0.,cx],[0.,fl_y,cy],[0.,0.,1.]],dtype=np.float64)

def make_pairs(frame_paths, mode='sequential', stride=1, window=5):
    pairs=[]
    n=len(frame_paths)
    if mode=='sequential':
        for i in range(n-stride): pairs.append((frame_paths[i],frame_paths[i+stride]))
    elif mode=='window':
        for i in range(n):
            for k in range(1,window+1):
                j=i+k
                if j<n: pairs.append((frame_paths[i],frame_paths[j]))
    elif mode=='all':
        for i in range(n):
            for j in range(i+1,n): pairs.append((frame_paths[i],frame_paths[j]))
    else: raise ValueError("pair_mode must be one of {'sequential','window','all'}")
    return pairs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--transforms",required=True)
    ap.add_argument("--images_root",required=True)
    ap.add_argument("--out_dir",required=True)
    ap.add_argument("--matrix_type",default="cam2world",choices=["cam2world","world2cam"])
    ap.add_argument("--pair_mode",default="sequential",choices=["sequential","window","all"])
    ap.add_argument("--stride",type=int,default=1)
    ap.add_argument("--window",type=int,default=5)
    ap.add_argument("--abs_path",action="store_true")
    args=ap.parse_args()

    os.makedirs(args.out_dir,exist_ok=True)
    data=load_transforms(args.transforms)
    frames=data.get("frames",[])
    if not frames: raise RuntimeError("No frames found in transforms.json")

    # 相机内参
    f0=frames[0]
    fl_x=f0.get("fl_x",f0.get("fl",None))
    fl_y=f0.get("fl_y",f0.get("fl",None))
    cx,cy=f0["cx"],f0["cy"]
    if fl_x is None or fl_y is None:
        raise RuntimeError("Missing fl_x/fl_y in transforms.json")
    K=build_K(fl_x,fl_y,cx,cy)
    np.savetxt(os.path.join(args.out_dir,"intrinsics.txt"),K,fmt="%.6f")

    valid_frames=[]
    missing=[]
    for f in frames:
        rel=f["file_path"]
        img_path=os.path.join(args.images_root,os.path.basename(rel)) if not os.path.isabs(rel) else rel
        if args.abs_path:
            img_path=os.path.abspath(img_path)
        if os.path.exists(img_path):
            valid_frames.append((img_path,f))
        else:
            missing.append(img_path)
    if missing:
        print(f"⚠️ Skipped {len(missing)} missing images.")
    print(f"✅ {len(valid_frames)} valid frames kept.")

    if not valid_frames:
        raise RuntimeError("No valid images found!")

    cam2world_dict={}
    img_paths=[]
    for img_path,f in valid_frames:
        bname=os.path.basename(img_path)
        img_paths.append(img_path)
        M=as_float_matrix(f["transform_matrix"],(4,4))
        M=ensure_cam2world(M,args.matrix_type)
        M[3,:]=[0,0,0,1]
        cam2world_dict[bname]=M

    # 写 gt_pose.txt
    gt_path=os.path.join(args.out_dir,"gt_pose.txt")
    with open(gt_path,"w") as g:
        for bname,M in cam2world_dict.items():
            flat=" ".join(f"{x:.6f}" for x in M.flatten())
            g.write(f"{bname} {flat}\n")

    # pairs.txt
    pairs=make_pairs(img_paths,args.pair_mode,args.stride,args.window)
    with open(os.path.join(args.out_dir,"pairs.txt"),"w") as g:
        for a,b in pairs:
            g.write(f"{a} {b}\n")
    print(f"✅ wrote pairs.txt ({len(pairs)} pairs)")
    print(f"📂 Output dir: {os.path.abspath(args.out_dir)}")

if __name__=="__main__":
    main()
