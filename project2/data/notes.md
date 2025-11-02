为了测试相机的问题，我拿了一个10秒的黄浦江周边景物的视频做测试。
默认电脑是安装了`ffmpeg`

1. 等间隔从视频里截取36张图
```shell
mkdir -p data/mobile_data/av_frames
ffmpeg -i data/mobile_data/ShanghaiCenter.mp4 -vf fps=3.6 data/mobile_data/av_frames/frame_%02d.png
```
---
100张图以上：图太少了，pairs没有足够的图，匹配不上
- 帧率=30
ffmpeg -i data/mobile_data/ShanghaiCenter.mp4 -vf fps=30 data/mobile_data/av_frames/frame_%03d.png


2. 等分抽样
根据视频长度计算间隔然后抽样
```shell
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 data/mobile_data/ShanghaiCenter.mp4
# 306
expr 306 / 36 # 8

# usage
mkdir -p data/mobile_data/av_frames_u
ffmpeg -i data/mobile_data/ShanghaiCenter.mp4 -vf "select=not(mod(n\,8))" -vsync vfr data/mobile_data/av_frames_u/frame_%02d.png
```
3. 取前36帧的图
```shell
mkdir -p data/mobile_data/front_frames
ffmpeg -i data/mobile_data/ShanghaiCenter.mp4 -vframes 36 data/mobile_data/front_frames/frame_%02d.png
```
----
取前300帧
ffmpeg -i data/mobile_data/ShanghaiCenter.mp4 -vframes 300 data/mobile_data/front_frames/frame_%02d.png




4. 全自动化脚本计算
```shell
python /home/cxx/HWs/CS290U/project2/data/mobile_data/extract_frames.py \
  --video data/mobile_data/ShanghaiCenter.mp4 \
  --out data/mobile_data/av_frames_u \
  --num 36 
```
有误差 取前36张就行。

# Colmap Part

```shell
mkdir colmap_workspace
colmap database_creator --database_path colmap_workspace/mobiledata.db
```
### Feature Extraction
```shell
# customized folder
cd ~/HWs/CS290U/project2/
colmap feature_extractor \
    --database_path colmap_workspace/mobiledata.db \
    --image_path data/mobile_data/av_frames \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1
```
使用了GPU加速
### Feature Matching
Sequential Match Mode:
```shell
colmap sequential_matcher --database_path colmap_workspace/mobiledata.db # --SiftMatching.guided_matching 1
```
> 这个模式开了会用上一轮的位姿来指导下一轮匹配，然后在之后的mapper里可能找不到关键帧来匹配

### Incremental Reconstruction
```shell
mkdir colmap_workspace/test
colmap mapper \
    --database_path colmap_workspace/mobiledata.db \
    --image_path data/mobile_data/av_frames \
    --output_path colmap_workspace/test

# 优化参数设计
colmap mapper \
    --database_path colmap_workspace/mobiledata.db \
    --image_path data/mobile_data/front_frames \
    --output_path colmap_workspace/sparse \
    --Mapper.min_num_matches 15 \
    --Mapper.init_min_num_inliers 60 \
    --Mapper.abs_pose_min_num_inliers 25 \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 0 \
    --Mapper.tri_ignore_two_view_tracks 1 \
    --Mapper.filter_max_reproj_error 4 \
    --Mapper.filter_min_tri_angle 0.5 \
    --Mapper.ba_local_max_num_iterations 30 \
    --Mapper.ba_global_max_num_iterations 50 \
    --Mapper.ba_global_images_ratio 1.1
```




这里会生成二进制文件，要转成txt:
```shell
colmap model_converter \
    --input_path colmap_workspace/test/2 \
    --output_path colmap_workspace/results/2 \
    --output_type TXT
```
| 文件名            | 含义   | 内容                                     |
| -------------- | ---- | -------------------------------------- |
| `cameras.txt`  | 相机内参 | fx, fy, cx, cy, 图像宽高等                  |
| `images.txt`   | 相机外参 | 四元数 (qw,qx,qy,qz), 平移向量 (tx,ty,tz)，图像名 |
| `points3D.txt` | 稀疏点云 | 每个点的坐标、颜色、被哪些图像观测到                     |

**Check Camera**:
```shell
grep -v "#" colmap_workspace/results/1/cameras.txt | awk '{print $1}'

grep -v "#" colmap_workspace/results/1/images.txt | awk '{print $9}' | sort | uniq -c
```
两处相机一致的话就没问题！最好指定同一个相机，否则colmap会根据图和图之间的差异生成不同的相机。

我的结果：
```shell
(pdos) cxx@cxx-Precision-3660:~/HWs/CS290U/project2$ grep -v "#" colmap_workspace/results/1/images.txt | awk '{print $9}' | sort | uniq -c
      2 -1
      2 1

(pdos) cxx@cxx-Precision-3660:~/HWs/CS290U/project2$ grep -v "#" colmap_workspace/results/1/cameras.txt | awk '{print $1}'
1
```

### Convert to Specific Format
`SuperGluePretrainedNetwork/demo_superglue.py`:
示例片段（大约在第 90 行附近）：
```python
pair = {
    'image0': image0_path,
    'image1': image1_path,
    'keypoints0': keypoints0,  # shape [N0, 2]
    'keypoints1': keypoints1,  # shape [N1, 2]
    'matches': matches,        # shape [N0], index of matching keypoint in image1 or -1
    'match_confidence': conf   # shape [N0], confidence scores
}
```
写一个脚本文件用于转化现有数据为输入这个项目的格式：
`~/HWs/CS290U/project2/colmap_workspace/convert2superglue.py`:
```shell
cd ~/HWs/CS290U/project2
# N-1对 旧版的只能支持no-eval
python colmap_workspace/convert2superglue.py \
  --input_path colmap_workspace/results/1 \
  --image_dir data/mobile_data/front_frames \
  --output colmap_workspace/results/1/test.json
```

新版的生成两种txt文件输入，带gt和不带的：
```shell
python colmap_workspace/convert2superglue.py \
  --input_path colmap_workspace/results/2 \
  --image_dir data/mobile_data/front_frames \
  --output colmap_workspace/results/2/test2.json
```

不评估：
```shell
mkdir colmap_workspace/results/2/superglue_outputs

python refs/SuperGluePretrainedNetwork/match_pairs.py \
  --input_dir . \
  --input_pairs colmap_workspace/results/2/test.txt \
  --output_dir colmap_workspace/results/2/superglue_outputs \
  --superglue outdoor --resize 640 --viz

```
评估：
```shell
mkdir colmap_workspace/results/2/superglue_outputs_eval

python refs/SuperGluePretrainedNetwork/match_pairs.py \
  --input_dir . \
  --input_pairs colmap_workspace/results/2/test_with_gt.txt \
  --output_dir colmap_workspace/results/2/superglue_outputs_eval \
  --superglue outdoor --resize 640 --viz --eval
```
-----

python refs/SuperGluePretrainedNetwork/match_pairs.py \
  --input_dir data/mobile_data/front_frames \
  --input_pairs colmap_workspace/results/1/test.txt \
  --output_dir refs/SuperGluePretrainedNetwork/outputs \
  --superglue outdoor \
  --resize 640 \
  --eval \
  --viz \
  --fast_viz

Evaluation Results (mean over 11 pairs):
AUC@5    AUC@10  AUC@20  Prec    MScore
7.84     8.46    13.42   90.86   65.48

python refs/SuperGluePretrainedNetwork/match_pairs.py \
  --input_dir data/mobile_data/front_frames \
  --input_pairs colmap_workspace/results/1/test.txt \
  --output_dir refs/SuperGluePretrainedNetwork/outputs/no_eval \
  --superglue outdoor \
  --resize 640 \
  --viz --fast_viz


---
colmap feature_extractor \
    --database_path colmap_workspace/finetune.db \
    --image_path data/mobile_data/front_frames \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1


colmap mapper \
  --database_path colmap_workspace/finetune.db \
  --image_path data/mobile_data/front_frames \
  --output_path colmap_workspace/finetune \
  --Mapper.min_num_matches 10 \
  --Mapper.init_min_num_inliers 30 \
  --Mapper.abs_pose_min_num_inliers 10 \
  --Mapper.tri_ignore_two_view_tracks 1 \
  --Mapper.ba_refine_focal_length 0 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 0 \
  --Mapper.filter_max_reproj_error 4 \
  --Mapper.filter_min_tri_angle 0.5 \
  --Mapper.ba_local_max_num_iterations 20 \
  --Mapper.ba_global_max_num_iterations 30


colmap model_converter \
    --input_path colmap_workspace/finetune/0 \
    --output_path colmap_workspace/results/2 \
    --output_type TXT




------


## Reproduction
37 images:
av_frames
test/0-2
results/0-2: 36, 32, 26 pairs

no-eval
```shell
python colmap_workspace/convertor.py \
  --colmap_dir colmap_workspace/results/0 \
  --image_root data/mobile_data/av_frames \
  --output colmap_workspace/results/0/demo.txt
```

eval:
```shell
python colmap_workspace/convertor2.py \
  --colmap_dir colmap_workspace/results/0 \
  --image_root data/mobile_data/av_frames \
  --output colmap_workspace/results/0/demo_eval.txt \
  --mode sequential
```

## 测试
不要指定outdoor模式 因为我们拍摄的照片是密集的连续的，没有很大视角变化，否则AUC=0
```shell
# 不带gt直接跑
python refs/SuperGluePretrainedNetwork/match_pairs.py \
  --input_dir . \
  --input_pairs colmap_workspace/results/0/demo.txt \
  --output_dir colmap_workspace/results/0/sample \
  --resize 640 \
  --superglue outdoor \
  --viz \
  --eval


# 效果最好的一个
python refs/SuperGluePretrainedNetwork/match_pairs.py \
  --input_dir . \
  --input_pairs colmap_workspace/results/0/demo_eval.txt \
  --output_dir colmap_workspace/results/0/eval \
  --resize -1 \
  --eval \
  --viz

# 第一版 eval 带issue
Evaluation Results (mean over 15 pairs):
AUC@5	 AUC@10	 AUC@20	 Prec	 MScore	
0.00	 7.25	 15.39	 98.01	 74.84	

# 第二版 eval 修改
Evaluation Results (mean over 15 pairs):
AUC@5	 AUC@10	 AUC@20	 Prec	 MScore	
0.00	 7.46	 15.45	 94.66	 73.26


----
# 失败案例：

python refs/SuperGluePretrainedNetwork/match_pairs.py \
  --input_dir . \
  --input_pairs colmap_workspace/results/0/demo_eval.txt \
  --output_dir colmap_workspace/results/0/eval \
  --resize -1 \
  --eval \
  --superglue outdoor \
  --viz

AUC@5	 AUC@10	 AUC@20	 Prec	 MScore	
0.00	 0.00	 0.00	 97.14	 59.00	
```
第三版修改：增加图和图之间时间间隔
```shell
python colmap_workspace/convertor3.py \
  --colmap_dir colmap_workspace/results/0 \
  --image_root data/mobile_data/av_frames \
  --output colmap_workspace/results/0/demo_eval-3.txt \
  --skip 2 \
  --min_angle 1.0 \
  --min_dist 0.02

python refs/SuperGluePretrainedNetwork/match_pairs.py \
  --input_dir . \
  --input_pairs colmap_workspace/results/0/demo_eval-3.txt \
  --output_dir colmap_workspace/results/0/eval \
  --resize -1 \
  --eval \
  --viz

Evaluation Results (mean over 14 pairs):
AUC@5	 AUC@10	 AUC@20	 Prec	 MScore	
10.79	 19.36	 34.31	 90.83	 64.75	
```



---
# 实战

## Superglue 匹配测试

3 种不同数据：

1. **无人机垂直视角（Drone Nadir View）**

   * 数据来源：MatrixCity 等数据集。
2. **无人机倾斜视角（Drone Oblique View）**

   * 来源：Horizon GS 或自采视频帧。
3. **空地视角（Air-to-Ground View）**

   * 匹配无人机视角与地面车辆视角图像。


仔细查看 HorizonGS: real/road.zip，里面已经给出了分类若干；
可以直接选：
- aerial_h 垂直
- aerial_q 倾斜
- street_cam1 车辆视角

由于自带的gt是全部包含的，这里要先做切分:
```shell
python datasets/exps/split_gt_json.py \
  --input datasets/GS/road/transforms.json \
  --output_dir datasets/GS/road/gt
# ---- aerial_h 注意有些图片缺失要去掉
python datasets/exps/gt_convertor.py \
  --json_path datasets/GS/road/gt/gt_aerial_h.json \
  --image_root datasets/GS/road/images/aerial_h \
  --output datasets/exps/inputs/gt_aerial_h.txt

python refs/SuperGluePretrainedNetwork/match_pairs.py \
  --input_dir . \
  --input_pairs datasets/exps/inputs/gt_aerial_h.txt \
  --output_dir datasets/exps/matches/superglue/aerial_h \
  --resize -1 \
  --superglue outdoor \
  --eval --viz

Evaluation Results (mean over 307 pairs):
AUC@5	 AUC@10	 AUC@20	 Prec	 MScore	
17.99	 26.35	 32.59	 25.58	 8.44	

# not fixed results
- 1
Evaluation Results (mean over 307 pairs):
AUC@5	 AUC@10	 AUC@20	 Prec	 MScore	
0.00	 0.00	 0.34	 21.20	 8.11
- 2
Evaluation Results (mean over 307 pairs):
AUC@5	 AUC@10	 AUC@20	 Prec	 MScore	
0.00	 0.00	 0.17	 18.32	 7.03	

# ---- aerial_q
python datasets/exps/gt_convertor.py \
  --json_path datasets/GS/road/gt/gt_aerial_q.json \
  --image_root datasets/GS/road/images/aerial_q \
  --output datasets/exps/inputs/gt_aerial_q.txt

python refs/SuperGluePretrainedNetwork/match_pairs.py \
  --input_dir . \
  --input_pairs datasets/exps/inputs/gt_aerial_q.txt \
  --output_dir datasets/exps/matches/superglue/aerial_q \
  --resize -1 \
  --superglue outdoor \
  --eval --viz 

Evaluation Results (mean over 307 pairs):
AUC@5	 AUC@10	 AUC@20	 Prec	 MScore	
18.02	 27.95	 34.62	 22.98	 7.99	

# ---- street_cam1
python datasets/exps/gt_convertor.py \
  --json_path datasets/GS/road/gt/gt_street_cam1.json \
  --image_root datasets/GS/road/images/street_cam1 \
  --output datasets/exps/inputs/gt_street_cam1.txt


python refs/SuperGluePretrainedNetwork/match_pairs.py \
  --input_dir . \
  --input_pairs datasets/exps/inputs/gt_street_cam1.txt \
  --output_dir datasets/exps/matches/superglue/street_cam1 \
  --resize -1 \
  --superglue outdoor \
  --eval \
  --viz

Evaluation Results (mean over 154 pairs):
AUC@5	 AUC@10	 AUC@20	 Prec	 MScore	
15.29	 40.60	 65.43	 31.77	 13.02	

```
## mast3r

使用预训练模型提取特征，之后模仿superglue的匹配方式写一个评估脚本。
```shell
# 小demo
python refs/mast3r/mast3r_eval.py \
    --gt_file datasets/exps/inputs/gt_aerial_h.txt \
    --output_dir datasets/exps/matches/mast3r/aerial_h \
    --model naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
    --img_size 512 \
    --device cuda

# 正式
# aerial_h
python refs/mast3r/mast3r_eval2.py \
    --gt_file datasets/exps/inputs/gt_aerial_h.txt \
    --output_dir datasets/exps/matches/mast3r/aerial_h \
    --model naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
    --img_size 512 \
    --device cuda


python refs/mast3r/mast3r_eval2.py \
    --gt_file datasets/exps/inputs/gt_aerial_q.txt \
    --output_dir datasets/exps/matches/mast3r/aerial_q \
    --model naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
    --img_size 512 \
    --device cuda


python refs/mast3r/mast3r_eval2.py \
    --gt_file datasets/exps/inputs/gt_street_cam1.txt \
    --output_dir datasets/exps/matches/mast3r/street_cam1 \
    --model naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
    --img_size 512 \
    --device cuda

## 结果

Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
[INFO] Loading model: naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
[INFO] Found 307 pairs to evaluate.

Evaluating: 100%|███████████████████████████████████████████████████████████| 307/307 [02:12<00:00,  2.32it/s]
[INFO] Per-pair results saved to datasets/exps/matches/mast3r/aerial_h/mast3r_results.csv

Evaluation Results (mean over 307 pairs):
AUC@1	AUC@2	AUC@5	AUC@10	Prec(=AUC@5)	MScore(=MeanConf)
 99.24	 99.70	 99.84	 99.92	 99.84	 88.94
```
这里的结果很大，因为没有转成在实际坐标系下的坐标，只是很大的特征数据，所以归一化之后的映射的数字还是很大，这里需要借用中间模块mapping来对数据做转化。

修改一版本最新的：
```shell
python refs/mast3r/mytest.py \
  --gt_file datasets/exps/inputs/gt_aerial_h.txt \
  --output_csv datasets/exps/matches/mast3r/mast3r_reproj10.csv \
  --num_samples 10

# full num_samples = 99999 > len(pairs)
python refs/mast3r/mytest.py \
  --gt_file datasets/exps/inputs/gt_aerial_h.txt \
  --output_csv datasets/exps/matches/mast3r/mast3r_aerial_h.csv \
  --num_samples 99999

python refs/mast3r/mytest.py \
  --gt_file datasets/exps/inputs/gt_aerial_q.txt \
  --output_csv datasets/exps/matches/mast3r/mast3r_aerial_q.csv \
  --num_samples 99999

python refs/mast3r/mytest.py \
  --gt_file datasets/exps/inputs/gt_street_cam1.txt \
  --output_csv datasets/exps/matches/mast3r/mast3r_street_cam1.csv \
  --num_samples 99999
```
- 10
```shell
MASt3R Match Pairs Evaluation (SuperGlue-aligned):
AUC@5   AUC@10  AUC@20  Prec@3px  MeanErr(px)
99.24   99.63   99.84   98.57   1.22
```
- 20
```shell
MASt3R Match Pairs Evaluation (SuperGlue-aligned):
AUC@5   AUC@10  AUC@20  Prec@3px  MeanErr(px)
99.56   99.79   99.91   99.14   1.05
```
- 100
```shell
MASt3R Match Pairs Evaluation (SuperGlue-aligned):
AUC@5   AUC@10  AUC@20  Prec@3px  MeanErr(px)
99.80   99.91   99.97   99.49   0.94
```
- aerial_h (307 pairs)
```shell
MASt3R Match Pairs Evaluation (SuperGlue-aligned):
AUC@5   AUC@10  AUC@20  Prec@3px  MeanErr(px)
99.66   99.78   99.91   99.40   1.10
```
- aerial_q (307 pairs)
```shell
MASt3R Match Pairs Evaluation (SuperGlue-aligned):
AUC@5	AUC@10	AUC@20	Prec@3px	MeanErr(px)
99.63	99.83	99.93	99.11	1.20
```
- street_cam1 (154 pairs)
```shell
MASt3R Match Pairs Evaluation (SuperGlue-aligned):
AUC@5	AUC@10	AUC@20	Prec@3px	MeanErr(px)
98.69	99.44	99.79	97.98	1.45
```

## vggt
使用预训练权重

小demo测试：
```shell
python refs/vggt/mytest.py \
  --gt_file datasets/exps/inputs/gt_aerial_h.txt \
  --output_csv datasets/exps/matches/vggt/vggt_10.csv \
  --num_samples 10

```
正式跑：
```shell
python refs/vggt/mytest.py \
  --gt_file datasets/exps/inputs/gt_aerial_h.txt \
  --output_csv datasets/exps/matches/vggt/aerial_h/vggt_aerial_h.csv \
  --num_samples 9999

python refs/vggt/mytest.py \
  --gt_file datasets/exps/inputs/gt_aerial_q.txt \
  --output_csv datasets/exps/matches/vggt/aerial_q/vggt_aerial_q.csv \
  --num_samples 9999

python refs/vggt/mytest.py \
  --gt_file datasets/exps/inputs/gt_street_cam1.txt \
  --output_csv datasets/exps/matches/vggt/street_cam1/vggt_street_cam1.csv \
  --num_samples 9999
```
- aerial_h
```shell
VGGT Match Pairs Evaluation (SuperGlue-aligned):
AUC@5	 AUC@10	 AUC@20	 Prec	 MError	
99.81	 99.92	 99.96	 99.64	 0.92	
```
- aerial_q
```shell
VGGT Match Pairs Evaluation (SuperGlue-aligned):
AUC@5	AUC@10	AUC@20	Prec@3px	MeanErr(px)
99.58	99.77	99.92	99.30	1.07
```
- street_cam1
```shell
VGGT Match Pairs Evaluation (SuperGlue-aligned):
AUC@5	 AUC@10	 AUC@20	 Prec	 MError	
97.57	 98.78	 99.40	 95.85	 2.69
```


# Localization and Reconstruction

## mast3r
demo.py只会生成一个glb文件，并且只提供简单的可视化交互，中间结果很多被忽略。
这里直接用它中间的组件来进行调用：
* project2/refs/mast3r/demo_glomap.py
* project2/refs/mast3r/visloc.py

但是mast3r这里的visloc.py指定了数据集进行一个评估，查看代码发现使用的dust3r的评估模块，然后它的输入是来源于：dust3r/dust3r_visloc/datasets/base_dataset.py。

基类具体定义：
```python
class BaseVislocDataset:
    def __init__(self):
        pass

    def set_resolution(self, model):
        self.maxdim = max(model.patch_embed.img_size)
        self.patch_size = model.patch_embed.patch_size

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        raise NotImplementedError()
```
自定义数据类用于接收数据, 放在相同路径下：`refs/mast3r/dust3r/dust3r_visloc/datasets/visloc_horizon.py`，并在__init__中加上引用`from .visloc_horizon import VislocHorizonGS`:
```python
# Copyright (C) 2025 WOODENMAN
# --------------------------------------------------------
# Lightweight HorizonGS dataset loader (SuperGlue GT format)
# Compatible with MASt3R visloc.py
# --------------------------------------------------------

import os
import numpy as np
from PIL import Image
import torch
from dust3r_visloc.datasets.base_dataset import BaseVislocDataset
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r_visloc.datasets.utils import get_resize_function


class VislocHorizonGS(BaseVislocDataset):
    def __init__(self, root, pairsfile='pairs.txt', intrinsics='intrinsics.txt', gt_pose='gt_pose.txt', topk=1):
        super().__init__()
        self.root = root
        self.topk = topk
        self.num_views = self.topk + 1
        self.maxdim = None
        self.patch_size = None

        # 读取 pairs.txt
        with open(os.path.join(root, pairsfile)) as f:
            self.pairs = [l.strip().split() for l in f.readlines() if l.strip()]

        # 读取 intrinsics.txt
        self.K = np.loadtxt(os.path.join(root, intrinsics)).reshape(3, 3)
        self.distortion = np.zeros(5, dtype=np.float32)

        # 可选：读取真值姿态（用于 AUC/pose error）
        self.gt_pose = {}
        gt_path = os.path.join(root, gt_pose)
        if os.path.exists(gt_path):
            for l in open(gt_path):
                name, *vals = l.strip().split()
                self.gt_pose[name] = np.array(vals, float).reshape(4, 4)
        else:
            print("⚠️ No gt_pose.txt found, will use identity matrices.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        assert self.maxdim is not None and self.patch_size is not None
        q_path, m_path = self.pairs[idx]
        q_name, m_name = os.path.basename(q_path), os.path.basename(m_path)

        q_img = Image.open(os.path.join(self.root, q_path)).convert('RGB')
        m_img = Image.open(os.path.join(self.root, m_path)).convert('RGB')

        resize_func_q, to_resize_q, to_orig_q = get_resize_function(self.maxdim, self.patch_size, *q_img.size[::-1])
        resize_func_m, to_resize_m, to_orig_m = get_resize_function(self.maxdim, self.patch_size, *m_img.size[::-1])

        q_tensor = resize_func_q(ImgNorm(q_img))
        m_tensor = resize_func_m(ImgNorm(m_img))

        q_pose = self.gt_pose.get(q_name, np.eye(4))
        m_pose = self.gt_pose.get(m_name, np.eye(4))

        # Query view
        query_view = {
            'intrinsics': self.K, 'distortion': self.distortion,
            'cam_to_world': q_pose,
            'rgb': q_img, 'rgb_rescaled': q_tensor,
            'to_orig': to_orig_q,
            'image_name': q_name
        }

        # Map view
        map_view = {
            'intrinsics': self.K, 'distortion': self.distortion,
            'cam_to_world': m_pose,
            'rgb': m_img, 'rgb_rescaled': m_tensor,
            'to_orig': to_orig_m,
            'image_name': m_name
        }

        return [query_view, map_view]
```
从之前的gt.txt转出合理的数据传入：
```shell
# python refs/mast3r/convertor.py \
#   -i datasets/exps/inputs/gt_aerial_h.txt \
#   -o datasets/exps/inputs/localization/aerial_h

python refs/mast3r/convertor.py \
  --transforms datasets/GS/road/gt/gt_aerial_h.json \
  --images_root datasets/GS/road/images/aerial_h \
  --out_dir datasets/exps/inputs/localization/aerial_h \
  --pair_mode sequential --stride 1 \
  --matrix_type cam2world \
  --abs_path

```

加入深度图：
```shell
python refs/mast3r/visloc.py \
  --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --dataset "VislocHorizonGS(
      'datasets/GS/road/images/aerial_h',
      pairsfile='datasets/exps/inputs/localization/aerial_h/pairs.txt',
      intrinsics='datasets/exps/inputs/localization/aerial_h/intrinsics.txt',
      gt_pose='datasets/exps/inputs/localization/aerial_h/gt_pose.txt',
      depth_root='datasets/GS/road/depths/aerial_h'
  )" \
  --pixel_tol 5 \
  --pnp_mode poselib \
  --output_dir project2/datasets/exps/localization/mast3r/aerial_h
```




------a
```shell
python refs/mast3r/mast3r_eval.py \
  --gt_file datasets/exps/inputs/gt_aerial_h.txt \
  --output_dir refs/mast3r/results/aerial_h \
  --num_samples -1 \
  --device cuda
```
Evaluation Results (mean over 5 pairs, original pixel scale):
AUC@1	AUC@2	AUC@5	AUC@10	Prec(=AUC@5)	MScore(=MeanConf)
  3.28	 27.64	 94.55	 98.14	 94.55	 82.35

Mean Reprojection Error: 4.1623 px
----

Evaluation Results (mean over 307 pairs, original pixel scale):
AUC@1	AUC@2	AUC@5	AUC@10	Prec(=AUC@5)	MScore(=MeanConf)
  0.59	  4.54	 98.37	 99.59	 98.37	 88.94

Mean Reprojection Error: 3.1433 px


python refs/mast3r/mast3r_eval.py \
  --gt_file datasets/exps/inputs/gt_aerial_q.txt \
  --output_dir refs/mast3r/results/aerial_q \
  --num_samples -1 \
  --device cuda

Evaluation Results (mean over 307 pairs, original pixel scale):
AUC@1   AUC@2   AUC@5   AUC@10  Prec(=AUC@5)    MScore(=MeanConf)
  0.58    1.83   97.19   99.36   97.19   90.75

Mean Reprojection Error: 3.4107 px


python refs/mast3r/mast3r_eval.py \
--gt_file datasets/exps/inputs/gt_street_cam1.txt \
--output_dir refs/mast3r/results/street_cam1 \
--num_samples -1 \
--device cuda

Evaluation Results (mean over 154 pairs, original pixel scale):
AUC@1   AUC@2   AUC@5   AUC@10  Prec(=AUC@5)    MScore(=MeanConf)
  0.12   89.65   97.45   98.60   97.45   85.67

Mean Reprojection Error: 3.0446 px




python refs/vggt/demo_viser.py --image_folder /home/cxx/HWs/CS290U/project2/datasets/GS/road/images/aerial_h