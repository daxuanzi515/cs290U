# 📘 项目二：图像匹配、相机姿态估计与三维重建

**授课教师：Prof. Yujiao Shi**
**截止日期：2025 年 11 月 2 日 23:59**


## 一、项目简介（Introduction）

本项目旨在让学生亲身实践计算机视觉中的基础概念，包括：

* 局部特征匹配（Local Feature Matching）
* 相机姿态估计（Camera Pose Estimation）
* 三维重建（3D Reconstruction）

学生需要：

1. 自行采集图像数据；
2. 使用 **COLMAP** 进行结构自运动（SfM）重建；
3. 评估 **SuperGlue** 等特征匹配算法；
4. 在进阶部分，比较多种最新方法（SuperGlue、MASt3R、VGGT）在不同场景下的表现。

---

## 二、基础任务（Basic Tasks）

### 2.1 数据采集（Data Acquisition）

* 使用手机或数码相机拍摄 **20 张连续场景图片**；备注：可以拍一个视频取连续20帧图作为数据。10s左右的视频，相机移动的位移不要太大，尽量保证图和图之间有重叠！
* 确保相邻图像之间有足够的重叠；
* 可以选择具有挑战性的场景（如光照变化、大视角差等）。

### 2.2 使用 COLMAP 进行三维重建（Structure from Motion）

* 工具： [COLMAP](https://colmap.github.io/)
* 目标：重建每张图像的 **内参** 与 **外参**（相机姿态）。
* 标准流程：

  1. 特征提取（Feature Extraction）
  2. 特征匹配（Feature Matching）
  3. 增量重建（Incremental Reconstruction）
* 输出：稀疏点云与对应的相机位姿。

### 2.3 测试集构建（Test Set Creation）

* 从 20 张图片中挑选 **10 对图像**，确保视角变化不同；备注：20张图里每两张进行两两匹配
* 使用 COLMAP 结果计算每对图像的 **相对位姿**（旋转和平移）；
* 将图像对、相机内参、相对位姿组织成测试集文件；
* 可参考 [SuperGlue 数据格式](https://github.com/magicleap/SuperGluePretrainedNetwork)。

### 2.4 使用 SuperGlue 进行评估（Evaluation with SuperGlue）

* 工具：[SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)
* 任务：

  1. 在自建测试集上运行 SuperGlue；
  2. 计算定量指标：如姿态误差；
  3. 绘制匹配可视化图；
  4. 分析结果优缺点与失败原因（例如大视角差、纹理不足、重复结构等）。

---

## 三、进阶任务（Advanced Tasks）
如果你只做basic task的话，是可以直接在本地跑的。
advanced task对显存有一定要求，需要你的笔记本显存>=8G。

### 3.1 匹配算法比较（Comparative Analysis of Matching Methods）

需比较以下三种方法：

| 方法                     | 来源                                                                |
| ---------------------- | ----------------------------------------------------------------- |
| SuperPoint + SuperGlue | [GitHub](https://github.com/magicleap/SuperGluePretrainedNetwork) |
| MASt3R                 | [GitHub](https://github.com/naver/mast3r)                         |
| VGGT                   | [GitHub](https://github.com/facebookresearch/vggt)                |

#### 三种场景：

1. **无人机垂直视角（Drone Nadir View）**

   * 数据来源：MatrixCity 等数据集。
2. **无人机倾斜视角（Drone Oblique View）**

   * 来源：Horizon GS 或自采视频帧。
3. **空地视角（Air-to-Ground View）**

   * 匹配无人机视角与地面车辆视角图像。

需对各方法进行 **定量与定性比较**，分析其在不同场景下的优劣势。

### 3.2 定位与三维重建对比（Comparative Analysis of Localization and Reconstruction）

* 比较 **MASt3R** 与 **VGGT** 在：

  * 图像定位；
  * 点云重建；
    上的表现。
* 同样覆盖三种场景。
* 对于空地视角（Scenario 3），需分别测试单对图像和多帧连续图像。

### 3.3 报告要求（Report and Analysis）

报告需包含：

* 所有方法与场景的图像匹配结果可视化；
* 三维点云重建可视化；
* 全面的性能分析。

---

## 四、提交要求（Submission）

提交一个压缩包：
`studentID-name-project2.zip`
内容包含：

* 报告 PDF（LaTeX 编写优先）；
* 代码文件与注释；
* 结果与可视化图表；
* 可在报告中附上代码仓库链接。

提交平台示例：
👉 [https://epan.shanghaitech.edu.cn/l/iFsj6h](https://epan.shanghaitech.edu.cn/l/iFsj6h)

---

## 📊 总结要点

| 模块     | 主要任务                       | 工具/资源                 |
| ------ | -------------------------- | --------------------- |
| 数据采集   | 拍摄 20 张重叠图像                | 相机/手机                 |
| SfM 重建 | 提取特征并重建姿态                  | COLMAP                |
| 测试集制作  | 计算相对姿态、整理数据                | COLMAP + SuperGlue 格式 |
| 算法评估   | 使用 SuperGlue 匹配与分析         | SuperGlue             |
| 进阶比较   | 对比 SuperGlue, MASt3R, VGGT | GitHub 官方实现           |
| 重建分析   | 对比 MASt3R 与 VGGT           | 多场景测试                 |
| 报告撰写   | 总结结果与可视化                   | LaTeX / PDF           |

---

## 📚 参考文献

1. Lihan Jiang et al. *Horizon-GS*, CVPR 2025.
2. Vincent Leroy et al. *MASt3R*, arXiv 2024.
3. Paul-Edouard Sarlin et al. *SuperGlue*, CVPR 2020.
4. Jianyuan Wang et al. *VGGT*, CVPR 2025.
5. Saining Zhang et al. *Drone-assisted Road Gaussian Splatting*, arXiv 2024.
