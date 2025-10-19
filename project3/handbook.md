# 🧠 CS290U Project 3：探索扩散模型的生成能力 — 从潜空间到多模态合成

**指导教师：** Prof. Yuijao Shi
**截止时间：** 2025 年 11 月 16 日 23:59
**提交格式：** PDF 报告 + 代码打包为 `学号-姓名-project3.zip` 上传至
[https://epan.shanghaitech.edu.cn/l/W11wly](https://epan.shanghaitech.edu.cn/l/W11wly)

---

## 一、介绍：基于 MNIST 的生成模型

本项目研究**生成模型（Generative Models）**，重点探索**扩散模型（Diffusion Models）**的强大能力。
整个实验基于 **MNIST 手写数字数据集**，作为验证生成与潜空间表征的理想测试平台。

项目分为两个部分：

* **基础部分（Basic）**：

  * 介绍 **变分自编码器（VAE）** 的重建与潜空间压缩。
  * 实现并比较 **DDPM（Denoising Diffusion Probabilistic Model）** 与其加速版本 **DDIM（Denoising Diffusion Implicit Model）**。
  * 将 VAE 与 DDPM 结合，构建高效的潜空间扩散管线。

* **进阶部分（Advanced）**：

  * 实现一个简化版的文本到图像模型 **tiny-mnist-sd**（类似 Stable Diffusion-3）。
  * 学习使用 **ComfyUI 框架**，比较其工作流与 tiny-mnist-sd 的结构差异。

📦 **注意**：压缩包内含基础代码，请务必阅读 `README` 文件并按说明配置环境。

---

## 二、基础探索：生成的基本原理

### 2.1 🧩 VAE：图像重建与潜空间编码

运行提供的 `train_vae.py`。
VAE 的损失由两部分组成：

* **重建损失（Reconstruction Loss）**：度量输入与重建图像的像素差异。
* **KL 散度损失（KL Divergence Loss）**：约束潜空间分布接近标准正态分布。

#### 📘 报告任务 1

解释上述两项损失的**数学意义与直觉**，说明它们如何在 VAE 中共同起作用。

#### ⚙️ 报告任务 2

修改代码中的 `β` 参数（控制 KL Loss 权重），比较不同 β 值下：

* 重建图像的清晰度；
* 潜空间采样生成图像的质量。
  展示**重建精度与潜空间正则性**之间的权衡。

#### ✅ 做法

1. 修改 `train_vae.py` 中的 `beta` 参数（如 0.1、1、5）。
2. 分别训练模型，保存 `reconstruction` 与 `latent samples`。
3. 在报告中展示图片对比与分析。

---

### 2.2 🌫️ DDPM 与 DDIM：加速扩散生成

基于 `train_diffusion.py`。
DDPM 是扩散模型的基础实现，但采样速度慢。
本部分要求你**在现有框架上实现 DDIM 采样器**，以提升生成效率。

#### 🧑‍💻 实现任务

扩展 DDPM 框架，添加 **DDIM 采样逻辑**（允许跳步非马尔可夫采样）。

#### 📘 报告任务

比较你实现的 **DDIM** 与 **DDPM**：

* 样本质量：DDIM 可能略差，但至少 **1/3 图像应可辨识**；
* 理论分析：说明 DDIM 的计算效率优势。

#### ✅ 做法

1. 在代码中实现 `sample_ddim()` 函数。
2. 选取相同步数生成图像，比较清晰度与生成时间。
3. 报告中给出对比图与运行时间表。

---

### 2.3 🔒 潜空间扩散：效率提升策略

基于 `train_latent_diffusion.py`。
直接在像素空间扩散成本高且学习困难，
因此采用 **潜空间扩散（Latent Diffusion）**：
先用 VAE 压缩图像，再在潜空间中扩散与采样。

#### 🧑‍💻 实现任务

选择你在 2.1 中训练的最佳 VAE，用作编码器/解码器。

#### 💡 提示

当 VAE 用作潜空间压缩器时，更接近普通自编码器（AE），
应**减小 β 参数**以提高重建质量。

#### 📘 报告任务

说明你选择的 VAE 参数，并展示潜空间扩散生成结果的质量。

#### ✅ 做法

1. 载入已训练的 VAE（encoder、decoder）。
2. 运行 `train_latent_diffusion.py` 生成图像。
3. 比较潜空间采样结果与像素空间采样结果。

---

## 三、进阶应用：可控生成

### 3.1 🧠 条件生成：训练 Tiny Multimodal SD

目标：实现一个简化的 **文本（标签）→图像** 模型 `tiny-mnist-sd`。

#### 📋 前置任务

运行 `train_clip.py`，训练一个基于 MNIST 图像与标签的 CLIP 模型（不需改动）。

#### 🧑‍💻 实现任务

在 `Unet.py` 中**补全 CLIP 条件注入逻辑**（将 CLIP 向量融入 UNet）。

#### ⚙️ 验证与报告

成功训练后，模型应生成：

* 10 列图（0–9），每列数字对应输入条件；
* 至少 1/3 的结果应正确对应条件标签。
  报告中展示对齐情况并分析可控生成效果。

#### ✅ 做法

1. 训练 CLIP：`python train_clip.py`
2. 修改 `Unet.py`，在 forward 中融合 CLIP 特征。
3. 运行 `train_tiny_mnist_sd.py`，生成 0–9 控制图。
4. 报告中贴出示例与准确率。

---

### 3.2 🧰 ComfyUI：了解扩散工作流

**ComfyUI** 是一个图形化的 Stable Diffusion 推理框架，
用户可通过拖拽节点构建复杂生成流程（如 CLIP 编码 → UNet 推理 → VAE 解码）。

#### 🧪 实践任务

1. 安装 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)。
2. 构建至少两种 text-to-image 流程并生成图像。
3. 可尝试不同的 LoRA 或采样器并分析失败原因。

#### 📘 报告任务

比较 **ComfyUI 工作流** 与 **tiny-mnist-sd 管线**：

* 相似点：CLIP 条件、UNet、VAE 等阶段；
* 不同点：模块灵活替换、显式控制流、更强可视化。

#### ✅ 做法

1. 在 ComfyUI 中加载基础 Stable Diffusion 模型；
2. 新建节点链（Prompt → CLIP → UNet → VAE）；
3. 截图工作流，与 tiny-mnist-sd 的 pipeline 进行结构对比；
4. 报告中分析两者在灵活性与可控性上的区别。

---

## 📚 参考文献

1. ComfyUI 官方仓库, 2023
2. Patrick Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*, ICML 2024
3. Jonathan Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
4. Diederik P. Kingma & Max Welling, *Auto-Encoding Variational Bayes*, arXiv 2013
5. Alec Radford et al., *CLIP: Learning Transferable Visual Models from Natural Language Supervision*, ICML 2021
6. Olaf Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
7. Jiaming Song et al., *Denoising Diffusion Implicit Models*, arXiv 2020

---

## ✅ 总结：报告与实现建议

| 模块        | 主要任务                               | 实现文件                               | 报告需展示内容      |
| --------- | ---------------------------------- | ---------------------------------- | ------------ |
| VAE       | 调整 β 比较重建与潜空间样本                    | `train_vae.py`                     | 结果图 + 理论解释   |
| DDPM/DDIM | 实现 DDIM 采样                         | `train_diffusion.py`               | 效率比较与图像展示    |
| 潜空间扩散     | 使用 VAE encoder 训练 latent diffusion | `train_latent_diffusion.py`        | 潜空间样本效果与参数说明 |
| 条件生成      | 在 UNet 中注入 CLIP 条件                 | `Unet.py`、`train_tiny_mnist_sd.py` | 0–9 控制图与对齐分析 |
| ComfyUI   | 构建两种工作流                            | ComfyUI GUI                        | 截图对比与总结分析    |
