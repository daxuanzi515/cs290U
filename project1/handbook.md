# Helper
## 🧾 一、作业概览（Overview）

**课程项目题目：**

> **Exploring Supervised and Foundation Models for Image Segmentation**
> 探索图像分割中的监督学习与基础模型（Foundation Models）

**指导老师：** Prof. Yuijao Shi
**截止日期：** 🕒 **2025年10月19日 23:59**

**提交格式要求：**

* 报告必须为 **PDF 格式**（可使用 LaTeX 或 Word 编写）；
* 将 **报告与代码** 一起打包为：

  ```
  studentID-name-project1.zip
  ```
* 提交地址：
  🔗 [https://epan.shanghaitech.edu.cn/l/JFrYpF](https://epan.shanghaitech.edu.cn/l/JFrYpF)

**数据集：**

* Carvana Dataset（用于汽车二值分割）
  🔗 [https://epan.shanghaitech.edu.cn/l/F2JPuU](https://epan.shanghaitech.edu.cn/l/F2JPuU)

---

## 📘 二、项目目的（Introduction）

该项目旨在让学生理解**从传统监督学习到基础模型（Foundation Models）**在图像分割中的演进。

分为两个层次：

1. **基础任务（Basic Task）**

   * 训练与评估一个 **U-Net 模型** 在 Carvana 数据集上进行汽车分割；
   * 了解监督式语义分割的完整流程（数据加载、超参数调节、训练、评估、可视化）；
   * 理解传统方法的**优点（高精度）**与**局限（仅限单任务）**。

2. **进阶任务（Advanced Tasks）**

   * 探索基础模型（如 **SAM、GroundingDINO**）在**零样本（Zero-Shot）**和**文本驱动（Text-driven）**分割任务中的表现；
   * 分析 SAM 的能力与局限；
   * 构建 **GroundingDINO + SAM** 管线，实现从“类别无关分割”到“语义分割”的升级。

---

## 🔧 三、基础任务（Basic Task）

**目标：**

> 训练并评估一个基于 PyTorch 的 U-Net 模型，对 Carvana 数据集进行汽车前景分割。

**要求步骤：**

1. 克隆官方代码库：
   🔗 [https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
2. 下载 Carvana 数据集并放置在合适路径。
3. 运行训练脚本前，**修改关键超参数**，避免出现 `loss = NaN` 问题。

   > ⚠️ 默认参数可能不稳定。建议重点调整：

   * 学习率（learning rate）
   * 批大小（batch size）
4. 达到稳定训练后：

   * 在验证集上评估 Dice score；
   * 用测试集（或自己拍摄的车图）推理；
   * 在报告中展示输入图像 vs. 预测分割图；
   * **目标：Dice 分数应 ≥ 0.90**

**报告内容建议：**

* 模型结构与参数说明；
* 训练配置与过程可视化；
* 损失曲线与验证指标；
* 推理结果（可视化对比图）；
* 对监督方法优缺点的分析。

---

## 🚀 四、进阶任务（Advanced Tasks）

### （1）SAM 零样本分割（Zero-Shot Segmentation）

**目标：**

> 理解 **Segment Anything Model (SAM)** 的“无标注自动分割”能力及局限性。

**要求：**

* 自制一个 **20 张照片**的小型数据集；
* 使用 **SAM 的 "everything" 模式** 进行自动分割；
* 分析以下内容：

  * SAM 的**成功与失败案例**；
  * **多掩码输出（multimask）** 的效果（不同尺度/细节层次）；
* 报告应展示：

  * SAM 的分割结果；
  * 对失败原因的分析（例如：复杂纹理、透明物体、小目标）。

---

### （2）GroundingDINO + SAM 语义分割（Text-driven Segmentation）

**目标：**

> 构建一个结合 **GroundingDINO 与 SAM** 的多模态管线，实现**文本驱动的语义分割**。

**核心思想：**

* 使用 GroundingDINO 的**文本→边界框检测**；
* 将边界框作为 **SAM 的 prompt** 输入；
* 输出语义相关的分割掩码。

**预期学习成果：**

* 理解如何让“类无关模型”通过文本提示变得“语义感知”；
* 实现**多模型联动与融合**；
* 探索基础模型在多模态场景下的扩展潜力。

**参考代码库：**

* GroundingDINO: [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
* SAM: [https://segment-anything.com/](https://segment-anything.com/)

---

## 📦 五、参考文献（References）

1. Carvana Image Masking Challenge – Kaggle
2. Grounding DINO – IDEA Research
3. PyTorch U-Net – Milesial
4. U-Net 原论文：*Ronneberger et al., 2015*
5. SAM 官方网站

---

## 🔑 六、重点与配置清单（Summary of Key Points）

| 项目环节      | 要点                                                                                     | 重要程度  |
| --------- | -------------------------------------------------------------------------------------- | ----- |
| **截止时间**  | 2025年10月19日 23:59                                                                      | ⭐⭐⭐⭐  |
| **报告语言**  | 必须使用 **英文**                                                                            | ⭐⭐⭐⭐  |
| **格式**    | PDF（建议使用 LaTeX）                                                                        | ⭐⭐⭐   |
| **文件名**   | `学号-姓名-project1.zip`                                                                   | ⭐⭐⭐   |
| **提交地址**  | [https://epan.shanghaitech.edu.cn/l/JFrYpF](https://epan.shanghaitech.edu.cn/l/JFrYpF) | ⭐⭐⭐⭐  |
| **数据集路径** | [https://epan.shanghaitech.edu.cn/l/F2JPuU](https://epan.shanghaitech.edu.cn/l/F2JPuU) | ⭐⭐⭐   |
| **框架要求**  | PyTorch + U-Net baseline                                                               | ⭐⭐⭐⭐  |
| **目标指标**  | Dice ≥ 0.90                                                                            | ⭐⭐⭐⭐  |
| **进阶要求**  | SAM 分析 + GroundingDINO+SAM 管线                                                          | ⭐⭐⭐⭐⭐ |
| **自建数据**  | 至少20张照片（SAM测试）                                                                         | ⭐⭐⭐   |


-------
10.11会议笔记：
report模版: => cpvr template （建议）
提高部分任务：SAM2代码是否能用？？ 最好先自己实现，因为是现成的，它希望在原先的代码上有所修改（）
Basic(可以不用，想交也可) + Advanced Codes（**）: 只需要代码，不需要模型，主要是Advanced部分，可以有选择的参考网上的代码
只写Basic => 训练的过程图+Dice分数+测试结果
Advanced=> 对数据做了什么处理？分析图，找共性问题； 网络结构的问题对最后结果的影响？
显示分数：模型训练完的分数 Dice Score
最后的效果看测试机或自己的图跑出的效果

要自己跑代码结果 multimask模式开启 在网站上简单测试是不行的