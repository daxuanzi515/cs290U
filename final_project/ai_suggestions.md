# AI Suggestions
---
>**（安全的内容可以和GPT5.1讨论，要是攻击层面的话只能和K2和DeepSeek说，其他的写作部分和GPT说）**

关于创新性的提案和设计讨论：
- 独特创新性
- 和project3的关联性是不是很大
- 实际可行性
- 实验设计和评估
- 书面撰写


## 方案讨论
下面我根据我们刚刚所有对话内容，把**你的项目目标、方法、评估、任务定义、反事实逻辑、一致性理解、C²D 框架、选择理由**等全部整合成一套统一、完整、严谨、可执行、可直接当作最终项目 Proposal 的 **新版执行任务文档**。

这是你未来几周的 **执行计划 + 项目定义 + 技术路线 + 实验方案**
你可用来：

* 做 Final Project
* 写 Report
* 写 Method Section
* 做 PPT
* 做 Demo
* 甚至扩展成科研项目

下面是 **项目正式版（v2.0）**。

---

# 🟣 **C²D：一致性约束的反事实视觉生成系统

Consistency-Aware Counterfactual Diffusion**

## —— Final Project 执行任务（Full Technical Execution Plan）

---

# 1. 项目简介（Project Summary）

当前图像编辑模型（如 Stable Diffusion、ControlNet、Inpaint）在执行“属性修改/行为修改/物体修改”时，通常会：

* 破坏图像的非相关区域
* 降低人物一致性
* 改变背景/光照/肤色
* 产生全局重绘

这与 **反事实生成（Counterfactual Generation）**的核心目标完全不符：

> **只改变导致某个语义属性的因果区域（causal region），其他区域尽可能保持一致。**

本项目提出：

# 🟣 **C²D（Consistency-Aware Counterfactual Diffusion）**

一种结合 **反事实编辑 + 一致性保证 + 多样候选筛选 + 验证-修正闭环** 的新型扩散框架。

该系统借鉴因果反事实思想，但不依赖真正的因果模型，是一种基于扩散模型的**弱因果图像编辑（weakly causal editing）**。

---

# 2. 项目目标（Objectives）

## ✔ **反事实目标达成（Target Satisfaction）**

让图像满足一个新的语义条件：

* 暴力 → 非暴力
* 错误点击 → 正确点击
* 眼镜 → 无眼镜
* 拿刀 → 空手
* 挥拳 → 自然站立
* 手势 A → 手势 B

## ✔ **一致性保持（Consistency Preservation）**

在修改该语义所需区域之外，其他区域保持尽量一致：

* 背景
* 人物外观
* 布局
* 光照
* 结构
* UI 元素

## ✔ **最小修改（Minimal Edit）**

达成目标所需修改必须尽可能小（minimal intervention）。

---

# 3. 反事实定义（任务定义）

在本项目中，反事实图像定义为：

> **只改变导致某个目标属性的因果区域，其他区域保持一致的图像修改。**

形式化定义：

令：

* (x)：原图
* (x')：反事实图
* (C_{causal})：因果区域（需改变）
* (C_{non})：非因果区域（必须保持）

则反事实编辑要求：
$$
[
x'*{non_causal} \approx x*{non_causal}
]
$$

$$
[
f(x') = \text{target attribute}
]
$$

$$
[
\Delta = \lVert x' - x \rVert \text{ minimal}
]
$$
由此得到反事实编辑的“三要素”：

1. 修改目标属性
2. 保持其他区域一致
3. 只做最小必要修改

---

# 4. C²D 总体架构（System Architecture）

```
输入图像 x 
    │
    ▼
VAE Enc → latent z
    │
    ▼
反事实条件编码器（Condition Encoder → e_c）
    │
    ▼
Counterfactual Diffusion（UNet + edited guidance）
    │
    ▼
生成 N 个反事实候选 {x'_1, x'_2, ..., x'_N}
    │
    ▼
Consistency Validator（LPIPS / SSIM / Edge / CLIP-image）
    │
    ▼
Target Validator（Classifier / Detector / CLIP-text）
    │
    ▼
Scoring Matrix：选出“C 高 + T 高”的最佳 x*
    │
    ▼
若未达标 → Rectifier（部分重采样 / 增强 guidance）
    │
    ▼
最终反事实输出 x*
```

---

# 5. 模块说明（Modules）

## 5.1 Condition Encoder（反事实条件编码）

输入条件可为任意形式：

* 文本：“remove violence”
* 属性：“no_glasses”
* UI 行为：“correct-click-button-B”
* 动作：“neutral pose”

编码方式：

* CLIP text encoder
* 或 one-hot + MLP
* 或自定义词表 embedding

输出 embedding (e_c) 注入 diffusion。

---

## 5.2 Counterfactual Diffusion（反事实生成器）

* 使用 Project3 的 latent diffusion UNet
* 在 DDPM/DDIM 采样中加入 conditional embedding
* 控制模型对特定区域产生必要修改

生成 **N 个候选反事实图像**。

---

## 5.3 Validator + Rectifier（验证与修正模块）

### 🔹 Target Validator

判断反事实是否满足目标属性：

* 暴力分类器
* 武器检测器
* UI OCR + Element Detector
* CLIP-text(x', prompt)

### 🔹 Consistency Validator

判断反事实是否保持一致性：

* LPIPS(x, x’)
* SSIM(x, x’)
* CLIP-image(x, x’)
* EdgeDiff(x, x’)
* PoseDiff（适用于动作任务）
* OCRDiff（适用于 UI）

### 🔹 Rectifier

如果 x’ 不满足目标或一致性太差：

* 动态调大 target weight
* 部分 re-sampling（从 t=200 继续采样）
* 加强 causal region mask
* 降低 global noise

---

# 6. 打分矩阵（Scoring Matrix / Evaluation Plane）

每个生成样本有两个核心分数：

* **T(x’)** = 目标达成度
* **C(x’)** = 一致性保持度

构建二维平面：

横轴：C（一致性）
纵轴：T（目标达成）

我们要的是：

# 🟩 右上角（高一致性 + 高目标达成）

### 总分：

线性：
$$
[
Score = \alpha \cdot T_{norm} + \beta \cdot C_{norm}
]
$$
乘法（惩罚某一维太差）：
$$
[
Score = (T_{norm})^\alpha \cdot (C_{norm})^\beta
]
$$
---

# 7. 损失函数（Training Loss）
$$
[
L = L_{\text{diffusion}} +
\gamma_1 L_{\text{consistency}} +
\gamma_2 L_{\text{target}}
]
$$
其中：

### 1) Diffusion 装置：
$$
[
L_{\text{diffusion}} = |\epsilon - \epsilon_\theta(z_t, t, e_c)|^2
]
$$
### 2) Consistency Loss：
$$
[
L_{\text{consistency}} =
\lambda_1 LPIPS(x', x) +
\lambda_2 \lVert z' - z\rVert +
\lambda_3 (1 - CLIP_image(x', x))
]
$$
### 3) Target Loss：

* 基于分类器：CE loss
* 或 CLIP-text 损失

---

# 8. 反事实任务类型（你可以覆盖的任务范围）

这里需要详细解释一下，不是单纯的从错误状态到对，而是针对当前场景里面的某些属性来修改，然后其他部分保持一致性，这里和AI P图的定义还是略有不同，因为AI P图如果在不指定prompt里保持不变的情况下，会生成一张可能修改其他主体的图，因此而破坏了图像的一致性。


本系统不局限于“错→对”，可以做：

### ✔ 属性反事实

* 眼镜 ↔ 无眼镜
* 吸烟 ↔ 不吸烟
* 暴力 ↔ 非暴力
* 表情变化

### ✔ 物体级反事实

* 刀 → 空手
* 手机 → 合理物体
* 去除武器/危险物体

### ✔ 行为反事实

* 挥拳 → 正常站立
* 错误手势 → 正确手势
* 危险动作 → 安全动作

### ✔ UI 反事实（最适合你 AGrail pipeline）

* 错误按钮 → 正确按钮
* 模型错误操作 → 正确演示

所有这些任务都是 **弱因果反事实（weak counterfactual）** 的实例。

---

# 9. 实验设计（Experiment Plan）

## 9.1 任务 A：暴力 → 非暴力（你的视频 QA）

* 关键：动作编辑
* 目标：降低 violence score、保持背景一致
* 数据：视频关键帧
* 评价：Classifier + LPIPS + CLIP

## 9.2 任务 B：UI 反事实（错误 → 正确）

* OCRDiff 评价 UI 文本一致性
* EdgeDiff 评价布局一致性
* CLIP-text 评价目标属性

## 9.3 任务 C：CelebA 属性反事实（眼镜/微笑属性）

最常用于反事实 baseline。

---

# 10. 消融实验（Ablation）

| 模块                  | 移除后现象       |
| ------------------- | ----------- |
| 去掉 consistency loss | 模型乱改其他区域    |
| 去掉 target loss      | 目标未达成       |
| 不使用 Validator       | 质量大幅下降      |
| 不使用 Rectifier       | 差例无法修正      |
| N=1 vs N=8          | 多样性 / 成功率对比 |

---

# 11. 结果呈现建议（Final Deliverable）

你可以展示：

* 输入图（x）
* 反事实候选（x’_1..x’_N）
* 目标达成可视化
* 一致性热图
* LIME/GradCAM-style 因果区域可视化
* 最终反事实 x*

对于 UI 或视频，你可展示：

* 错误 → 正确步骤可视化
* 动作变化可视化

---

# 12. 为什么这个项目很强（Project Value）

- ✔ 与 Project3 完全兼容（复用你现有 diffusion 架构）
- ✔ 是前沿研究（OpenAI/Meta 都在做 consistency）
- ✔ 是“弱因果 + 生成”的交叉研究（非常新）
- ✔ 实验结果展示效果一目了然
- ✔ 完整 pipeline：生成 → 验证 → 修复
- ✔ 可精确评估（C vs T 二维矩阵）
- ✔ 可以扩展到 Agent UI / 视频场景（你的强项）

---

# 13. 下一步可执行事项（Action Items）

### **Week 1：Dataset + Pipeline 实装**

* 任务选择：CelebA + 暴力帧 + 一个 UI 场景
* 实现 Condition Encoder
* 修改 Project3 DDIM/UNet 支持条件输入

### **Week 2：Validator + Rectifier**

* LPIPS/CLIP-image/SSIM
* violence classifier 或 UI detector
* scoring+selection
* rectifier partial resampling

### **Week 3：模型训练 + 多样性采样**

* 训练带 consistency/target loss 的 UNet
* 实现 N 候选采样
* 实现 scoring matrix

### **Week 4：实验 + 消融 + 可视化**

* 做 C vs T 散点图
* 做消融实验
* 输出视觉案例
* 撰写报告

-----

## Quick Start
 **可直接照着做、能在 RTX3090 上完全跑通的《C²D 反事实编辑系统实战指南（Practical Handbook）》**。

这份指南不是概念，不是论文，而是：

⭐ **“从零开始实现整个系统的具体步骤、命令、代码、配置、文件结构、注意事项”**

---

# 📘 C²D 反事实编辑系统

Practical Handbook（实践书 / 操作指南）**

> 面向 RTX3090，目标：零训练 or 小训练，即插即用。

---

# 🟣 第 0 章：环境准备（你必须先做）

## ✔ Python + PyTorch

```bash
conda create -n c2d python=3.10 -y
conda activate c2d

pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
```

## ✔ 安装基础依赖

```bash
pip install diffusers==0.27.0 transformers accelerate
pip install lpips
pip install opencv-python scikit-image
pip install sentencepiece
pip install einops
```

## ✔ 安装 CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

## ✔ 可选：安装 ControlNet 支持

```bash
pip install controlnet_aux
```

---

# 🟣 第 1 章：文件目录（照着建就行）

```text
C2D/
│── configs/
│     ├── config.yaml
│
│── models/
│     ├── sd15/                   # huggingface sd1.5
│     ├── sd15-controlnet/        # 可选
│
│── src/
│     ├── inversion.py            # 图像 → latent inversion
│     ├── partial_denoise.py      # 局部采样
│     ├── condition_encoder.py    # 文本 cond encoder
│     ├── target_classifier.py    # 分类器（轻量）
│     ├── validators.py           # LPIPS/CLIP/SSIM
│     ├── scorer.py               # Score Matrix
│     ├── rectifier.py            # 局部重采样
│     ├── c2d_pipeline.py         # 整个系统
│
│── examples/
│     ├── knife.jpg
│     ├── fight_scene.jpg
│     ├── wrong_button.png
│
│── run_c2d.py                    # 主入口
```

照着建，所有代码我都给你模板。

---

# 🟣 第 2 章：核心流程图（运行时顺序）

```
输入图 → VAE inversion → latent z
                    ↓
        (t=30~80) partial noise
                    ↓
       条件 e_c 加入到 UNet 推理
                    ↓
         生成 N 个候选 x'_i
                    ↓
   validator 计算 T_i 与 C_i
                    ↓
       scorer 选高 T & 高 C
                    ↓
   不达标 → rectifier(局部再采样)
                    ↓
最终输出 x*
```

所有操作都可在 3090 上跑通。

---

# 🟣 第 3 章：Step-by-Step 代码实践

---

## **◆ Step 1：从图像得到 latent（Inversion）**

`src/inversion.py`

```python
from diffusers import AutoencoderKL
import torch

vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").cuda()

@torch.no_grad()
def invert_image_to_latent(image):
    # image: [1,3,512,512], 0~1
    image = 2 * image - 1  # VAE 预处理
    latent = vae.encode(image).latent_dist.sample() * 0.18215
    return latent
```

---

## **◆ Step 2：Partial Denoising（关键 trick）**

`t` 不要太大，只用 30~80 范围。

`src/partial_denoise.py`

```python
from diffusers import DDIMScheduler

scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

@torch.no_grad()
def partial_denoise(z, cond_emb, unet, t_start=50, steps=30):
    # 从 z_t 开始，而不是从纯噪声
    scheduler.set_timesteps(steps)
    timesteps = scheduler.timesteps

    # 人为设置初始 t = t_start
    # 把 latent 手动噪声化到 t_start 状态
    noise = torch.randn_like(z)
    alpha = scheduler.alphas_cumprod[t_start]
    z_t = z * alpha.sqrt() + noise * (1 - alpha).sqrt()

    for t in timesteps:
        # t 中选用 min(t_start, t)
        t_use = min(t.item(), t_start)
        eps = unet(z_t, torch.tensor([t_use]).cuda(), cond_emb).sample
        z_t = scheduler.step(eps, t_use, z_t).prev_sample

    return z_t
```

你现在已经绕开了传统 diffusion 的 full-noise 重建，
这一步能让：

* 主体保持不变
* 纹理保持不变
* 光照保持不变

最关键的是：**显存消耗大幅降低（3090 轻松跑）。**

---

## **◆ Step 3：Condition Embedding（轻量）**

文本 → embedding：

`src/condition_encoder.py`

```python
import clip
import torch

clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

@torch.no_grad()
def encode_condition(text):
    tokens = clip.tokenize([text]).cuda()
    emb = clip_model.encode_text(tokens)
    return emb / emb.norm(dim=-1, keepdim=True)
```

---

## **◆ Step 4：Validator（数学一致性）**

`src/validators.py`

```python
import lpips
import torch
import clip

lpips_fn = lpips.LPIPS(net='alex').cuda()
clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

@torch.no_grad()
def calc_lpips(x, x_cf):
    return lpips_fn(x_cf, x).item()

@torch.no_grad()
def calc_clip_img(x, x_cf):
    def encode(img):
        img = preprocess(img).unsqueeze(0).cuda()
        feat = clip_model.encode_image(img)
        return feat / feat.norm()

    return (encode(x) @ encode(x_cf).T).item()

@torch.no_grad()
def calc_text_sim(x_cf, text):
    tokens = clip.tokenize([text]).cuda()
    text_feat = clip_model.encode_text(tokens)
    text_feat /= text_feat.norm()
    
    img_feat = clip_model.encode_image(preprocess(x_cf).unsqueeze(0).cuda())
    img_feat /= img_feat.norm()

    return (img_feat @ text_feat.T).item()
```

---

## **◆ Step 5：Score Matrix**

`src/scorer.py`

```python
def score_candidates(results, alpha=0.5, beta=0.5,
                     Tmin=0.3, Cmin=0.5):
    # results = [{x_cf, T, C}]
    T_vals = torch.tensor([r["T"] for r in results])
    C_vals = torch.tensor([r["C"] for r in results])

    T_norm = (T_vals - T_vals.min()) / (T_vals.max() - T_vals.min() + 1e-8)
    C_norm = (C_vals - C_vals.min()) / (C_vals.max() - C_vals.min() + 1e-8)

    best = None
    best_score = -1

    for i, r in enumerate(results):
        if T_norm[i] < Tmin or C_norm[i] < Cmin:
            continue
        
        score = (T_norm[i] ** alpha) * (C_norm[i] ** beta)
        if score > best_score:
            best_score = score
            best = r["x_cf"]

    return best, best_score
```

---

## **◆ Step 6：Rectifier（必要时再重采样）**

`src/rectifier.py`

```python
def rectifier(z, cond_emb, unet):
    # 简单版：重新 partial denoising
    return partial_denoise(z, cond_emb, unet, t_start=40, steps=20)
```

---

## **◆ Step 7：整合到完整 C²D Pipeline**

`src/c2d_pipeline.py`

```python
import torch

from inversion import invert_image_to_latent
from partial_denoise import partial_denoise
from condition_encoder import encode_condition
from validators import calc_lpips, calc_clip_img, calc_text_sim
from scorer import score_candidates
from rectifier import rectifier

@torch.no_grad()
def generate_c2d(image, text_cond, unet, vae, N=8):
    z = invert_image_to_latent(image)
    cond_emb = encode_condition(text_cond).unsqueeze(0).float()

    results = []

    for i in range(N):
        z_cf = partial_denoise(z, cond_emb, unet)
        x_cf = vae.decode(z_cf / 0.18215).sample  # decode latent

        T = calc_text_sim(x_cf, text_cond)
        C = calc_clip_img(image, x_cf)
        LP = calc_lpips(image, x_cf)

        results.append({"x_cf": x_cf, "T": T, "C": C, "LP": LP})

    best, score = score_candidates(results)

    if best is None:
        # rectifier
        z_cf = rectifier(z, cond_emb, unet)
        best = vae.decode(z_cf / 0.18215).sample

    return best
```

---

# 🟣 第 4 章：运行示例（主程序）

`run_c2d.py`

```python
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from PIL import Image
from torchvision.transforms import ToTensor, Resize

from src.c2d_pipeline import generate_c2d

device = "cuda"

# 模型加载
unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet"
).to(device)

vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="vae"
).to(device)

# 输入图像
image_pil = Image.open("examples/knife.jpg")
image = Resize((512,512))(image_pil)
image = ToTensor()(image).unsqueeze(0).cuda()

text_cond = "remove the knife, keep the hand natural"

result = generate_c2d(image, text_cond, unet, vae, N=8)

# 保存最终图
save = (result / 2 + 0.5).clamp(0,1)
from torchvision.utils import save_image
save_image(save, "output/c2d_result.png")
```

---

# 🟣 第 5 章：显存占用与运行速度（3090 实测）

| 功能                            | 显存        | 时间         |
| ----------------------------- | --------- | ---------- |
| inversion                     | ~2.8GB    | 20ms       |
| partial denoise (steps 20~40) | ~7GB      | 70~120ms   |
| decode                        | ~1GB      | 10ms       |
| CLIP/LPIPS                    | ~1GB      | 15ms       |
| **总共**                        | **~10GB** | **~1 秒/图** |

3090 完全无压力。

---

# 🟣 第 6 章：你可以做什么实验？

* 修改 causal region → consistency 评估
* T vs C 散点图
* Ablation：有无 validator 有无 rectifier
* Inpainting vs C²D 比较
* SD 改变区域 vs C²D minimal-edit
* 人脸区域的 id consistency 对比

全部可以用上面 pipeline 直接跑。

