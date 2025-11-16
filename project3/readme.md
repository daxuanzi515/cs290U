# RECORDS
Only for project 3
----
# Setup
Follow [TA_Helper](TA_Helper.md)

# Basic Task
## 2.1 kl-weight analysis
命令只修改kl-weight, 其他设置为默认值：
```shell
cd project3
python train_vae.py \
	--data-dir data/MNIST \
	--output-dir result/vae \
	--epochs 40 \
	--batch-size 128 \
	--lr 2e-4 \
	--latent-dim 100 \
	--kl-weight 1 \
	--loss-type mse \
	--num-workers 4 \
	--seed 42 \
	--save-recon                # add --no-save-recon to disable
```
beta分别设置为0.1, 0.5, 1.0, 5.0测试 （0.1~1应该最合理）
```shell
# 0.1
python train_vae.py --epochs 40 --kl-weight 0.1 --output-dir results/vae_beta_0.1
# 0.5 quick_start already done
python train_vae.py --epochs 40 --kl-weight 0.5 --output-dir results/vae_beta_0.5
# 1.0
python train_vae.py --epochs 40 --kl-weight 1.0 --output-dir results/vae_beta_1.0
# 5.0
python train_vae.py --epochs 40 --kl-weight 5.0 --output-dir results/vae_beta_5.0
```
选取第40个epoch来分析：

| β 值 | 重建图像质量 | 观察 | 重建图 | 采样图 |
|------|--------------|------|------|------|
| 0.1  | 清晰         | 潜空间不规则，容易过拟合 | ![](results/vae_beta_0.1/recon_epoch_040.png)    | ![](results/vae_beta_0.1/samples_epoch_040.png)   |
| 0.5  | 平衡         | 最佳重建质量和采样平衡 | ![](results/vae_beta_0.5/recon_epoch_040.png)    | ![](results/vae_beta_0.5/samples_epoch_040.png)   |
| 1.0  | 稍模糊       | 正则化开始强，重建模糊 | ![](results/vae_beta_1.0/recon_epoch_040.png)    | ![](results/vae_beta_1.0/samples_epoch_040.png)   |
| 5.0  | 模糊         | 强正则化，丢失细节 | ![](results/vae_beta_5.0/recon_epoch_040.png)    | ![](results/vae_beta_5.0/samples_epoch_040.png)   |

- β=0.1：重建图像非常清晰，但采样图像缺乏多样性，潜空间正则性不足。

- β=0.5：提供了最佳的平衡，重建质量和生成样本质量都很不错。

- β=1.0：潜空间正则性增强，但重建图像有轻微模糊，采样结果可能缺乏多样性。

- β=5.0：过度正则化，导致重建图像模糊，生成的采样图质量差。

---

VAE 的总损失：
$$
L = \text{ReconLoss} + \beta \cdot D_{KL}(q_\phi(z \mid x) \parallel \mathcal{N}(0, I))
$$

当 $β$ 太小 (<0.1), $KL$项几乎不起作用; 潜空间不服从标准正态分布；重建清晰，但随机采样图像容易崩掉（因为潜空间不连续）。

当$β$太大 (>2), 模型被迫强制正则化；潜空间“塌缩”（encoder输出接近0）；重建模糊、特征丢失，但采样更“均匀”。

因此，在实践中
$β$ ≈ 0.3–0.7 → 最稳定；
$β$ ≈ 1.0 → 理论标准；
$β$ > 3.0 → 仅用于展示过度正则现象。

## 2.2 DDPM and DDIM
在模块`project3/HW3/diffusion.py`120行添加完整内容。

```shell
cd project3
python train_diffusion.py \
	--data_dir ./data \
	--batch_size 128 \
	--epochs 20 \
	--lr 1e-4 \
	--timesteps 1000 \
	--ddim_steps 200 \
	--ddim_eta 0.0 \
	--out_dir results/diffusion
```
Loss:
```shell
Epoch 001 | train_loss: 0.0768
Epoch 002 | train_loss: 0.0286
Epoch 003 | train_loss: 0.0248
Epoch 004 | train_loss: 0.0228
Epoch 005 | train_loss: 0.0217
Epoch 006 | train_loss: 0.0208
Epoch 007 | train_loss: 0.0197
Epoch 008 | train_loss: 0.0193
Epoch 009 | train_loss: 0.0190
Epoch 010 | train_loss: 0.0186
Epoch 011 | train_loss: 0.0181
Epoch 012 | train_loss: 0.0181
Epoch 013 | train_loss: 0.0177
Epoch 014 | train_loss: 0.0173
Epoch 015 | train_loss: 0.0171
Epoch 016 | train_loss: 0.0171
Epoch 017 | train_loss: 0.0170
Epoch 018 | train_loss: 0.0168
Epoch 019 | train_loss: 0.0165
Epoch 020 | train_loss: 0.0167
```

## 2.3 Latent Space
VAE-beta=0.5
```shell
python train_latent_diffusion.py \
	--data_dir ./data \
	--vae_ckpt results/vae_beta_0.5/convae_latest.pt \
	--epochs 40 \
	--lr 1e-4 \
	--timesteps 1000 \
	--ddim_steps 200 \
	--ddim_eta 0.0 \
	--out_dir results/latent_diffusion_0.5 \
	--canonicalize 
```
Loss:
```shell
Epoch 001 | latent train_loss: 0.5907
Epoch 002 | latent train_loss: 0.3140
Epoch 003 | latent train_loss: 0.2809
Epoch 004 | latent train_loss: 0.2805
Epoch 005 | latent train_loss: 0.2765
Epoch 006 | latent train_loss: 0.2804
Epoch 007 | latent train_loss: 0.2786
Epoch 008 | latent train_loss: 0.2770
Epoch 009 | latent train_loss: 0.2799
Epoch 010 | latent train_loss: 0.2805
Epoch 011 | latent train_loss: 0.2788
Epoch 012 | latent train_loss: 0.2786
Epoch 013 | latent train_loss: 0.2795
Epoch 014 | latent train_loss: 0.2806
Epoch 015 | latent train_loss: 0.2787
Epoch 016 | latent train_loss: 0.2805
Epoch 017 | latent train_loss: 0.2754
Epoch 018 | latent train_loss: 0.2783
Epoch 019 | latent train_loss: 0.2813
Epoch 020 | latent train_loss: 0.2800
Epoch 021 | latent train_loss: 0.2781
Epoch 022 | latent train_loss: 0.2805
Epoch 023 | latent train_loss: 0.2776
Epoch 024 | latent train_loss: 0.2796
Epoch 025 | latent train_loss: 0.2767
Epoch 026 | latent train_loss: 0.2798
Epoch 027 | latent train_loss: 0.2791
Epoch 028 | latent train_loss: 0.2769
Epoch 029 | latent train_loss: 0.2774
Epoch 030 | latent train_loss: 0.2804
Epoch 031 | latent train_loss: 0.2791
Epoch 032 | latent train_loss: 0.2775
Epoch 033 | latent train_loss: 0.2804
Epoch 034 | latent train_loss: 0.2781
Epoch 035 | latent train_loss: 0.2770
Epoch 036 | latent train_loss: 0.2787
Epoch 037 | latent train_loss: 0.2806
Epoch 038 | latent train_loss: 0.2767
Epoch 039 | latent train_loss: 0.2795
Epoch 040 | latent train_loss: 0.2798
```

VAE-beta=1.0
```shell
python train_latent_diffusion.py \
	--data_dir ./data \
	--vae_ckpt results/vae_beta_1.0/convae_latest.pt \
	--epochs 40 \
	--lr 1e-4 \
	--timesteps 1000 \
	--ddim_steps 200 \
	--ddim_eta 0.0 \
	--out_dir results/latent_diffusion_1.0 \
	--canonicalize 
```
Loss:
```shell
Epoch 001 | latent train_loss: 0.5909
Epoch 002 | latent train_loss: 0.3142
Epoch 003 | latent train_loss: 0.2812
Epoch 004 | latent train_loss: 0.2807
Epoch 005 | latent train_loss: 0.2767
Epoch 006 | latent train_loss: 0.2807
Epoch 007 | latent train_loss: 0.2790
Epoch 008 | latent train_loss: 0.2773
Epoch 009 | latent train_loss: 0.2800
Epoch 010 | latent train_loss: 0.2806
Epoch 011 | latent train_loss: 0.2790
Epoch 012 | latent train_loss: 0.2789
Epoch 013 | latent train_loss: 0.2797
Epoch 014 | latent train_loss: 0.2809
Epoch 015 | latent train_loss: 0.2788
Epoch 016 | latent train_loss: 0.2808
Epoch 017 | latent train_loss: 0.2757
Epoch 018 | latent train_loss: 0.2786
Epoch 019 | latent train_loss: 0.2815
Epoch 020 | latent train_loss: 0.2803
Epoch 021 | latent train_loss: 0.2784
Epoch 022 | latent train_loss: 0.2808
Epoch 023 | latent train_loss: 0.2778
Epoch 024 | latent train_loss: 0.2799
Epoch 025 | latent train_loss: 0.2770
Epoch 026 | latent train_loss: 0.2800
Epoch 027 | latent train_loss: 0.2792
Epoch 028 | latent train_loss: 0.2772
Epoch 029 | latent train_loss: 0.2775
Epoch 030 | latent train_loss: 0.2806
Epoch 031 | latent train_loss: 0.2793
Epoch 032 | latent train_loss: 0.2777
Epoch 033 | latent train_loss: 0.2806
Epoch 034 | latent train_loss: 0.2784
Epoch 035 | latent train_loss: 0.2772
Epoch 036 | latent train_loss: 0.2789
Epoch 037 | latent train_loss: 0.2808
Epoch 038 | latent train_loss: 0.2771
Epoch 039 | latent train_loss: 0.2797
Epoch 040 | latent train_loss: 0.2799
```
VAE-beta=0.5 is better.
# Advanced Task
## 3.1 Tiny MNIST Stable Diffusion (tiny_minist_sd)
先训练clip之后再训练tiny_sd.

Clip: 
```shell
python train_clip.py \
	--data_dir ./data \
	--epochs 10 \
	--batch_size 256 \
	--embed_dim 128 \
	--lr 1e-3 \
	--weight_decay 1e-4 \
	--out_dir results/clip

python train_clip.py \
	--data_dir ./data \
	--epochs 50 \
	--batch_size 256 \
	--embed_dim 128 \
	--lr 1e-3 \
	--weight_decay 1e-4 \
	--out_dir results/clip_50
```
Results:
```shell
Epoch 001 | train_loss: 0.4007 | val_acc: 0.9492
Epoch 002 | train_loss: 0.1084 | val_acc: 0.9658
Epoch 003 | train_loss: 0.0753 | val_acc: 0.9708
Epoch 004 | train_loss: 0.0573 | val_acc: 0.9718
Epoch 005 | train_loss: 0.0459 | val_acc: 0.9720
Epoch 006 | train_loss: 0.0398 | val_acc: 0.9718
Epoch 007 | train_loss: 0.0314 | val_acc: 0.9798
Epoch 008 | train_loss: 0.0271 | val_acc: 0.9762
Epoch 009 | train_loss: 0.0239 | val_acc: 0.9773
Epoch 010 | train_loss: 0.0252 | val_acc: 0.9755
Test | acc: 0.9810

...
Epoch 045 | train_loss: 0.0127 | val_acc: 0.9775
Epoch 046 | train_loss: 0.0067 | val_acc: 0.9842
Epoch 047 | train_loss: 0.0026 | val_acc: 0.9848
Epoch 048 | train_loss: 0.0009 | val_acc: 0.9858
Epoch 049 | train_loss: 0.0068 | val_acc: 0.9688
Epoch 050 | train_loss: 0.0127 | val_acc: 0.9832
Test | acc: 0.9857
```

Tiny_SD: 
```shell
python train_tiny_sd.py \
	--data_dir ./data \
	--clip_ckpt results/clip/clip_latest.pt \
	--embed_dim 128 \
	--epochs 40 \
	--lr 2e-4 \
	--timesteps 1000 \
	--ddim_steps 50 \
	--ddim_eta 0.0 \
	--out_dir results/tiny_sd
```

Results:
```shell
Epoch 001 | train_loss: 0.4686
Epoch 002 | train_loss: 0.2827
Epoch 003 | train_loss: 0.2787
Epoch 004 | train_loss: 0.2817
Epoch 005 | train_loss: 0.2800
Epoch 006 | train_loss: 0.2794
Epoch 007 | train_loss: 0.2790
Epoch 008 | train_loss: 0.2797
Epoch 009 | train_loss: 0.2792
Epoch 010 | train_loss: 0.2815
Epoch 011 | train_loss: 0.2787
Epoch 012 | train_loss: 0.2798
Epoch 013 | train_loss: 0.2782
Epoch 014 | train_loss: 0.2805
Epoch 015 | train_loss: 0.2800
Epoch 016 | train_loss: 0.2802
Epoch 017 | train_loss: 0.2795
Epoch 018 | train_loss: 0.2792
Epoch 019 | train_loss: 0.2786
Epoch 020 | train_loss: 0.2801
Epoch 021 | train_loss: 0.2793
Epoch 022 | train_loss: 0.2785
Epoch 023 | train_loss: 0.2785
Epoch 024 | train_loss: 0.2790
Epoch 025 | train_loss: 0.2820
Epoch 026 | train_loss: 0.2797
Epoch 027 | train_loss: 0.2779
Epoch 028 | train_loss: 0.2789
Epoch 029 | train_loss: 0.2782
Epoch 030 | train_loss: 0.2790
Epoch 031 | train_loss: 0.2787
Epoch 032 | train_loss: 0.2771
Epoch 033 | train_loss: 0.2800
Epoch 034 | train_loss: 0.2766
Epoch 035 | train_loss: 0.2774
Epoch 036 | train_loss: 0.2799
Epoch 037 | train_loss: 0.2771
Epoch 038 | train_loss: 0.2788
Epoch 039 | train_loss: 0.2761
Epoch 040 | train_loss: 0.2768
```

- 查看这次训练的采样图，发现模糊而且识别不清：
![](results/tiny_sd_ddpm/samples_epoch_040.png)

- 修改diffusion的代码之后纠正就可以了：
![](results/tiny_sd/samples_epoch_040.png)


写一个0-shot来检测一下训练的clip模型：
```shell
python eval_clip.py \
  --data_dir ./data \
  --clip_ckpt ./results/clip_shot/clip_latest.pt \
  --embed_dim 128 \
  --batch_size 128

# results
Validation accuracy: 0.9840
Zero-shot accuracy: 0.1562
```
但是发现准确率很低，模型没有学到语义对齐的特征，只是学到像素映射关系；
因此在采样的图里做不到竖列都是一样的数字的图的效果，只能做到乱七八糟数字的图：
![](results/tiny_sd/samples_epoch_040.png)

我怀疑我代码写错了，然后修改了一下：

```python
# train_tiny_sd.py:
time_emb = self.sinusoidal_time_embedding(t, self.time_dim)
# 将时间嵌入和条件嵌入与潜在向量 x 进行拼接
h = torch.cat([x, time_emb, cond], dim=1)  # (B, D + time_dim + cond_dim)


# diffision.py:
            eps_pred = self.model(img, t, cond) if cond is not None else self.model(img, t)
            x0_pred = (img - torch.sqrt(1.0 - a_bar_t) * eps_pred) / torch.sqrt(a_bar_t)

            if i == 0:
                # 最后一步直接到 x0
                img = x0_pred
                break

            # 上一个子时间步（而非 t-1）
            t_prev = ts[i - 1].expand(batch_size)
            a_bar_prev = self.alpha_bars[t_prev].view(-1, 1, 1, 1).clamp(1e-12, 1.0)

            # DDIM sigma：eta * sqrt((1-ā_prev)/(1-ā_t) * (1 - ā_t/ā_prev))
            sigma_t = eta * torch.sqrt(
                (1.0 - a_bar_prev) / (1.0 - a_bar_t) * (1.0 - a_bar_t / a_bar_prev)
            ).clamp_min(0.0)

            # 均值项：确定性部分 + 残差项（与 eps_pred 同向）
            # 注意：当 eta=0 时，第二项与第三项（噪声）都应消失
            c = torch.sqrt(1.0 - a_bar_prev - sigma_t ** 2).clamp_min(0.0)
            mean = torch.sqrt(a_bar_prev) * x0_pred + c * eps_pred  
```

40轮里选出2张比较满意的图：

![](candidates/40epch/samples_epoch_023.png)

![](candidates/40epch/samples_epoch_024.png)

200轮里选出其中三张比较满意的图：

![](candidates/samples_epoch_186.png)

![](candidates/samples_epoch_194.png)

![](candidates/samples_epoch_195.png)

## 3.2 ComfyUI (exploration)

1. 安装 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)。
2. 构建至少两种 text-to-image 流程并生成图像。
3. 可尝试不同的 LoRA 或采样器并分析失败原因。

### ENV
```shell
cd ComfyUI
conda activate cs290u_hw3
pip install -r requirements.txt
python main.py --listen 0.0.0.0 --port 8188
```
访问：http://127.0.0.1:8188

权重放哪里？这个是结构：
```
ComfyUI/
├── models/
│   ├── checkpoints/        ← 主扩散模型（Stable Diffusion / SDXL）
│   ├── vae/                ← VAE 解码模型
│   ├── clip/               ← CLIP 文本编码器
│   ├── loras/              ← LoRA 权重
│   ├── embeddings/         ← Textual Inversion 嵌入（.pt / .bin）
│   ├── controlnet/         ← ControlNet 模型
│   ├── upscalers/          ← 放超分模型（如 ESRGAN）
│   └── style_models/       ← 风格模型（部分节点使用）
```

这里的位置映射，假设你当前路径为 project3/：
```shell
cp -r results/clip* ComfyUI/models/clip/
cp -r results/vae_beta_* ComfyUI/models/vae/
cp -r results/diffusion ComfyUI/models/checkpoints/
cp -r results/latent_diffusion_* ComfyUI/models/checkpoints/
cp -r results/tiny_sd* ComfyUI/models/checkpoints/
```

### Customized Nodes Manager for ComfyUI
直接用官方的VAE节点导入权重，发现我们的vae的权重不被接受，尝试安装`nodes_manager`来自定义节点，否则按照官方定义的VAE我们是无法自适应的。

![](imgs/2025-11-14%2017-52-29屏幕截图.png)


参考链接：https://www.uisdc.com/comfyui-3

下载Linux版：https://github.com/Comfy-Org/ComfyUI-Manager

手动按脚本操作：
```shell
cd ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager
pip install -r custom_nodes/comfyui-manager/requirements.txt
cd ..
python main.py --listen 0.0.0.0 --port 8188 
```
成功的看到如图1变为图2：
![图1](imgs/2025-11-13%2019-56-18屏幕截图.png)
![图2](imgs/2025-11-14%2017-48-09屏幕截图.png)

根据指南安装汉化节点。

### 自定义插件

**之后是自定义节点的部分，可以在`custom_nodes`里找到一个示例`example_node.py.example`, 我们根据自己的VAE需要编写一个新的节点插件来用于专门导入特殊权重。**

参考资料: 
- https://zhuanlan.zhihu.com/p/7033713672
- https://github.com/liubai-liubai/ComfyUI-ImgSeg-LB
- https://github.com/Pal-dont-want-to-work/comfyui-custom_nodes-tutorial
- https://h0zkh0f8v2a.feishu.cn/wiki/KUnlwgJxSidQi7k0Iq3cUvGpnLS

### 使用默认模版构建工作流
v1-5-pruned-emaonly-fp16.safetensors: 
https://cas-bridge.xethub.hf.co/xet-bridge-us/66d0dccb3866d4d3087d3a9f/908c39bfdfec888e295ba04e974b6342f3c15776760edd46838240a8d455525d?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20251116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251116T101354Z&X-Amz-Expires=3600&X-Amz-Signature=aa1135302f3f18f3314fa40099f8787613a3b89ee19e1f8a4ed157153b44546d&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=684fe4ca662a7bb7a96e6b80&response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27v1-5-pruned-emaonly-fp16.safetensors%3B+filename%3D%22v1-5-pruned-emaonly-fp16.safetensors%22%3B&x-id=GetObject&Expires=1763291634&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MzI5MTYzNH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82NmQwZGNjYjM4NjZkNGQzMDg3ZDNhOWYvOTA4YzM5YmZkZmVjODg4ZTI5NWJhMDRlOTc0YjYzNDJmM2MxNTc3Njc2MGVkZDQ2ODM4MjQwYThkNDU1NTI1ZCoifV19&Signature=AvFib5-D1NPg0GyJxPt7MyjQuVGZQ9I9P7C7z-2AeqEdVzpCrIvrpSsoFjoorSkx3M9UgmMwVovLHaOVNi7NNphnsMzExcfxTyN7ZaT1W5zvFquukSsjQhJb5q-4bBFhSzspvvts-JLYvlv9exHH86KBmMVm0blzKH3CbnXEtJc2ZHQDYC7darsUUKLmJzkjhMEUHPfVqOHwQW5ENh5b8IhAoCc36d0wUHH0UpbE1Ab%7EyFFbv4qE7yW0%7EcPz0h0B7yV78xkhdLthJ2Afh6ZLG9FMpnq6z5AbabS%7EaF6BjjTgxVrjBJpd4erbrrSIkqnzHlxzPusgzOkGcm-Rw5r56g__&Key-Pair-Id=K2L8F4GPSG1IFC


dreamshaper_8.safetensors: 
https://civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com/53515/model/dreamshaper8Pruned.hz5Q.safetensors?X-Amz-Expires=86400&response-content-disposition=attachment%3B%20filename%3D%22dreamshaper_8.safetensors%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=e01358d793ad6966166af8b3064953ad/20251116/us-east-1/s3/aws4_request&X-Amz-Date=20251116T101151Z&X-Amz-SignedHeaders=host&X-Amz-Signature=c1135793233f651111e90e2facbfc69b79fdb14617997a7c5ad22e7b120904f9

Mix_V1:
https://civitai.com/api/download/models/14856?type=Model&format=SafeTensor&size=full&fp=fp16

**Default**

成功导入模型之后运行第一个工作流default:

使用初始例子：![](./imgs/default_test_0.png)

positive prompt:
`generate 8*10 matrix, from zero to nine numbers, a black and white image.`
negative prompt:
`people and object, blur`

跑结果：![](./imgs/default_test_02_SD1.5_00003_.png)

提示词不够强，更换提示词：

`Black-and-white bitmap, 8×10 matrix of Arabic numerals 0-9, each column contains 8 different digits without repetition, all cells clearly legible, high-contrast white background black digits, no shading no color no grid lines, crisp sans-serif font, 300 dpi, pure monochrome, 1-bit depth`

`No color, no grayscale, no grid, no frame, no shadow, no anti-aliasing, no blur, no distortion, no repetition within any column, no decorative elements`

跑结果：
- ![](./imgs/default_test_03_SD1.5_00031_.png)
- ![](./imgs/default_test_04_SD1.5_00035_.png)

虽然生成了数字但是没有完全按照要求生成，原因是采用的Comfy的Default使用的是v1-5-pruned-emaonly-fp16.safetensors，专注于文生图，但是生成的图像集中于彩色和多样性，并不擅长生成准确的像素数字的黑白图像。


**Lora Multiple**:

![](imgs/lora_multi_test_0.png)

发现lora强引导生成人，以下几张勉强可用：
- ![](./imgs/2loras_test__00014_.png)
- ![](./imgs/2loras_test__00111_.png)
- ![](./imgs/2loras_test__00140_.png)
- ![](./imgs/2loras_test__00148_.png)

loras允许使用多个不一样风格的loras模型，但是这些loras模型大部分专注生成动漫人物的各种风格，比如古风、宫崎骏、东京塔、星空、日系、日系人物等等，并不擅长生成像素数字的黑白图像。而且无论你如何调整提示词，它其实都默认你生成的对象包含人物在内的元素，这样就算在negative提示词编写了不允许人的元素，基本还是会生成人物相关的图像。
