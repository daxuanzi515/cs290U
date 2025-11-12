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
Epoch 002 | train_loss: 0.0289
Epoch 003 | train_loss: 0.0248
Epoch 004 | train_loss: 0.0227
Epoch 005 | train_loss: 0.0214
Epoch 006 | train_loss: 0.0205
Epoch 007 | train_loss: 0.0201
Epoch 008 | train_loss: 0.0193
Epoch 009 | train_loss: 0.0189
Epoch 010 | train_loss: 0.0186
Epoch 011 | train_loss: 0.0182
Epoch 012 | train_loss: 0.0180
Epoch 013 | train_loss: 0.0176
Epoch 014 | train_loss: 0.0173
Epoch 015 | train_loss: 0.0172
Epoch 016 | train_loss: 0.0169
Epoch 017 | train_loss: 0.0169
Epoch 018 | train_loss: 0.0168
Epoch 019 | train_loss: 0.0168
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
## Tiny MNIST Stable Diffusion (tiny_minist_sd)
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