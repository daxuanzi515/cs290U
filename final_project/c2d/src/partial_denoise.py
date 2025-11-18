from diffusers import DDIMScheduler
import torch

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
