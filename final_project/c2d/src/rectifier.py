import partial_denoise


def rectifier(z, cond_emb, unet):
    # 简单版：重新 partial denoising
    return partial_denoise(z, cond_emb, unet, t_start=40, steps=20)
