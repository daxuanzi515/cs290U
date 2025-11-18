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
