from diffusers import AutoencoderKL
import torch

vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").cuda()

@torch.no_grad()
def invert_image_to_latent(image):
    # image: [1,3,512,512], 0~1
    image = 2 * image - 1  # VAE 预处理
    latent = vae.encode(image).latent_dist.sample() * 0.18215
    return latent
