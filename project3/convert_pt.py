import torch
from HW3.vae import ConvVAE

ckpt = torch.load("./results/vae_beta_0.5/convae_latest.pt", map_location="cpu")

latent_dim = ckpt.get("args", {}).get("latent_dim", 100)
vae = ConvVAE(latent_dim=latent_dim)
vae.load_state_dict(ckpt["model_state"], strict=True)

# === 模拟 Stable Diffusion VAE 接口 ===
# ComfyUI 期望以下字段存在：
# {"encoder": {...}, "decoder": {...}}
vae_for_comfy = {
    "encoder": vae.encoder.state_dict() if hasattr(vae, "encoder") else vae.state_dict(),
    "decoder": vae.decoder.state_dict() if hasattr(vae, "decoder") else vae.state_dict(),
}

out_path = "./ComfyUI/models/vae/vae_tiny_sd_compatible.pt"
torch.save(vae_for_comfy, out_path)
print(f"✅ 已包装为兼容格式: {out_path}")
