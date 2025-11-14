# export_comfy_weights.py
import torch, json
from pathlib import Path
from safetensors.torch import save_file

root   = Path("./results")          # 训练输出根目录
out_dir= Path("./comfy_weights")    # 导出目录
out_dir.mkdir(exist_ok=True)

# # 1. UNet（扩散模型）
# ckpt = torch.load(root/"tiny_sd"/"tiny_sd_latest.pt", map_location="cpu")
# save_file(ckpt["model_state"],  out_dir/"tiny_sd_unet.safetensors")
# print("✔ UNet  →  tiny_sd_unet.safetensors")

# # 2. CLIP
# ckpt = torch.load(root/"clip"/"clip_latest.pt", map_location="cpu")
# save_file(ckpt["model_state"],  out_dir/"tiny_sd_clip.safetensors")
# print("✔ CLIP  →  tiny_sd_clip.safetensors")

# 3. VAE（前面已导出的可直接用；这里再转一次确保纯净）
# ckpt = torch.load(root/"vae_beta_0.5"/"convae_best.pt", map_location="cpu")
# sd   = ckpt["model_state"]
# # 把 1→4 通道 & key 改成 SD 风格（和前面脚本一致）
# import copy
# out_sd = {}
# for k,v in sd.items():
#     if k=="encoder.0.weight":
#         out_sd["encoder.conv_in.weight"] = v.repeat(1,4,1,1)/4.0
#     elif k=="encoder.0.bias":
#         out_sd["encoder.conv_in.bias"]   = v
#     elif k=="decoder.last.weight":
#         out_sd["decoder.conv_out.weight"]= v.repeat(4,1,1,1)
#     elif k=="decoder.last.bias":
#         out_sd["decoder.conv_out.bias"]  = v.repeat(4)
#     else:
#         out_sd[k] = v
# # 必须补的占位
# out_sd["quant_conv.weight"]      = torch.zeros((32,32,1,1))
# out_sd["quant_conv.bias"]        = torch.zeros(32)
# out_sd["post_quant_conv.weight"] = torch.zeros((32,4,1,1))
# out_sd["post_quant_conv.bias"]   = torch.zeros(4)

# save_file(out_sd, out_dir/"tiny_sd_vae.safetensors")
# print("✔ VAE   →  tiny_sd_vae.safetensors")
# print(f"完成！请把 {out_dir} 里的三个文件分别放到对应 models 子目录即可")



from safetensors.torch import save_file
import torch
save_file({
    "encoder.conv_in.weight": torch.randn(128, 4, 3, 3),
    "encoder.conv_in.bias":   torch.randn(128),
    "decoder.conv_out.weight":torch.randn(4, 128, 3, 3),
    "decoder.conv_out.bias":  torch.randn(4),
    "quant_conv.weight":      torch.randn(32, 32, 1, 1),
    "quant_conv.bias":        torch.randn(32),
    "post_quant_conv.weight": torch.randn(32, 4, 1, 1),
    "post_quant_conv.bias":   torch.randn(4),
}, "dummy_vae.safetensors")