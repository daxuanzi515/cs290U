#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a ConvVAE on MNIST and export ComfyUI-compatible VAE every epoch.
Compatible with the original HW3 folder structure.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from safetensors.torch import save_file   # 新增 1
import re

import torchvision                                 # 新增 2（备用）

# --------------- 你的原 import -----------------
from HW3.vae import ConvVAE, vae_loss, save_random_samples
from HW3.data import get_mnist_dataloaders
# ---------------------------------------------


# =========================================================
# 工具：把 MNIST 单通道权重 -> SD 4 通道兼容形状 + 官方 key 名
# =========================================================
def convert_to_sd_shape(state_dict: dict) -> dict:
    """
    1. 将首层/末层 Cin/Cout 由 1 扩到 4（repeat 并归一化，防止数值爆炸）
    2. 添加 SD VAE 必须的 quant_conv / post_quant_conv 空层
    3. key 名保持 encoder.x.x / decoder.x.x 即可，ComfyUI 会自动识别
    """
    new_sd = {}
    for k, v in state_dict.items():
        # 首层卷积
        if k == "encoder.0.weight":
            v = v.repeat(1, 4, 1, 1) / 4.0
        # 末层卷积
        if k == "decoder.last.weight":
            v = v.repeat(4, 1, 1, 1)
        if k == "decoder.last.bias":
            v = v.repeat(4)
        new_sd[k] = v

    # 占位层（形状与 SD 官方 VAE 一致，内容为空）
    new_sd["quant_conv.weight"]        = torch.zeros((32, 32, 1, 1))
    new_sd["quant_conv.bias"]          = torch.zeros(32)
    new_sd["post_quant_conv.weight"]   = torch.zeros((32, 4, 1, 1))
    new_sd["post_quant_conv.bias"]     = torch.zeros(4)
    return new_sd


# =========================================================
# 原封不动的 save_recon_only
# =========================================================
@torch.no_grad()
def save_recon_only(
    model: ConvVAE,
    device: torch.device,
    out_dir: Path,
    epoch: int,
    batch: torch.Tensor,
    n_show: int = 64,
    nrow: int = 8,
):
    model.eval()
    x = batch[:n_show].to(device)
    recon, _, _ = model(x)
    grid = torchvision.utils.make_grid(recon.cpu(), nrow=nrow, padding=2)
    out_path = out_dir / f"recon_epoch_{epoch:03d}.png"
    torchvision.utils.save_image(grid, out_path)
    return out_path


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Train ConvVAE on MNIST & export ComfyUI VAE")
    parser.add_argument("--data-dir", type=str, default="data/MNIST")
    parser.add_argument("--output-dir", type=str, default="results/vae")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--kl-weight", type=float, default=1)
    parser.add_argument("--loss-type", type=str, choices=["bce", "mse"], default="mse")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--save-recon", dest="save_recon", action="store_true", default=True)
    parser.add_argument("--no-save-recon", dest="save_recon", action="store_false")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, _, _ = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.0,
        download=True,
        resize_to_32=True,
    )

    model = ConvVAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_rec, total_kl, total_count = 0.0, 0.0, 0.0, 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss, rec, kl = vae_loss(recon, data, mu, logvar,
                                     kl_weight=args.kl_weight, loss_type=args.loss_type)
            loss.backward()
            optimizer.step()

            bs = data.size(0)
            total_loss += loss.item() * bs
            total_rec  += rec * bs
            total_kl   += kl * bs
            total_count += bs

        # ---------- 采样 ----------
        sample_path = save_random_samples(
            model=model, device=device, out_dir=out_dir, epoch=epoch,
            n_samples=64, latent_dim=args.latent_dim, nrow=8,
        )

        # ---------- 重建 ----------
        recon_path = None
        try:
            first_batch, _ = next(iter(train_loader))
            if args.save_recon:
                recon_path = save_recon_only(model, device, out_dir, epoch, first_batch)
        except StopIteration:
            pass

        # ---------- 日志 ----------
        epoch_avg_loss = total_loss / total_count
        epoch_avg_rec  = total_rec  / total_count
        epoch_avg_kl   = total_kl   / total_count
        info = f"Epoch {epoch:03d} | loss={epoch_avg_loss:.4f} " \
               f"rec={epoch_avg_rec:.4f} kl={epoch_avg_kl:.4f} | samples: {sample_path.name}"
        if recon_path:
            info += f" | recon: {recon_path.name}"
        print(info)

        # ---------- 原断点保存 ----------
        latest_ckpt = out_dir / "convae_latest.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "epoch_avg_loss": epoch_avg_loss,
        }, latest_ckpt)

        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            best_ckpt = out_dir / "convae_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "epoch_avg_loss": epoch_avg_loss,
            }, best_ckpt)

        # ========== 新增：导出 ComfyUI VAE ==========
        vae_dir = Path("models/vae")          # 可自定义路径
        vae_dir.mkdir(parents=True, exist_ok=True)
        sd = model.state_dict()
        sd = convert_to_sd_shape(sd)
        save_file(sd, vae_dir / f"vae_epoch_{epoch:03d}.safetensors")


if __name__ == "__main__":
    main()