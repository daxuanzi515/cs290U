import argparse
from pathlib import Path

import torch
from torch import optim

from HW3.data import get_mnist_dataloaders
from HW3.Unet import UNet
from HW3.diffusion import GaussianDiffusion, DiffusionConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train a DDPM/UNet on MNIST (32x32)")
    # Data
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.0)
    # Model
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--base_channels", type=int, default=32)
    # Diffusion
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.02)
    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    # Logging / Checkpoints
    p.add_argument("--out_dir", type=str, default="./result/diffusion", help="Directory to save checkpoints & samples")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Sampling settings
    p.add_argument("--ddim_steps", type=int, default=200)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    return p.parse_args()


@torch.no_grad()
def save_samples(diffusion: GaussianDiffusion, out_dir: Path, epoch: int, device: torch.device, ddim_steps: int, ddim_eta: float):
    diffusion.model.eval()
    # DDPM
    imgs_ddpm = diffusion.sample_ddpm(batch_size=64, device=device)
    imgs_ddpm = (imgs_ddpm.clamp(-1, 1) + 1) / 2.0
    # DDIM
    imgs_ddim = diffusion.sample_ddim(batch_size=64, device=device, steps=ddim_steps, eta=ddim_eta)
    imgs_ddim = (imgs_ddim.clamp(-1, 1) + 1) / 2.0
    try:
        from torchvision.utils import save_image, make_grid

        grid_ddpm = make_grid(imgs_ddpm, nrow=8)
        grid_ddim = make_grid(imgs_ddim, nrow=8)
        save_image(grid_ddpm, out_dir / f"samples_ddpm_epoch_{epoch:03d}.png")
        save_image(grid_ddim, out_dir / f"samples_ddim_epoch_{epoch:03d}.png")
    except Exception:
        pass


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Data (keep [0,1]) then map to [-1,1] before feeding the model
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        resize_to_32=True,
    )

    # Model + Diffusion
    unet = UNet(in_channels=args.in_channels, base_channels=args.base_channels)
    diff_conf = DiffusionConfig(image_size=32, channels=args.in_channels, timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end)
    diffusion = GaussianDiffusion(unet, diff_conf).to(device)

    opt = optim.Adam(diffusion.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(1, args.epochs + 1):
        diffusion.train()
        epoch_loss = 0.0
        num = 0
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            # map to [-1,1]
            x = x * 2 - 1
            loss = diffusion.training_loss(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = x.size(0)
            epoch_loss += loss.item() * bs
            num += bs
            step += 1


        avg = epoch_loss / max(1, num)
        print(f"Epoch {epoch:03d} | train_loss: {avg:.4f}")

        # Save sample and checkpoint
        save_samples(diffusion, out_dir, epoch, device, args.ddim_steps, args.ddim_eta)

        ckpt = {
            "epoch": epoch,
            "model_state": diffusion.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "diffusion_latest.pt")

    # final sample
    save_samples(diffusion, out_dir, args.epochs, device, args.ddim_steps, args.ddim_eta)


if __name__ == "__main__":
    main()
