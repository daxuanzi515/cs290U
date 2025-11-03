import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch import optim

from HW3.data import get_mnist_dataloaders
from HW3.vae import ConvVAE
from HW3.diffusion import GaussianDiffusion, DiffusionConfig


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal time embedding identical to the one used in UNet.
    timesteps: (B,) int64 or float tensor of values in [0, T)
    returns: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    if half == 0:
        return torch.zeros((timesteps.shape[0], dim), device=device)
    freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * torch.arange(0, half, device=device).float() / max(half - 1, 1))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class EpsMLP(nn.Module):
    """
    Small MLP to predict epsilon in VAE latent vector space.
    Accepts inputs shaped (B, D, 1, 1) to be compatible with the image-oriented diffusion wrapper,
    flattens to (B, D), conditions on sinusoidal time embedding, and outputs (B, D, 1, 1).
    """

    def __init__(self, latent_dim: int, time_dim: int = 256, hidden_dims: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        in_dim = latent_dim + time_dim

        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.SiLU(),
            nn.Linear(h1, h2), nn.SiLU(),
            nn.Linear(h2, latent_dim),
        )

        # a small MLP on time for better capacity
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, D, 1, 1) or (B, D)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        elif x.dim() != 2:
            raise ValueError("EpsMLP expects input of shape (B, D) or (B, D, 1, 1)")

        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        h = torch.cat([x, t_emb], dim=1)
        out = self.net(h)
        return out.view(out.size(0), self.latent_dim, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Train diffusion in VAE latent space on MNIST (32x32)")
    # Data
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.0)
    # VAE
    p.add_argument("--vae_ckpt", type=str, default="./result/vae/convae_latest.pt", help="Path to ConvVAE checkpoint")
    p.add_argument("--latent_dim", type=int, default=-1, help="Latent dim; if <1, inferred from VAE checkpoint args")
    p.add_argument("--canonicalize", action="store_true", help="Canonicalize latent sign per-sample during training & sampling")
    # Diffusion
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.02)
    # Model (MLP epsilon)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    # Training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Logging / Checkpoints
    p.add_argument("--out_dir", type=str, default="./result/latent_diffusion", help="Directory to save checkpoints & samples")
    # Sampling settings
    p.add_argument("--ddim_steps", type=int, default=200)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    return p.parse_args()


@torch.no_grad()
def save_samples_latent(
    diffusion: GaussianDiffusion,
    vae: ConvVAE,
    out_dir: Path,
    epoch: int,
    device: torch.device,
    ddim_steps: int,
    ddim_eta: float,
    latent_dim: int,
    canonicalize: bool,
):
    diffusion.model.eval()
    vae.eval()

    # DDPM samples in latent space
    z_ddpm = diffusion.sample_ddpm(batch_size=64, device=device)  # (B, D, 1, 1)
    z_ddpm = z_ddpm.view(z_ddpm.size(0), latent_dim)
    if canonicalize:
        z_ddpm = vae.canonicalize_latent(z_ddpm)
    x_ddpm = vae.decode(z_ddpm).clamp(0.0, 1.0)

    # DDIM samples in latent space
    z_ddim = diffusion.sample_ddim(batch_size=64, device=device, steps=ddim_steps, eta=ddim_eta)
    z_ddim = z_ddim.view(z_ddim.size(0), latent_dim)
    if canonicalize:
        z_ddim = vae.canonicalize_latent(z_ddim)
    x_ddim = vae.decode(z_ddim).clamp(0.0, 1.0)

    try:
        from torchvision.utils import save_image, make_grid

        grid_ddpm = make_grid(x_ddpm, nrow=8)
        grid_ddim = make_grid(x_ddim, nrow=8)
        save_image(grid_ddpm, out_dir / f"samples_ddpm_epoch_{epoch:03d}.png")
        save_image(grid_ddim, out_dir / f"samples_ddim_epoch_{epoch:03d}.png")
    except Exception:
        pass


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Load data (images in [0,1])
    train_loader, _, _ = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        resize_to_32=True,
    )

    # Load VAE checkpoint (determine latent_dim first)
    try:
        ckpt = torch.load(args.vae_ckpt, map_location=device)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"VAE checkpoint not found at {args.vae_ckpt}. Train VAE first using train_vae.py or set --vae_ckpt.") from e

    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    latent_dim = args.latent_dim if args.latent_dim and args.latent_dim > 0 else int(ckpt_args.get("latent_dim", 100))

    vae = ConvVAE(latent_dim=latent_dim)
    vae.load_state_dict(ckpt["model_state"], strict=True)
    vae.to(device)
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    # Epsilon MLP model and Gaussian Diffusion configured for latent vector as (D,1,1)
    eps_model = EpsMLP(latent_dim=latent_dim, time_dim=args.time_dim, hidden_dims=(args.hidden_dim, args.hidden_dim))
    diff_conf = DiffusionConfig(image_size=1, channels=latent_dim, timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end)
    diffusion = GaussianDiffusion(eps_model, diff_conf).to(device)

    opt = optim.Adam(diffusion.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        diffusion.train()
        epoch_loss = 0.0
        num = 0

        for x, _ in train_loader:
            x = x.to(device)  # [0,1]
            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # (B, D)
                if args.canonicalize:
                    z = vae.canonicalize_latent(z)

            z_img = z.unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)
            loss = diffusion.training_loss(z_img)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = z.size(0)
            epoch_loss += loss.item() * bs
            num += bs

        avg_loss = epoch_loss / max(1, num)
        print(f"Epoch {epoch:03d} | latent train_loss: {avg_loss:.4f}")

        # Save samples and checkpoint
        save_samples_latent(diffusion, vae, out_dir, epoch, device, args.ddim_steps, args.ddim_eta, latent_dim, args.canonicalize)

        ckpt_diff = {
            "epoch": epoch,
            "model_state": diffusion.state_dict(),
            "args": vars(args),
            "latent_dim": latent_dim,
        }
        torch.save(ckpt_diff, out_dir / "latent_diffusion_latest.pt")

    # Final samples
    save_samples_latent(diffusion, vae, out_dir, args.epochs, device, args.ddim_steps, args.ddim_eta, latent_dim, args.canonicalize)


if __name__ == "__main__":
    main()
