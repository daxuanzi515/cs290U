import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from HW3.data import get_mnist_dataloaders
from HW3.diffusion import GaussianDiffusion, DiffusionConfig
from HW3.clip import SimpleCLIP
from HW3.vae import ConvVAE


class EpsMLPCond(nn.Module):
    """
    Small MLP epsilon-predictor that operates in VAE latent vector space and
    is conditioned on a (possibly hybrid) embedding. Input/Output tensors are
    shaped (B, D, 1, 1) to be compatible with the GaussianDiffusion wrapper.
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        time_dim: int = 256,
        hidden_dims: Tuple[int, int] = (512, 512),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        in_dim = latent_dim + time_dim + cond_dim


        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.SiLU(),
            nn.Linear(h1, h2), nn.SiLU(),
            nn.Linear(h2, latent_dim),
        )

        # small MLP on time for better capacity
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )

    def sinusoidal_time_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, D, 1, 1) or (B, D)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        elif x.dim() != 2:
            raise ValueError("EpsMLPCond expects input of shape (B, D) or (B, D, 1, 1)")

        # cond: (B, C)
        if cond.dim() != 2:
            cond = cond.view(cond.size(0), -1)

        # time embedding + conditioning concat
        time_emb = self.sinusoidal_time_embedding(t, self.time_dim)
        time_emb = self.time_mlp(time_emb)  # small MLP as in many DDPMs
        h = torch.cat([x, time_emb, cond], dim=1)  # (B, D + time_dim + cond_dim)
        out = self.net(h)
        return out.view(out.size(0), self.latent_dim, 1, 1)

def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny (class/text)-conditioned diffusion on MNIST in VAE latent space")
    # Data
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.0)
    # Model / Conditioning
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--embed_dim", type=int, default=128, help="CLIP embedding dimension (must match checkpoint when using clip/hybrid)")
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--cond_mode", type=str, default="onehot", choices=["onehot", "clip", "hybrid"],
                   help="Condition type: onehot (recommended), clip, or hybrid (concat onehot|clip)")
    p.add_argument("--alpha_onehot", type=float, default=1.0, help="Scale for one-hot part when cond_mode=hybrid")
    p.add_argument("--alpha_clip", type=float, default=1.0, help="Scale for CLIP part when cond_mode=hybrid")
    # VAE latent
    p.add_argument("--vae_ckpt", type=str, default="./results/vae_beta_0.5/convae_latest.pt", help="Path to ConvVAE checkpoint")
    p.add_argument("--latent_dim", type=int, default=-1, help="Latent dim; if <1, inferred from VAE checkpoint args")
    p.add_argument("--canonicalize", action="store_true", help="Canonicalize latent sign per-sample during training & sampling")
    # Eps MLP
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    # Diffusion
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.02)
    # Training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    # Logging / Checkpoints
    p.add_argument("--out_dir", type=str, default="./results/tiny_minist_sd", help="Directory to save checkpoints & samples")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # CLIP
    p.add_argument("--clip_ckpt", type=str, default="./results/clip/clip_latest.pt")
    # Sampling settings
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    return p.parse_args()


def build_cond_from_labels(labels: torch.Tensor,
                           clip: SimpleCLIP,
                           mode: str,
                           num_classes: int,
                           alpha_onehot: float = 1.0,
                           alpha_clip: float = 1.0) -> torch.Tensor:
    """
    Build conditioning vector according to the chosen mode.
    - onehot:      one-hot(num_classes)
    - clip:        CLIP text embedding
    - hybrid:      concat[ alpha_onehot * onehot , alpha_clip * clip_emb ]
    """
    if mode == "onehot":
        cond = F.one_hot(labels, num_classes=num_classes).float()
        return cond
    elif mode == "clip":
        cond = clip.text_encoder(labels)  # already normalized in SimpleCLIP
        return cond
    elif mode == "hybrid":
        onehot = F.one_hot(labels, num_classes=num_classes).float() * alpha_onehot
        clip_emb = clip.text_encoder(labels) * alpha_clip
        cond = torch.cat([onehot, clip_emb], dim=-1)
        return cond
    else:
        raise ValueError(f"Unknown cond_mode: {mode}")


@torch.no_grad()
def save_grid_by_class(
    diffusion: GaussianDiffusion,
    out_dir: Path,
    epoch: int,
    device: torch.device,
    clip: SimpleCLIP,
    steps: int,
    eta: float,
    vae: ConvVAE,
    latent_dim: int,
    canonicalize: bool,
    cond_mode: str,
    num_classes: int,
    alpha_onehot: float,
    alpha_clip: float,
):
    diffusion.model.eval()
    vae.eval()
    if clip is not None:
        clip.eval()

    # 8 rows × 10 cols = 80 samples; each row cycles 0..9
    labels = torch.tensor(list(range(num_classes)) * 8, device=device, dtype=torch.long)

    cond = build_cond_from_labels(
        labels, clip=clip, mode=cond_mode, num_classes=num_classes,
        alpha_onehot=alpha_onehot, alpha_clip=alpha_clip
    )

    # Sample latent z with DDIM, then decode via VAE
    z = diffusion.sample_ddim(batch_size=labels.size(0), device=device, steps=steps, eta=eta, cond=cond)
    z = z.view(z.size(0), latent_dim)
    if canonicalize:
        z = vae.canonicalize_latent(z)
    imgs = vae.decode(z).clamp(0.0, 1.0)

    try:
        from torchvision.utils import save_image
        out_dir.mkdir(parents=True, exist_ok=True)
        save_image(imgs, out_dir / f"samples_epoch_{epoch:03d}.png", nrow=num_classes)
    except Exception:
        pass


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Data
    train_loader, _, _ = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        resize_to_32=True,
    )

    # Load CLIP (only if mode needs it) and freeze
    clip = None
    if args.cond_mode in ("clip", "hybrid"):
        clip = SimpleCLIP(in_channels=args.in_channels, embed_dim=args.embed_dim).to(device)
        ckpt_path = Path(args.clip_ckpt)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            clip.load_state_dict(ckpt["model_state"], strict=False)
        for p in clip.parameters():
            p.requires_grad = False
        clip.eval()

    # Load VAE and freeze
    try:
        vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"VAE checkpoint not found at {args.vae_ckpt}. Train VAE first using train_vae.py or set --vae_ckpt."
        ) from e
    ckpt_args = vae_ckpt.get("args", {}) if isinstance(vae_ckpt.get("args", {}), dict) else {}
    latent_dim = args.latent_dim if args.latent_dim and args.latent_dim > 0 else int(ckpt_args.get("latent_dim", 100))
    vae = ConvVAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(vae_ckpt["model_state"], strict=True)
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    # Determine cond_dim according to mode
    if args.cond_mode == "onehot":
        cond_dim = args.num_classes
    elif args.cond_mode == "clip":
        cond_dim = args.embed_dim
    else:  # hybrid
        cond_dim = args.num_classes + args.embed_dim

    # Conditional epsilon MLP + Diffusion in latent vector space (D,1,1)
    eps_model = EpsMLPCond(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        time_dim=args.time_dim,
        hidden_dims=(args.hidden_dim, args.hidden_dim),
    )
    diff_conf = DiffusionConfig(
        image_size=1, channels=latent_dim, timesteps=args.timesteps,
        beta_start=args.beta_start, beta_end=args.beta_end
    )
    diffusion = GaussianDiffusion(eps_model, diff_conf).to(device)

    opt = optim.Adam(diffusion.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        diffusion.train()
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x = x.to(device)  # images in [0,1]
            y = y.to(device)

            with torch.no_grad():
                # Encode to latent z ~ q(z|x)
                mu, logvar = vae.encode(x)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # (B, D)
                if args.canonicalize:
                    z = vae.canonicalize_latent(z)

                # Build conditioning (onehot/clip/hybrid)
                cond = build_cond_from_labels(
                    y, clip=clip, mode=args.cond_mode, num_classes=args.num_classes,
                    alpha_onehot=args.alpha_onehot, alpha_clip=args.alpha_clip
                )

            # Diffusion on latent as (B, D, 1, 1)
            z_img = z.unsqueeze(-1).unsqueeze(-1)
            loss = diffusion.training_loss(z_img, cond=cond)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total += bs

        avg = total_loss / max(1, total)
        print(f"Epoch {epoch:03d} | train_loss: {avg:.4f}")

        # Save samples grid 8x10 with rows 0..9 (decode latent through VAE)
        save_grid_by_class(
            diffusion, out_dir, epoch, device, clip, args.ddim_steps, args.ddim_eta,
            vae, latent_dim, args.canonicalize, args.cond_mode, args.num_classes,
            args.alpha_onehot, args.alpha_clip
        )

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": diffusion.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "tiny_sd_latest.pt")

    # final sample
    save_grid_by_class(
        diffusion, out_dir, args.epochs, device, clip, args.ddim_steps, args.ddim_eta,
        vae, latent_dim, args.canonicalize, args.cond_mode, args.num_classes,
        args.alpha_onehot, args.alpha_clip
    )


if __name__ == "__main__":
    main()
