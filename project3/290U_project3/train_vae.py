import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torchvision.utils import save_image, make_grid
from HW3.vae import ConvVAE, vae_loss, save_random_samples
from HW3.data import get_mnist_dataloaders


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
    """
    Save a grid of reconstructed images only (no originals), n_show images.
    Returns the saved path.
    """
    model.eval()
    x = batch[:n_show].to(device)
    recon, _, _ = model(x)
    grid = make_grid(recon.cpu(), nrow=nrow, padding=2)
    out_path = out_dir / f"recon_epoch_{epoch:03d}.png"
    save_image(grid, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Train ConvVAE on MNIST and save per-epoch samples")
    parser.add_argument("--data-dir", type=str, default="data/MNIST", help="MNIST data directory (root for torchvision MNIST)")
    parser.add_argument("--output-dir", type=str, default="result/vae", help="Directory to save checkpoints and images")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--kl-weight", type=float, default=1, help="Weight for KL term")
    parser.add_argument("--loss-type", type=str, choices=["bce", "mse"], default="mse")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    # Save reconstructions each epoch by default, with an option to disable
    parser.add_argument("--save-recon", dest="save_recon", action="store_true", default=True, help="Also save reconstructions each epoch")
    parser.add_argument("--no-save-recon", dest="save_recon", action="store_false", help="Disable saving reconstructions")
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
            loss, rec, kl = vae_loss(recon, data, mu, logvar, kl_weight=args.kl_weight, loss_type=args.loss_type)
            loss.backward()
            optimizer.step()

            bs = data.size(0)
            total_loss += loss.item()
            total_rec += rec
            total_kl += kl
            total_count += bs

        sample_path = save_random_samples(
            model=model,
            device=device,
            out_dir=out_dir,
            epoch=epoch,
            n_samples=64,
            latent_dim=args.latent_dim,
            nrow=8,
        )

        recon_path = None
        try:
            first_batch, _ = next(iter(train_loader))
            if args.save_recon:
                recon_path = save_recon_only(model, device, out_dir, epoch, first_batch, n_show=64, nrow=8)
        except StopIteration:
            pass

        epoch_avg_loss = total_loss / total_count
        epoch_avg_rec = total_rec / total_count
        epoch_avg_kl = total_kl / total_count

        info = f"Epoch {epoch:03d} | loss={epoch_avg_loss:.4f} rec={epoch_avg_rec:.4f} kl={epoch_avg_kl:.4f} | samples: {sample_path.name}"
        if recon_path is not None:
            info += f" | recon: {recon_path.name}"
        print(info)

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


if __name__ == "__main__":
    main()