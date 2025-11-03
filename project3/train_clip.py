import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim

from HW3.data import get_mnist_dataloaders
from HW3.clip import SimpleCLIP


def parse_args():
    p = argparse.ArgumentParser(description="Train a simple CLIP on MNIST (image<->label)")
    # Data
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--resize_to_32", action="store_true", help="Resize images to 32x32 (default True)")
    # Model/Train
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    # Logging
    p.add_argument("--out_dir", type=str, default="./result/clip")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def evaluate(model: SimpleCLIP, loader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        class_embs = model.text_encoder.all_class_embeddings().to(device)
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            img_z = model.forward_image(x)
            logits = model.compute_logits(img_z, class_embs)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    acc = correct / max(1, total)
    return acc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        resize_to_32=True,  # keep consistent with other modules
    )

    model = SimpleCLIP(in_channels=args.in_channels, embed_dim=args.embed_dim).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # Forward
            img_z = model.forward_image(x)  # (B, D)
            class_embs = model.text_encoder.all_class_embeddings().to(device)  # (C, D)
            logits = model.compute_logits(img_z, class_embs)  # (B, C)

            # Cross-entropy to true label
            loss = F.cross_entropy(logits, y)

            # Optional symmetric loss (text->image) could be added, but simple CE works well here

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total += bs

        train_loss = total_loss / max(1, total)
        msg = f"Epoch {epoch:03d} | train_loss: {train_loss:.4f}"

        # Evaluate
        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device)
            msg += f" | val_acc: {val_acc:.4f}"
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "args": vars(args),
                }, out_dir / "clip_best.pt")

        print(msg)

        # Save latest every epoch
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "args": vars(args),
        }, out_dir / "clip_latest.pt")

    # Final test accuracy
    test_acc = evaluate(model, test_loader, device)
    print(f"Test | acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
