#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train SemanticCLIP on MNIST (image ↔ semantic text alignment)
"""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import optim
from HW3.data import get_mnist_dataloaders
from HW3.clip_u import SemanticCLIP


# -------------------------
# Argument parsing
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Semantic CLIP on MNIST")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--out_dir", type=str, default="./results/clip_semantic")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# -------------------------
# Symmetric contrastive loss
# -------------------------
def clip_loss(img_emb, txt_emb, logit_scale):
    logits_i2t = logit_scale * img_emb @ txt_emb.T
    logits_t2i = logit_scale * txt_emb @ img_emb.T
    targets = torch.arange(len(img_emb), device=img_emb.device)
    loss_i2t = F.cross_entropy(logits_i2t, targets)
    loss_t2i = F.cross_entropy(logits_t2i, targets)
    return (loss_i2t + loss_t2i) / 2


# -------------------------
# Evaluate classification accuracy
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    class_embs = model.text_encoder.all_class_embeddings().to(device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        img_emb = model.forward_image(x)
        logits = model.compute_logits(img_emb, class_embs)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


# -------------------------
# Main training loop
# -------------------------
def main():
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        resize_to_32=True,
    )

    model = SemanticCLIP(in_channels=1, embed_dim=args.embed_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total = 0.0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            img_emb, txt_emb = model(imgs, labels)
            loss = clip_loss(img_emb, txt_emb, model.logit_scale.exp())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        train_loss = total_loss / total
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        # Save best checkpoint
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "args": vars(args),
            }, out_dir / "clip_best.pt")

        # Always save latest
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "args": vars(args),
        }, out_dir / "clip_latest.pt")

    # Final test accuracy
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
