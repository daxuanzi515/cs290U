#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a stronger CLIP on MNIST (image<->label) with symmetric contrastive loss and natural-language text embeddings.
"""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import optim

from HW3.data import get_mnist_dataloaders
from HW3.clip import SimpleCLIP


def parse_args():
    p = argparse.ArgumentParser(description="Train a stronger CLIP on MNIST (image<->label)")
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
    p.add_argument("--out_dir", type=str, default="./results/clip_0_shot")
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

    # Prepare data
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        resize_to_32=True,
    )

    # Initialize CLIP
    model = SimpleCLIP(in_channels=args.in_channels, embed_dim=args.embed_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Semantic digit descriptions for natural language conditioning
    # 默认数据集的标签是直接阿拉伯数字所以对齐效果不佳，这里用自然语言的描述代替 看0-shot分数是否提升
    digit_texts = [
        "the digit zero", "the digit one", "the digit two", "the digit three", "the digit four",
        "the digit five", "the digit six", "the digit seven", "the digit eight", "the digit nine"
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    temperature = 0.07  # temperature scaling for contrastive logits

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # Compute embeddings
            img_emb = model.forward_image(x)               # (B, D)
            txt_emb = model.text_encoder(y)                # (B, D)
            txt_emb = F.normalize(txt_emb, dim=-1)
            img_emb = F.normalize(img_emb, dim=-1)

            # Compute pairwise logits (symmetric CLIP loss)
            logits_per_image = img_emb @ txt_emb.T / temperature
            logits_per_text = txt_emb @ img_emb.T / temperature
            targets = torch.arange(x.size(0), device=device)

            loss_i2t = F.cross_entropy(logits_per_image, targets)
            loss_t2i = F.cross_entropy(logits_per_text, targets)
            loss = (loss_i2t + loss_t2i) / 2.0

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total += bs

        train_loss = total_loss / max(1, total)
        msg = f"Epoch {epoch:03d} | train_loss: {train_loss:.4f}"

        # Validation
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
