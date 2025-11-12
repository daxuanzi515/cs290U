import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from HW3.clip import SimpleCLIP
from HW3.data import get_mnist_dataloaders

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a pre-trained CLIP model on MNIST (image<->label)")
    # Data
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--clip_ckpt", type=str, default="./results/clip/clip_latest.pt")  # CLIP checkpoint path
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def evaluate(model: SimpleCLIP, loader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        class_embs = model.text_encoder.all_class_embeddings().to(device)
        for x, y in loader:
            x = x.to(device)  # No normalization here, leave it as it is
            y = y.to(device)

            img_z = model.forward_image(x)  # (B, D)
            logits = model.compute_logits(img_z, class_embs)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    acc = correct / max(1, total)
    return acc

def visualize_clip_embeddings(model: SimpleCLIP, loader, device: torch.device):
    model.eval()
    img_embeddings = []
    txt_embeddings = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            img_emb = model.forward_image(images)  # (B, D)
            txt_emb = model.text_encoder(labels)  # (B, D)

            img_embeddings.append(img_emb.cpu().numpy())
            txt_embeddings.append(txt_emb.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    img_embeddings = np.concatenate(img_embeddings, axis=0)
    txt_embeddings = np.concatenate(txt_embeddings, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    # **Here we perform the conversion for visualization**
    # Convert embeddings to float32 for T-SNE visualization
    img_embeddings = img_embeddings.astype(np.float32)
    txt_embeddings = txt_embeddings.astype(np.float32)

    # Combine image and text embeddings
    all_embeddings = np.concatenate([img_embeddings, txt_embeddings], axis=0)
    all_labels = np.concatenate([labels_list, labels_list], axis=0)

    # Apply T-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:len(img_embeddings), 0], tsne_results[:len(img_embeddings), 1], c='blue', label='Images')
    plt.scatter(tsne_results[len(img_embeddings):, 0], tsne_results[len(img_embeddings):, 1], c='red', label='Texts')
    plt.title("T-SNE visualization of CLIP embeddings")
    plt.legend()
    plt.show()

# def zero_shot_classification(model: SimpleCLIP, images, labels, device: torch.device):
#     model.eval()
#     images = images.to(device)  # No normalization here
#     labels = labels.to(device)

#     img_emb = model.forward_image(images)
#     txt_emb = model.text_encoder(labels)

#     logits = model.compute_logits(img_emb, txt_emb)
#     pred = logits.argmax(dim=-1)

#     correct = (pred == labels).sum().item()
#     accuracy = correct / labels.size(0)
#     print(f"Zero-shot accuracy: {accuracy:.4f}")

def zero_shot_classification(model: SimpleCLIP, images, labels, device: torch.device):
    model.eval()
    
    # Convert images to float32 and normalize to [0, 1]
    images = images.to(device).float() / 255.0  # Convert uint8 to float32 and scale to [0, 1]
    
    # Ensure the images have the correct shape: (B, 1, H, W)
    if images.dim() == 3:  # Check if the image has shape (B, H, W) and add a channel dimension
        images = images.unsqueeze(1)  # Now shape (B, 1, H, W)
    
    labels = labels.to(device)

    img_emb = model.forward_image(images)
    txt_emb = model.text_encoder(labels)

    logits = model.compute_logits(img_emb, txt_emb)
    pred = logits.argmax(dim=-1)

    correct = (pred == labels).sum().item()
    accuracy = correct / labels.size(0)
    print(f"Zero-shot accuracy: {accuracy:.4f}")



def main():
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device(args.device)

    # Load data
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.1,  # You can adjust this as needed
        resize_to_32=True,  # Resize images to 32x32 (keep consistent with other modules)
    )

    # Load CLIP model
    clip = SimpleCLIP(in_channels=1, embed_dim=args.embed_dim).to(device)
    ckpt_path = Path(args.clip_ckpt)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        clip.load_state_dict(ckpt["model_state"], strict=False)
    for p in clip.parameters():
        p.requires_grad = False
    clip.eval()

    # 1. Evaluate the model's accuracy on validation set
    val_acc = evaluate(clip, val_loader, device)
    print(f"Validation accuracy: {val_acc:.4f}")

    # # 2. Visualize image and text embeddings using T-SNE
    # visualize_clip_embeddings(clip, test_loader, device)

    # 3. Test zero-shot classification accuracy on test set
    zero_shot_classification(clip, test_loader.dataset.data[:args.batch_size], test_loader.dataset.targets[:args.batch_size], device)

if __name__ == "__main__":
    main()
