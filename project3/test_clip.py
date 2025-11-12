import torch
import torch.nn.functional as F
from HW3.clip import SimpleCLIP

clip = SimpleCLIP(embed_dim=128)
ckpt = torch.load("./results/clip_50/clip_latest.pt", map_location="cpu")
clip.load_state_dict(ckpt["model_state"], strict=False)

emb = clip.text_encoder.all_class_embeddings()  # (10, 128)
sim = F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
print(sim)
