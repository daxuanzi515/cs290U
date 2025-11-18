import clip
import torch

clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

@torch.no_grad()
def encode_condition(text):
    tokens = clip.tokenize([text]).cuda()
    emb = clip_model.encode_text(tokens)
    return emb / emb.norm(dim=-1, keepdim=True)
