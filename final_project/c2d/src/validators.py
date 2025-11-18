import lpips
import torch
import clip

lpips_fn = lpips.LPIPS(net='alex').cuda()
clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

@torch.no_grad()
def calc_lpips(x, x_cf):
    return lpips_fn(x_cf, x).item()

@torch.no_grad()
def calc_clip_img(x, x_cf):
    def encode(img):
        img = preprocess(img).unsqueeze(0).cuda()
        feat = clip_model.encode_image(img)
        return feat / feat.norm()

    return (encode(x) @ encode(x_cf).T).item()

@torch.no_grad()
def calc_text_sim(x_cf, text):
    tokens = clip.tokenize([text]).cuda()
    text_feat = clip_model.encode_text(tokens)
    text_feat /= text_feat.norm()
    
    img_feat = clip_model.encode_image(preprocess(x_cf).unsqueeze(0).cuda())
    img_feat /= img_feat.norm()

    return (img_feat @ text_feat.T).item()
