import argparse
from PIL import Image
from torchvision.transforms import ToTensor, Resize
from diffusers import UNet2DConditionModel, AutoencoderKL

from c2d_pipeline import generate_c2d
from torchvision.utils import save_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Counterfactual instruction")
    parser.add_argument("--output", type=str, default="output.png",
                        help="Output path")
    parser.add_argument("--N", type=int, default=8,
                        help="Number of candidates to sample")

    args = parser.parse_args()

    device = "cuda"

    # Load models
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    ).to(device)

    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae"
    ).to(device)

    # Load image
    img = Image.open(args.input).convert("RGB")
    img = Resize((512, 512))(img)
    img = ToTensor()(img).unsqueeze(0).to(device)

    # Run C²D
    result = generate_c2d(img, args.prompt, unet, vae, N=args.N)

    # Save result
    save = (result / 2 + 0.5).clamp(0, 1)
    save_image(save, args.output)

    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
