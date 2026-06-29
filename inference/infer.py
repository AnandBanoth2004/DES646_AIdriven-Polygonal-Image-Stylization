import os
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms
import yaml
import glob

# Ensure project root is on sys.path when running this file directly
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.pix2pix_train import UNetGenerator

def load_model(ckpt_path: str, device: str):
    net = UNetGenerator().to(device)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state["G"])
    net.eval()
    return net

def infer_on_image(img_path: str, ckpt_path: str, out_path: str, device: Optional[str] = None, image_size: int = 256):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(ckpt_path, device)
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        y = net(x)
    y = (y.clamp(-1,1) + 1)/2
    y_img = transforms.ToPILImage()(y.squeeze(0).cpu())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    y_img.save(out_path)
    return out_path

def infer_on_folder(input_dir: str, ckpt_path: str, out_dir: str, device: Optional[str] = None, image_size: int = 256):
    paths = []
    for ext in ("*.jpg","*.jpeg","*.png"):
        paths.extend(glob.glob(os.path.join(input_dir, ext)))
    for p in paths:
        base = os.path.basename(p)
        outp = os.path.join(out_dir, base)
        infer_on_image(p, ckpt_path, outp, device=device, image_size=image_size)
    return out_dir

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    ap.add_argument("--input", type=str, required=True, help="Image file or folder")
    ap.add_argument("--output", type=str, required=True, help="Output file or folder")
    ap.add_argument("--image_size", type=int, default=256)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.isdir(args.input):
        infer_on_folder(args.input, args.ckpt, args.output, device=dev, image_size=args.image_size)
        print(f"Saved folder outputs to {args.output}")
    else:
        infer_on_image(args.input, args.ckpt, args.output, device=dev, image_size=args.image_size)
        print(f"Saved: {args.output}")
