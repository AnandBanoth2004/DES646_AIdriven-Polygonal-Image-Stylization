import os
import glob
import shutil
from typing import Optional, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure project root is on sys.path when running this file directly
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Optional imports
try:
    import fiftyone as fo
    FIFTYONE_OK = True
except Exception:
    FIFTYONE_OK = False

from tools.triangulate import stylize_image
from utils.image_ops import read_image, write_image

def download_openimages_small(dataset_dir: str, limit: int = 100) -> str:
    """
    Download a small OpenImages subset using FiftyOne (if available).
    """
    if not FIFTYONE_OK:
        raise ImportError("fiftyone is not installed; install it to use OpenImages downloader.")
    os.makedirs(dataset_dir, exist_ok=True)
    # Avoid MongoDB validation errors on older local MongoDBs
    try:
        fo.config.database_validation = False
    except Exception:
        pass
    # Do NOT pass dataset_dir here to avoid conflicting kwargs in current FiftyOne version
    dataset = fo.zoo.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        max_samples=limit,
        label_types=[],
    )
    # Export/copy raw images into a flat folder under the requested dataset_dir for our pipeline
    export_dir = os.path.join(dataset_dir, "images")
    os.makedirs(export_dir, exist_ok=True)
    for sample in dataset:
        src = sample.filepath
        base = os.path.basename(src)
        dst = os.path.join(export_dir, base)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    return export_dir

def _save_resized(img_np: np.ndarray, out_path: str, size: int):
    H, W = img_np.shape[:2]
    scale = size / max(H, W)
    nh, nw = int(H*scale), int(W*scale)
    img_resized = np.array(Image.fromarray(img_np).resize((nw, nh), Image.BICUBIC))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh)//2
    x0 = (size - nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = img_resized
    write_image(out_path, canvas)

def make_polygon_targets(
    image_dir: str,
    out_dir: str,
    mode: str = "triangle",
    params: Optional[Dict] = None,
    save_sizes=(256, 512)
):
    """
    Build polygon targets for each image.
    mode: 'triangle' or 'rectangle'
    params: dict with keys for stylize_image
    """
    if params is None:
        params = {}
    use_rect = (mode == "rectangle")
    os.makedirs(out_dir, exist_ok=True)
    img_paths = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        img_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    img_paths = sorted(img_paths)
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in {image_dir}")
    for p in tqdm(img_paths, desc="Building polygon targets"):
        try:
            img = read_image(p)
            # Avoid passing 'rectangles' twice if present in params
            _p = dict(params)
            _p.pop("rectangles", None)
            out = stylize_image(
                img,
                rectangles=use_rect,
                **_p
            )
            base = os.path.splitext(os.path.basename(p))[0]
            for s in save_sizes:
                outp = os.path.join(out_dir, f"{base}_{mode}_{s}.png")
                _save_resized(out, outp, size=s)
        except Exception as e:
            print(f"Skip {p}: {e}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="openimages", choices=["openimages"])  # only OpenImages supported
    ap.add_argument("--dataset_dir", type=str, default="data/demo")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out_dir", type=str, default="data/targets/triangles_256")
    ap.add_argument("--mode", type=str, default="triangle", choices=["triangle","rectangle"])
    ap.add_argument("--point_count", type=int, default=800)
    ap.add_argument("--edge_threshold", type=float, default=50.0)
    ap.add_argument("--color_mode", type=str, default="mean")
    ap.add_argument("--rectangles", action="store_true")
    args = ap.parse_args()

    if args.dataset == "openimages":
        images_dir = download_openimages_small(args.dataset_dir, limit=args.limit)

    params = dict(
        point_count=args.point_count,
        edge_threshold=args.edge_threshold,
        color_mode=args.color_mode,
        rectangles=args.rectangles or (args.mode == "rectangle")
    )
    os.makedirs(args.out_dir, exist_ok=True)
    make_polygon_targets(images_dir, args.out_dir, mode=args.mode, params=params)
    print(f"Targets saved in {args.out_dir}")
