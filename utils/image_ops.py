import cv2
import numpy as np
from skimage.filters import sobel
from skimage.color import rgb2gray

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def write_image(path, img_rgb):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def resize_max(img, max_side=512):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    nh, nw = int(h*scale), int(w*scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def edge_map(img_rgb, blur_sigma=1.0):
    # convert to grayscale and blur
    img = img_rgb.astype(np.float32) / 255.0
    gray = rgb2gray(img)
    if blur_sigma > 0:
        gray_blur = cv2.GaussianBlur((gray*255).astype(np.uint8), (0,0), blur_sigma)
        gray = gray_blur.astype(np.float32) / 255.0
    edges = sobel(gray)  # [0,1]
    return edges

def entropy_like(img_rgb, ksize=9):
    # Local std dev as entropy-like texture magnitude
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (ksize, ksize))
    sqmean = cv2.blur(gray*gray, (ksize, ksize))
    var = np.maximum(sqmean - mean*mean, 0)
    std = np.sqrt(var)
    std = std / (std.max() + 1e-6)
    return std
