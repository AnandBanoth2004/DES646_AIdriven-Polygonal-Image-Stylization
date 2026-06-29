import argparse
import os
import random
from typing import Tuple, List

import numpy as np
import cv2

try:
    from scipy.spatial import Delaunay
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Ensure project root is on sys.path when running this file directly
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.image_ops import read_image, write_image, edge_map

def sample_points(img: np.ndarray, point_count: int, edge_threshold: float, jitter: float = 1.0) -> np.ndarray:
    """Sample points from edges + random + border. Returns Nx2 float32 [x,y]."""
    h, w = img.shape[:2]
    edges = edge_map(img, blur_sigma=1.0)
    edges = (edges / (edges.max() + 1e-6))  # [0,1]
    # threshold for stronger edges
    edge_pts = np.argwhere(edges > (edge_threshold/255.0))  # (y,x)
    edge_pts = edge_pts[:, [1,0]]  # -> (x,y)
    # random subsample
    if len(edge_pts) > point_count//2:
        idx = np.random.choice(len(edge_pts), size=point_count//2, replace=False)
        edge_pts = edge_pts[idx]
    # random points
    rand_n = max(0, point_count - len(edge_pts) - 30)
    rand_x = np.random.randint(0, w, size=rand_n)
    rand_y = np.random.randint(0, h, size=rand_n)
    rand_pts = np.stack([rand_x, rand_y], axis=1)

    # grid jitter points for coverage
    grid_step = int(max(8, min(h, w) / 32))
    gx = np.arange(0, w, grid_step)
    gy = np.arange(0, h, grid_step)
    gxx, gyy = np.meshgrid(gx, gy)
    grid_pts = np.stack([gxx.ravel(), gyy.ravel()], axis=1)
    # jitter
    if jitter > 0:
        j = (np.random.rand(len(grid_pts), 2) - 0.5) * grid_step * 0.5
        grid_pts = np.clip(grid_pts + j, [0,0], [w-1, h-1])

    # border points
    border_pts = np.array([
        [0,0],[w-1,0],[0,h-1],[w-1,h-1],
        [w//2,0],[0,h//2],[w-1,h//2],[w//2,h-1]
    ])

    pts = np.concatenate([edge_pts, rand_pts, grid_pts.astype(np.int32), border_pts], axis=0)
    pts = np.unique(pts, axis=0).astype(np.float32)
    return pts

def triangles_from_points(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    """Return faces as Mx3 int indices into pts array."""
    if SCIPY_AVAILABLE and len(pts) >= 3:
        tri = Delaunay(pts)
        return tri.simplices
    # Fallback: OpenCV Subdiv2D (slower to parse)
    subdiv = cv2.Subdiv2D((0,0,w,h))
    for (x,y) in pts:
        try:
            subdiv.insert((float(x), float(y)))
        except Exception:
            pass
    triangleList = subdiv.getTriangleList()
    faces = []
    # Map triangle vertices back to nearest sampled points
    for t in triangleList:
        x1,y1,x2,y2,x3,y3 = t
        T = np.array([[x1,y1],[x2,y2],[x3,y3]])
        # reject triangles outside
        if np.any(T[:,0] < 0) or np.any(T[:,0] >= w) or np.any(T[:,1] < 0) or np.any(T[:,1] >= h):
            continue
        idxs = []
        for vx,vy in T:
            d = np.sum((pts - np.array([vx,vy]))**2, axis=1)
            idxs.append(int(np.argmin(d)))
        if len(set(idxs)) == 3:
            faces.append(idxs)
    return np.array(faces, dtype=np.int32)

def color_triangle(img: np.ndarray, poly: np.ndarray, color_mode: str = "mean") -> Tuple[int,int,int]:
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly.astype(np.int32), 1)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0,0,0)
    if color_mode == "centroid":
        cx, cy = np.mean(poly, axis=0).astype(int)
        cx = np.clip(cx, 0, img.shape[1]-1)
        cy = np.clip(cy, 0, img.shape[0]-1)
        col = img[cy, cx, :]
    else:  # mean
        col = img[ys, xs, :].mean(axis=0)
    return tuple([int(c) for c in col])

def render_triangles(img: np.ndarray, pts: np.ndarray, faces: np.ndarray, color_mode: str = "mean") -> np.ndarray:
    out = np.zeros_like(img)
    for tri in faces:
        poly = pts[tri]  # (3,2)
        col = color_triangle(img, poly, color_mode)
        cv2.fillConvexPoly(out, poly.astype(np.int32), col)
        cv2.polylines(out, [poly.astype(np.int32)], isClosed=True, color=col, thickness=1)
    return out

def quadtree_tiling(img: np.ndarray, min_size: int = 8, variance_thresh: float = 12.0) -> np.ndarray:
    """Adaptive rectangles by recursive variance threshold."""
    h, w = img.shape[:2]
    out = np.zeros_like(img)

    def process(x0, y0, x1, y1):
        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            return
        if (x1 - x0) <= min_size or (y1 - y0) <= min_size:
            col = patch.reshape(-1,3).mean(axis=0).astype(np.uint8)
        else:
            var = patch.astype(np.float32).var()
            if var > variance_thresh:
                mx = (x0 + x1)//2
                my = (y0 + y1)//2
                process(x0, y0, mx, my)
                process(mx, y0, x1, my)
                process(x0, my, mx, y1)
                process(mx, my, x1, y1)
                return
            else:
                col = patch.reshape(-1,3).mean(axis=0).astype(np.uint8)
        out[y0:y1, x0:x1] = col

    process(0, 0, w, h)
    return out

def stylize_image(
    img: np.ndarray,
    point_count: int = 800,
    edge_threshold: float = 50.0,
    blur_sigma: float = 1.0,
    color_mode: str = "mean",
    rectangles: bool = False,
    quadtree_min_size: int = 8,
    quadtree_variance_thresh: float = 12.0,
) -> np.ndarray:
    """Main entry point: triangle or rectangle stylization."""
    img_proc = img.copy()
    if rectangles:
        return quadtree_tiling(img_proc, min_size=quadtree_min_size, variance_thresh=quadtree_variance_thresh)
    pts = sample_points(img_proc, point_count=point_count, edge_threshold=edge_threshold, jitter=1.0)
    h, w = img_proc.shape[:2]
    faces = triangles_from_points(pts, w, h)
    out = render_triangles(img_proc, pts, faces, color_mode=color_mode)
    return out

def main():
    ap = argparse.ArgumentParser(description="Triangle/Rectangle polygon stylizer")
    ap.add_argument("--input", required=True, help="Path to input image")
    ap.add_argument("--output", required=True, help="Path to save stylized image")
    ap.add_argument("--point_count", type=int, default=800)
    ap.add_argument("--edge_threshold", type=float, default=50.0)
    ap.add_argument("--blur_sigma", type=float, default=1.0)
    ap.add_argument("--color_mode", type=str, default="mean", choices=["mean","centroid"])
    ap.add_argument("--rectangles", action="store_true", help="Use adaptive rectangles (quadtree)")
    ap.add_argument("--qt_min", type=int, default=8, help="Quadtree min block size")
    ap.add_argument("--qt_var", type=float, default=12.0, help="Quadtree variance threshold")
    args = ap.parse_args()

    img = read_image(args.input)
    out = stylize_image(
        img,
        point_count=args.point_count,
        edge_threshold=args.edge_threshold,
        blur_sigma=args.blur_sigma,
        color_mode=args.color_mode,
        rectangles=args.rectangles,
        quadtree_min_size=args.qt_min,
        quadtree_variance_thresh=args.qt_var
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_image(args.output, out)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
