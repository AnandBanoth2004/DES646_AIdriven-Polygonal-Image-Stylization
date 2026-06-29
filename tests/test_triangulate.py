import numpy as np
from tools.triangulate import stylize_image

def test_stylize_triangle_basic():
    # Create synthetic image
    img = np.zeros((128,128,3), dtype=np.uint8)
    img[32:96, 32:96, :] = (255, 128, 64)
    out = stylize_image(img, point_count=200, edge_threshold=20.0, color_mode="mean", rectangles=False)
    assert out.shape == img.shape
    assert out.mean() > 0  # not empty

def test_stylize_rectangle_basic():
    img = np.zeros((128,128,3), dtype=np.uint8)
    img[:, :64, :] = (50, 200, 50)
    img[:, 64:, :] = (200, 50, 200)
    out = stylize_image(img, rectangles=True, quadtree_min_size=8, quadtree_variance_thresh=5.0)
    assert out.shape == img.shape
    assert out.mean() > 0
