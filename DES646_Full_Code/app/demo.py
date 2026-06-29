import os
import tempfile

import streamlit as st

# Ensure project root is on sys.path when running via Streamlit
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.triangulate import stylize_image
from utils.image_ops import read_image, write_image
from inference.infer import infer_on_image

st.set_page_config(page_title="Polygonal Stylizer", layout="wide")

st.title("Image → Polygonal Stylizer (Triangles / Rectangles / Pix2Pix)")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"])
col1, col2 = st.columns(2)

with st.sidebar:
    st.header("Parameters")
    style_mode = st.selectbox("Style Mode", ["triangle", "rectangle", "learned (pix2pix)"])
    if style_mode in ("triangle","rectangle"):
        point_count = st.slider("Point Count", min_value=100, max_value=5000, value=800, step=100)
        edge_threshold = st.slider("Edge Threshold", min_value=1, max_value=255, value=50, step=1)
        color_mode = st.selectbox("Triangle Color", ["mean","centroid"])
        qt_min = st.slider("Quadtree Min Size", min_value=4, max_value=64, value=8, step=2)
        qt_var = st.slider("Quadtree Var Threshold", min_value=1.0, max_value=50.0, value=12.0, step=0.5)
    else:
        ckpt_path = st.text_input("Checkpoint Path", "checkpoints/pix2pix_unet_patchgan.pt")
        infer_size = st.slider("Inference Size", min_value=256, max_value=1024, value=512, step=64)

if uploaded is not None:
    tmpdir = tempfile.mkdtemp()
    in_path = os.path.join(tmpdir, uploaded.name)
    with open(in_path, "wb") as f:
        f.write(uploaded.getbuffer())
    img = read_image(in_path)
    with col1:
        st.subheader("Original")
        st.image(img, channels="RGB", use_column_width=True)

    if style_mode in ("triangle","rectangle"):
        out = stylize_image(
            img,
            point_count=point_count,
            edge_threshold=float(edge_threshold),
            color_mode=color_mode,
            rectangles=(style_mode == "rectangle"),
            quadtree_min_size=int(qt_min),
            quadtree_variance_thresh=float(qt_var),
        )
        with col2:
            st.subheader("Algorithmic Stylization")
            st.image(out, channels="RGB", use_column_width=True)
        if st.button("Download result", type="primary"):
            out_path = os.path.join(tmpdir, f"stylized_{style_mode}.png")
            write_image(out_path, out)
            with open(out_path, "rb") as f:
                st.download_button("Download stylized image", data=f, file_name=os.path.basename(out_path), mime="image/png")
    else:
        if st.button("Run learned model"):
            if not os.path.isfile(ckpt_path):
                st.error(f"Checkpoint not found: {ckpt_path}")
            else:
                out_path = os.path.join(tmpdir, "stylized_pix2pix.png")
                # infer_on_image(img_path, ckpt_path, out_path, device=None, image_size=...)
                infer_on_image(in_path, ckpt_path, out_path, image_size=infer_size)
                out = read_image(out_path)
                with col2:
                    st.subheader("Learned Stylization (Pix2Pix)")
                    st.image(out, channels="RGB", use_column_width=True)
                with open(out_path, "rb") as f:
                    st.download_button("Download stylized image", data=f, file_name=os.path.basename(out_path), mime="image/png")
else:
    st.info("Upload an image to begin.")
