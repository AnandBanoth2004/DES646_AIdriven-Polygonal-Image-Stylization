<<<<<<< HEAD
# AI-driven Polygonal Image Stylization
=======
# DES646_AI-driven-Polygonal-Image-Stylization
>>>>>>> 1b49c274573f719208de76a87dc2a88c88518537

**Bridging Algorithmic Art and Machine Learning Design**

This project transforms natural images into artistic polygonal mosaics using both algorithmic (Delaunay triangulation and adaptive quadtree) and learned (Pix2Pix GAN-based) stylization techniques for creative design applications.

## Project Overview

This system provides three stylization modes:
- **Triangular**: Delaunay triangulation-based polygonal stylization
- **Rectangular**: Adaptive quadtree-based rectangular stylization  
- **Learned (Pix2Pix)**: AI-powered style transfer using U-Net generator and PatchGAN discriminator

## Project Structure

```
DES646_IDEA1/
├── app/
│   └── demo.py              # Streamlit interactive UI
├── checkpoints/
│   └── pix2pix_unet_patchgan.pt  # Trained model checkpoint
├── configs/
│   └── default.yaml         # Training and stylization parameters
├── data/
│   ├── images/              # Input photos for training
│   ├── openimages/          # OpenImages dataset (optional)
│   ├── targets/             # Generated polygonal target images
│   └── prepare_dataset.py   # Dataset preparation script
├── inference/
│   └── infer.py             # Pix2Pix inference on images/folders
├── models/
│   └── pix2pix_train.py     # Pix2Pix training (U-Net + PatchGAN)
├── tools/
│   └── triangulate.py       # Algorithmic stylization (triangles/rectangles)
├── utils/
│   └── image_ops.py         # Image I/O and processing utilities
└── tests/
    └── test_triangulate.py  # Unit tests for triangulation
```

## Installation

**Requirements:**
- Python 3.9+
- Windows Terminal (or Command Prompt)
- GPU optional (CUDA for faster training, CPU fallback available)

**Setup:**

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
   - **Windows Terminal/CMD:**
   ```bash
   .venv\Scripts\activate
   ```
   - **PowerShell:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Note:** `fiftyone` is optional and used only for downloading OpenImages dataset. If installation fails, you can skip it and use your own images in `data/images/`.

## Quick Start

### 1. Algorithmic Stylization (No Training Required)

**Triangular Stylization:**
```bash
python tools/triangulate.py --input "data/images/your_image.png" --output out/triangles.png --point_count 800 --edge_threshold 50 --color_mode mean
```

**Rectangular Stylization (Quadtree):**
```bash
python tools/triangulate.py --input "data/images/your_image.png" --output out/rectangles.png --rectangles --qt_min 8 --qt_var 12
```

**Parameters:**
- `--point_count`: Triangle density (more points = more detail)
- `--edge_threshold`: Weight towards strong edges when sampling points (1-255)
- `--color_mode`: `mean` or `centroid` color per polygon
- `--qt_min`: Quadtree minimum block size (4-64)
- `--qt_var`: Quadtree variance threshold for splitting

### 2. Prepare Training Dataset

**Option A: Use Your Own Images (Recommended)**

1. Place your photos in `data/images/` (supports: jpg, jpeg, png, webp, bmp)
2. Generate polygonal targets:
```bash
python -m data.prepare_dataset --dataset_dir data/images --out_dir data/targets/triangles_256 --mode triangle --point_count 800 --edge_threshold 50 --color_mode mean
```

**Option B: Download OpenImages Dataset (Requires fiftyone)**

```bash
python -m data.prepare_dataset --dataset openimages --dataset_dir data/openimages --limit 100 --out_dir data/targets/triangles_256 --mode triangle --point_count 800
```

The script generates target images at 256×256 and 512×512 with letterboxing to square format.

### 3. Train Pix2Pix Model

Adjust `configs/default.yaml` for training parameters (epochs, image size, batch size, learning rate, etc.).

**Training Command:**
```bash
python -m models.pix2pix_train --config configs/default.yaml --photos_dir data/images --targets_dir data/targets/triangles_256 --out_dir checkpoints
```

This saves the trained generator to `checkpoints/pix2pix_unet_patchgan.pt`.

**Training Tips:**
- For quick testing, set `epochs: 1` and `image_size: 256` in `configs/default.yaml`
- For production, use `epochs: 20+` and `image_size: 512`
- Training automatically uses GPU if CUDA is available, otherwise falls back to CPU

### 4. Run Inference (Learned Model)

**Single Image:**
```bash
python -m inference.infer --ckpt checkpoints/pix2pix_unet_patchgan.pt --input "path/to/image.jpg" --output out/pix2pix.png --image_size 512
```

**Batch Processing (Folder):**
```bash
python -m inference.infer --ckpt checkpoints/pix2pix_unet_patchgan.pt --input data/images --output out/preds --image_size 512
```

### 5. Interactive Streamlit App

Launch the interactive demo:
```bash
streamlit run app/demo.py
```

The app allows you to:
- Upload any image
- Choose stylization mode: `triangle`, `rectangle`, or `learned (pix2pix)`
- Adjust parameters in real-time
- Preview and download stylized results

**Note:** For learned mode, ensure `checkpoints/pix2pix_unet_patchgan.pt` exists (train the model first if needed).

## Running the Project from Start

**If you have a trained checkpoint (`checkpoints/pix2pix_unet_patchgan.pt`):**
- You can skip training and go directly to inference or the Streamlit app
- The `out/preds/` folder (if it existed) contained previous inference outputs and can be regenerated

**If you're starting fresh:**
1. Prepare your dataset (Step 2 above)
2. Train the model (Step 3 above)
3. Run inference or use the Streamlit app (Steps 4-5)

## Testing

Run unit tests:
```bash
pytest -q
```

This verifies the triangle and rectangle stylization algorithms.

## Configuration

Edit `configs/default.yaml` to customize:
- Training hyperparameters (learning rate, batch size, epochs)
- Image size and augmentation settings
- Triangulation parameters
- Device selection (CPU/CUDA)

## Troubleshooting

**Windows-specific:**
- If `fiftyone` installation fails, skip it and use your own images
- Use Windows Terminal or CMD (not PowerShell) for running commands
- Ensure Python 3.9+ is installed

**General:**
- **SciPy missing?** Triangulation automatically falls back to OpenCV's `Subdiv2D`
- **CUDA not available?** Training/inference automatically use CPU
- **File matching errors?** Ensure photo and target filenames match (or use synthesized naming patterns)

## Expected Deliverables

✅ Working stylization prototype (Streamlit app)  
✅ Trained Pix2Pix model checkpoint (`pix2pix_unet_patchgan.pt`)  
✅ Stylized image dataset (triangle and rectangle modes)  
✅ Concept demo video showing transformation results  
✅ Final design documentation (PDF & GitHub repo)

## References

- `pytorch-CycleGAN-and-pix2pix` — Pix2Pix reference implementation
- `Open Images V7` — Optional dataset source
- Delaunay Triangulation — Geometric stylization algorithm
- Adaptive Quadtree — Rectangular partitioning algorithm

---

**Project developed for DES646 - AI-driven Polygonal Image Stylization**
