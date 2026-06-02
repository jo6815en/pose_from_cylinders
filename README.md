# Pose from Cylinders

This repository trains an image-pair model that uses cylinders in a scene as geometric anchors. Given two images, the model predicts both:

- a `vision` representation for each image: `[occupancy, radius, depth]` per horizontal bin
- a relative 2D pose between the images: `[tx, ty, sin(yaw), cos(yaw)]`

The main workflow lives in `train.ipynb`. It creates the datasets, model, losses, training loops, validation plots, inference views, and visualizations of predicted cylinders.

## Repository Overview

- `train.ipynb`: main notebook for training, validation, inference, and plotting.
- `analyze_dataset.ipynb`: dataset and camera-pose analysis/debugging.
- `model.py`: pair model with patch embedding, Transformer backbone, vision head, and pose head.
- `dataset.py`: dataset class for synthetic cylinder scenes with `labels.json` annotations.
- `colmap_pair_dataset.py`: dataset class for image pairs with relative COLMAP poses.
- `losses.py`: supervised loss, pose loss, vision loss, Sinkhorn matching, radius consistency, and reprojection loss.
- `utils.py`: plotting utilities and small pose/vision helper functions.
- `debugger.py`: helpers for inspecting cameras, yaw, and dataset entries.
- `ex_train_loop.py`: minimal example for reading `ColmapPairDataset`.
- `working_requirements.txt`: dependencies used in the working environment.

## Data

The data directories are ignored by git and are expected to exist locally in the repository root.

### Synthetic Cylinder Data

`SceneTwoPairsDataset` reads directories with this structure:

```text
dataset/
  scene_000/
    labels.json
    images/
      pair_000_cam1.png
      pair_000_cam2.png
      ...
```

The same format is used for directories such as:

```text
dataset/
valdataset/
testdataset/
```

Each `labels.json` should contain a `pairs` list where each pair has:

- `image1`, `image2`
- `vision1`, `vision2`
- `camera1`, `camera2`, either on the pair object or at scene level

The labeled `vision` array has shape `(num_bins, 4)`:

```text
[occupancy, radius, depth, cylinder_id]
```

The model predicts the corresponding `(num_bins, 3)` representation:

```text
[occupancy, radius, depth]
```

### Forest/COLMAP Data

`ColmapPairDataset` reads:

```text
forestdataset/
  images/
    ...
  relative_poses.json
```

Each entry in `relative_poses.json` should contain:

- `from_image`
- `to_image`
- `T_ab`, a relative 4x4 transform from camera/frame A to B

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r working_requirements.txt
```

Start Jupyter and open `train.ipynb`:

```bash
jupyter notebook
```

If `jupyter` is not available in the environment, install it separately:

```bash
pip install notebook
```

## Training

The recommended workflow is to run `train.ipynb` from top to bottom:

1. Import dependencies and create `device`.
2. Create `PairImageCylinderModel`.
3. Create `train_loader`, `val_loader`, and `forest_loader`.
4. Run the training cell.
5. Run the plotting cells for loss curves and inference.

The notebook currently uses a configuration like:

```python
PairImageCylinderModel(
    img_size=128,
    patch_size=8,
    embed_dim=384,
    depth=6,
    num_heads=6,
    num_bins=128,
)
```

The main loss terms are:

- supervised vision loss against synthetic labels
- supervised relative pose loss
- matched radius consistency between views
- matched 2D reprojection loss between views
- additional forest/COLMAP consistency after a start epoch

## Visualization

`utils.py` provides:

```python
plot_estimated_cylinders_on_images(images, visions, occ_threshold=0.6, flip_x=True)
```

It draws predicted cylinders as vertical overlay bands on top of the images. `flip_x=True` is used because the vision-bin direction is mirrored relative to the image x-coordinate.

`train.ipynb` includes cells for:

- plotting ground-truth vs predicted `occupancy`, `radius`, and `depth`
- plotting predicted cylinders on synthetic test images
- plotting predicted cylinders on a sample from `forest_loader`
- plotting relative pose with `plot_relative_pose`

## Quick Checks

Check that the Python files compile:

```bash
venv/bin/python -m py_compile model.py dataset.py colmap_pair_dataset.py losses.py utils.py debugger.py
```

Check that the notebook is valid JSON:

```bash
python3 -c "import json; json.load(open('train.ipynb')); print('notebook json ok')"
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Notes

- Data directories are not versioned according to `.gitignore`.
- `train.ipynb` is the active experiment workflow and may contain executed cell outputs.
- The forest data has no cylinder ground truth; it uses model predictions and geometric consistency between views.
