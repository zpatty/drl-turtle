# Crush: Autonomous Sea Turtle Robot for Marine Fieldwork

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code and software for the paper: **"Autonomous Sea Turtle Robot for Marine Fieldwork"**  
---

## System Requirements

### Software Dependencies

| Package | Version Tested | Notes |
|---------|---------------|-------|
| Python | 3.12 | Primary runtime (ROS2 node) |
| Python | 3.10 | Required for TVM inference server |
| ROS2 | Jazzy | Robot operating system |
| OpenCV | 4.12.0.88 | Image processing |
| NumPy | 1.26.4 | Numerical computation |
| Edge Impulse Linux SDK | 1.2.2 | FOMO object detection |
| PyYAML | 6.0.1 | Configuration loading |
| micromamba / conda | 26.1.1 | Environment management |

### Operating Systems Tested

- Ubuntu 22.04 (development)
- Ubuntu 24.04 (Raspberry Pi 5 deployment)

### Required Hardware

- **Standard demo/evaluation:** Any x86_64 or ARM64 machine with ≥4 GB RAM  
- **Full deployment (as in paper):** Raspberry Pi 5 (8 GB RAM), stereo camera pair (640×480 per eye), Dynamixel motor chain (10 motors, 1 Mbps), XIAO nRF52840 IMU/depth sensor

---

## Installation Guide

### 1. Clone the repository

```bash
git clone https://github.com/zpatty/drl-turtle.git
cd drl-turtle
```

### 2. Set up Python environments

Two separate environments are required due to TVM/PyTorch compatibility:

**Environment A — ROS2 tracker client (Python 3.12):**
```bash
micromamba create -n crush_ros python=3.12
micromamba activate crush_ros
pip install opencv-python numpy pyyaml edge-impulse-linux rclpy cv_bridge
```

**Environment B — TVM inference server (Python 3.10):**
```bash
micromamba create -n crush_tvm python=3.10
micromamba activate crush_tvm
pip install torch numpy opencv-python
# Install TVM runtime (pre-compiled for your platform):
pip install [TVM_WHEEL_PATH]
# Install pysot tracker:
cd pysot && python setup.py build_ext --inplace && cd ..
```

### 3. Download model weights

```bash
# Download from [DATA REPOSITORY LINK]:
wget [URL]/tracker_models.zip
unzip tracker_models.zip
```

**Typical install time:** ~15 minutes on a standard desktop computer.

---

## Demo

A small demo dataset (10-second video clip, ~30 frames) is included in `demo/` to verify correct installation.

### Run the demo

**Step 1 — Start the TVM inference server** (in Environment B):
```bash
micromamba activate crush_tvm
python tvm_inference_server_with_detector.py \
    --config configs/siamrpn_r50_l234_dwxcorr.yaml \
    --snapshot tracker_models/siamrpn_r50.pth \
    --port 9999
```

**Step 2 — Run tracker on demo video** (in a second terminal, Environment A):
```bash
micromamba activate crush_ros
python demo_offline.py \
    --input demo/demo_clip.mp4 \
    --confidence-threshold 0.7 \
    --scan-density 6 \
    --save-data \
    --output-dir demo_output/
```

**Step 3 — Test FOMO detection model:**
```bash
python debug_with_image_RGB.py ./models/turtle-detector.eim demo/test_frame.jpg
```

### Expected Output

- Terminal: per-frame centroid coordinates `[cx, cy]`, confidence scores, and FPS
- `demo_output/detection/`: annotated frames with bounding boxes
- `demo_output/centroids.csv`: tracking log with columns `frame, timestamp, cx, cy, bbox_x, bbox_y, bbox_w, bbox_h, fps, confidence, tracking_status`
- `detection_visualization.jpg`: bounding box overlay from FOMO model

**Expected run time for demo:** ~30 seconds on a standard desktop computer.

---

## Instructions for Use

### Running on your own data (offline video)

```bash
python demo_offline.py \
    --input /path/to/your/video.mp4 \
    --confidence-threshold 0.7 \
    --save-data \
    --output-dir results/
```

### Running on live ROS2 camera stream

```bash
# Terminal 1: inference server
python tvm_inference_server_with_detector.py \
    --config configs/[CONFIG].yaml \
    --snapshot tracker_models/[WEIGHTS].pth

# Terminal 2: ROS2 tracker node
python ros2_tvm_client_grid_search.py \
    --camera-topic frames \
    --message-type turtlecam \
    --confidence-threshold 0.7 \
    --scan-density 6 \
    --save-data \
    --output-dir results/
```

### Using pre-prepared templates for re-detection

```bash
python ros2_tvm_client_grid_search.py \
    --load-templates /path/to/templates.npz \
    --confidence-threshold 0.7
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--confidence-threshold` | 0.7 | Minimum score before triggering grid re-detection |
| `--scan-density` | 6 | Grid density for full-frame scan (4=sparse, 8=dense) |
| `--message-type` | turtlecam | Camera message type: `turtlecam`, `compressed`, or `image` |
| `--save-data` | off | Enable saving of frames and CSV log |
| `--show-display` | off | Enable live display window |
| `--debug-grid-search` | off | Save per-position heatmaps for grid search debugging |

---

## Reproduction Instructions

To reproduce the quantitative results in the paper:

```bash
# [TODO]
# e.g.:
wget [DATASET_URL]/aquarium_tracking_trials.zip
unzip aquarium_tracking_trials.zip

python scripts/reproduce_tracking_analysis.py \
    --data-dir aquarium_tracking_trials/ \
    --output results_paper/
```

This will regenerate the depth tracking performance metrics and obstacle avoidance statistics reported in the paper. Expected run time: ~[X] hours on a standard desktop.

---

## License

This software is released under the [MIT License](LICENSE). See `LICENSE` for details.

---

## Citation

If you use this software, please cite:

```bibtex
@article{crush2026,
  title   = {Autonomous Sea Turtle Robot for Marine Fieldwork},
  author  = {Zach J. Patterson, Emily Sologuren, Levi Cai, Daniel Kim, Alaa Maalouf, Pascal Spino, Daniela Rus},
  year    = {2026},
  doi     = {https://doi.org/10.48550/arXiv.2602.21389}
}
```

---

## Contact
Emily Sologuren — esolo@mit.edu  
MIT Computer Science and Artificial Intelligence Laboratory (CSAIL)
