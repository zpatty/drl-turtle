# Crush: Autonomous Sea Turtle Robot for Marine Fieldwork

**License:** MIT  
Code and software for the paper: *"Autonomous Sea Turtle Robot for Marine Fieldwork"*

---

## Overview

This repository contains the vision and control software for Crush, a biomimetic autonomous underwater robot designed for non-disruptive marine animal tracking. The software stack includes:

- **FOMO-based visual detection** (`crush_tracker.py`) — real-time turtle detection using an Edge Impulse model deployed on a Raspberry Pi 5
- **Autonomous dive planning** (`untethered_planning_node.py`) — centroid-to-motor control with obstacle avoidance
- **Offline demo** (`demo_offline.py`) — run the detector on any video file without ROS2 or hardware
- **Model verification** (`test_detector.py`) — quickly verify the detector on included test images

---

## Repository Structure

```
turtle_hardware/
├── turtle-tracking.eim                  # FOMO model (ARM64 — Raspberry Pi 5)
├── turtle-tracking-linux-x86_64.eim     # FOMO model (x86_64 — laptop/desktop)
├── template_frames/                     # Test images for model verification
│   ├── turtle_front.png
│   ├── turtle_down.png
│   ├── turtle_angle.png
│   ├── turtle_angle_2.png
│   └── turtle_angle_3.png
├── crush_tracker.py                     # ROS2 FOMO tracker node (live deployment)
├── untethered_planning_node.py          # Autonomous dive planner (live deployment)
├── demo_offline.py                      # Standalone offline demo (no ROS2 required)
├── demo_cam.py                          # ROS2 camera replay node (offline ROS2 demo)
├── demo_planner.py                      # ROS2 planner node (offline ROS2 demo)
├── demo_logger.py                       # ROS2 logger node (offline ROS2 demo)
├── test_detector.py                     # Model verification on test images
├── launch.sh                            # Full robot deployment launcher
└── demo_launch.sh                       # Offline ROS2 demo launcher
```

> **Note on model architecture:** The `.eim` files are compiled binaries. Use
> `turtle-tracking-linux-x86_64.eim` on a laptop or desktop (x86_64) and
> `turtle-tracking.eim` on the Raspberry Pi 5 (ARM64). They are functionally
> identical — only the target architecture differs.

---

## Data Availability

All raw experimental data are publicly available at:

**https://cwru.box.com/s/8zfq3esnuja9srlxorn7i5mo2dbmhnkp**

The repository is organized into four folders corresponding to the experimental
conditions reported in the paper:

| Folder | Description |
|---|---|
| `Tetherless_Tracking` | Untethered autonomous turtle-following trials |
| `Alumni_Pool_Obstacle_Avoidance` | Pool-based obstacle avoidance trials |
| `NEA_Tracking` | New England Aquarium Giant Ocean Tank tracking trials |
| `NEA_Obstacle_Avoidance` | New England Aquarium obstacle avoidance trials |

Each folder contains three subdirectories:

| Subfolder | Description |
|---|---|
| `Turtle_Sensor_Data/` | IMU readings, motor positions, and depth/altitude logs |
| `Time_Synced_Videos/` | Time-synchronized Turtle POV and/or GoPro footage |
| `Turtle_POV_Camera_Data/` | Raw stereo camera frames and detection outputs, with the following structure: |

`Turtle_POV_Camera_Data/` is organized as:

| File/Folder | Description |
|---|---|
| `left/` | Raw left camera frames (`.jpg`, one per frame) |
| `right/` | Raw right camera frames (`.jpg`, one per frame) |
| `stitched/` | Stitched stereo frames used as tracker input |
| `detection/` | Annotated frames with detection crosshair overlaid |
| `centroids.csv` | Per-frame centroid coordinates, confidence, and tracking status |
| `detection*.mp4` | Annotated output video |

The `left/` and `right/` folders from any trial can be used directly as input
to `demo_launch.sh` for offline ROS2 replay. Any trial `.mp4` from
`Time_Synced_Videos/` or `Turtle_POV_Camera_Data/` can be used as input to
`demo_offline.py`.

---

## System Requirements

### For the offline demo and model verification (no ROS2, no hardware)

Any x86_64 machine with ≥4 GB RAM running Linux or macOS. Windows users should use WSL2.

| Package | Version Tested | Notes |
|---|---|---|
| Python | 3.12.3 | |
| OpenCV | 4.12.0.88 | |
| NumPy | 2.2.6 | |
| Edge Impulse Linux SDK | 1.2.2 | |
| pyaudio | 0.2.14 | Required by Edge Impulse SDK |
| six | 1.17.0 | Required by Edge Impulse SDK |

### For live robot deployment (Raspberry Pi 5)

| Package | Version Tested | Notes |
|---|---|---|
| Python | 3.12.3 | System Python |
| ROS2 | Jazzy | Installed system-wide |
| OpenCV | 4.10.0.84 | |
| NumPy | 1.26.4 | |
| Edge Impulse Linux SDK | 1.2.1 | |
| PyYAML | 6.0.1 | |
| rclpy | 7.1.1 | Installed with ROS2 Jazzy |
| cv_bridge | 1.13.0 | Installed with ROS2 Jazzy |

**Deployment hardware (as in paper):** Raspberry Pi 5 (8 GB RAM), stereo camera pair (640×480 per eye), Dynamixel motor chain (10 motors, 1 Mbps), XIAO nRF52840 IMU/depth sensor.

**Operating systems tested:** Ubuntu 22.04 (development), Ubuntu 24.04 (Raspberry Pi 5 deployment).

---

## Installation

### Option A — Conda/micromamba environment (recommended for laptop/desktop)

This installs everything needed to run `demo_offline.py` and `test_detector.py`
without ROS2 or robot hardware.

**With micromamba:**
```bash
micromamba create -n crush python=3.12
micromamba activate crush
micromamba install -c conda-forge pyaudio
pip install opencv-python "numpy>=1.26,<3" six edge-impulse-linux
```

**With conda:**
```bash
conda create -n crush python=3.12
conda activate crush
conda install -c conda-forge pyaudio
pip install opencv-python "numpy>=1.26,<3" six edge-impulse-linux
```

> `pyaudio` and `six` are transitive dependencies of the Edge Impulse Linux SDK.
> Installing `pyaudio` via conda/micromamba avoids the need for system-level
> `portaudio` headers that `pip install pyaudio` requires.

Clone the repository and navigate to the working directory:
```bash
git clone https://github.com/zpatty/drl-turtle.git
cd drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware
```

Make the model files executable (required by the Edge Impulse SDK):
```bash
chmod +x turtle-tracking.eim
chmod +x turtle-tracking-linux-x86_64.eim
```

**Typical install time:** ~10 minutes on a standard desktop computer.

### Option B — System-wide install (Raspberry Pi / Ubuntu, for live deployment)

ROS2 Jazzy provides `rclpy` and `cv_bridge` system-wide. Install the remaining
Python dependencies with:

```bash
pip3 install opencv-python numpy pyyaml six edge-impulse-linux
```

Follow the [official ROS2 Jazzy installation guide](https://docs.ros.org/en/jazzy/Installation.html)
for ROS2 itself.

---

## Model Verification

Verify the FOMO detector is working correctly on the included test images.
Run this first to confirm the model and environment are set up correctly
before proceeding to the demo.

```bash
# On laptop/desktop (x86_64):
python3 test_detector.py --model ./turtle-tracking-linux-x86_64.eim

# On Raspberry Pi (ARM64):
python3 test_detector.py --model ./turtle-tracking.eim
```

**Expected terminal output:**
```
  ✅ turtle_front.png    — 1 detection(s)  [312 ms]
       Detection 1: centroid=(481.2, 241.0)  conf=0.874
  ✅ turtle_down.png     — 1 detection(s)  [308 ms]
  ...
```

To also save annotated images to `test_detector_output/`:
```bash
python3 test_detector.py --model ./turtle-tracking-linux-x86_64.eim --save-vis
```

If a detection is missing, the script reports the best raw confidence score below
threshold to help diagnose whether the issue is threshold, model, or preprocessing.

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `--model` | — | Path to `.eim` FOMO model (required) |
| `--images` | `template_frames/` | Folder of test images |
| `--confidence-threshold` | `0.5` | Minimum confidence to report |
| `--save-vis` | off | Save annotated images to `test_detector_output/` |

---

## Demo

Runs the FOMO tracker on a recorded trial video, producing annotated frames and
a centroid log. **No ROS2 or hardware required.**

Download any trial video from the data repository above, then run:

```bash
# Activate your environment first:
micromamba activate crush   # or: conda activate crush

python3 demo_offline.py \
    --model ./turtle-tracking-linux-x86_64.eim \
    --input /path/to/trial_video.mp4 \
    --confidence-threshold 0.7 \
    --save-data \
    --output-dir demo_output/
```

**Expected output** (written to `demo_output/YYYYMMDD_HHMMSS/`):

| File | Description |
|---|---|
| `centroids.csv` | Per-frame centroid coordinates, confidence, and tracking status |
| `detection/frame00001.jpg` … | Annotated frames with detection crosshair |
| `detection_output.mp4` | Annotated video |

**Expected columns in `centroids.csv`:**
`frame, timestamp, cx, cy, bbox_x, bbox_y, bbox_w, bbox_h, fps, confidence, tracking_status`

**Expected run time:** ~30 seconds for a 10-second clip on a standard desktop computer.

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `--model` | — | Path to `.eim` FOMO model (required) |
| `--input` | — | Input video file (required) |
| `--confidence-threshold` | `0.7` | Minimum detection confidence |
| `--save-data` | off | Save annotated frames, CSV, and output video |
| `--show-display` | off | Show live preview window while processing |
| `--output-dir` | `results/` | Output directory |

---

## Instructions for Use

### Live robot deployment (full stack)

The full autonomous deployment stack is launched via a single tmux script that
starts all seven ROS2 nodes concurrently on the robot:

```bash
bash launch.sh
```

| Node | Script | Role |
|---|---|---|
| EdgeImpulse_Tracker | `crush_tracker.py` | FOMO-based visual detection and centroid publishing |
| TurtleRobot | `TurtleRobot_other.py` | Motor interface |
| Camera | `Cam.py` | Stereo camera publishing |
| Sensors | `TurtleSensors.py` | IMU and depth sensing |
| Controller | `TurtleController.py` | Low-level motor control |
| Planner | `untethered_planning_node.py` | Autonomous dive planning |
| Logger | `logger.py` | Full sensor and control data logging |

**Session management:**
```bash
bash launch.sh --attach     # Reconnect to a running session
bash launch.sh --terminate  # Gracefully stop all nodes
bash launch.sh --kill       # Force kill the session
bash launch.sh --live       # Launch without writing log files
```

By default, each node's stdout and stderr are written to timestamped log files
under `logs/turtle_YYYYMMDD_HHMMSS/`.

---

## License

This software is released under the MIT License. See `LICENSE` for details.

---

## Citation

If you use this software, please cite:

```bibtex
@article{crush2026,
  title   = {Autonomous Sea Turtle Robot for Marine Fieldwork},
  author  = {Zach J. Patterson, Emily Sologuren, Levi Cai, Daniel Kim,
             Alaa Maalouf, Pascal Spino, Daniela Rus},
  year    = {2026},
  doi     = {https://doi.org/10.48550/arXiv.2602.21389}
}
```

---

## Contact

Emily Sologuren — esolo@mit.edu  
MIT Computer Science and Artificial Intelligence Laboratory (CSAIL)