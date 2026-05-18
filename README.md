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
crush/
├── models/
│   └── turtle-detector.eim       # Edge Impulse FOMO model
├── template_frames/              # Test images for model verification
│   ├── turtle_front.png
│   ├── turtle_down.png
│   ├── turtle_angle.png
│   ├── turtle_angle_2.png
│   └── turtle_angle_3.png
├── demo/
│   └── demo_clip.mp4             # Short demo video clip [see Data Availability]
├── crush_tracker.py              # ROS2 FOMO tracker node (live deployment)
├── untethered_planning_node.py   # Autonomous dive planner (live deployment)
├── demo_offline.py               # Standalone offline demo (no ROS2 required)
├── demo_cam.py                   # ROS2 camera replay node (offline ROS2 demo)
├── demo_planner.py               # ROS2 planner node (offline ROS2 demo)
├── demo_logger.py                # ROS2 logger node (offline ROS2 demo)
├── test_detector.py              # Model verification on test images
├── launch.sh                     # Full robot deployment launcher
└── demo_launch.sh                # Offline ROS2 demo launcher
```

---

## System Requirements

### For the offline demo and model verification (no ROS2, no hardware)

Any x86_64 or ARM64 machine with ≥4 GB RAM running Linux, macOS, or Windows (WSL2).

| Package | Version Tested |
|---|---|
| Python | 3.10–3.12 |
| OpenCV | 4.10.0 |
| NumPy | 1.26.4 |
| Edge Impulse Linux SDK | 1.2.1 |

### For live robot deployment

| Package | Version Tested | Notes |
|---|---|---|
| Python | 3.12.3 | System Python on Raspberry Pi 5 |
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
pip install opencv-python numpy edge-impulse-linux
```

**With conda:**
```bash
conda create -n crush python=3.12
conda activate crush
pip install opencv-python numpy edge-impulse-linux
```

Then clone the repository:
```bash
git clone https://github.com/zpatty/drl-turtle.git
cd drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware
```

**Typical install time:** ~10 minutes on a standard desktop computer.

### Option B — System-wide install (Raspberry Pi / Ubuntu, for live deployment)

ROS2 Jazzy provides `rclpy` and `cv_bridge` system-wide. Install the remaining
Python dependencies with:

```bash
pip3 install opencv-python numpy pyyaml edge-impulse-linux
```

Follow the [official ROS2 Jazzy installation guide](https://docs.ros.org/en/jazzy/Installation.html)
for ROS2 itself.

### Download model and demo data

```bash
# [TODO — final URL to be added prior to publication]
wget [DATA_REPOSITORY_URL]/models.zip
unzip models.zip

wget [DATA_REPOSITORY_URL]/demo.zip
unzip demo.zip
```

---

## Demo

Runs the FOMO tracker on a short recorded trial video, producing annotated
frames and a centroid log. **No ROS2 or hardware required.**

```bash
# Activate your environment first:
micromamba activate crush   # or: conda activate crush

python3 demo_offline.py \
    --model ./models/turtle-detector.eim \
    --input demo/demo_clip.mp4 \
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
| `--model` | `./models/turtle-detector.eim` | Path to `.eim` FOMO model |
| `--input` | — | Input video file (required) |
| `--confidence-threshold` | `0.7` | Minimum detection confidence |
| `--save-data` | off | Save annotated frames, CSV, and output video |
| `--show-display` | off | Show live preview window while processing |
| `--output-dir` | `results/` | Output directory |

---

## Model Verification

Verify the FOMO detector is working correctly on the included test images
before running the demo or deploying on the robot.

```bash
python3 test_detector.py \
    --model ./models/turtle-detector.eim \
    --images template_frames/ \
    --save-vis
```

**Expected terminal output:**
```
  ✅ turtle_front.png    — 1 detection(s)  [312 ms]
       Detection 1: centroid=(481.2, 241.0)  conf=0.874
  ✅ turtle_down.png     — 1 detection(s)  [308 ms]
  ...
```

Annotated images are saved to `test_detector_output/` when `--save-vis` is used.
If a detection is missing, the script reports the best raw confidence score below
threshold to help diagnose whether the issue is threshold, model, or preprocessing.

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `--model` | `./models/turtle-detector.eim` | Path to `.eim` FOMO model |
| `--images` | `template_frames/` | Folder of test images |
| `--confidence-threshold` | `0.5` | Minimum confidence to report |
| `--save-vis` | off | Save annotated images to `test_detector_output/` |

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

### Offline ROS2 replay (recorded trial data)

To replay a recorded trial through the full tracker and planner stack without
robot hardware, using the provided trial dataset:

```bash
bash demo_launch.sh --data-dir /path/to/trial_folder/
```

The trial folder must contain `left/` and `right/` subdirectories of stereo
frame images (the format produced by `launch.sh --save-data`). This launches
four ROS2 nodes — camera replay, tracker, planner, and logger — in a tmux session.

```bash
bash demo_launch.sh --attach                     # Reconnect to running session
bash demo_launch.sh --kill                       # Stop session
bash demo_launch.sh --data-dir /path/ \
    --output-dir my_results/ \
    --model ./models/turtle-detector.eim
```

### Tracker standalone (live camera, no full robot stack)

```bash
python3 crush_tracker.py \
    --model ./models/turtle-detector.eim \
    --camera-topic frames \
    --message-type turtlecam \
    --target-class turtle \
    --confidence-threshold 0.5 \
    --save-data \
    --output-dir results/
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `--model` | — | Path to `.eim` FOMO model file |
| `--confidence-threshold` | `0.5` | Minimum detection confidence to publish centroid |
| `--target-class` | `turtle` | Object class label to track |
| `--message-type` | `turtlecam` | Camera message type: `turtlecam`, `compressed`, or `image` |
| `--save-data` | off | Save annotated frames and CSV centroid log |
| `--show-display` | off | Enable live display window |

---

## Reproduction Instructions

To reproduce the quantitative tracking results reported in the paper:

```bash
# [TODO — dataset and analysis script to be deposited prior to publication]
wget [DATASET_URL]/aquarium_tracking_trials.zip
unzip aquarium_tracking_trials.zip

python3 scripts/reproduce_tracking_analysis.py \
    --data-dir aquarium_tracking_trials/ \
    --output results_paper/
```

This will regenerate the tracking performance metrics reported in the paper.
Expected run time: ~[X] hours on a standard desktop computer.

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